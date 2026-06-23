"""Step 11: Statistical Analysis - IPCW + Weighted Cox + Competing Risks
Implements primary analysis of the target trial emulation:
- IPCW weight estimation for per-protocol effect
- Weighted cause-specific Cox proportional hazards
- Cumulative incidence functions (Aalen-Johansen) for competing risks
- Cluster-robust standard errors at subject_id level
- 7 sensitivity analyses (S1-S7)
- Multivariable-adjusted Cox model
- Propensity score analysis
"""
import pandas as pd
import numpy as np
import sys, os, warnings
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from utils import parse_datetime, hours_between, log

warnings.filterwarnings("ignore")

# ── IPCW helpers ──────────────────────────────────────────────

def _prepare_baseline_for_ipcw(baseline_cov):
    """Select and prepare baseline covariates for IPCW logistic models."""
    cov_cols = [
        "episode_id", "landmark_hour",
        "age_years", "gender",
        "current_n_abx", "current_max_spectrum_score",
        "current_anti_mrsa",
        "current_carbapenem",
        "current_combination_therapy", "current_last_resort",
        "cumulative_systemic_abx_hours", "cumulative_broad_abx_hours",
        "any_culture_obtained_before_landmark",
        "culture_result_known_before_landmark",
        "positive_culture_known_before_landmark",
        "blood_culture_before_landmark",
        "mechanical_ventilation_at_landmark",
        "vasopressor_at_landmark",
        "wbc_max_24h", "neutrophil_max_24h",
        "crp_max_24h", "lactate_max_24h",
        "platelet_min_24h", "ph_min_24h",
        "base_excess_min_24h", "spo2_min_24h",
        "congenital_heart_disease",
        "malignancy", "hematologic_disease",
        "prematurity",
    ]
    available = [c for c in cov_cols if c in baseline_cov.columns]
    df = baseline_cov[available].copy()

    if "gender" in df.columns:
        df["gender_male"] = (df["gender"].astype(str).str.upper() == "M").astype(int)
        df = df.drop(columns=["gender"])

    for col in df.columns:
        if col in ["episode_id", "landmark_hour"]:
            continue
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
    return df


def estimate_ipcw(cloned, baseline_cov, logfile=None):
    """Estimate stabilized IPCW weights for per-protocol analysis."""
    from statsmodels.api import Logit

    bl = _prepare_baseline_for_ipcw(baseline_cov)
    merged = cloned.merge(bl, on=["episode_id", "landmark_hour"], how="left")
    merged["uncensored"] = 1 - merged["artificial_censored"].astype(int)

    pred_cols = [c for c in bl.columns
                 if c not in ["episode_id", "landmark_hour"]]
    merged["ipcw_weight"] = 1.0

    for arm in ["spectrum_reduction", "continue_broad"]:
        mask = merged["assigned_strategy"] == arm
        arm_data = merged[mask].copy()
        if len(arm_data) == 0:
            continue

        y = arm_data["uncensored"].values
        p_marginal = y.mean()
        if p_marginal == 0 or p_marginal == 1:
            merged.loc[mask, "ipcw_weight"] = 1.0
            continue

        X_cols = [c for c in pred_cols if c in arm_data.columns
                  and arm_data[c].nunique() > 1]
        if len(X_cols) == 0:
            merged.loc[mask, "ipcw_weight"] = 1.0
            continue

        X = arm_data[X_cols].astype(float)
        X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)

        try:
            model = Logit(y, X).fit(disp=0, maxiter=200)
            p_cond = model.predict(X)
            p_cond = np.clip(p_cond, 0.01, 0.99)
            w = p_marginal / p_cond
            q99 = np.percentile(w, 99)
            w = np.clip(w, 0, q99)
            merged.loc[mask, "ipcw_weight"] = w
            if logfile:
                log(f"  IPCW [{arm}]: P(uncensored)={p_marginal:.3f}, "
                    f"weight mean={w.mean():.2f} sd={w.std():.2f} "
                    f"range=[{w.min():.2f}, {w.max():.2f}]", logfile)
        except Exception as e:
            if logfile:
                log(f"  IPCW [{arm}]: model failed ({e}), using unweighted", logfile)
            merged.loc[mask, "ipcw_weight"] = 1.0

    return merged


# ── Primary analysis ──────────────────────────────────────────

def run_weighted_cox(cloned, logfile=None, label="Primary"):
    """Weighted cause-specific Cox PH with cluster-robust SE."""
    from lifelines import CoxPHFitter

    df = cloned.copy()
    fup_start = parse_datetime(df["followup_start_time"])
    fup_end = parse_datetime(df["followup_end_time"])
    max_fup_hours = (fup_end - fup_start).dt.total_seconds() / 3600.0

    event_time = np.where(
        df["event_status"] == 1,
        df["outcome_hour_from_landmark"].values,
        np.where(
            df["event_status"] == 2,
            np.nan,
            max_fup_hours.values
        )
    )

    mask_death = df["event_status"] == 2
    if mask_death.any():
        death_time_series = parse_datetime(df.loc[mask_death, "death_time"].astype(str))
        lm_series = parse_datetime(df.loc[mask_death, "followup_start_time"].astype(str))
        death_hours = (death_time_series - lm_series).dt.total_seconds() / 3600.0
        event_time[mask_death.values] = death_hours.values

    event_time = np.maximum(event_time, 0.1)
    event_observed = (df["event_status"] == 1).astype(int).values
    treatment = (df["assigned_strategy"] == "spectrum_reduction").astype(int)
    weights = df["ipcw_weight"].values
    subject_id = df["subject_id"].values

    cox_df = pd.DataFrame({
        "time": event_time,
        "event": event_observed,
        "treatment": treatment,
        "weight": weights,
        "subject_id": subject_id,
    })

    cph = CoxPHFitter()
    fit_df = cox_df[["time", "event", "treatment", "weight", "subject_id"]].copy()
    try:
        cph.fit(fit_df, duration_col="time", event_col="event",
                weights_col="weight", cluster_col="subject_id", robust=True)
        if logfile:
            log("", logfile)
            log(f"  === {label}: Weighted Cause-Specific Cox PH ===", logfile)
            log(f"  Outcome events: {event_observed.sum()}", logfile)
            log(f"  Deaths (competing): {mask_death.sum()}", logfile)
            log(f"  Censored: {(event_observed == 0).sum() - mask_death.sum()}", logfile)
            log("", logfile)
            log(cph.summary.to_string(), logfile)
            log("", logfile)

        hr = cph.summary.loc["treatment", "exp(coef)"]
        ci_lower = cph.summary.loc["treatment", "exp(coef) lower 95%"]
        ci_upper = cph.summary.loc["treatment", "exp(coef) upper 95%"]
        p_value = cph.summary.loc["treatment", "p"]
        if logfile:
            log(f"  HR (reduction vs continue): {hr:.3f} "
                f"(95% CI {ci_lower:.3f}-{ci_upper:.3f}), P={p_value:.4f}", logfile)
    except Exception as e:
        if logfile:
            log(f"  [{label}] Cox model failed: {e}", logfile)
        hr, ci_lower, ci_upper, p_value = np.nan, np.nan, np.nan, np.nan

    return hr, ci_lower, ci_upper, p_value


def run_aalen_johansen(cloned, logfile=None):
    """Aalen-Johansen estimator for cumulative incidence (competing risks)."""
    from lifelines import AalenJohansenFitter

    log("", logfile)
    log("  === Cumulative Incidence: Aalen-Johansen Estimator ===", logfile)

    results = {}
    for arm in ["spectrum_reduction", "continue_broad"]:
        arm_df = cloned[cloned["assigned_strategy"] == arm].copy()
        if len(arm_df) == 0:
            continue

        fup_start = parse_datetime(arm_df["followup_start_time"])
        fup_end = parse_datetime(arm_df["followup_end_time"])
        max_hours = (fup_end - fup_start).dt.total_seconds() / 3600.0

        event_type = arm_df["event_status"].astype(int).values.copy()
        time_arr = np.zeros(len(arm_df))

        oc_mask = arm_df["outcome_occurred"] == 1
        if oc_mask.any():
            time_arr[oc_mask.values] = arm_df.loc[oc_mask, "outcome_hour_from_landmark"].values

        death_mask = arm_df["event_status"] == 2
        if death_mask.any():
            dtime = parse_datetime(arm_df.loc[death_mask, "death_time"].astype(str))
            ltime = parse_datetime(arm_df.loc[death_mask, "followup_start_time"].astype(str))
            time_arr[death_mask.values] = ((dtime - ltime).dt.total_seconds() / 3600.0).values

        cens_mask = (event_type == 0) & (arm_df["event_status"] != 2)
        if cens_mask.any():
            time_arr[cens_mask.values] = max_hours[cens_mask.values].values

        time_arr = np.maximum(time_arr, 0.1)
        time_days = time_arr / 24.0
        weights = arm_df["ipcw_weight"].values

        try:
            aj = AalenJohansenFitter(calculate_variance=False)
            aj.fit(durations=time_days, event_observed=event_type,
                   event_of_interest=1, weights=weights, label=arm)
            cif = aj.cumulative_density_
            times = cif.index.values
            mask = times <= 28
            ci_28d = cif.values[mask][-1][0] if mask.any() else 0.0
            results[arm] = {"aj": aj, "ci_28d": ci_28d}
            log(f"  [{arm}] 28-day cumulative incidence: {ci_28d*100:.2f}%", logfile)
        except Exception as e:
            log(f"  [{arm}] AJ estimator failed: {e}", logfile)
            results[arm] = None

    if results.get("spectrum_reduction") and results.get("continue_broad"):
        rd = results["spectrum_reduction"]["ci_28d"] - results["continue_broad"]["ci_28d"]
        log(f"  Risk difference at 28d: {rd*100:.2f} percentage points", logfile)
        log("  (Negative = reduction arm has lower incidence)", logfile)

    return results


def run_weighted_km(cloned, logfile=None):
    """Weighted Kaplan-Meier survival curves by treatment arm."""
    from lifelines import KaplanMeierFitter

    log("", logfile)
    log("  === Weighted Kaplan-Meier (Cause-Specific) ===", logfile)

    km_results = {}
    for arm in ["spectrum_reduction", "continue_broad"]:
        arm_df = cloned[cloned["assigned_strategy"] == arm].copy()
        if len(arm_df) == 0:
            continue

        fup_start = parse_datetime(arm_df["followup_start_time"])
        fup_end = parse_datetime(arm_df["followup_end_time"])
        max_hours = (fup_end - fup_start).dt.total_seconds() / 3600.0

        time_arr = np.zeros(len(arm_df))
        event_obs = np.zeros(len(arm_df))

        oc_mask = arm_df["outcome_occurred"] == 1
        if oc_mask.any():
            time_arr[oc_mask.values] = arm_df.loc[oc_mask, "outcome_hour_from_landmark"].values
            event_obs[oc_mask.values] = 1

        death_mask = arm_df["event_status"] == 2
        if death_mask.any():
            dtime = parse_datetime(arm_df.loc[death_mask, "death_time"].astype(str))
            ltime = parse_datetime(arm_df.loc[death_mask, "followup_start_time"].astype(str))
            time_arr[death_mask.values] = ((dtime - ltime).dt.total_seconds() / 3600.0).values
            event_obs[death_mask.values] = 0

        cens_mask = (arm_df["event_status"] == 0)
        if cens_mask.any():
            time_arr[cens_mask.values] = max_hours[cens_mask.values].values

        time_arr = np.maximum(time_arr, 0.1)
        time_days = time_arr / 24.0
        weights = arm_df["ipcw_weight"].values

        try:
            kmf = KaplanMeierFitter()
            kmf.fit(durations=time_days, event_observed=event_obs,
                    weights=weights, label=arm)
            s_28d = kmf.survival_function_at_times([28]).values
            s_28d = s_28d[0] if len(s_28d) > 0 else np.nan
            km_results[arm] = {"kmf": kmf, "survival_28d": s_28d}
            log(f"  [{arm}] 28-day event-free survival: {s_28d*100:.1f}%", logfile)
        except Exception as e:
            log(f"  [{arm}] KM failed: {e}", logfile)
            km_results[arm] = None

    return km_results


def run_baseline_table(cloned, logfile=None):
    """Baseline characteristics by treatment arm (Table 1)."""
    log("", logfile)
    log("  === Baseline Characteristics by Assigned Strategy ===", logfile)

    a = cloned[cloned["assigned_strategy"] == "spectrum_reduction"]
    b = cloned[cloned["assigned_strategy"] == "continue_broad"]

    log(f"  {'Characteristic':<40} {'Reduction':>12} {'Continue':>12} {'Total':>12}", logfile)
    log(f"  {'':->40} {'':->12} {'':->12} {'':->12}", logfile)
    log(f"  {'N':<40} {len(a):>12,} {len(b):>12,} {len(cloned):>12,}", logfile)
    log(f"  {'Outcomes, n (%)':<40} "
        f"{(a['outcome_occurred']==1).sum():>5} ({((a['outcome_occurred']==1).sum()/len(a)*100):>.1f}%)"
        f"{(b['outcome_occurred']==1).sum():>7} ({((b['outcome_occurred']==1).sum()/len(b)*100):>.1f}%)"
        f"{(cloned['outcome_occurred']==1).sum():>7} ({((cloned['outcome_occurred']==1).sum()/len(cloned)*100):>.1f}%)",
        logfile)
    log(f"  {'Deaths, n (%)':<40} "
        f"{(a['event_status']==2).sum():>5} ({((a['event_status']==2).sum()/len(a)*100):>.1f}%)"
        f"{(b['event_status']==2).sum():>7} ({((b['event_status']==2).sum()/len(b)*100):>.1f}%)"
        f"{(cloned['event_status']==2).sum():>7} ({((cloned['event_status']==2).sum()/len(cloned)*100):>.1f}%)",
        logfile)
    log(f"  {'CONS outcomes, n':<40} "
        f"{(a['cons_outcome']==1).sum():>12,} "
        f"{(b['cons_outcome']==1).sum():>12,} "
        f"{(cloned['cons_outcome']==1).sum():>12,}", logfile)
    log(f"  {'MDRO outcomes, n':<40} "
        f"{(a['mdro_outcome']==1).sum():>12,} "
        f"{(b['mdro_outcome']==1).sum():>12,} "
        f"{(cloned['mdro_outcome']==1).sum():>12,}", logfile)
    log(f"  {'Artificial censored, n (%)':<40} "
        f"{(a['artificial_censored']==1).sum():>5} ({((a['artificial_censored']==1).sum()/len(a)*100):>.1f}%)"
        f"{(b['artificial_censored']==1).sum():>7} ({((b['artificial_censored']==1).sum()/len(b)*100):>.1f}%)"
        f"{(cloned['artificial_censored']==1).sum():>7} ({((cloned['artificial_censored']==1).sum()/len(cloned)*100):>.1f}%)",
        logfile)


# ── Data merging helpers for sensitivity analyses ─────────────

def _merge_eligible_flags(cloned):
    """Merge c7_source, unrecognized_baseline_resistant_isolate, date_only_excluded
    from eligible trials data into cloned trials."""
    eli = pd.read_csv(LANDMARK_ELIGIBLE_CSV, low_memory=False)
    eli = eli[eli["is_eligible"] == 1].copy()
    flag_cols = ["episode_id", "landmark_hour"]
    for c in ["c7_source", "unrecognized_baseline_resistant_isolate", "date_only_excluded"]:
        if c in eli.columns:
            flag_cols.append(c)
    eli_flags = eli[flag_cols].copy()
    for col in eli_flags.columns:
        if col not in ["episode_id", "landmark_hour"]:
            eli_flags[col] = eli_flags[col].astype(str)
    eli_flags["episode_id"] = eli_flags["episode_id"].astype(str)
    eli_flags["landmark_hour"] = eli_flags["landmark_hour"].astype(int)
    cloned["episode_id"] = cloned["episode_id"].astype(str)
    cloned["landmark_hour"] = cloned["landmark_hour"].astype(int)
    return cloned.merge(eli_flags, on=["episode_id", "landmark_hour"], how="left")


def _merge_episode_sequence(cloned):
    """Merge episode_sequence from icu_base into cloned trials."""
    icu = pd.read_csv(ICU_BASE_CSV, low_memory=False)
    icu["episode_id"] = icu["episode_id"].astype(str)
    cloned["episode_id"] = cloned["episode_id"].astype(str)
    return cloned.merge(icu[["episode_id", "episode_sequence"]], on="episode_id", how="left")


def _get_mar_absent_episodes():
    """Return set of episode_ids where ALL abx orders lack MAR data (mar_data_absent_flag=1).

    Note: In the current pipeline, MAR data (L1 activity hierarchy) is not processed,
    so mar_data_absent_flag=1 for all orders. This makes S2 exclude all trials.
    We define 'mar_data_absent' at the episode level: an episode is flagged if
    ALL its systemic broad-spectrum abx orders have mar_data_absent_flag=1.
    """
    abx = pd.read_csv(ABX_ORDERS_CLEAN_CSV, low_memory=False)
    abx["episode_id"] = abx["episode_id"].astype(str)
    # Only consider systemic broad-spectrum orders (relevant for continue_broad judgment)
    broad_systemic = abx[(abx["is_systemic"] == 1) & (abx["is_broad_spectrum"] == 1)]
    # An episode lacks MAR data if ALL its orders have mar_data_absent_flag=1
    by_ep = broad_systemic.groupby("episode_id")["mar_data_absent_flag"].agg(
        ["mean", "count", "sum"])
    # Episode flagged if mean == 1 (all orders flagged)
    flagged = set(by_ep[by_ep["mean"] == 1.0].index)
    return flagged


# ── Helper for single sensitivity Cox ─────────────────────────

def _run_single_sensitivity(cloned, baseline_cov, subset_mask, label, logfile=None):
    """Run IPCW+Cox on a subset of clones defined by subset_mask.

    Parameters:
        cloned: full cloned DataFrame with ipcw_weight
        baseline_cov: baseline covariates
        subset_mask: boolean mask (same length as cloned) for rows to KEEP
        label: label for logging
        logfile: log file path

    Returns: dict with description, n_excluded, HR, CI, P
    """
    subset = cloned[subset_mask].copy()
    n_excluded = len(cloned) - len(subset)
    if len(subset) == 0:
        if logfile:
            log(f"  [{label}] All trials excluded (n={n_excluded}), skipping", logfile)
        return {"description": label, "n_excluded": n_excluded,
                "HR": np.nan, "CI_lower": np.nan, "CI_upper": np.nan, "P": np.nan}

    # Re-estimate IPCW on the subset
    subset_w = estimate_ipcw(subset, baseline_cov, None)
    hr, lo, hi, p = run_weighted_cox(subset_w, None, label=label)

    if logfile:
        log(f"  [{label}] n={len(subset)}, excluded={n_excluded}, "
            f"HR={hr:.3f} (95%CI {lo:.3f}-{hi:.3f}), P={p:.4f}", logfile)

    return {"description": label, "n_excluded": n_excluded,
            "HR": hr, "CI_lower": lo, "CI_upper": hi, "P": p}


# ── All sensitivity analyses ──────────────────────────────────

def run_all_sensitivity_analyses(cloned_w, baseline_cov, logfile=None):
    """Run all 7 pre-specified sensitivity analyses."""
    log("", logfile)
    log("  " + "=" * 70, logfile)
    log("  SENSITIVITY ANALYSES (S1-S7)", logfile)
    log("  " + "=" * 70, logfile)

    analyses = {}
    n_total = len(cloned_w)

    # ── S1: Exclude CONS outcomes ──
    log("", logfile)
    log("  --- S1: Exclude CONS (Coagulase-Negative Staphylococci) Outcomes ---", logfile)
    s1 = cloned_w.copy()
    s1_outcome_mask = s1["outcome_occurred"] == 1
    n_cons_outcomes = (s1_outcome_mask & (s1["cons_outcome"] == 1)).sum()
    s1.loc[s1_outcome_mask & (s1["cons_outcome"] == 1), "event_status"] = 0
    s1.loc[s1_outcome_mask & (s1["cons_outcome"] == 1), "outcome_occurred"] = 0
    hr1, lo1, hi1, p1 = run_weighted_cox(s1, None, label="S1_Exclude_CONS")
    analyses["S1_Exclude_CONS"] = {
        "description": f"Exclude CONS outcomes ({n_cons_outcomes} outcomes reclassified)",
        "HR": hr1, "CI_lower": lo1, "CI_upper": hi1, "P": p1,
    }
    log(f"  [S1_Exclude_CONS] {n_cons_outcomes} CONS outcomes → censored, "
        f"HR={hr1:.3f} (95%CI {lo1:.3f}-{hi1:.3f}), P={p1:.4f}", logfile)

    # ── S2: Exclude mar_data_absent_flag trials ──
    log("", logfile)
    log("  --- S2: Exclude Trials Without MAR Verification Data ---", logfile)
    mar_eps = _get_mar_absent_episodes()
    s2_keep = ~cloned_w["episode_id"].astype(str).isin(mar_eps)
    n_s2_excluded = (~s2_keep).sum()
    log(f"  [S2] Episodes with all-abx mar_data_absent: {len(mar_eps)}", logfile)
    log(f"  [S2] Clones excluded: {n_s2_excluded} / {n_total}", logfile)
    if n_s2_excluded < n_total:
        s2_result = _run_single_sensitivity(cloned_w, baseline_cov, s2_keep,
                                            "S2_Exclude_MAR_Absent", logfile)
        analyses["S2_Exclude_MAR_Absent"] = s2_result
    else:
        log(f"  [S2_Exclude_MAR_Absent] NOT APPLICABLE: all {n_total} clones excluded "
            f"(all trials lack MAR verification — pipeline limitation)", logfile)
        analyses["S2_Exclude_MAR_Absent"] = {
            "description": "Exclude MAR-absent trials (N/A: all excluded)",
            "HR": np.nan, "CI_lower": np.nan, "CI_upper": np.nan, "P": np.nan,
        }

    # ── S3: Extend Allowable Gap to 12h ──
    log("", logfile)
    log("  --- S3: Extend Allowable Gap 6h → 12h ---", logfile)
    s3_result = _run_s3_gap12(baseline_cov, logfile)
    analyses["S3_Gap12h"] = s3_result

    # ── S4: Exclude date_only trials ──
    log("", logfile)
    log("  --- S4: Exclude date_only Trials ---", logfile)
    merged = _merge_eligible_flags(cloned_w)
    n_date_only = (merged["date_only_excluded"].astype(str) == "1").sum()
    log(f"  [S4] date_only_excluded clones: {n_date_only}", logfile)
    if n_date_only > 0:
        s4_keep = merged["date_only_excluded"].astype(str) != "1"
        s4_result = _run_single_sensitivity(cloned_w, baseline_cov, s4_keep.values,
                                            "S4_Exclude_DateOnly", logfile)
        analyses["S4_Exclude_DateOnly"] = s4_result
    else:
        log(f"  [S4_Exclude_DateOnly] No date_only trials exist → identical to primary", logfile)
        analyses["S4_Exclude_DateOnly"] = {
            "description": "Exclude date_only trials (0 excluded — no date_only data)",
            "HR": np.nan, "CI_lower": np.nan, "CI_upper": np.nan, "P": np.nan,
        }

    # ── S5: Exclude unrecognized_baseline_resistant_isolate ──
    log("", logfile)
    log("  --- S5: Exclude Trials With Unrecognized Baseline Resistance ---", logfile)
    if "unrecognized_baseline_resistant_isolate" not in merged.columns:
        merged = _merge_eligible_flags(cloned_w)
    s5_flagged = merged["unrecognized_baseline_resistant_isolate"].astype(str) == "1"
    n_s5_flagged = s5_flagged.sum()
    log(f"  [S5] Clones with unrecognized baseline resistant isolate: {n_s5_flagged}", logfile)
    s5_keep = ~s5_flagged
    s5_result = _run_single_sensitivity(cloned_w, baseline_cov, s5_keep.values,
                                        "S5_Exclude_UnrecognizedBaseline", logfile)
    analyses["S5_Exclude_UnrecognizedBaseline"] = s5_result

    # ── S6: Restrict to first ICU episode ──
    log("", logfile)
    log("  --- S6: Restrict to First ICU Episode Only ---", logfile)
    cloned_w_seq = _merge_episode_sequence(cloned_w)
    s6_keep = cloned_w_seq["episode_sequence"] == 1
    n_s6_excluded = (~s6_keep).sum()
    log(f"  [S6] Clones excluded (episode_sequence > 1): {n_s6_excluded}", logfile)
    s6_result = _run_single_sensitivity(cloned_w, baseline_cov, s6_keep.values,
                                        "S6_FirstEpisodeOnly", logfile)
    analyses["S6_FirstEpisodeOnly"] = s6_result

    # ── S7: Exclude ICD-only infection context ──
    log("", logfile)
    log("  --- S7: Exclude ICD-Only Infection Context Trials ---", logfile)
    if "c7_source" not in merged.columns:
        merged = _merge_eligible_flags(cloned_w)
    s7_flagged = merged["c7_source"] == "icd_only"
    n_s7_flagged = s7_flagged.sum()
    log(f"  [S7] Clones with ICD-only infection context: {n_s7_flagged}", logfile)
    s7_keep = ~s7_flagged
    s7_result = _run_single_sensitivity(cloned_w, baseline_cov, s7_keep.values,
                                        "S7_Exclude_ICD_Only", logfile)
    analyses["S7_Exclude_ICD_Only"] = s7_result

    return analyses


def _run_s3_gap12(baseline_cov, logfile=None):
    """S3: Recompute continue_broad with allowable gap = 12h, then re-analyze.

    Changes ALLOWABLE_GAP_HOURS from 6 to 12, re-runs Steps 8-10,
    then estimates IPCW and runs weighted Cox.
    """
    import config
    from step8_reduction import run_step8
    from step9_clone import run_step9
    from step10_outcome import run_step10

    original_gap = config.ALLOWABLE_GAP_HOURS
    config.ALLOWABLE_GAP_HOURS = 12

    # Backup main output files (step8/9/10 import paths at module load,
    # so they write to the original paths regardless of config changes)
    import tempfile, shutil
    red_backup = config.SPECTRUM_REDUCTION_CSV + ".s3_backup"
    cloned_backup = config.CLONED_TRIALS_CSV + ".s3_backup"
    shutil.copy2(config.SPECTRUM_REDUCTION_CSV, red_backup)
    shutil.copy2(config.CLONED_TRIALS_CSV, cloned_backup)

    try:
        # Load required input data
        eligible = pd.read_csv(LANDMARK_ELIGIBLE_CSV, low_memory=False)
        for col in ["subject_id", "hadm_id", "episode_id"]:
            eligible[col] = eligible[col].astype(str)
        eligible = eligible[eligible["is_eligible"] == 1].copy()
        # Parse datetime columns required by step8
        for dt_col in ["landmark_time", "episode_intime", "episode_outtime"]:
            if dt_col in eligible.columns:
                eligible[dt_col] = parse_datetime(eligible[dt_col])

        icu_base = pd.read_csv(ICU_BASE_CSV, low_memory=False)
        for col in ["subject_id", "hadm_id", "episode_id"]:
            icu_base[col] = icu_base[col].astype(str)
        for dt_col in ["episode_intime", "episode_outtime", "death_time"]:
            if dt_col in icu_base.columns:
                icu_base[dt_col] = parse_datetime(icu_base[dt_col])

        abx_orders = pd.read_csv(ABX_ORDERS_CLEAN_CSV, low_memory=False)
        for col in ["subject_id", "hadm_id", "episode_id"]:
            abx_orders[col] = abx_orders[col].astype(str)
        if "start_time" in abx_orders.columns:
            abx_orders["start_time"] = parse_datetime(abx_orders["start_time"])
        if "end_time" in abx_orders.columns:
            abx_orders["end_time"] = parse_datetime(abx_orders["end_time"])

        micro_isolates = None
        if os.path.exists(MICRO_ISOLATES_CLEAN_CSV):
            micro_isolates = pd.read_csv(MICRO_ISOLATES_CLEAN_CSV, low_memory=False)
            for col in ["subject_id", "hadm_id", "episode_id"]:
                micro_isolates[col] = micro_isolates[col].astype(str)
            for dt_col in ["charttime", "culture_time", "imputed_report_time",
                          "specimen_collection_time"]:
                if dt_col in micro_isolates.columns:
                    micro_isolates[dt_col] = parse_datetime(micro_isolates[dt_col])

        # Re-run step 8 with gap=12h
        log("  [S3] Re-running Step 8 with ALLOWABLE_GAP_HOURS=12...", logfile)
        reduction_s3 = run_step8(None, eligible_trials=eligible,
                                 baseline_cov=baseline_cov,
                                 abx_orders=abx_orders, icu_base=icu_base)

        # Re-run step 9 (clone-censor)
        log("  [S3] Re-running Step 9 (clone-censor)...", logfile)
        cloned_s3 = run_step9(None, reduction_events=reduction_s3,
                              eligible_trials=eligible,
                              micro_isolates=micro_isolates, icu_base=icu_base)

        # Re-run step 10 (outcomes)
        log("  [S3] Re-running Step 10 (outcome determination)...", logfile)
        cloned_s3 = run_step10(None, cloned_trials=cloned_s3,
                               micro_isolates=micro_isolates, icu_base=icu_base)

        # Count reclassifications
        n_reduction = (reduction_s3["reduction_occurred"] == 1).sum()
        n_continue = (reduction_s3["continue_broad_48h"] == 1).sum()
        n_escalation = (reduction_s3["escalation_occurred"] == 1).sum()
        n_death = (reduction_s3["death_in_grace_period"] == 1).sum()
        log(f"  [S3] Gap=12h classification: reduction={n_reduction}, "
            f"continue={n_continue}, escalation={n_escalation}, death={n_death}", logfile)

        # Compare with original (read from backup, since run_step8 overwrote SPECTRUM_REDUCTION_CSV)
        orig_reduction = pd.read_csv(red_backup, low_memory=False)
        n_reclassify = ((orig_reduction["continue_broad_48h"] != reduction_s3["continue_broad_48h"]) |
                        (orig_reduction["escalation_occurred"] != reduction_s3["escalation_occurred"])).sum()
        log(f"  [S3] Trials reclassified: {n_reclassify} / {len(reduction_s3)}", logfile)

        # Estimate IPCW and run Cox
        log("  [S3] Estimating IPCW weights (gap=12h)...", logfile)
        cloned_s3_w = estimate_ipcw(cloned_s3, baseline_cov, None)
        hr, lo, hi, p = run_weighted_cox(cloned_s3_w, None, label="S3_Gap12h")

        log(f"  [S3_Gap12h] HR={hr:.3f} (95%CI {lo:.3f}-{hi:.3f}), P={p:.4f}", logfile)

        result = {
            "description": f"Gap=12h ({n_reclassify} trials reclassified)",
            "HR": hr, "CI_lower": lo, "CI_upper": hi, "P": p,
            "n_reclassified": n_reclassify,
            "n_reduction": n_reduction, "n_continue": n_continue,
            "n_escalation": n_escalation, "n_death": n_death,
        }

        return result

    except Exception as e:
        log(f"  [S3_Gap12h] FAILED: {e}", logfile)
        import traceback
        log(traceback.format_exc(), logfile)
        return {
            "description": "Gap=12h (FAILED)",
            "HR": np.nan, "CI_lower": np.nan, "CI_upper": np.nan, "P": np.nan,
        }
    finally:
        config.ALLOWABLE_GAP_HOURS = original_gap
        # Restore original output files
        shutil.copy2(red_backup, config.SPECTRUM_REDUCTION_CSV)
        shutil.copy2(cloned_backup, config.CLONED_TRIALS_CSV)
        os.remove(red_backup)
        os.remove(cloned_backup)


# ── Multivariable-Adjusted Cox ─────────────────────────────────

def run_adjusted_cox(cloned_w, baseline_cov, logfile=None):
    """Multivariable-adjusted Cox: treatment + baseline covariates.

    Tries covariate sets hierarchically from most adjusted to simplest.
    Excludes near-zero-variance covariates (current_anti_pseudomonal 99.4%,
    current_anti_anaerobe 99.6%) and collinear pairs (current_combination_therapy
    with current_n_abx, r=0.86). Missing values imputed to median (not zero).
    """
    from lifelines import CoxPHFitter

    log("", logfile)
    log("  " + "=" * 70, logfile)
    log("  MULTIVARIABLE-ADJUSTED COX MODEL", logfile)
    log("  " + "=" * 70, logfile)

    # ── Covariate groups ──
    demo_covs = ["age_years"]

    abx_covs = [
        "current_n_abx", "current_max_spectrum_score",
        "current_anti_mrsa", "current_carbapenem",
        "cumulative_systemic_abx_hours", "cumulative_broad_abx_hours",
    ]

    organ_covs = ["mechanical_ventilation_at_landmark", "vasopressor_at_landmark"]

    micro_covs = [
        "culture_result_known_before_landmark", "positive_culture_known_before_landmark",
    ]

    comorb_covs = ["congenital_heart_disease", "malignancy", "hematologic_disease", "prematurity"]

    lab_covs = [
        "wbc_max_24h", "neutrophil_max_24h", "crp_max_24h", "lactate_max_24h",
        "platelet_min_24h", "ph_min_24h", "base_excess_min_24h", "spo2_min_24h",
    ]

    # All covariates (demographics separated for gender handling)
    all_adj_covariates = (demo_covs + abx_covs + organ_covs + micro_covs +
                          comorb_covs + lab_covs)

    # ── Merge baseline covariates ──
    bl = baseline_cov.copy()
    for col in ["subject_id", "hadm_id", "episode_id"]:
        bl[col] = bl[col].astype(str)

    available_covs = ["episode_id", "landmark_hour", "gender"] + \
                     [c for c in all_adj_covariates if c in bl.columns]
    bl_sub = bl[available_covs].copy()

    # Encode gender
    if "gender" in bl_sub.columns:
        bl_sub["gender_male"] = (bl_sub["gender"].astype(str).str.upper() == "M").astype(int)
        bl_sub = bl_sub.drop(columns=["gender"])

    # Ensure numeric types
    for c in all_adj_covariates + ["gender_male"]:
        if c in bl_sub.columns and bl_sub[c].dtype == object:
            bl_sub[c] = pd.to_numeric(bl_sub[c], errors="coerce")

    # Merge into cloned data
    cloned_w["episode_id"] = cloned_w["episode_id"].astype(str)
    cloned_w["landmark_hour"] = cloned_w["landmark_hour"].astype(int)
    bl_sub["episode_id"] = bl_sub["episode_id"].astype(str)
    bl_sub["landmark_hour"] = bl_sub["landmark_hour"].astype(int)

    overlap_cols = [c for c in bl_sub.columns
                    if c in cloned_w.columns
                    and c not in ["episode_id", "landmark_hour"]]
    cloned_clean = cloned_w.drop(columns=overlap_cols, errors="ignore")
    merged = cloned_clean.merge(bl_sub, on=["episode_id", "landmark_hour"], how="left")

    # ── Build time-to-event ──
    fup_start = parse_datetime(merged["followup_start_time"])
    fup_end = parse_datetime(merged["followup_end_time"])
    max_fup_hours = (fup_end - fup_start).dt.total_seconds() / 3600.0

    event_time = np.where(
        merged["event_status"] == 1,
        merged["outcome_hour_from_landmark"].values,
        np.where(merged["event_status"] == 2, np.nan, max_fup_hours.values)
    )
    mask_death = merged["event_status"] == 2
    if mask_death.any():
        dtime = parse_datetime(merged.loc[mask_death, "death_time"].astype(str))
        ltime = parse_datetime(merged.loc[mask_death, "followup_start_time"].astype(str))
        event_time[mask_death.values] = ((dtime - ltime).dt.total_seconds() / 3600.0).values

    event_time = np.maximum(event_time, 0.1)
    event_observed = (merged["event_status"] == 1).astype(int).values
    treatment = (merged["assigned_strategy"] == "spectrum_reduction").astype(int)
    weights = merged["ipcw_weight"].values
    subject_id = merged["subject_id"].values

    # ── Build Cox DataFrame ──
    cox_df = pd.DataFrame({
        "time": event_time, "event": event_observed,
        "treatment": treatment, "weight": weights,
        "subject_id": subject_id,
    })

    # Add standardized covariates (median imputation for missing, skip zero-variance)
    all_covs_with_gender = all_adj_covariates + ["gender_male"]
    covs_in_model = []
    for c in all_covs_with_gender:
        if c not in merged.columns:
            continue
        vals = pd.to_numeric(merged[c], errors="coerce")
        n_miss = vals.isna().sum()
        if n_miss > 0:
            vals = vals.fillna(vals.median())
        std = np.std(vals)
        if std < 1e-8:
            continue
        vals = (vals - np.mean(vals)) / std
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        cox_df[c] = vals
        covs_in_model.append(c)

    log(f"  Covariates after zero-variance filter: {len(covs_in_model)}", logfile)

    adj_result = None

    # Single adjusted model: all valid covariates
    covs_use = [c for c in covs_in_model]
    model_name = f"Expanded adjusted Cox ({len(covs_use)} covariates)"

    try:
        cph = CoxPHFitter()
        cols_use = ["time", "event", "treatment", "subject_id"] + covs_use
        mdf = cox_df[cols_use].copy().dropna()

        cph.fit(mdf, duration_col="time", event_col="event",
                cluster_col="subject_id", robust=True)

        hr = cph.summary.loc["treatment", "exp(coef)"]
        ci_lower = cph.summary.loc["treatment", "exp(coef) lower 95%"]
        ci_upper = cph.summary.loc["treatment", "exp(coef) upper 95%"]
        p_value = cph.summary.loc["treatment", "p"]

        n_cov = len(covs_use)
        log(f"  [{model_name}] HR={hr:.3f} "
            f"(95%CI {ci_lower:.3f}-{ci_upper:.3f}), P={p_value:.4f}, n_cov={n_cov}", logfile)

        adj_result = {"HR": hr, "CI_lower": ci_lower, "CI_upper": ci_upper,
                     "P": p_value, "n_covariates": n_cov, "model": model_name}
    except Exception as e:
        log(f"  [{model_name}] Failed: {e}", logfile)

    if adj_result is None:
        log("  All adjusted Cox models failed", logfile)
        adj_result = {"HR": np.nan, "CI_lower": np.nan, "CI_upper": np.nan,
                     "P": np.nan, "n_covariates": 0, "model": "none"}

    return adj_result


# ── Propensity Score Analysis ──────────────────────────────────

def run_propensity_score_analysis(cloned_w, baseline_cov, logfile=None):
    """Propensity score analysis: PS-based IPTW as alternative to IPCW.

    1. Estimate propensity score: P(spectrum_reduction | baseline covariates)
    2. Compute IPTW weights: 1/PS for reduction arm, 1/(1-PS) for continue arm
    3. Stabilize and truncate weights
    4. Run weighted Cox with IPTW weights
    """
    from sklearn.linear_model import LogisticRegression

    log("", logfile)
    log("  " + "=" * 70, logfile)
    log("  PROPENSITY SCORE ANALYSIS (IPTW)", logfile)
    log("  " + "=" * 70, logfile)

    # Only use unique trials (one clone per trial to avoid double-counting in PS model)
    # Use Clone B (continue_broad) as reference for each trial
    clone_a = cloned_w[cloned_w["assigned_strategy"] == "spectrum_reduction"].copy()
    clone_b = cloned_w[cloned_w["assigned_strategy"] == "continue_broad"].copy()

    # Build trial-level dataset for PS estimation
    trials = clone_b[["episode_id", "landmark_hour"]].copy()
    trials["episode_id"] = trials["episode_id"].astype(str)
    trials["landmark_hour"] = trials["landmark_hour"].astype(int)

    # Merge baseline covariates
    bl = baseline_cov.copy()
    for col in ["episode_id"]:
        bl[col] = bl[col].astype(str)
    bl["landmark_hour"] = bl["landmark_hour"].astype(int)

    ps_covs = [
        "age_years", "gender",
        "current_n_abx", "current_max_spectrum_score",
        "current_anti_mrsa",
        "current_carbapenem",
        "current_combination_therapy", "current_last_resort",
        "cumulative_systemic_abx_hours", "cumulative_broad_abx_hours",
        "any_culture_obtained_before_landmark",
        "culture_result_known_before_landmark",
        "positive_culture_known_before_landmark",
        "blood_culture_before_landmark",
        "mechanical_ventilation_at_landmark",
        "vasopressor_at_landmark",
        "wbc_max_24h", "neutrophil_max_24h",
        "crp_max_24h", "lactate_max_24h",
        "platelet_min_24h", "ph_min_24h",
        "base_excess_min_24h", "spo2_min_24h",
        "congenital_heart_disease",
        "malignancy", "hematologic_disease",
        "prematurity",
    ]
    ps_avail = ["episode_id", "landmark_hour"] + [c for c in ps_covs if c in bl.columns]
    ps_data = trials.merge(bl[ps_avail], on=["episode_id", "landmark_hour"], how="left")

    # Treatment indicator: 1 = spectrum_reduction
    # Get this from the reduction events data
    red = pd.read_csv(SPECTRUM_REDUCTION_CSV, low_memory=False)
    red["episode_id"] = red["episode_id"].astype(str)
    red["landmark_hour"] = red["landmark_hour"].astype(int)
    ps_data = ps_data.merge(
        red[["episode_id", "landmark_hour", "reduction_occurred", "escalation_occurred"]],
        on=["episode_id", "landmark_hour"], how="left")
    ps_data["treated"] = ps_data["reduction_occurred"].fillna(0).astype(int)
    ps_data["escalated"] = ps_data["escalation_occurred"].fillna(0).astype(int)

    # Exclude escalation trials from PS analysis: the clinical decision
    # "de-escalate vs escalate" is not the same as "de-escalate vs continue-broad."
    # Restricting to reduction + continue-broad aligns with the primary estimand.
    ps_data = ps_data[ps_data["escalated"] == 0].copy()
    if "gender" in ps_data.columns:
        ps_data["gender_male"] = (ps_data["gender"].astype(str).str.upper() == "M").astype(int)
        ps_data = ps_data.drop(columns=["gender"])
        # Update ps_avail: replace "gender" with "gender_male"
        ps_avail = ["episode_id", "landmark_hour"] + [
            "gender_male" if c == "gender" else c
            for c in ps_avail if c not in ["episode_id", "landmark_hour"]
        ]

    # Prepare X (covariates) and y (treatment)
    X_cols = [c for c in ps_avail if c not in ["episode_id", "landmark_hour"]]
    # Ensure numeric
    for c in X_cols:
        if c in ps_data.columns:
            ps_data[c] = pd.to_numeric(ps_data[c], errors="coerce").fillna(0)
    X = ps_data[X_cols].values
    y = ps_data["treated"].values

    n_treated = y.sum()
    n_control = len(y) - n_treated
    log(f"  Trials: {len(y)}, Treated (reduction): {n_treated}, "
        f"Control (continue-broad): {n_control}", logfile)

    if n_treated < 10 or n_control < 10:
        log("  Insufficient sample for PS analysis, skipping", logfile)
        return None

    try:
        # Fit propensity score model
        ps_model = LogisticRegression(max_iter=1000, random_state=42)
        ps_model.fit(X, y)
        ps = ps_model.predict_proba(X)[:, 1]  # P(treated)
        ps = np.clip(ps, 0.01, 0.99)

        # Compute stabilized IPTW weights
        p_treated = y.mean()
        iptw = np.where(y == 1, p_treated / ps, (1 - p_treated) / (1 - ps))
        # Truncate at 99th percentile
        q99 = np.percentile(iptw, 99)
        iptw = np.clip(iptw, 0, q99)

        log(f"  PS distribution: mean={ps.mean():.3f} sd={ps.std():.3f} "
            f"range=[{ps.min():.3f}, {ps.max():.3f}]", logfile)
        log(f"  IPTW: mean={iptw.mean():.2f} sd={iptw.std():.2f} "
            f"range=[{iptw.min():.2f}, {iptw.max():.2f}]", logfile)

        # Map IPTW weights back to clones
        trial_weight_map = dict(zip(
            ps_data["episode_id"] + "_" + ps_data["landmark_hour"].astype(str),
            iptw
        ))

        cloned_ps = cloned_w.copy()
        cloned_ps["episode_id"] = cloned_ps["episode_id"].astype(str)
        cloned_ps["landmark_hour"] = cloned_ps["landmark_hour"].astype(int)
        cloned_ps["trial_key"] = cloned_ps["episode_id"] + "_" + \
                                 cloned_ps["landmark_hour"].astype(str)
        cloned_ps["iptw_weight"] = cloned_ps["trial_key"].map(trial_weight_map).fillna(1.0)

        # Combine IPTW with IPCW (multiply)
        cloned_ps["combined_weight"] = cloned_ps["ipcw_weight"] * cloned_ps["iptw_weight"]

        # Run Cox with different weight schemes
        log("", logfile)
        log("  --- PS-Only IPTW Weighted Cox ---", logfile)

        # Temporarily swap weights
        orig_w = cloned_ps["ipcw_weight"].copy()
        cloned_ps["ipcw_weight"] = cloned_ps["iptw_weight"]
        hr_ps, lo_ps, hi_ps, p_ps = run_weighted_cox(cloned_ps, None, label="PS_IPTW")
        cloned_ps["ipcw_weight"] = orig_w

        log("", logfile)
        log("  --- Combined IPCW×IPTW Weighted Cox ---", logfile)
        cloned_ps["ipcw_weight"] = cloned_ps["combined_weight"]
        hr_comb, lo_comb, hi_comb, p_comb = run_weighted_cox(cloned_ps, None, label="IPCWxIPTW")
        cloned_ps["ipcw_weight"] = orig_w

        log(f"  [PS_IPTW] HR={hr_ps:.3f} (95%CI {lo_ps:.3f}-{hi_ps:.3f}), P={p_ps:.4f}", logfile)
        log(f"  [IPCWxIPTW] HR={hr_comb:.3f} (95%CI {lo_comb:.3f}-{hi_comb:.3f}), P={p_comb:.4f}", logfile)

        return {
            "PS_IPTW_HR": hr_ps, "PS_IPTW_CI_lower": lo_ps,
            "PS_IPTW_CI_upper": hi_ps, "PS_IPTW_P": p_ps,
            "IPCWxIPTW_HR": hr_comb, "IPCWxIPTW_CI_lower": lo_comb,
            "IPCWxIPTW_CI_upper": hi_comb, "IPCWxIPTW_P": p_comb,
            "n_treated": n_treated, "n_control": n_control,
        }

    except Exception as e:
        log(f"  PS analysis failed: {e}", logfile)
        import traceback
        log(traceback.format_exc(), logfile)
        return None


# ── Main ──────────────────────────────────────────────────────

def run_rmst(cloned, km_results, logfile=None, n_bootstrap=2000):
    """Restricted Mean Event-Free Survival Time at 28 days (672 hours).

    Computes RMST from IPCW-weighted Kaplan-Meier curves as area under the
    survival curve, with clustered bootstrap CIs for the between-arm difference.
    RMST does not rely on the proportional hazards assumption.
    """
    from lifelines import KaplanMeierFitter

    TAU_DAYS = 28.0
    TAU_HOURS = 672.0

    # --- Helper: compute RMST from a lifelines KM object ---
    def _rmst_from_kmf(kmf, tau=TAU_DAYS):
        """Area under KM survival curve from 0 to tau using trapezoidal rule."""
        sf = kmf.survival_function_
        t = sf.index.values
        s = sf.iloc[:, 0].values
        # Clip to tau
        mask = t <= tau
        t_clip = np.append(t[mask], tau)
        s_clip = np.append(s[mask], kmf.predict(tau))
        return float(np.trapz(s_clip, t_clip))

    # --- Compute RMST from existing KM fits ---
    rmst = {}
    for arm in ["spectrum_reduction", "continue_broad"]:
        if arm in km_results and km_results[arm] is not None:
            rmst[arm] = _rmst_from_kmf(km_results[arm]["kmf"])
        else:
            rmst[arm] = np.nan

    diff = rmst.get("spectrum_reduction", np.nan) - rmst.get("continue_broad", np.nan)
    ratio = rmst.get("spectrum_reduction", np.nan) / rmst.get("continue_broad", np.nan) if rmst.get("continue_broad", np.nan) > 0 else np.nan

    log("", logfile)
    log("  === Restricted Mean Event-Free Survival Time (28 days) ===", logfile)
    log(f"  IPCW-weighted KM area under survival curve:", logfile)
    for arm in ["spectrum_reduction", "continue_broad"]:
        pct = rmst[arm] / TAU_DAYS * 100 if not np.isnan(rmst[arm]) else np.nan
        log(f"    [{arm}] RMST = {rmst[arm]:.1f} days ({rmst[arm]*24:.1f} h, {pct:.1f}% of max {TAU_DAYS} days)", logfile)
    log(f"    RMST difference (Reduction - Continue): {diff:.1f} days ({diff*24:.1f} h)", logfile)
    log(f"    RMST ratio (Reduction / Continue):    {ratio:.4f}", logfile)

    # --- Clustered bootstrap for CI of RMST difference ---
    log("", logfile)
    log(f"  Bootstrapping RMST (B={n_bootstrap}, cluster=subject_id)...", logfile)

    subject_ids = cloned["subject_id"].unique()
    n_subjects = len(subject_ids)
    rng = np.random.RandomState(42)

    boot_diffs = np.empty(n_bootstrap)
    boot_rmst_red = np.empty(n_bootstrap)
    boot_rmst_cont = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        # Sample subjects with replacement
        boot_sids = rng.choice(subject_ids, size=n_subjects, replace=True)
        boot_df = cloned[cloned["subject_id"].isin(boot_sids)].copy()

        arm_rmst = {}
        for arm in ["spectrum_reduction", "continue_broad"]:
            arm_df = boot_df[boot_df["assigned_strategy"] == arm]
            if len(arm_df) < 10:
                arm_rmst[arm] = np.nan
                continue

            fup_start = pd.to_datetime(arm_df["followup_start_time"])
            fup_end = pd.to_datetime(arm_df["followup_end_time"])
            max_hours = (fup_end - fup_start).dt.total_seconds() / 3600.0

            time_arr = np.zeros(len(arm_df))
            event_obs = np.zeros(len(arm_df))

            oc_mask = arm_df["outcome_occurred"] == 1
            if oc_mask.any():
                time_arr[oc_mask.values] = arm_df.loc[oc_mask, "outcome_hour_from_landmark"].values
                event_obs[oc_mask.values] = 1

            death_mask = arm_df["event_status"] == 2
            if death_mask.any():
                dtime = pd.to_datetime(arm_df.loc[death_mask, "death_time"].astype(str))
                ltime = pd.to_datetime(arm_df.loc[death_mask, "followup_start_time"].astype(str))
                time_arr[death_mask.values] = ((dtime - ltime).dt.total_seconds() / 3600.0).values

            cens_mask = arm_df["event_status"] == 0
            if cens_mask.any():
                time_arr[cens_mask.values] = max_hours[cens_mask.values].values

            time_arr = np.maximum(time_arr, 0.1) / 24.0
            weights = arm_df["ipcw_weight"].values

            try:
                kmf = KaplanMeierFitter()
                kmf.fit(durations=time_arr, event_observed=event_obs,
                        weights=weights, label=arm)
                arm_rmst[arm] = _rmst_from_kmf(kmf)
            except Exception:
                arm_rmst[arm] = np.nan

        boot_rmst_red[b] = arm_rmst.get("spectrum_reduction", np.nan)
        boot_rmst_cont[b] = arm_rmst.get("continue_broad", np.nan)
        boot_diffs[b] = (arm_rmst.get("spectrum_reduction", np.nan)
                         - arm_rmst.get("continue_broad", np.nan))

    # --- Bootstrap inference ---
    valid = ~np.isnan(boot_diffs)
    boot_diffs_valid = boot_diffs[valid]
    n_valid = valid.sum()

    boot_se = np.std(boot_diffs_valid, ddof=1)
    ci_lo_diff = np.percentile(boot_diffs_valid, 2.5)
    ci_hi_diff = np.percentile(boot_diffs_valid, 97.5)
    # Bootstrap P: proportion of boot_diffs <= 0 (two-sided)
    boot_p = 2 * min(np.mean(boot_diffs_valid <= 0), np.mean(boot_diffs_valid >= 0))
    boot_p = max(boot_p, 1.0 / n_bootstrap)

    log(f"  Bootstrap results ({n_valid}/{n_bootstrap} valid reps):", logfile)
    log(f"    RMST difference: {diff:.1f} days ({diff*24:.1f} h)", logfile)
    log(f"    95% CI (bootstrap): {ci_lo_diff:.1f} - {ci_hi_diff:.1f} days", logfile)
    log(f"    SE (bootstrap): {boot_se:.1f} days", logfile)
    log(f"    P (bootstrap): {boot_p:.4f}", logfile)

    # RMST ratio bootstrap
    boot_ratio = boot_rmst_red[valid] / boot_rmst_cont[valid]
    boot_ratio_se = np.std(boot_ratio, ddof=1)
    ci_lo_ratio = np.percentile(boot_ratio, 2.5)
    ci_hi_ratio = np.percentile(boot_ratio, 97.5)
    log(f"    RMST ratio: {ratio:.4f}", logfile)
    log(f"    95% CI (bootstrap): {ci_lo_ratio:.4f} - {ci_hi_ratio:.4f}", logfile)

    return {
        "rmst_reduction_days": rmst.get("spectrum_reduction", np.nan),
        "rmst_continue_days": rmst.get("continue_broad", np.nan),
        "rmst_difference_days": diff,
        "rmst_difference_hours": diff * 24,
        "rmst_ratio": ratio,
        "diff_ci_lower": ci_lo_diff,
        "diff_ci_upper": ci_hi_diff,
        "diff_se": boot_se,
        "diff_p": float(boot_p),
        "ratio_ci_lower": ci_lo_ratio,
        "ratio_ci_upper": ci_hi_ratio,
        "boot_valid": n_valid,
    }


def run_step11(logfile=None, cloned_trials=None, baseline_cov=None):
    log("=" * 60, logfile)
    log("Step 11: Statistical Analysis (IPCW + Cox + Competing Risks "
        "+ Sensitivity + Adjusted + PS)", logfile)
    log("=" * 60, logfile)

    # 1. Load inputs
    if cloned_trials is None:
        cloned_trials = pd.read_csv(CLONED_TRIALS_CSV, low_memory=False)
        cloned_trials["followup_start_time"] = parse_datetime(
            cloned_trials["followup_start_time"])
        cloned_trials["followup_end_time"] = parse_datetime(
            cloned_trials["followup_end_time"])
        if "death_time" in cloned_trials.columns:
            cloned_trials["death_time"] = parse_datetime(cloned_trials["death_time"])
        if "outcome_time" in cloned_trials.columns:
            cloned_trials["outcome_time"] = parse_datetime(cloned_trials["outcome_time"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            cloned_trials[col] = cloned_trials[col].astype(str)
    log(f"  Cloned trials: {len(cloned_trials):,}", logfile)

    if baseline_cov is None:
        baseline_cov = pd.read_csv(LANDMARK_BASELINE_CSV, low_memory=False)
        for col in ["subject_id", "hadm_id", "episode_id"]:
            baseline_cov[col] = baseline_cov[col].astype(str)
    log(f"  Baseline covariates: {len(baseline_cov)} rows", logfile)

    # 2. Baseline table
    run_baseline_table(cloned_trials, logfile)

    # 3. Estimate IPCW weights
    log("", logfile)
    log("Estimating IPCW weights...", logfile)
    cloned_w = estimate_ipcw(cloned_trials, baseline_cov, logfile)

    log(f"  IPCW weight - mean={cloned_w['ipcw_weight'].mean():.2f} "
        f"sd={cloned_w['ipcw_weight'].std():.2f} "
        f"median={cloned_w['ipcw_weight'].median():.2f}", logfile)
    for arm in ["spectrum_reduction", "continue_broad"]:
        w = cloned_w.loc[cloned_w["assigned_strategy"] == arm, "ipcw_weight"]
        log(f"    [{arm}] mean={w.mean():.2f} sd={w.std():.2f}", logfile)

    # 4. Primary: Weighted cause-specific Cox
    log("", logfile)
    log("Running primary analysis...", logfile)
    hr, ci_lo, ci_hi, p_val = run_weighted_cox(cloned_w, logfile)

    # 5. Weighted KM curves
    km_results = run_weighted_km(cloned_w, logfile)

    # 5b. Restricted Mean Survival Time (complementary to KM, no PH assumption)
    rmst_results = run_rmst(cloned_w, km_results, logfile)

    # 6. Cumulative incidence (Aalen-Johansen)
    aj_results = run_aalen_johansen(cloned_w, logfile)

    # 7. ALL sensitivity analyses (S1-S7)
    sensitivity_results = run_all_sensitivity_analyses(cloned_w, baseline_cov, logfile)

    # 8. Multivariable-adjusted Cox
    adj_cox_results = run_adjusted_cox(cloned_w, baseline_cov, logfile)

    # 9. Propensity score analysis
    ps_results = run_propensity_score_analysis(cloned_w, baseline_cov, logfile)

    # 10. Summary table
    log("", logfile)
    log("  " + "=" * 80, logfile)
    log("  COMPREHENSIVE ANALYSIS SUMMARY", logfile)
    log("  " + "=" * 80, logfile)

    def _fmt_hr(hr_val, lo, hi, p):
        if np.isnan(hr_val):
            return "   N/A (see notes)"
        return f"  {hr_val:>8.3f}  ({lo:.3f}-{hi:.3f})  {p:>8.4f}"

    log(f"  {'Analysis':<40} {'HR':>8} {'95% CI':>20} {'P':>8}", logfile)
    log(f"  {'':->40} {'':->8} {'':->20} {'':->8}", logfile)

    # Primary
    log(f"  {'PRIMARY (cause-specific Cox)':<40}" +
        _fmt_hr(hr, ci_lo, ci_hi, p_val), logfile)
    log(f"  {'':->40} {'':->8} {'':->20} {'':->8}", logfile)

    # Sensitivity analyses
    for key, res in sensitivity_results.items():
        label = res.get("description", key)
        log(f"  {label:<40}" +
            _fmt_hr(res["HR"], res["CI_lower"], res["CI_upper"], res["P"]), logfile)

    log(f"  {'':->40} {'':->8} {'':->20} {'':->8}", logfile)

    # Adjusted Cox
    label_adj = f"Adjusted Cox ({adj_cox_results.get('n_covariates', '?')} covariates)"
    log(f"  {label_adj:<40}" +
        _fmt_hr(adj_cox_results["HR"], adj_cox_results["CI_lower"],
                adj_cox_results["CI_upper"], adj_cox_results["P"]), logfile)

    # Propensity score
    if ps_results:
        log(f"  {'PS-IPTW (propensity score only)':<40}" +
            _fmt_hr(ps_results["PS_IPTW_HR"], ps_results["PS_IPTW_CI_lower"],
                    ps_results["PS_IPTW_CI_upper"], ps_results["PS_IPTW_P"]), logfile)
        log(f"  {'IPCW × IPTW (doubly weighted)':<40}" +
            _fmt_hr(ps_results["IPCWxIPTW_HR"], ps_results["IPCWxIPTW_CI_lower"],
                    ps_results["IPCWxIPTW_CI_upper"], ps_results["IPCWxIPTW_P"]), logfile)

    log("  " + "=" * 80, logfile)
    log("  HR < 1 = spectrum_reduction protective (fewer outcomes)", logfile)
    log("  HR > 1 = spectrum_reduction harmful (more outcomes)", logfile)
    log("  All CIs use cluster-robust sandwich SE at subject_id level", logfile)
    log("  " + "=" * 80, logfile)

    # 11. Save enriched output
    output_path = os.path.join(OUTPUT_DIR, "cloned_trials_weighted.csv")
    cloned_w.to_csv(output_path, index=False, encoding="utf-8")
    log(f"-> Saved {output_path} with IPCW weights", logfile)

    log("", logfile)
    log("Step 11 complete!", logfile)

    return cloned_w, {
        "primary_hr": hr, "primary_ci_lower": ci_lo, "primary_ci_upper": ci_hi,
        "primary_p": p_val,
        "km_results": km_results,
        "rmst_results": rmst_results,
        "aj_results": aj_results,
        "sensitivity": sensitivity_results,
        "adjusted_cox": adj_cox_results,
        "propensity_score": ps_results,
    }


if __name__ == "__main__":
    logfile = os.path.join(OUTPUT_DIR, "processing_log.txt")
    run_step11(logfile)
