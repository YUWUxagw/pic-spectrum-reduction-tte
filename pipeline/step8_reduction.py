"""Step 8: Spectrum Reduction Events (spectrum_reduction_events.csv)
Implements Section 8 - 7 reduction dimensions with Net Intensity Constraint.
D1: spectrum_score reduction >=1 for >=24h
D2: anti_pseudomonal de-escalation (>=24h sustained)
D3: anti_mrsa de-escalation (>=24h sustained)
D4: carbapenem de-escalation (>=24h sustained)
D5: anti_anaerobe de-escalation (>=24h sustained)
D6: n_abx reduction
D7: stop all antibiotics (>=24h sustained)
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from utils import parse_datetime, hours_between, log


def run_step8(logfile=None, eligible_trials=None, baseline_cov=None,
              abx_orders=None, icu_base=None):
    log("=" * 60, logfile)
    log("Step 8: Spectrum Reduction Events", logfile)
    log("=" * 60, logfile)

    # 1. Load inputs
    if eligible_trials is None:
        eligible_trials = pd.read_csv(LANDMARK_ELIGIBLE_CSV, low_memory=False)
        eligible_trials["landmark_time"] = parse_datetime(eligible_trials["landmark_time"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            eligible_trials[col] = eligible_trials[col].astype(str)
    eligible = eligible_trials[eligible_trials["is_eligible"]].copy()

    if baseline_cov is None:
        baseline_cov = pd.read_csv(LANDMARK_BASELINE_CSV, low_memory=False)
        baseline_cov["landmark_time"] = parse_datetime(baseline_cov["landmark_time"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            baseline_cov[col] = baseline_cov[col].astype(str)

    if abx_orders is None:
        abx_orders = pd.read_csv(ABX_ORDERS_CLEAN_CSV, low_memory=False)
        abx_orders["start_time"] = parse_datetime(abx_orders["start_time"])
        abx_orders["end_time"] = parse_datetime(abx_orders["end_time"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            abx_orders[col] = abx_orders[col].astype(str)

    if icu_base is None:
        icu_base = pd.read_csv(ICU_BASE_CSV, low_memory=False)
        icu_base["episode_intime"] = parse_datetime(icu_base["episode_intime"])
        icu_base["death_time"] = parse_datetime(icu_base["death_time"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            icu_base[col] = icu_base[col].astype(str)

    log(f"  Eligible trials: {len(eligible)}", logfile)

    # 2. Merge baseline covariates + death_time from icu_base
    base_cols = [c for c in baseline_cov.columns if c in [
        "episode_id", "landmark_hour",
        "current_max_spectrum_score", "current_n_abx",
        "current_anti_pseudomonal", "current_anti_mrsa",
        "current_carbapenem", "current_anti_anaerobe", "current_last_resort",
        "cov_gnb", "cov_pseudomonas", "cov_mrsa", "cov_anaerobe",
        "cov_atypical", "cov_fungal", "cov_salvage",
    ]]
    events = eligible.merge(baseline_cov[base_cols],
                            on=["episode_id", "landmark_hour"], how="left")

    # Merge death_time from icu_base
    events = events.merge(
        icu_base[["episode_id", "death_time"]],
        on="episode_id", how="left"
    )

    # Calculate grace period timing (Section 8.2)
    events["grace_end"] = events["landmark_time"] + pd.Timedelta(hours=GRACE_PERIOD_HOURS)
    events["ascertainment_end"] = events["landmark_time"] + pd.Timedelta(hours=ASCERTAINMENT_HOURS)

    columns_need = [
        "baseline_max_spectrum_score", "baseline_n_abx",
        "baseline_anti_pseudomonal", "baseline_anti_mrsa",
        "baseline_carbapenem", "baseline_anti_anaerobe", "baseline_current_last_resort",
        "baseline_cov_gnb", "baseline_cov_pseudomonas", "baseline_cov_mrsa",
        "baseline_cov_anaerobe", "baseline_cov_atypical", "baseline_cov_fungal",
        "baseline_cov_salvage",
    ]
    for col in columns_need:
        if col not in events.columns:
            events[col] = 0

    # Initialize output columns
    events["reduction_occurred"] = 0
    events["reduction_time"] = pd.NaT
    events["reduction_hour_from_landmark"] = np.nan
    events["reduction_type"] = ""
    events["all_reduction_types"] = ""
    events["reduction_with_concurrent_escalation"] = 0
    events["continue_broad_48h"] = 0
    events["escalation_occurred"] = 0
    events["escalation_time"] = pd.NaT
    events["death_in_grace_period"] = 0
    events["stop_followed_by_death_24h"] = 0
    events["single_drug_stop"] = 0

    log("Analyzing antibiotic trajectories in grace period...", logfile)

    # 3. Process each trial
    n_reductions = 0
    n_continues = 0
    n_escalations = 0
    n_death_grace = 0

    for idx, row in events.iterrows():
        ep = row["episode_id"]
        lm = row["landmark_time"]
        grace_end = row["grace_end"]
        death_time = row.get("death_time")

        # Get antibiotic orders for this episode in grace period
        ep_abx = abx_orders[abx_orders["episode_id"] == ep].copy()
        grace_abx = ep_abx[(ep_abx["start_time"] < grace_end) &
                            ((ep_abx["end_time"].isna()) | (ep_abx["end_time"] > lm))]

        if len(grace_abx) == 0:
            continue

        # ── Baseline values ──
        bl_score = float(row.get("current_max_spectrum_score", 0) or 0)
        bl_n_abx = int(row.get("current_n_abx", 0) or 0)
        bl_anti_pseud = int(row.get("current_anti_pseudomonal", 0) or 0)
        bl_anti_mrsa = int(row.get("current_anti_mrsa", 0) or 0)
        bl_carb = int(row.get("current_carbapenem", 0) or 0)
        bl_anti_anaer = int(row.get("current_anti_anaerobe", 0) or 0)
        bl_salvage = int(row.get("current_last_resort", 0) or 0)
        bl_cov_pseud = int(row.get("cov_pseudomonas", 0) or 0)
        bl_cov_mrsa = int(row.get("cov_mrsa", 0) or 0)
        bl_cov_anaer = int(row.get("cov_anaerobe", 0) or 0)
        bl_cov_fungal = int(row.get("cov_fungal", 0) or 0)
        bl_cov_gnb = int(row.get("cov_gnb", 0) or 0)
        bl_cov_atyp = int(row.get("cov_atypical", 0) or 0)

        # Store baseline
        events.at[idx, "baseline_max_spectrum_score"] = bl_score
        events.at[idx, "baseline_n_abx"] = bl_n_abx
        events.at[idx, "baseline_anti_pseudomonal"] = bl_anti_pseud
        events.at[idx, "baseline_anti_mrsa"] = bl_anti_mrsa
        events.at[idx, "baseline_carbapenem"] = bl_carb
        events.at[idx, "baseline_anti_anaerobe"] = bl_anti_anaer
        events.at[idx, "baseline_current_last_resort"] = bl_salvage
        events.at[idx, "baseline_cov_gnb"] = bl_cov_gnb
        events.at[idx, "baseline_cov_pseudomonas"] = bl_cov_pseud
        events.at[idx, "baseline_cov_mrsa"] = bl_cov_mrsa
        events.at[idx, "baseline_cov_anaerobe"] = bl_cov_anaer
        events.at[idx, "baseline_cov_atypical"] = bl_cov_atyp
        events.at[idx, "baseline_cov_fungal"] = bl_cov_fungal
        events.at[idx, "baseline_cov_salvage"] = bl_salvage

        # ── Check death in grace period (Section 8.3.7) ──
        death_in_grace = 0
        if pd.notna(death_time) and death_time <= grace_end:
            death_in_grace = 1
            events.at[idx, "death_in_grace_period"] = 1
            n_death_grace += 1
            continue  # Trial has death in grace period, cannot continue or reduce

        # ── Track changes over time ──
        time_points = pd.date_range(lm, grace_end, freq="1h")
        if len(time_points) == 0:
            continue

        # Phase 1: Track net intensity and escalation across ALL time points
        net_intensity_violated = False
        escalation_found = False
        escalation_time = None
        peak_score = bl_score
        peak_cov_pseud = bl_cov_pseud
        peak_cov_mrsa = bl_cov_mrsa
        peak_cov_fungal = bl_cov_fungal
        peak_salvage = bl_salvage
        peak_carb = bl_carb

        for t in time_points:
            active_at_t = grace_abx[
                (grace_abx["start_time"] <= t) & (grace_abx["end_time"] > t)
            ]

            if len(active_at_t) == 0:
                continue

            curr_score = active_at_t["spectrum_score"].max()
            curr_cov_pseud = active_at_t["coverage_pseudomonas"].max() \
                if "coverage_pseudomonas" in active_at_t.columns else 0
            curr_cov_mrsa = active_at_t["coverage_mrsa"].max() \
                if "coverage_mrsa" in active_at_t.columns else 0
            curr_cov_fungal = active_at_t["coverage_fungal"].max() \
                if "coverage_fungal" in active_at_t.columns else 0
            curr_salvage = active_at_t["salvage_therapy"].max()
            curr_carb = active_at_t["carbapenem"].max()

            # Track peak values across grace period
            if curr_score > peak_score:
                peak_score = curr_score
                if escalation_time is None:
                    escalation_time = t
            if curr_cov_pseud > peak_cov_pseud:
                peak_cov_pseud = curr_cov_pseud
                if escalation_time is None:
                    escalation_time = t
            if curr_cov_mrsa > peak_cov_mrsa:
                peak_cov_mrsa = curr_cov_mrsa
                if escalation_time is None:
                    escalation_time = t
            if curr_cov_fungal > peak_cov_fungal:
                peak_cov_fungal = curr_cov_fungal
                if escalation_time is None:
                    escalation_time = t
            if curr_salvage > peak_salvage:
                peak_salvage = curr_salvage
                if escalation_time is None:
                    escalation_time = t
            if curr_carb > peak_carb:
                peak_carb = curr_carb
                if escalation_time is None:
                    escalation_time = t

        # Determine net intensity violation (Section 8.3.2)
        # Any increase in coverage dimensions = violation
        if peak_score > bl_score:
            net_intensity_violated = True
            escalation_found = True
        if peak_cov_pseud > bl_cov_pseud:
            net_intensity_violated = True
            escalation_found = True
        if peak_cov_mrsa > bl_cov_mrsa:
            net_intensity_violated = True
            escalation_found = True
        if peak_salvage > bl_salvage:
            net_intensity_violated = True
            escalation_found = True
        if peak_cov_fungal > bl_cov_fungal:
            net_intensity_violated = True
            escalation_found = True
        if peak_carb > bl_carb:
            net_intensity_violated = True
            escalation_found = True

        # Check escalation E1: spectrum_score increase >=1 sustained >=24h (Section 8.3.6)
        escalation_e1_sustained = False
        if peak_score >= bl_score + 1:
            for t in time_points:
                sustained_count = 0
                for check_t in pd.date_range(t, t + pd.Timedelta(hours=24), freq="1h"):
                    if check_t > grace_end:
                        break
                    check_active = grace_abx[
                        (grace_abx["start_time"] <= check_t) &
                        (grace_abx["end_time"] > check_t)
                    ]
                    check_score = check_active["spectrum_score"].max() if len(check_active) > 0 else 0
                    if check_score >= bl_score + 1:
                        sustained_count += 1
                if sustained_count >= 24:
                    escalation_e1_sustained = True
                    escalation_found = True
                    break

        # Phase 2: Detect reductions (only valid if no net intensity violation)
        reductions_found = set()
        reduction_time_found = None

        # Track allowable gap for continue_broad (Section 8.2/8.3.5)
        max_broad_gap_hours = 0.0
        last_broad_time = lm
        any_reduction_during_gap = False

        if not net_intensity_violated:
            for t in time_points:
                active_at_t = grace_abx[
                    (grace_abx["start_time"] <= t) & (grace_abx["end_time"] > t)
                ]

                if len(active_at_t) == 0:
                    # All antibiotics stopped
                    if bl_n_abx > 0:
                        stop_start = t
                        sustained = True
                        for check_t in pd.date_range(t, t + pd.Timedelta(hours=24), freq="1h"):
                            if check_t > grace_end:
                                break
                            check_active = grace_abx[
                                (grace_abx["start_time"] <= check_t) &
                                (grace_abx["end_time"] > check_t)
                            ]
                            if len(check_active) > 0:
                                sustained = False
                                break
                        if sustained:
                            reductions_found.add("stop_all_antibiotics")
                            if reduction_time_found is None:
                                reduction_time_found = t
                    continue

                curr_score = active_at_t["spectrum_score"].max()
                curr_n = active_at_t["drug_std"].nunique()
                curr_cov_pseud = active_at_t["coverage_pseudomonas"].max() \
                    if "coverage_pseudomonas" in active_at_t.columns else 0
                curr_cov_mrsa = active_at_t["coverage_mrsa"].max() \
                    if "coverage_mrsa" in active_at_t.columns else 0
                curr_cov_anaer = active_at_t["coverage_anaerobe"].max() \
                    if "coverage_anaerobe" in active_at_t.columns else 0
                curr_salvage = active_at_t["salvage_therapy"].max()
                curr_carb = active_at_t["carbapenem"].max()

                # Track broad coverage gap
                has_broad = (curr_score >= 3)
                if has_broad:
                    gap = (t - last_broad_time).total_seconds() / 3600.0
                    if gap > max_broad_gap_hours:
                        max_broad_gap_hours = gap
                    last_broad_time = t

                # ── D1: spectrum_score reduction >= 1, sustained >=24h ──
                if curr_score <= bl_score - 1 and bl_score > 0:
                    sustained_count = 0
                    for check_t in pd.date_range(t, t + pd.Timedelta(hours=24), freq="1h"):
                        if check_t > grace_end:
                            break
                        check_active = grace_abx[
                            (grace_abx["start_time"] <= check_t) &
                            (grace_abx["end_time"] > check_t)
                        ]
                        check_score = check_active["spectrum_score"].max() if len(check_active) > 0 else 0
                        if check_score <= bl_score - 1:
                            sustained_count += 1
                    if sustained_count >= 24:
                        reductions_found.add("spectrum_score_reduction")
                        if reduction_time_found is None:
                            reduction_time_found = t

                # ── D2: anti_pseudomonal de-escalation, sustained >=24h (Section 8.3.3) ──
                if bl_cov_pseud == 1 and curr_cov_pseud == 0 and curr_salvage == 0:
                    sustained = _check_dimension_sustained(
                        grace_abx, t, grace_end, bl_cov_pseud,
                        lambda act: act["coverage_pseudomonas"].max()
                        if "coverage_pseudomonas" in act.columns else 0,
                        target_value=0, comparison="eq"
                    )
                    if sustained:
                        reductions_found.add("anti_pseudomonal_de_escalation")
                        if reduction_time_found is None:
                            reduction_time_found = t

                # ── D3: anti_mrsa de-escalation, sustained >=24h ──
                if bl_cov_mrsa == 1 and curr_cov_mrsa == 0:
                    sustained = _check_dimension_sustained(
                        grace_abx, t, grace_end, bl_cov_mrsa,
                        lambda act: act["coverage_mrsa"].max()
                        if "coverage_mrsa" in act.columns else 0,
                        target_value=0, comparison="eq"
                    )
                    if sustained:
                        reductions_found.add("anti_mrsa_de_escalation")
                        if reduction_time_found is None:
                            reduction_time_found = t

                # ── D4: carbapenem de-escalation, sustained >=24h ──
                if bl_carb == 1 and curr_carb == 0 and curr_salvage == 0:
                    sustained = _check_dimension_sustained(
                        grace_abx, t, grace_end, bl_carb,
                        lambda act: act["carbapenem"].max(),
                        target_value=0, comparison="eq"
                    )
                    if sustained:
                        reductions_found.add("carbapenem_de_escalation")
                        if reduction_time_found is None:
                            reduction_time_found = t

                # ── D5: anti_anaerobe de-escalation, sustained >=24h ──
                if bl_cov_anaer == 1 and curr_cov_anaer == 0:
                    sustained = _check_dimension_sustained(
                        grace_abx, t, grace_end, bl_cov_anaer,
                        lambda act: act["coverage_anaerobe"].max()
                        if "coverage_anaerobe" in act.columns else 0,
                        target_value=0, comparison="eq"
                    )
                    if sustained:
                        reductions_found.add("anti_anaerobe_de_escalation")
                        if reduction_time_found is None:
                            reduction_time_found = t

                # ── D6: n_abx reduction (maintain same/higher score & coverage) ──
                if curr_n < bl_n_abx and curr_score >= bl_score:
                    reductions_found.add("n_abx_reduction")
                    if reduction_time_found is None:
                        reduction_time_found = t

        else:
            # Net intensity violated - track broad coverage gap anyway for logging
            for t in time_points:
                active_at_t = grace_abx[
                    (grace_abx["start_time"] <= t) & (grace_abx["end_time"] > t)
                ]
                if len(active_at_t) > 0:
                    curr_score = active_at_t["spectrum_score"].max()
                    has_broad = (curr_score >= 3)
                    if has_broad:
                        gap = (t - last_broad_time).total_seconds() / 3600.0
                        if gap > max_broad_gap_hours:
                            max_broad_gap_hours = gap
                        last_broad_time = t

        # ── Record findings ──
        if reductions_found:
            if not net_intensity_violated:
                events.at[idx, "reduction_occurred"] = 1
                events.at[idx, "reduction_with_concurrent_escalation"] = 0
                events.at[idx, "reduction_type"] = _pick_highest_priority(reductions_found)
                events.at[idx, "all_reduction_types"] = "|".join(sorted(reductions_found))
                if reduction_time_found is not None:
                    events.at[idx, "reduction_time"] = reduction_time_found
                    events.at[idx, "reduction_hour_from_landmark"] = \
                        (reduction_time_found - lm).total_seconds() / 3600.0

                # Check D7 stop_followed_by_death_24h (Section 8.3.4)
                if "stop_all_antibiotics" in reductions_found and pd.notna(death_time):
                    if reduction_time_found is not None:
                        hours_to_death = (death_time - reduction_time_found).total_seconds() / 3600.0
                        if 0 < hours_to_death <= 24:
                            events.at[idx, "stop_followed_by_death_24h"] = 1
                            events.at[idx, "reduction_occurred"] = 0
                            events.at[idx, "reduction_type"] = ""
                            events.at[idx, "all_reduction_types"] = ""
                            events.at[idx, "reduction_time"] = pd.NaT
                            events.at[idx, "reduction_hour_from_landmark"] = np.nan
                            reductions_found = set()

                if events.at[idx, "reduction_occurred"] == 1:
                    n_reductions += 1
                    # Flag single-drug stop for sensitivity analysis
                    if events.at[idx, "reduction_type"] == "stop_all_antibiotics" and bl_n_abx == 1:
                        events.at[idx, "single_drug_stop"] = 1

            else:
                # Reduction found but net intensity violated (Section 8.3.2)
                events.at[idx, "reduction_occurred"] = 0
                events.at[idx, "reduction_with_concurrent_escalation"] = 1
                events.at[idx, "reduction_type"] = ""
                events.at[idx, "all_reduction_types"] = "|".join(sorted(reductions_found))
                events.at[idx, "escalation_occurred"] = 0

        if events.at[idx, "reduction_occurred"] == 0 and \
           events.at[idx, "reduction_with_concurrent_escalation"] == 0:
            if escalation_found and not reductions_found:
                events.at[idx, "escalation_occurred"] = 1
                if escalation_time is not None:
                    events.at[idx, "escalation_time"] = escalation_time
                n_escalations += 1
            elif not escalation_found:
                # Check continue_broad_48h (Section 8.3.5) with allowable gap
                if max_broad_gap_hours <= ALLOWABLE_GAP_HOURS:
                    events.at[idx, "continue_broad_48h"] = 1
                    n_continues += 1
                else:
                    # Broad coverage gap exceeded allowable - cannot classify as continue
                    events.at[idx, "escalation_occurred"] = 1
                    n_escalations += 1

    log(f"  Reductions: {n_reductions:,}", logfile)
    log(f"  Continue broad: {n_continues:,}", logfile)
    log(f"  Escalations: {n_escalations:,}", logfile)
    log(f"  Death in grace period: {n_death_grace:,}", logfile)

    # 4. Output (Section 8.4)
    output_cols = [
        "subject_id", "hadm_id", "episode_id", "landmark_hour", "landmark_time",
        "grace_end", "ascertainment_end",
        "reduction_occurred", "reduction_time", "reduction_hour_from_landmark",
        "reduction_type", "all_reduction_types",
        "reduction_with_concurrent_escalation",
        "continue_broad_48h", "escalation_occurred", "escalation_time",
        "death_in_grace_period", "stop_followed_by_death_24h",
        "baseline_max_spectrum_score", "baseline_n_abx",
        "baseline_carbapenem", "baseline_anti_pseudomonal",
        "baseline_anti_mrsa", "baseline_anti_anaerobe",
        "baseline_current_last_resort",
        "baseline_cov_gnb", "baseline_cov_pseudomonas",
        "baseline_cov_mrsa", "baseline_cov_anaerobe",
        "baseline_cov_atypical", "baseline_cov_fungal",
        "baseline_cov_salvage",
        "single_drug_stop",
    ]
    output = events[[c for c in output_cols if c in events.columns]].reset_index(drop=True)

    output.to_csv(SPECTRUM_REDUCTION_CSV, index=False, encoding="utf-8")
    log(f"-> Saved {SPECTRUM_REDUCTION_CSV}: {len(output):,} rows", logfile)

    # Type breakdown
    type_counts = output["reduction_type"].value_counts()
    log(f"  Reduction types: {type_counts.to_dict()}", logfile)
    # Single-drug stop flag for sensitivity analysis
    n_single_drug = (output["single_drug_stop"] == 1).sum()
    if n_single_drug > 0:
        log(f"  Single-drug stops (sensitivity flag): {n_single_drug} reductions", logfile)

    log("Step 8 complete!", logfile)
    return output


def _check_dimension_sustained(grace_abx, start_t, grace_end, baseline_value,
                                value_fn, target_value, comparison="eq"):
    """Check if a dimension reduction is sustained for >=24 hours.
    comparison: 'eq' (value == target) or 'lte' (value <= target)
    """
    sustained_count = 0
    for check_t in pd.date_range(start_t, start_t + pd.Timedelta(hours=24), freq="1h"):
        if check_t > grace_end:
            break
        check_active = grace_abx[
            (grace_abx["start_time"] <= check_t) &
            (grace_abx["end_time"] > check_t)
        ]
        if len(check_active) == 0:
            curr_val = 0
        else:
            curr_val = value_fn(check_active)

        if comparison == "eq":
            if curr_val == target_value:
                sustained_count += 1
        elif comparison == "lte":
            if curr_val <= target_value:
                sustained_count += 1

    return sustained_count >= 24


def _pick_highest_priority(reduction_set):
    """Pick highest priority reduction type (Section 8.3.3)."""
    priority = [
        "stop_all_antibiotics",
        "carbapenem_de_escalation",
        "anti_pseudomonal_de_escalation",
        "anti_mrsa_de_escalation",
        "anti_anaerobe_de_escalation",
        "spectrum_score_reduction",
        "n_abx_reduction",
    ]
    for p in priority:
        if p in reduction_set:
            return p
    return "|".join(sorted(reduction_set))


if __name__ == "__main__":
    logfile = os.path.join(OUTPUT_DIR, "processing_log.txt")
    run_step8(logfile)
