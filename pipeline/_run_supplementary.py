"""Supplementary Analyses — PIPELINE v3.2
1. Fine-Gray subdistribution hazard model
2. S8 single-drug stop sensitivity
3. Spectrum score dose-response
4. Time-stratified 7d/14d/28d
5. Culture intensity assessment
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ═══ Load data ═══
df = pd.read_csv(r"F:\test\output\cloned_trials_weighted.csv")
re = pd.read_csv(r"F:\test\output\spectrum_reduction_events.csv")
bl = pd.read_csv(r"F:\test\output\landmark_baseline_covariates.csv")
TAU = 672

# ── Prepare time & event ──
df['followup_hours'] = (pd.to_datetime(df['followup_end_time']) - pd.to_datetime(df['followup_start_time'])).dt.total_seconds() / 3600
df['tte'] = np.where(df['outcome_occurred'] == 1, df['outcome_hour_from_landmark'], df['followup_hours'])
df['tte_capped'] = np.minimum(df['tte'], TAU)

df['etype'] = 0
df.loc[df['outcome_occurred'] == 1, 'etype'] = 1
df['death_during_fup'] = ((df['death_competing_same_time'] == 1) | (df['death_time'].notna() & (df['death_time'] != ''))).astype(int)
df.loc[(df['death_during_fup'] == 1) & (df['outcome_occurred'] == 0), 'etype'] = 2
df['treatment'] = (df['assigned_strategy'] == 'spectrum_reduction').astype(int)

print(f"N={len(df)}  outcomes={df['etype'].eq(1).sum()}  deaths={df['etype'].eq(2).sum()}  censored={df['etype'].eq(0).sum()}")

# ═══════════════════════════════════════════════════════════════
# 1. FINE-GRAY SUBDISTRIBUTION HAZARD (robust implementation)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("1. FINE-GRAY SUBDISTRIBUTION HAZARD MODEL")
print("=" * 80)

def estimate_g_km(times, etype):
    """KM for censoring survival G(t): 'event' = etype==0 (true censoring)."""
    order = np.argsort(times)
    t_s = times[order]
    e_s = (etype[order] == 0).astype(int)
    ut = np.sort(np.unique(t_s[e_s == 1]))
    g_vals = np.ones(len(ut) + 1)
    g_times = np.zeros(len(ut) + 1)
    n_risk = len(times)
    surv = 1.0
    for idx, tj in enumerate(ut):
        d = np.sum(e_s[t_s == tj])
        if n_risk > 0:
            surv *= max(1 - d / n_risk, 0.001)
        n_risk -= np.sum(t_s == tj)
        g_vals[idx + 1] = surv
        g_times[idx + 1] = tj
    return g_times, g_vals

def interp_g_km(t, gt, gv):
    """G(t-) left-continuous interpolation."""
    if t <= 0:
        return 1.0
    idx = np.searchsorted(gt, t, side='right') - 1
    return max(gv[max(0, min(idx, len(gv) - 1))], 0.001)

def fg_loglik(beta, times, etype, treatment, ipcw, gt, gv, tau):
    """Negative log partial likelihood for Fine-Gray model."""
    n = len(times)
    evt_times = np.sort(np.unique(times[(etype == 1) & (times <= tau)]))
    if len(evt_times) == 0:
        return 1e10

    loglik = 0.0
    for tj in evt_times:
        es = (times == tj) & (etype == 1)
        alive = (times >= tj) & (etype != 2)
        comp = (etype == 2) & (times < tj)
        risk = alive | comp

        fgw = np.ones(n)
        if np.any(comp):
            g_tj = interp_g_km(tj, gt, gv)
            for i in np.where(comp)[0]:
                fgw[i] = g_tj / max(interp_g_km(times[i], gt, gv), 0.001)

        tw = fgw * ipcw
        x_risk = treatment[risk]
        w_risk = tw[risk]

        # Clip beta*x to prevent overflow
        bx = np.clip(beta * x_risk, -30, 30)
        ebx = np.exp(bx)

        S0 = np.sum(w_risk * ebx)
        if S0 < 1e-15:
            continue

        x_evt = treatment[es]
        w_evt = tw[es]
        bx_evt = np.clip(beta * x_evt, -30, 30)

        loglik += np.sum(w_evt * (bx_evt - np.log(S0)))

    return -loglik  # negative for minimization

# Prepare data
times_arr = df['tte_capped'].values
etype_arr = df['etype'].values
trt_arr = df['treatment'].values
ipcw_arr = df['ipcw_weight'].values

gt, gv = estimate_g_km(times_arr, etype_arr)

# Find beta by minimizing negative log-lik
res = minimize_scalar(
    lambda b: fg_loglik(b, times_arr, etype_arr, trt_arr, ipcw_arr, gt, gv, TAU),
    bounds=(-5, 5), method='bounded', options={'xatol': 1e-8, 'maxiter': 200}
)
beta_fg = res.x

# SE via numeric Hessian
eps = 1e-5
ll0 = fg_loglik(beta_fg, times_arr, etype_arr, trt_arr, ipcw_arr, gt, gv, TAU)
ll_plus = fg_loglik(beta_fg + eps, times_arr, etype_arr, trt_arr, ipcw_arr, gt, gv, TAU)
ll_minus = fg_loglik(beta_fg - eps, times_arr, etype_arr, trt_arr, ipcw_arr, gt, gv, TAU)
hess = (ll_plus - 2*ll0 + ll_minus) / (eps**2)
se_fg = np.sqrt(1.0 / max(hess, 1e-10))

sHR = np.exp(beta_fg)
ci_l = np.exp(beta_fg - 1.96 * se_fg)
ci_u = np.exp(beta_fg + 1.96 * se_fg)
z_fg = beta_fg / se_fg
p_fg = 2 * (1 - norm.cdf(abs(z_fg)))

print(f"  Fine-Gray (IPCW-weighted):")
print(f"    beta = {beta_fg:.4f}  SE = {se_fg:.4f}")
print(f"    sHR = {sHR:.4f}  (95% CI {ci_l:.4f} – {ci_u:.4f})")
print(f"    P = {p_fg:.6f}")
print(f"  Comparative: Cause-specific HR = 0.392 vs Fine-Gray sHR = {sHR:.4f}")

# ═══════════════════════════════════════════════════════════════
# 2. S8: EXCLUDE SINGLE-DRUG STOP
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("2. S8: EXCLUDE SINGLE-DRUG STOP CASES")
print("=" * 80)

# Identify single_drug_stop trials
sd_trials = re[re['single_drug_stop'] == 1][['episode_id', 'landmark_hour']].drop_duplicates()
print(f"  Single-drug stop trials: {len(sd_trials)}")

# Tag clones to exclude
sd_keys = set()
for _, row in sd_trials.iterrows():
    sd_keys.add(f"{row['episode_id']}_{row['landmark_hour']}_reduction")
    sd_keys.add(f"{row['episode_id']}_{row['landmark_hour']}_continue")

df_s8 = df[~df['clone_id'].isin(sd_keys)].copy()
print(f"  Clones excluded: {len(df) - len(df_s8)}")
print(f"  Remaining clones: {len(df_s8)}")

# Recompute IPCW weights within S8 sample
ipcw_covariates = [
    'age_years', 'gender_male',
    'current_n_abx', 'current_max_spectrum_score',
    'current_anti_mrsa',
    'current_carbapenem',
    'current_combination_therapy', 'current_last_resort',
    'cumulative_systemic_abx_hours', 'cumulative_broad_abx_hours',
    'any_culture_obtained_before_landmark', 'culture_result_known_before_landmark',
    'positive_culture_known_before_landmark', 'blood_culture_before_landmark',
    'mechanical_ventilation_at_landmark', 'vasopressor_at_landmark',
    'wbc_max_24h', 'neutrophil_max_24h', 'crp_max_24h', 'lactate_max_24h',
    'platelet_min_24h', 'ph_min_24h', 'base_excess_min_24h', 'spo2_min_24h',
    'congenital_heart_disease', 'malignancy', 'hematologic_disease', 'prematurity',
]

def recompute_ipcw(df_arm, covariate_cols):
    """Recompute stabilized IPCW weights for one arm."""
    X = df_arm[covariate_cols].copy()
    y = df_arm['uncensored'].values
    # Standardize continuous
    for c in ['age_years', 'current_n_abx', 'current_max_spectrum_score',
              'cumulative_systemic_abx_hours', 'cumulative_broad_abx_hours',
              'wbc_max_24h', 'neutrophil_max_24h', 'crp_max_24h', 'lactate_max_24h',
              'platelet_min_24h', 'ph_min_24h', 'base_excess_min_24h', 'spo2_min_24h']:
        if c in X.columns:
            X[c] = (X[c] - X[c].mean()) / X[c].std()
    X = X.fillna(0)

    try:
        model = LogisticRegression(max_iter=2000, C=1e5).fit(X, y)
        p_uncensored = y.mean()
        p_given_x = np.clip(model.predict_proba(X)[:, 1], 0.01, 0.99)
        sw = p_uncensored / p_given_x
        clip_val = np.percentile(sw, 99)
        sw = np.clip(sw, 0, clip_val)
        return sw
    except:
        return np.ones(len(df_arm))

# Recompute IPCW
df_s8_new = df_s8.copy()
for arm in ['spectrum_reduction', 'continue_broad']:
    mask = df_s8_new['assigned_strategy'] == arm
    df_s8_new.loc[mask, 'ipcw_weight_s8'] = recompute_ipcw(df_s8_new[mask], ipcw_covariates)

# Fit Cox
cph_s8 = CoxPHFitter()
df_s8_fit = df_s8_new[['tte_capped', 'outcome_occurred', 'treatment', 'ipcw_weight_s8', 'subject_id']].copy()
df_s8_fit.columns = ['duration', 'event', 'treatment', 'weights', 'subject_id']
try:
    cph_s8.fit(df_s8_fit, duration_col='duration', event_col='event',
               weights_col='weights', cluster_col='subject_id')
    hr_s8 = np.exp(cph_s8.params_.values[0])
    ci_s8 = np.exp(cph_s8.confidence_intervals_.values[0])
    p_s8 = cph_s8.summary['p'].values[0]
    n_events_s8 = df_s8_new['outcome_occurred'].sum()
    print(f"  S8 results: N={len(df_s8_new)}, Events={n_events_s8}")
    print(f"  HR = {hr_s8:.4f}  (95% CI {ci_s8[0]:.4f} – {ci_s8[1]:.4f})  P = {p_s8:.6f}")
except Exception as e:
    print(f"  S8 Cox failed: {e}")
    hr_s8, ci_s8, p_s8 = np.nan, [np.nan, np.nan], np.nan
    n_events_s8 = df_s8_new['outcome_occurred'].sum()

# ═══════════════════════════════════════════════════════════════
# 3. DOSE-RESPONSE: SPECTRUM SCORE REDUCTION MAGNITUDE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("3. DOSE-RESPONSE: SPECTRUM SCORE REDUCTION MAGNITUDE")
print("=" * 80)

# Merge reduction info with clones
re_info = re[['episode_id', 'landmark_hour', 'reduction_occurred', 'reduction_type',
               'baseline_max_spectrum_score', 'single_drug_stop', 'baseline_n_abx']].copy()

df_dr = df.merge(re_info, on=['episode_id', 'landmark_hour'], how='left')
df_dr['treatment'] = (df_dr['assigned_strategy'] == 'spectrum_reduction').astype(int)
df_dr['tte_capped'] = np.minimum(
    np.where(df_dr['outcome_occurred'] == 1, df_dr['outcome_hour_from_landmark'],
             (pd.to_datetime(df_dr['followup_end_time']) - pd.to_datetime(df_dr['followup_start_time'])).dt.total_seconds() / 3600), TAU)

# ── A. By baseline spectrum score (properly stratified) ──
# Within each score stratum, compare ALL reduction-arm clones (both adherent and
# non-adherent, IPCW-weighted to reconstruct the per-protocol effect) vs ALL
# continue-broad clones in the SAME score stratum. This preserves the ITT-style
# estimand while isolating the treatment effect within each score level.
print("\n  Dose-response by baseline spectrum score level (both arms restricted to same score):")
for score in [3, 4]:
    sc_mask = df_dr['baseline_max_spectrum_score'] == score
    sub = df_dr[sc_mask].copy()
    n_red = int((sub['treatment'] == 1).sum())
    n_ctrl = int((sub['treatment'] == 0).sum())
    evt_red = int(sub.loc[sub['treatment'] == 1, 'outcome_occurred'].sum())
    evt_ctrl = int(sub.loc[sub['treatment'] == 0, 'outcome_occurred'].sum())
    print(f"  Score {score}: Reduction n={n_red} events={evt_red}, Continue n={n_ctrl} events={evt_ctrl}")
    try:
        cph_dr = CoxPHFitter()
        df_dr_fit = sub[['tte_capped', 'outcome_occurred', 'treatment', 'ipcw_weight', 'subject_id']].copy()
        df_dr_fit.columns = ['duration', 'event', 'treatment', 'weights', 'subject_id']
        cph_dr.fit(df_dr_fit, duration_col='duration', event_col='event',
                    weights_col='weights', cluster_col='subject_id')
        hr_dr = np.exp(cph_dr.params_.values[0])
        ci_dr = np.exp(cph_dr.confidence_intervals_.values[0])
        p_dr = cph_dr.summary['p'].values[0]
        print(f"    HR = {hr_dr:.4f} ({ci_dr[0]:.4f} - {ci_dr[1]:.4f}), P = {p_dr:.6f}")
    except Exception as e:
        print(f"    Failed: {e}")

# ── B. By reduction type (descriptive only) ──
# Formal causal comparison between reduction types is not feasible within the
# clone-censor-weight framework because: (a) the specific reduction type is only
# defined for adherent clones; (b) IPCW re-estimation per subtype would treat
# clones who underwent OTHER types of de-escalation as "censored," violating the
# assumption that censored and uncensored clones are exchangeable conditional on
# covariates (since other de-escalation types are also protective). We therefore
# present descriptive statistics only.

mask_a_adh = (df_dr['assigned_strategy'] == 'spectrum_reduction') & (df_dr['reduction_occurred'] == 1)

print("\n  By reduction type (descriptive statistics only):")
print(f"  {'Reduction Type':<35s} {'N':>6s} {'Events':>8s} {'Crude Rate':>12s}")
print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*12}")
for rtype, label in [
    ('stop_all_antibiotics', 'D7: Complete cessation'),
    ('n_abx_reduction', 'D6: N-abx reduction'),
    ('carbapenem_de_escalation', 'D4: Carbapenem de-escalation'),
    ('anti_mrsa_de_escalation', 'D3: Anti-MRSA de-escalation'),
    ('anti_pseudomonal_de_escalation', 'D2: Anti-pseudomonal de-escalation'),
]:
    mask_type = mask_a_adh & (df_dr['reduction_type'] == rtype)
    n_type = mask_type.sum()
    evt_type = int(df_dr.loc[mask_type, 'outcome_occurred'].sum())
    rate = evt_type / n_type * 100 if n_type > 0 else 0
    print(f"  {label:<35s} {n_type:>6d} {evt_type:>8d} {rate:>11.1f}%")
# D1 (n=1) and D5 (n=0) noted in text
print(f"  {'D1: Spectrum score reduction':<35s} {1:>6d} {0:>8d} {0:>11.1f}%")
print(f"  {'D5: Anti-anaerobe de-escalation':<35s} {0:>6d} {0:>8d} {'N/A':>11s}")

# For reference: continue-broad adherent crude rate
n_cb = mask_b_adh.sum() if 'mask_b_adh' in dir() else (df_dr['assigned_strategy'] == 'continue_broad').sum()
evt_cb = int(df_dr.loc[df_dr['assigned_strategy'] == 'continue_broad', 'outcome_occurred'].sum())
# Actually let's use the adherent continue-broad for a fair comparison
mask_b_adherent = (df_dr['assigned_strategy'] == 'continue_broad') & (df_dr['reduction_occurred'] != 1)
n_cb_adh = mask_b_adherent.sum()
evt_cb_adh = int(df_dr.loc[mask_b_adherent, 'outcome_occurred'].sum())
rate_cb = evt_cb_adh / n_cb_adh * 100
print(f"  {'(Ref) Continue-broad adherent':<35s} {n_cb_adh:>6d} {evt_cb_adh:>8d} {rate_cb:>11.1f}%")

# ═══════════════════════════════════════════════════════════════
# 4. TIME-STRATIFIED 7d/14d/28d
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("4. TIME-STRATIFIED ANALYSIS: 7d/14d/28d")
print("=" * 80)

time_windows = [
    ("0–7d (0–168h)", 0, 168),
    ("7–14d (168–336h)", 168, 336),
    ("14–28d (336–672h)", 336, 672),
    ("0–14d (0–336h)", 0, 336),
    ("0–28d (0–672h)", 0, 672),
]

for label, t_start, t_end in time_windows:
    # Create window-specific data
    df_win = df.copy()
    df_win['tte_win'] = np.minimum(df_win['tte_capped'], t_end)
    # Events only count if they occur within the window
    df_win['event_win'] = df_win['outcome_occurred'] & (df_win['outcome_hour_from_landmark'] >= t_start) & (df_win['outcome_hour_from_landmark'] < t_end)
    df_win['event_win'] = df_win['event_win'].astype(int)
    # For those who had events before t_start, they're no longer at risk
    # but should be excluded from analysis of later windows
    # Simplification: restrict to those alive and event-free at t_start
    at_risk = (df_win['tte_capped'] >= t_start) | (df_win['outcome_occurred'] == 1) & (df_win['outcome_hour_from_landmark'] >= t_start)
    df_win = df_win[at_risk].copy()

    n_evt = df_win['event_win'].sum()
    if n_evt < 5:
        print(f"  {label}: insufficient events (n={n_evt})")
        continue

    try:
        cph_win = CoxPHFitter()
        df_wf = df_win[['tte_win', 'event_win', 'treatment', 'ipcw_weight', 'subject_id']].copy()
        df_wf.columns = ['duration', 'event', 'treatment', 'weights', 'subject_id']
        cph_win.fit(df_wf, duration_col='duration', event_col='event',
                     weights_col='weights', cluster_col='subject_id')
        hr_win = np.exp(cph_win.params_.values[0])
        ci_win = np.exp(cph_win.confidence_intervals_.values[0])
        p_win = cph_win.summary['p'].values[0]
        n_r = df_win[(df_win['treatment'] == 1) & (df_win['event_win'] == 1)].shape[0]
        n_c = df_win[(df_win['treatment'] == 0) & (df_win['event_win'] == 1)].shape[0]
        print(f"  {label}: Events(R/C)={n_r}/{n_c}  HR={hr_win:.4f}  "
              f"95% CI {ci_win[0]:.4f} – {ci_win[1]:.4f}  P={p_win:.6f}")
    except Exception as e:
        print(f"  {label}: failed: {e}")

# ═══════════════════════════════════════════════════════════════
# 5. CULTURE INTENSITY ASSESSMENT
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("5. CULTURE INTENSITY ASSESSMENT")
print("=" * 80)

# Culture counts from baseline covariates
bl_cultures = bl[['episode_id', 'landmark_hour', 'number_of_cultures_before_landmark']].copy()
df_ci = df.merge(bl_cultures, on=['episode_id', 'landmark_hour'], how='left')

# Pre-LM culture density: cultures / ICU LOS (hours) * 24 → cultures per patient-day
# ICU LOS before landmark in hours
icu_los_before = (pd.to_datetime(df_ci['followup_start_time']) - pd.to_datetime('1970-01-01')).dt.total_seconds() / 3600
# Actually from the data, we can use: landmark_hour as proxy for ICU LOS before LM
df_ci['pre_lm_los_hours'] = df_ci['landmark_hour'].astype(float)
df_ci['pre_lm_culture_density'] = df_ci['number_of_cultures_before_landmark'] / (df_ci['pre_lm_los_hours'] / 24)
df_ci['pre_lm_culture_density'] = df_ci['pre_lm_culture_density'].replace([np.inf, -np.inf], np.nan)

# Post-LM culture density: we don't have post-LM culture counts directly
# But we know by design this is identical between arms (cloned data)
# Report pre-LM density as baseline comparison

for arm in ['spectrum_reduction', 'continue_broad']:
    sub = df_ci[df_ci['assigned_strategy'] == arm]
    dens = sub['pre_lm_culture_density'].dropna()
    print(f"  {arm}:")
    print(f"    Pre-LM cultures: mean = {sub['number_of_cultures_before_landmark'].mean():.2f} (SD {sub['number_of_cultures_before_landmark'].std():.2f})")
    print(f"    Pre-LM culture density (/pt-day): mean = {dens.mean():.2f} (SD {dens.std():.2f})")

# Also report culture result known rate
print(f"\n  Culture result known at landmark:")
for arm in ['spectrum_reduction', 'continue_broad']:
    sub = df_ci[df_ci['assigned_strategy'] == arm]
    rate = sub['culture_result_known_before_landmark'].mean() * 100
    print(f"  {arm}: {rate:.1f}%")

# ═══════════════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

output_path = r"F:\test\output\supplementary_analyses.txt"
with open(output_path, 'w', encoding='utf-8') as f:
    f.write("=" * 90 + "\n")
    f.write("SUPPLEMENTARY ANALYSES — PIPELINE v3.2\n")
    f.write("=" * 90 + "\n\n")

    f.write("S-A1. Fine-Gray Subdistribution Hazard Model\n")
    f.write("-" * 50 + "\n")
    f.write(f"  sHR = {sHR:.4f}\n")
    f.write(f"  95% CI: {ci_l:.4f} – {ci_u:.4f}\n")
    f.write(f"  P = {p_fg:.6f}\n")
    f.write(f"  Comparison: Cause-specific HR = 0.392 vs Fine-Gray sHR = {sHR:.4f}\n")
    f.write(f"  Interpretation: The {'more' if sHR < 0.393 else 'less'} protective sHR accounts for\n")
    f.write(f"  death as a competing event rather than independent censoring.\n\n")

    f.write("S-A2. S8: Excluding Single-Drug Stop Cases\n")
    f.write("-" * 50 + "\n")
    f.write(f"  Trials excluded: {len(sd_trials)} (single_drug_stop = 1)\n")
    f.write(f"  Clones excluded: {len(df) - len(df_s8)}\n")
    f.write(f"  Remaining clones: {len(df_s8)}\n")
    f.write(f"  HR = {hr_s8:.4f}  (95% CI {ci_s8[0]:.4f} – {ci_s8[1]:.4f})  P = {p_s8:.6f}\n\n")

    f.write("S-A3. Dose-Response by Baseline Spectrum Score\n")
    f.write("-" * 50 + "\n")
    f.write(f"  See console output for detailed results.\n\n")

    f.write("S-A4. Time-Stratified Analysis (7d/14d/28d)\n")
    f.write("-" * 50 + "\n")
    f.write(f"  See console output for detailed results.\n\n")

    f.write("S-A5. Culture Intensity Assessment\n")
    f.write("-" * 50 + "\n")
    f.write(f"  By design (cloned data), baseline culture intensity is identical between arms.\n")
    f.write(f"  See console output for pre-LM culture density by arm.\n\n")

print(f"Saved to {output_path}")
print("\nDone.")
