"""S7: Follow-up from grace period end (landmark + 48h)"""
import pandas as pd, numpy as np
from lifelines import CoxPHFitter

df = pd.read_csv(r'F:\test\output\cloned_trials_weighted.csv')
GRACE = 48; TAU = 672

df['treatment'] = (df['assigned_strategy']=='spectrum_reduction').astype(int)
# followup_start_time IS the landmark time (verified: landmark_hour matches, dur = end-start gives time from landmark)
df['fu_hours'] = (pd.to_datetime(df['followup_end_time']) - pd.to_datetime(df['followup_start_time'])).dt.total_seconds()/3600

def run_cox(data, label=''):
    fc = data[['tte','outcome_occurred','treatment','ipcw_weight','subject_id']].copy()
    fc.columns = ['duration','event','treatment','weights','subject_id']
    c = CoxPHFitter()
    c.fit(fc, duration_col='duration', event_col='event',
          weights_col='weights', cluster_col='subject_id')
    hr = np.exp(c.params_.values[0])
    ci = np.exp(c.confidence_intervals_.values[0])
    p  = c.summary['p'].values[0]
    print(f'  {label}: HR={hr:.4f} ({ci[0]:.4f}-{ci[1]:.4f}), P={p:.6f}, events={fc["event"].sum()}, N={len(fc)}')
    return hr, ci, p

# ---- Primary (reference) ----
df['tte'] = np.minimum(
    np.where(df['outcome_occurred']==1, df['outcome_hour_from_landmark'], df['fu_hours']), TAU)
print('Primary (landmark start, 0-672h):')
hr0, ci0, p0 = run_cox(df)

# ---- Grace period boundaries ----
# Grace period = [0, 48] hours inclusive, consistent with Methods:
# "48-hour grace period" with sustained fulfillment "for at least 24 continuous hours within"
mask_gp_event  = (df['outcome_occurred']==1) & (df['outcome_hour_from_landmark'] <= GRACE)
mask_gp_censor = (df['outcome_occurred']==0) & (df['fu_hours'] <= GRACE)

n_gp_event  = mask_gp_event.sum()
n_gp_censor = mask_gp_censor.sum()
print(f'\nGrace period [0, 48h] inclusive:')
print(f'  Events within GP: {n_gp_event} ({n_gp_event/505*100:.1f}% of all events)')
print(f'  Censored within GP: {n_gp_censor} ({n_gp_censor/len(df)*100:.1f}% of clones)')
print(f'  Median fup of GP-censored clones: {df.loc[mask_gp_censor, "fu_hours"].median():.1f}h')
print(f'  These contribute risk time only (no events), excluded from S7a, kept in S7b')

for arm in ['spectrum_reduction','continue_broad']:
    m_evt = mask_gp_event & (df['assigned_strategy']==arm)
    m_cen = mask_gp_censor & (df['assigned_strategy']==arm)
    n_arm = (df['assigned_strategy']==arm).sum()
    print(f'  {arm}: GP events={m_evt.sum()} ({m_evt.sum()/n_arm*100:.1f}%), GP censored={m_cen.sum()} ({m_cen.sum()/n_arm*100:.1f}%)')

# ---- S7a: Exclude GP events + GP-censored clones; follow-up starts at 48h ----
mask_drop = mask_gp_event | mask_gp_censor
df_s7a = df[~mask_drop].copy()
df_s7a['tte'] = np.minimum(
    np.where(df_s7a['outcome_occurred']==1,
             df_s7a['outcome_hour_from_landmark'] - GRACE,
             df_s7a['fu_hours'] - GRACE), TAU - GRACE)

# Assert no negative or zero tte (all retained clones have fu_hours > 48)
n_bad = (df_s7a['tte'] <= 0).sum()
if n_bad > 0:
    print(f'\n  WARNING: {n_bad} clones with tte <= 0 after shift, applying floor at 0.1h')
df_s7a['tte'] = np.maximum(df_s7a['tte'], 0.1)

print(f'\nS7a: Excluded {mask_drop.sum()} clones ({n_gp_event} GP events + {n_gp_censor} GP censored), fup from 48h to 624h:')
print(f'  Tte range: {df_s7a["tte"].min():.2f}h - {df_s7a["tte"].max():.2f}h (expected max: {TAU-GRACE}h)')
hr_s7a, ci_s7a, p_s7a = run_cox(df_s7a)

# ---- S7b: Exclude only GP events; GP-censored clones kept with landmark start ----
# These clones (fu_hours <= 48) contribute short risk time with zero events,
# which minimally affects the early-period hazard estimate
mask_drop2 = mask_gp_event
df_s7b = df[~mask_drop2].copy()
df_s7b['tte'] = np.minimum(
    np.where(df_s7b['outcome_occurred']==1, df_s7b['outcome_hour_from_landmark'], df_s7b['fu_hours']), TAU)

print(f'\nS7b: Excluded {n_gp_event} GP events only, GP-censored clones kept (risk time only), landmark start:')
run_cox(df_s7b)
