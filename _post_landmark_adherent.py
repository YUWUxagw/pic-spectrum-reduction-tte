"""Post-landmark culture density: adherent patients (real comparison)"""
import pandas as pd, numpy as np
from scipy.stats import mannwhitneyu

df = pd.read_csv(r'F:\test\output\cloned_trials_weighted.csv')
micro = pd.read_csv(r'F:\test\output\micro_isolates_clean.csv')
red = pd.read_csv(r'F:\test\output\spectrum_reduction_events.csv')

df['episode_id'] = df['episode_id'].astype(str)
micro['episode_id'] = micro['episode_id'].astype(str)
red['episode_id'] = red['episode_id'].astype(str)
red['landmark_hour'] = red['landmark_hour'].astype(int)

# Merge reduction info
df = df.merge(red[['episode_id','landmark_hour','reduction_occurred','escalation_occurred']],
              on=['episode_id','landmark_hour'], how='left')

# Adherent clones (unique trials):
# Reduction arm adherent: assigned to reduction AND reduction_occurred == 1
mask_red_adh = (df['assigned_strategy'] == 'spectrum_reduction') & (df['reduction_occurred'] == 1)
# Continue-broad arm adherent: assigned to continue AND no reduction AND no escalation
mask_cb_adh = (df['assigned_strategy'] == 'continue_broad') & (df['reduction_occurred'] != 1) & (df['escalation_occurred'] != 1)

# Get unique episode-landmark combinations
red_trials = df[mask_red_adh][['episode_id','landmark_hour']].drop_duplicates()
cb_trials  = df[mask_cb_adh][['episode_id','landmark_hour']].drop_duplicates()

print(f'Reduction adherent trials: {len(red_trials)}')
print(f'Continue-broad adherent trials: {len(cb_trials)}')

# Merge with micro data
red_micro = red_trials.merge(micro, on='episode_id', how='inner')
cb_micro  = cb_trials.merge(micro, on='episode_id', how='inner')

# Post-landmark: culture_hour_from_icu > landmark_hour AND <= landmark_hour + 672
red_micro['post'] = (red_micro['culture_hour_from_icu'] > red_micro['landmark_hour']) & \
                     (red_micro['culture_hour_from_icu'] <= red_micro['landmark_hour'] + 672)
cb_micro['post'] = (cb_micro['culture_hour_from_icu'] > cb_micro['landmark_hour']) & \
                     (cb_micro['culture_hour_from_icu'] <= cb_micro['landmark_hour'] + 672)

# Per-trial post-landmark culture count
red_post = red_micro.groupby(['episode_id','landmark_hour'])['post'].sum()
cb_post  = cb_micro.groupby(['episode_id','landmark_hour'])['post'].sum()

print('\n=== Post-landmark cultures within 28-day follow-up ===')
for label, data in [('Spectrum Reduction (adherent)', red_post), ('Continue Broad (adherent)', cb_post)]:
    n = len(data)
    mean_v = data.mean()
    sd_v = data.std()
    med_v = data.median()
    q25 = data.quantile(0.25)
    q75 = data.quantile(0.75)
    density = data / 28.0
    print(f'\n{label} (n={n} trials):')
    print(f'  Cultures: mean={mean_v:.2f} (SD={sd_v:.2f}), median={med_v:.1f}')
    print(f'  IQR: {q25:.1f} - {q75:.1f}')
    print(f'  Range: {data.min()} - {data.max()}')
    print(f'  Density/day: mean={density.mean():.3f} (SD={density.std():.3f})')

# Statistical comparison
stat, p = mannwhitneyu(red_post.values, cb_post.values)
cliff_d = stat / (len(red_post) * len(cb_post))
print(f'\nMann-Whitney: U={stat:.1f}, P={p:.4f}')
print(f'Cliff delta: {cliff_d:.4f} (0.5 = no difference)')

# Also compare crude event detection rates
red_evt = df[mask_red_adh]['outcome_occurred'].sum()
cb_evt  = df[mask_cb_adh]['outcome_occurred'].sum()
n_red = mask_red_adh.sum()
n_cb  = mask_cb_adh.sum()
print(f'\nOutcome events per adherent clone:')
print(f'  Reduction: {red_evt}/{n_red} ({red_evt/n_red*100:.1f}%)')
print(f'  Continue:  {cb_evt}/{n_cb} ({cb_evt/n_cb*100:.1f}%)')
