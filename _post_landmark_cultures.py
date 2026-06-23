"""Post-landmark culture intensity analysis"""
import pandas as pd, numpy as np
from scipy.stats import mannwhitneyu

df = pd.read_csv(r'F:\test\output\cloned_trials_weighted.csv')
micro = pd.read_csv(r'F:\test\output\micro_isolates_clean.csv')

df['episode_id'] = df['episode_id'].astype(str)
micro['episode_id'] = micro['episode_id'].astype(str)

merged = df[['episode_id','landmark_hour','assigned_strategy','subject_id']].merge(
    micro[['episode_id','culture_hour_from_icu']], on='episode_id', how='inner')

print(f'Clones with culture data: {merged["episode_id"].nunique()} / {df["episode_id"].nunique()}')

# Post-landmark = culture_hour_from_icu > landmark_hour AND within 28d follow-up
merged['post'] = (merged['culture_hour_from_icu'] > merged['landmark_hour']) & \
                 (merged['culture_hour_from_icu'] <= merged['landmark_hour'] + 672)

# Aggregate per clone
clone_post = merged.groupby(['episode_id','landmark_hour','assigned_strategy'])['post'].sum().reset_index()
clone_post.columns = ['episode_id','landmark_hour','assigned_strategy','n_post_cultures']

print('\n=== Post-landmark cultures within 28-day follow-up ===')
for arm in ['spectrum_reduction', 'continue_broad']:
    sub = clone_post[clone_post['assigned_strategy'] == arm]
    n = sub['n_post_cultures']
    density = n / 28.0
    print(f'\n{arm} (n={len(sub)} clones):')
    print(f'  Cultures: mean={n.mean():.2f} (SD={n.std():.2f}), median={n.median():.1f}')
    print(f'  IQR: {n.quantile(0.25):.1f} - {n.quantile(0.75):.1f}')
    print(f'  Range: {n.min()} - {n.max()}')
    print(f'  Density/day: mean={density.mean():.3f} (SD={density.std():.3f})')

# Test
d_red = clone_post[clone_post['assigned_strategy']=='spectrum_reduction']['n_post_cultures']
d_cb  = clone_post[clone_post['assigned_strategy']=='continue_broad']['n_post_cultures']
stat, p = mannwhitneyu(d_red, d_cb)
print(f'\nMann-Whitney: U={stat:.1f}, P={p:.4f}')
print(f'Effect size (Cliff delta):', (stat / (len(d_red) * len(d_cb))))
