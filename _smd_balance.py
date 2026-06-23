"""IPCW-weighted covariate balance: SMD before/after weighting"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv(r'F:\test\output\cloned_trials_weighted.csv')
TAU = 672

# 28 IPCW covariates
covs = [
    "age_years","gender","current_n_abx","current_max_spectrum_score",
    "current_anti_mrsa","current_carbapenem","current_combination_therapy",
    "current_last_resort","cumulative_systemic_abx_hours","cumulative_broad_abx_hours",
    "any_culture_obtained_before_landmark","culture_result_known_before_landmark",
    "positive_culture_known_before_landmark","blood_culture_before_landmark",
    "mechanical_ventilation_at_landmark","vasopressor_at_landmark",
    "wbc_max_24h","neutrophil_max_24h","crp_max_24h","lactate_max_24h",
    "platelet_min_24h","ph_min_24h","base_excess_min_24h","spo2_min_24h",
    "congenital_heart_disease","malignancy","hematologic_disease","prematurity",
]

# Clean names for plotting
clean_names = {
    "age_years":"Age (years)","gender":"Male sex","current_n_abx":"N active antibiotics",
    "current_max_spectrum_score":"Max spectrum score","current_anti_mrsa":"Anti-MRSA coverage",
    "current_carbapenem":"Carbapenem coverage","current_combination_therapy":"Combination therapy",
    "current_last_resort":"Salvage therapy","cumulative_systemic_abx_hours":"Cumulative abx hours",
    "cumulative_broad_abx_hours":"Cumulative broad abx hours",
    "any_culture_obtained_before_landmark":"Any culture obtained",
    "culture_result_known_before_landmark":"Culture result known",
    "positive_culture_known_before_landmark":"Positive culture known",
    "blood_culture_before_landmark":"Blood culture obtained",
    "mechanical_ventilation_at_landmark":"Mechanical ventilation",
    "vasopressor_at_landmark":"Vasopressor support","wbc_max_24h":"WBC (max)",
    "neutrophil_max_24h":"Neutrophil (max)","crp_max_24h":"CRP (max)","lactate_max_24h":"Lactate (max)",
    "platelet_min_24h":"Platelet (min)","ph_min_24h":"pH (min)","base_excess_min_24h":"Base excess (min)",
    "spo2_min_24h":"SpO2 (min)","congenital_heart_disease":"Congenital heart disease",
    "malignancy":"Malignancy","hematologic_disease":"Hematologic disease","prematurity":"Prematurity",
}

# Prepare covariates
# gender already converted to gender_male in step11 pipeline
if 'gender' not in df.columns and 'gender_male' in df.columns:
    covs = [c for c in covs if c != 'gender'] + ['gender_male']
if 'gender' in df.columns:
    df['gender_male'] = (df['gender'].astype(str).str.upper()=='M').astype(int)
    covs = [c for c in covs if c != 'gender'] + ['gender_male']
clean_names['gender_male'] = 'Male sex'

mask_red = df['assigned_strategy'] == 'spectrum_reduction'
mask_cb  = df['assigned_strategy'] == 'continue_broad'

smd_results = []
for cov in covs:
    if cov not in df.columns:
        continue
    x_red = df.loc[mask_red, cov]
    x_cb  = df.loc[mask_cb, cov]
    w_red = df.loc[mask_red, 'ipcw_weight']
    w_cb  = df.loc[mask_cb, 'ipcw_weight']

    # Raw SMD
    m1, m2 = x_red.mean(), x_cb.mean()
    s1, s2 = x_red.std(), x_cb.std()
    sp = np.sqrt((s1**2 + s2**2) / 2)
    raw_smd = abs(m1 - m2) / sp if sp > 0 else 0

    # Weighted SMD
    wm1 = np.average(x_red, weights=w_red)
    wm2 = np.average(x_cb, weights=w_cb)
    wv1 = np.average((x_red - wm1)**2, weights=w_red)
    wv2 = np.average((x_cb - wm2)**2, weights=w_cb)
    wsp = np.sqrt((wv1 + wv2) / 2)
    wtd_smd = abs(wm1 - wm2) / wsp if wsp > 0 else 0

    smd_results.append({
        'variable': clean_names.get(cov, cov),
        'raw_SMD': raw_smd,
        'weighted_SMD': wtd_smd,
    })

smd_df = pd.DataFrame(smd_results)
smd_df = smd_df.sort_values('raw_SMD', ascending=False)

# Print table
print(f"{'Variable':<35s} {'Raw SMD':>8s} {'Weighted SMD':>12s}")
print("-"*57)
for _, row in smd_df.iterrows():
    flag = ' <0.1' if row['weighted_SMD'] < 0.1 else ''
    print(f"{row['variable']:<35s} {row['raw_SMD']:8.3f} {row['weighted_SMD']:12.3f}{flag}")

n_below_01 = (smd_df['weighted_SMD'] < 0.1).sum()
print(f"\n{len(smd_df)} covariates: {n_below_01}/{len(smd_df)} with weighted SMD < 0.1 after IPCW")

# === Love Plot ===
fig, ax = plt.subplots(figsize=(8, 7), facecolor='white')
ax.set_facecolor('white')

y_pos = range(len(smd_df))
vars_ordered = smd_df['variable'].values

ax.scatter(smd_df['raw_SMD'], y_pos, color='#E74C3C', s=40, zorder=3, label='Unweighted (raw)')
ax.scatter(smd_df['weighted_SMD'], y_pos, color='#2980B9', s=40, zorder=3, label='IPCW-weighted')

# Lines connecting raw -> weighted
for i, (raw, wtd) in enumerate(zip(smd_df['raw_SMD'], smd_df['weighted_SMD'])):
    ax.plot([raw, wtd], [i, i], color='#BDC3C7', linewidth=0.8, zorder=1)

ax.axvline(x=0.1, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
ax.text(0.105, len(smd_df)+0.5, 'SMD = 0.1', fontsize=9, color='gray', va='bottom')

ax.set_yticks(y_pos)
ax.set_yticklabels(vars_ordered, fontsize=8)
ax.set_xlabel('Standardized Mean Difference', fontsize=11)
ax.set_xlim(0, max(smd_df['raw_SMD'].max(), smd_df['weighted_SMD'].max()) * 1.15)
ax.set_ylim(-1, len(smd_df))
ax.invert_yaxis()
ax.legend(fontsize=10, loc='lower right')
ax.set_title('Covariate Balance Before and After IPCW Weighting', fontsize=13, fontweight='bold')

fig.tight_layout()
fig.savefig(r'F:\test\FigureS_LovePlot.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('\nLove plot saved: F:/test/FigureS_LovePlot.png')
print('SMD table saved to smd_df')
