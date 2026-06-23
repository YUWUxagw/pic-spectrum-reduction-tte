"""Figures 2: KM survival + AJ CIF (combined) — using pipeline event_status"""
import pandas as pd, numpy as np
from lifelines import KaplanMeierFitter, AalenJohansenFitter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_csv(r'F:\test\output\cloned_trials_weighted.csv')
TAU = 672

# Time-to-event using pipeline's own event_status (0=censor, 1=outcome, 2=competing death)
df['fu'] = (pd.to_datetime(df['followup_end_time']) - pd.to_datetime(df['followup_start_time'])).dt.total_seconds()/3600
df['tte'] = np.minimum(np.where(df['outcome_occurred']==1, df['outcome_hour_from_landmark'], df['fu']), TAU)

mask_red = df['assigned_strategy'] == 'spectrum_reduction'
mask_cb  = df['assigned_strategy'] == 'continue_broad'

colors = {'red': '#27AE60', 'cb': '#E74C3C'}
lbls   = {'red': 'Spectrum Reduction', 'cb': 'Continue Broad'}

fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

# === Panel A: KM survival ===
for arm, mask in [('red',mask_red), ('cb',mask_cb)]:
    sub = df[mask].copy()
    kmf = KaplanMeierFitter()
    kmf.fit(sub['tte'], event_observed=(sub['event_status']==1).astype(int),
            weights=sub['ipcw_weight'], label=lbls[arm])
    kmf.plot_survival_function(ax=axA, color=colors[arm], linewidth=2.5)

axA.set_xlim(0,TAU); axA.set_ylim(0.72,1.0)
axA.set_xlabel('Time from Landmark (hours)', fontsize=11)
axA.set_ylabel('Event-Free Survival Probability', fontsize=11)
axA.set_title('A. IPCW-Weighted Kaplan-Meier Event-Free Survival', fontsize=12, fontweight='bold')
axA.legend(fontsize=10)
for x,l in [(168,'7d'),(336,'14d'),(672,'28d')]:
    axA.axvline(x=x,color='gray',linestyle='--',alpha=0.3)
    axA.text(x-35,0.725,l,fontsize=8,color='gray')
axA.annotate('94.7%',xy=(672,0.96),xytext=(690,0.96),fontsize=10,color=colors['red'],fontweight='bold',ha='left')
axA.annotate('76.7%',xy=(672,0.755),xytext=(690,0.755),fontsize=10,color=colors['cb'],fontweight='bold',ha='left')

# === Panel B: AJ CIF ===
for arm, mask in [('red',mask_red), ('cb',mask_cb)]:
    sub = df[mask].copy()
    ajf = AalenJohansenFitter(calculate_variance=False)
    ajf.fit(durations=sub['tte'].values/24, event_observed=sub['event_status'].values,
            weights=sub['ipcw_weight'].values, label=lbls[arm], event_of_interest=1)
    c = ajf.cumulative_density_
    t = c.index.values
    v = c.values.flatten() * 100
    axB.fill_between(t*24, v, alpha=0.12, color=colors[arm])
    axB.plot(t*24, v, color=colors[arm], linewidth=2.5, label=lbls[arm])

axB.set_xlim(0,TAU); axB.set_ylim(0,25)
axB.set_xlabel('Time from Landmark (hours)', fontsize=11)
axB.set_ylabel('Cumulative Incidence (%)', fontsize=11)
axB.set_title('B. Aalen-Johansen Cumulative Incidence\n(Death as Competing Event)', fontsize=12, fontweight='bold')
axB.legend(fontsize=10)
for x,l in [(168,'7d'),(336,'14d'),(672,'28d')]:
    axB.axvline(x=x,color='gray',linestyle='--',alpha=0.3)
    axB.text(x-35,0.7,l,fontsize=8,color='gray')
axB.annotate('21.6%',xy=(672,21.55),xytext=(690,22.2),fontsize=10,color=colors['cb'],fontweight='bold',ha='left')
axB.annotate('5.2%',xy=(672,5.23),xytext=(690,5.9),fontsize=10,color=colors['red'],fontweight='bold',ha='left')

fig.suptitle('Figure 3. IPCW-Weighted Kaplan-Meier Event-Free Survival and Aalen-Johansen Cumulative Incidence \nof Resistant-Organism Detection, by Assigned Antibiotic Strategy', fontsize=14, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(r'F:\test\Figure3_Combined.tiff', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print('Figure 3 saved — AJ CIF now uses event_status (5.23% / 21.55%)')
