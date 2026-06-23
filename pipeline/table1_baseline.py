"""Generate Table 1: Baseline Characteristics by Assigned Strategy (Cloned Population)

In the clone-censor-weight design, each eligible trial contributes one clone to each arm,
so baseline covariates are identically distributed (SMD = 0 for all variables).
"""
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from config import *

# Load data
cloned = pd.read_csv(os.path.join(OUTPUT_DIR, "cloned_trials_weighted.csv"), low_memory=False)
bl = pd.read_csv(LANDMARK_BASELINE_CSV, low_memory=False)

# Merge baseline covariates into cloned data
bl["episode_id"] = bl["episode_id"].astype(str)
bl["landmark_hour"] = bl["landmark_hour"].astype(int)
cloned["episode_id"] = cloned["episode_id"].astype(str)
cloned["landmark_hour"] = cloned["landmark_hour"].astype(int)

# Drop columns that exist in both to avoid _x/_y suffixes
overlap = [c for c in bl.columns if c in cloned.columns and c not in ["episode_id", "landmark_hour"]]
cloned_clean = cloned.drop(columns=overlap, errors="ignore")
merged = cloned_clean.merge(bl, on=["episode_id", "landmark_hour"], how="left")

A = merged[merged["assigned_strategy"] == "spectrum_reduction"]
B = merged[merged["assigned_strategy"] == "continue_broad"]
assert len(A) == len(B), f"Clone arms unequal: {len(A)} vs {len(B)}"

n = len(A)
n_total = len(merged)

# Helpers
def fmt_n_pct(num, denom):
    return f"{num} ({num/denom*100:.1f}%)"

def fmt_median_iqr(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    return f"{series.median():.1f} [{q1:.1f}–{q3:.1f}]"

def fmt_mean_sd(series):
    return f"{series.mean():.1f} ({series.std():.1f})"

def fmt_lab_full(series, decimal=1):
    valid = series.dropna()
    n_valid = len(valid)
    pct = n_valid / len(series) * 100
    med = valid.median()
    q1 = valid.quantile(0.25)
    q3 = valid.quantile(0.75)
    return f"{med:.{decimal}f} [{q1:.{decimal}f}–{q3:.{decimal}f}]", n_valid, pct

# Age groups
def age_group(age):
    if age < 28/365.25: return "neonate"
    elif age < 1: return "infant"
    elif age < 3: return "toddler"
    elif age < 12: return "child"
    else: return "adolescent"

a_ag = A["age_years"].apply(age_group).value_counts()
b_ag = B["age_years"].apply(age_group).value_counts()

lines = []
W = 44

def section(title):
    lines.append("")
    lines.append(f"  {title}")

def row(label, a_val, b_val, t_val):
    lines.append(f"  {label:<{W-2}} {a_val:>24} {b_val:>24} {t_val:>18}")

def bin_row(label, col):
    an, bn, tn = int(A[col].sum()), int(B[col].sum()), int(merged[col].sum())
    row(label, fmt_n_pct(an, n), fmt_n_pct(bn, n), fmt_n_pct(tn, n_total))

def lab_row(label, col, decimal=1):
    a_fmt, a_n, a_pct = fmt_lab_full(A[col], decimal)
    b_fmt, b_n, b_pct = fmt_lab_full(B[col], decimal)
    t_fmt, t_n, t_pct = fmt_lab_full(merged[col], decimal)
    row(label, a_fmt, b_fmt, t_fmt)
    row("  available, n (%)", fmt_n_pct(a_n, n), fmt_n_pct(b_n, n), fmt_n_pct(t_n, n_total))

# === BUILD TABLE ===
lines.append("=" * 118)
lines.append("TABLE 1. Baseline Characteristics of the Cloned Study Population by Assigned Strategy")
lines.append("=" * 118)
lines.append("")
hdr = f"{'Characteristic':<{W}} {'Spectrum Reduction':>24} {'Continue Broad':>24} {'Overall':>18}"
lines.append(hdr)
lines.append(f"{'':<{W}} {f'(n = {n})':>24} {f'(n = {n})':>24} {f'(N = {n_total})':>18}")
lines.append("-" * 118)

# DEMOGRAPHICS
section("DEMOGRAPHICS")
row("Age, years", fmt_median_iqr(A["age_years"]), fmt_median_iqr(B["age_years"]), fmt_median_iqr(merged["age_years"]))
a_m = (A["gender"] == "M").sum()
b_m = (B["gender"] == "M").sum()
t_m = (merged["gender"] == "M").sum()
row("Male sex", fmt_n_pct(a_m, n), fmt_n_pct(b_m, n), fmt_n_pct(t_m, n_total))
for ag in ["neonate", "infant", "toddler", "child", "adolescent"]:
    an = a_ag.get(ag, 0)
    bn = b_ag.get(ag, 0)
    tn = an + bn
    row(f"  {ag}", fmt_n_pct(an, n), fmt_n_pct(bn, n), fmt_n_pct(tn, n_total))
row("Calendar year", fmt_median_iqr(A["calendar_year"]), fmt_median_iqr(B["calendar_year"]), fmt_median_iqr(merged["calendar_year"]))
row("ICU LOS before landmark, h",
    fmt_median_iqr(A["icu_los_before_landmark_hours"]),
    fmt_median_iqr(B["icu_los_before_landmark_hours"]),
    fmt_median_iqr(merged["icu_los_before_landmark_hours"]))

# ANTIBIOTIC EXPOSURE
section("ANTIBIOTIC EXPOSURE AT LANDMARK")
row("Active systemic antibiotics, n",
    fmt_median_iqr(A["current_n_abx"]), fmt_median_iqr(B["current_n_abx"]),
    fmt_median_iqr(merged["current_n_abx"]))
row("Max spectrum score",
    fmt_mean_sd(A["current_max_spectrum_score"]), fmt_mean_sd(B["current_max_spectrum_score"]),
    fmt_mean_sd(merged["current_max_spectrum_score"]))
for lbl, col in [
    ("Anti-pseudomonal coverage", "current_anti_pseudomonal"),
    ("Anti-MRSA coverage", "current_anti_mrsa"),
    ("Carbapenem", "current_carbapenem"),
    ("Anti-anaerobe coverage", "current_anti_anaerobe"),
    ("Combination therapy (≥2 agents)", "current_combination_therapy"),
    ("Salvage therapy", "current_last_resort"),
]:
    bin_row(lbl, col)

# CUMULATIVE ABX
section("CUMULATIVE ANTIBIOTIC EXPOSURE BEFORE LANDMARK")
row("Systemic antibiotics, h",
    fmt_median_iqr(A["cumulative_systemic_abx_hours"]),
    fmt_median_iqr(B["cumulative_systemic_abx_hours"]),
    fmt_median_iqr(merged["cumulative_systemic_abx_hours"]))
row("Broad-spectrum antibiotics, h",
    fmt_median_iqr(A["cumulative_broad_abx_hours"]),
    fmt_median_iqr(B["cumulative_broad_abx_hours"]),
    fmt_median_iqr(merged["cumulative_broad_abx_hours"]))

# MICROBIOLOGY
section("MICROBIOLOGY BASELINE")
bin_row("Any culture obtained", "any_culture_obtained_before_landmark")
bin_row("Culture result known at landmark", "culture_result_known_before_landmark")
bin_row("Positive culture known", "positive_culture_known_before_landmark")
bin_row("Blood culture obtained", "blood_culture_before_landmark")
row("Number of cultures",
    fmt_median_iqr(A["number_of_cultures_before_landmark"]),
    fmt_median_iqr(B["number_of_cultures_before_landmark"]),
    fmt_median_iqr(merged["number_of_cultures_before_landmark"]))

# LABORATORY VALUES
section("LABORATORY VALUES (24h before landmark)")
for lbl, col, dec in [
    ("WBC, ×10⁹/L", "wbc_max_24h", 1),
    ("Neutrophil, %", "neutrophil_max_24h", 1),
    ("C-reactive protein, mg/L", "crp_max_24h", 1),
    ("Lactate, mmol/L", "lactate_max_24h", 1),
    ("Platelet, ×10⁹/L", "platelet_min_24h", 1),
    ("pH", "ph_min_24h", 2),
    ("Base excess, mmol/L", "base_excess_min_24h", 1),
    ("SpO₂, %", "spo2_min_24h", 1),
]:
    lab_row(lbl, col, dec)

# ORGAN SUPPORT
section("ORGAN SUPPORT AT LANDMARK")
bin_row("Mechanical ventilation", "mechanical_ventilation_at_landmark")
bin_row("Vasopressor support", "vasopressor_at_landmark")

# WEIGHT
section("ANTHROPOMETRICS")
lab_row("Weight, kg", "nearest_weight_pre_landmark_kg", 1)

# COMORBIDITIES
section("COMORBIDITIES (ICD-10-CN)")
for lbl, col in [
    ("Congenital heart disease", "congenital_heart_disease"),
    ("Prematurity", "prematurity"),
    ("Malignancy", "malignancy"),
    ("Hematologic disease", "hematologic_disease"),
    ("Chronic kidney disease", "chronic_kidney_disease"),
    ("Neurologic disease", "neurologic_disease"),
    ("Chronic lung disease", "chronic_lung_disease"),
    ("Immunodeficiency", "immunodeficiency"),
    ("Postoperative status", "postoperative_status"),
]:
    bin_row(lbl, col)

# FOOTER
lines.append("")
lines.append("-" * 118)
lines.append("Values are median [IQR], mean (SD), or n (%). Laboratory values are median [IQR] for")
lines.append("available data; coverage reported as 'available, n (%)'. By design of the clone-censor-")
lines.append("weight method, each eligible trial contributes one clone to each strategy. All baseline")
lines.append("covariates are therefore identically distributed between assigned strategy arms")
lines.append("(standardized mean difference [SMD] = 0 for all variables).")
lines.append("=" * 118)

output = "\n".join(lines)

# Save main table
outpath = os.path.join(OUTPUT_DIR, "baseline_table_1.txt")
with open(outpath, "w", encoding="utf-8") as f:
    f.write(output)
print(f"Saved: {outpath}")
print(f"Table 1: {n} per arm, {n_total} total clones ({n} unique trials, each cloned × 2 arms)")
print("All covariates identically distributed by design (SMD = 0).")
