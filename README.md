# Code for "Antibiotic Spectrum Reduction Versus Continued Broad-Spectrum Therapy in Critically Ill Children: A Target Trial Emulation"

## Overview

This repository contains the complete analysis pipeline for a target trial emulation of antibiotic spectrum reduction in a pediatric intensive care unit, using the clone-censor-weight design. All analyses were performed in Python 3.11.

## Repository Structure

### Core Pipeline (`pipeline/`)

Sequential steps of the emulation. Run in order via `run_pipeline.py`.

| File | Description |
|------|-------------|
| `config.py` | Global parameters (grace period, tau, spectrum score thresholds, file paths) |
| `run_pipeline.py` | Master script orchestrating steps 3–11 |
| `antibiotic_dictionary.py` | Hand-curated drug-to-spectrum-score mapping; route and coverage classification |
| `utils.py` | Shared utilities (datetime parsing, logging, path resolution) |
| `step3_icu_base.py` | ICU stay extraction and episode merging |
| `step4_abx_clean.py` | Antibiotic order cleaning; three-tier missing end-time imputation |
| `step5_micro.py` | Microbiology data processing; organism classification; resistance phenotyping |
| `step6_landmark.py` | Sequential landmarking at 48, 72, and 96 hours; eligibility assessment |
| `step7_baseline.py` | Baseline covariate extraction (demographics, labs, comorbidities, microbiology) |
| `step8_reduction.py` | Spectrum reduction classification (D1–D7 hierarchy); escalation and continue-broad assignment |
| `step9_clone.py` | Clone-censor design: cloning eligible trials and artificial censoring |
| `step10_outcome.py` | Incident resistant-organism detection; deduplication; specimen-type-specific reporting lags |
| `step11_analysis.py` | IPCW estimation; primary Cox model with cluster-robust standard errors; RMST; Aalen-Johansen CIF; Fine-Gray; robustness checks |
| `_run_supplementary.py` | Supplementary analyses: dose-response, time-stratified HRs, S6 (single-drug stop) |
| `table1_baseline.py` | Baseline characteristics table generation |

### Sensitivity and Diagnostic Analyses (`root/`)

| File | Description |
|------|-------------|
| `_grace_period_sensitivity.py` | S7: Follow-up re-zeroed at grace-period end (48 h); grace-period events excluded |
| `_post_landmark_adherent.py` | Post-landmark culture intensity comparison between adherent groups |
| `_post_landmark_cultures.py` | Post-landmark culture density computation |
| `_smd_balance.py` | Covariate balance diagnostics (Love plot; SMD table before/after IPCW weighting) |

### Figure Generation (`root/`)

| File | Description |
|------|-------------|
| `figure1_strobe.py` | Figure 1: STROBE flow diagram |
| `figure2_survival_cif.py` | Figures 2–3: IPCW-weighted Kaplan-Meier and Aalen-Johansen CIF |
| `figure3_forest.py` | Figure 4: Forest plot of primary and sensitivity HRs |

## Usage

1. **Data Acquisition**: Request and download the source CSV files from the Pediatric Intensive Care (PIC) database on PhysioNet. Place the files into your local data path and update the directories in `pipeline/config.py`.
2. **Environment Setup**: Install dependencies via `pip`:
   ```bash
   pip install -r requirements.txt

## Data Availability

The Pediatric Intensive Care (PIC) database is available at PhysioNet. Access requires completion of human-subjects research training and a signed data use agreement. The PIC database was approved by the Institutional Review Board of the Children's Hospital of Zhejiang University School of Medicine (Hangzhou, China).

## Key Design Features

- **Clone-censor-weight design: Each eligible trial is cloned into two copies assigned to opposite strategies; artificial censoring is applied when observed treatment deviates from assignment; IPCW corrects for selection bias.
- **D1–D7 de-escalation hierarchy: Seven hierarchically prioritized dimensions from complete cessation (D7) to spectrum score reduction without other criteria (D1).
- **Co-primary estimands: Cause-specific HR (IPCW-weighted Cox with cluster-robust standard errors) and 28-day RMST difference (assumption-free).
- **28-covariate IPCW model: Six domains (demographics, antibiotic exposure, microbiology, organ support, laboratory values, comorbidities).
- **Extensive Sensitivity Checks: Features seven pre-specified sensitivity protocols (S1–S7) alongside conventional multivariable-adjusted Cox and propensity score analyses to guarantee econometric and clinical robustness.

## Requirements

```
python >= 3.10
pandas >= 2.0
numpy >= 1.24
lifelines >= 0.30
scikit-learn >= 1.3
scipy >= 1.10
statsmodels >= 0.14
matplotlib >= 3.7
```

## Citation

[To be added upon publication]

## License

[MIT License](LICENSE)
