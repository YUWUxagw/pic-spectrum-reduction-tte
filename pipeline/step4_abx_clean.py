"""Step 4: Antibiotic Data Cleaning (abx_orders_clean.csv)
Implements Sections 4.1-4.2 from the processing plan:
- Build antibiotic dictionary from PRESCRIPTIONS unique drugs
- End_time imputation (3 priority levels)
- Route filtering (exclude topical, ophthalmic, otic, inhaled)
- Join with ICU base cohort
- Activity hierarchy (L1-L4)
"""
import pandas as pd
import numpy as np
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from utils import parse_datetime, hours_between, log, is_active_at
from antibiotic_dictionary import build_antibiotic_dictionary, load_antibiotic_dictionary


def run_step4(logfile=None, icu_base=None):
    log("=" * 60, logfile)
    log("Step 4: Antibiotic Data Cleaning", logfile)
    log("=" * 60, logfile)

    # 1. Load ICU base
    if icu_base is None:
        log("Loading ICU base cohort...", logfile)
        icu_base = pd.read_csv(ICU_BASE_CSV, low_memory=False)
        icu_base["episode_intime"] = parse_datetime(icu_base["episode_intime"])
        icu_base["episode_outtime"] = parse_datetime(icu_base["episode_outtime"])
    for col in ["subject_id", "hadm_id", "episode_id"]:
        icu_base[col] = icu_base[col].astype(str)
    log(f"  {len(icu_base)} episodes", logfile)

    # 2. Load PRESCRIPTIONS
    log("Loading PRESCRIPTIONS.csv ...", logfile)
    rx = pd.read_csv(PRESCRIPTIONS_CSV, low_memory=False)
    rx.columns = [c.lower() for c in rx.columns]
    log(f"  Total prescriptions: {len(rx):,}", logfile)

    # Clean drug names
    rx["drug_name"] = rx["drug_name"].str.strip()

    # 3. Build / load antibiotic dictionary
    log("Building antibiotic dictionary...", logfile)
    abx_dict = build_antibiotic_dictionary(rx, ABX_DICT_CSV)
    n_abx = abx_dict["is_antibiotic"].sum()
    n_sys = abx_dict["is_systemic_antibiotic"].sum()
    n_unknown = abx_dict["needs_review"].sum()
    log(f"  Dictionary: {len(abx_dict)} unique drugs, {n_abx} antibiotics, "
        f"{n_sys} systemic, {n_unknown} need review", logfile)

    # 4. Map PRESCRIPTIONS to dictionary
    rx = rx.merge(
        abx_dict[["drug_raw", "drug_std", "is_antibiotic", "is_systemic_antibiotic",
                   "route", "class", "spectrum_score", "aware_group",
                   "anti_pseudomonal", "anti_mrsa", "carbapenem", "anti_anaerobe",
                   "salvage_therapy", "glycopeptide_or_oxazolidinone",
                   "coverage_gnb", "coverage_pseudomonas", "coverage_mrsa",
                   "coverage_anaerobe", "coverage_atypical", "coverage_fungal",
                   "standard_dosing_interval_hours"]],
        left_on="drug_name", right_on="drug_raw", how="left"
    )

    # 5. Filter to systemic antibiotics only
    log("Filtering to systemic antibiotics...", logfile)
    abx = rx[rx["is_systemic_antibiotic"] == 1].copy()
    log(f"  Systemic antibiotic orders: {len(abx):,}", logfile)

    # 6. Time standardization (Section 2.1)
    log("Time standardization...", logfile)
    # PRESCRIPTIONS has STARTDATE, ENDDATE (date columns, sometimes with time in them)
    abx["start_time"] = parse_datetime(abx["startdate"])
    abx["end_time"] = parse_datetime(abx["enddate"])

    # Date-only check (Section 2.2)
    abx["start_precision"] = "datetime"
    abx["end_precision"] = "datetime"
    start_valid = abx["start_time"].dropna()
    if len(start_valid) > 0:
        date_only_mask = (start_valid.dt.hour == 0) & (start_valid.dt.minute == 0) & \
                         (start_valid.dt.second == 0)
        date_only_idx = start_valid[date_only_mask].index
        abx.loc[date_only_idx, "start_precision"] = "date_only"

    end_valid = abx["end_time"].dropna()
    if len(end_valid) > 0:
        date_only_mask = (end_valid.dt.hour == 0) & (end_valid.dt.minute == 0) & \
                         (end_valid.dt.second == 0)
        date_only_idx = end_valid[date_only_mask].index
        abx.loc[date_only_idx, "end_precision"] = "date_only"

    abx["date_only_flag"] = ((abx["start_precision"] == "date_only") |
                              (abx["end_precision"] == "date_only")).astype(int)

    log(f"  ENDDATE missing: {abx['end_time'].isna().sum():,} ({abx['end_time'].isna().mean()*100:.1f}%)", logfile)
    log(f"  Date-only precision: {abx['date_only_flag'].sum():,}", logfile)

    # 7. End_time imputation (Section 4.2.1)
    log("Imputing missing end_time...", logfile)
    abx["end_time_imputed"] = 0
    abx["end_time_imputed_type"] = "original"

    # Priority 1: Next order's start_time as end_time
    abx = abx.sort_values(["subject_id", "hadm_id", "drug_std", "start_time"])
    for (subj, hadm, drug), grp in abx.groupby(["subject_id", "hadm_id", "drug_std"],
                                                sort=False):
        if len(grp) > 1:
            grp = grp.sort_values("start_time")
            for i in range(len(grp) - 1):
                curr_idx = grp.index[i]
                next_idx = grp.index[i + 1]
                if pd.isna(abx.at[curr_idx, "end_time"]) or \
                   abx.at[curr_idx, "end_time_imputed"] != 2:
                    abx.at[curr_idx, "end_time"] = abx.at[next_idx, "start_time"]
                    abx.at[curr_idx, "end_time_imputed"] = 1
                    abx.at[curr_idx, "end_time_imputed_type"] = "next_order_start"

    # Priority 2 (MAR-based imputation, Section 4.2.1) - requires CHARTEVENTS MAR data
    # Framework: look up last MAR administration time for same patient + drug_std,
    # impute end_time = last_mar_time + dosing_interval/2
    # Currently gated behind CHARTEVENTS availability.
    # When MAR data is available, uncomment and adapt the block below.
    # mar_data_available = False
    # if mar_data_available:
    #     for idx in abx[abx["end_time"].isna()].index:
    #         mar_rows = chartevents_mar[
    #             (chartevents_mar["subject_id"] == abx.at[idx, "subject_id"]) &
    #             (chartevents_mar["drug_std"] == abx.at[idx, "drug_std"])
    #         ]
    #         if len(mar_rows) > 0:
    #             last_mar = mar_rows["mar_admin_time"].max()
    #             dose_h = abx.at[idx, "standard_dosing_interval_hours"] or 8
    #             abx.at[idx, "end_time"] = last_mar + pd.Timedelta(hours=dose_h / 2)
    #             abx.at[idx, "end_time_imputed"] = 1
    #             abx.at[idx, "end_time_imputed_type"] = "mar_based"

    # Priority 3 (simplified - use ICU-level median when unavailable)
    # Group by ICU stay median for each drug_std
    log("  Computing Level 3 imputations (median duration per ICU)...", logfile)
    abx["duration_hours"] = hours_between(abx["end_time"], abx["start_time"])
    valid_dur = abx[abx["duration_hours"].notna() & (abx["duration_hours"] > 0) &
                    (abx["duration_hours"] < 720)]  # exclude > 30 day durations
    medians = valid_dur.groupby(["icustay_id", "drug_std"])["duration_hours"].median()

    still_missing = abx["end_time"].isna()
    if still_missing.any():
        for idx in abx[still_missing].index:
            key = (abx.at[idx, "icustay_id"], abx.at[idx, "drug_std"])
            if key in medians.index:
                med_val = medians[key]
                abx.at[idx, "end_time"] = abx.at[idx, "start_time"] + pd.Timedelta(
                    hours=float(med_val))
                abx.at[idx, "end_time_imputed"] = 1
                abx.at[idx, "end_time_imputed_type"] = "level3_median"

    # Drop remaining rows with no end_time
    before = len(abx)
    abx = abx[abx["end_time"].notna()].copy()
    log(f"  Dropped {before - len(abx)} orders with unresolvable end_time", logfile)

    # 8. Join with ICU base cohort (Section 4.2.4)
    log("Joining with ICU base cohort...", logfile)
    abx["subject_id"] = abx["subject_id"].astype(str)
    abx["hadm_id"] = abx["hadm_id"].astype(str)

    icu_join = icu_base[["subject_id", "hadm_id", "episode_id",
                          "episode_intime", "episode_outtime"]].copy()
    abx = abx.merge(icu_join, on=["subject_id", "hadm_id"], how="inner")
    log(f"  After ICU join: {len(abx):,} orders", logfile)

    # 9. Calculate relative times (Section 2.3)
    abx["abx_start_hour"] = hours_between(abx["start_time"], abx["episode_intime"])
    abx["abx_end_hour"] = hours_between(abx["end_time"], abx["episode_intime"])

    # 10. Mark broad-spectrum
    abx["is_broad_spectrum"] = (abx["spectrum_score"] >= 3).astype(int)
    abx["is_systemic"] = 1

    # 11. Activity Hierarchy (Section 4.2.5)
    # L1 (MAR) and L2 (Infusion Stop) require CHARTEVENTS/PROCEDUREEVENTS data.
    # When unavailable, default to L3 (order interval) or L4 (imputed).
    abx["abx_activity_source"] = "order"  # L3 default
    abx.loc[abx["end_time_imputed"] == 1, "abx_activity_source"] = "imputed"  # L4
    abx["mar_data_absent_flag"] = 1  # L1 MAR data not processed (Section 4.2.5 Note)

    # 12. Select output columns (Section 4.2.6)
    output_cols = [
        "subject_id", "hadm_id", "episode_id", "icustay_id",
        "drug_name", "drug_std", "class", "spectrum_score",
        "anti_pseudomonal", "anti_mrsa", "carbapenem", "anti_anaerobe",
        "salvage_therapy", "glycopeptide_or_oxazolidinone",
        "is_systemic", "is_broad_spectrum",
        "start_time", "end_time", "abx_start_hour", "abx_end_hour",
        "date_only_flag", "end_time_imputed", "end_time_imputed_type",
        "abx_activity_source", "mar_data_absent_flag",
        "standard_dosing_interval_hours",
        "coverage_gnb", "coverage_pseudomonas", "coverage_mrsa",
        "coverage_anaerobe", "coverage_atypical", "coverage_fungal",
    ]
    output = abx[output_cols].reset_index(drop=True)

    # 13. Save
    output.to_csv(ABX_ORDERS_CLEAN_CSV, index=False, encoding="utf-8")
    log(f"-> Saved {ABX_ORDERS_CLEAN_CSV}: {len(output):,} rows, {len(output.columns)} cols", logfile)

    # 14. Statistics
    imputed_pct = output["end_time_imputed"].mean() * 100
    log(f"  End_time imputed: {imputed_pct:.1f}%", logfile)
    impute_types = output["end_time_imputed_type"].value_counts().to_dict()
    log(f"  Imputation types: {impute_types}", logfile)
    log(f"  Date-only flagged: {output['date_only_flag'].sum():,}", logfile)
    log("Step 4 complete!", logfile)

    return output


if __name__ == "__main__":
    logfile = os.path.join(OUTPUT_DIR, "processing_log.txt")
    run_step4(logfile)
