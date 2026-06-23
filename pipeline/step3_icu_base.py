"""Step 3: Build ICU Base Cohort (icu_base.csv)
Implements Sections 3.1-3.4 from the processing plan:
- Build independent ICU episodes with merge rule (gap <= 48h)
- Apply eligibility criteria (age < 18, LOS >= 48h)
- Handle death times with priority hierarchy
- Output icu_base.csv with raw_icu_intervals preserved
"""
import pandas as pd
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from utils import parse_datetime, classify_age, build_episodes, hours_between, log


def run_step3(logfile=None):
    log("=" * 60, logfile)
    log("Step 3: Building ICU Base Cohort", logfile)
    log("=" * 60, logfile)

    # 1. Load tables
    log("Loading PATIENTS.csv ...", logfile)
    patients = pd.read_csv(PATIENTS_CSV, low_memory=False)
    patients.columns = [c.lower() for c in patients.columns]
    log(f"  {len(patients)} patients", logfile)

    log("Loading ADMISSIONS.csv ...", logfile)
    admissions = pd.read_csv(ADMISSIONS_CSV, low_memory=False)
    admissions.columns = [c.lower() for c in admissions.columns]
    log(f"  {len(admissions)} admissions", logfile)

    log("Loading ICUSTAYS.csv ...", logfile)
    icustays = pd.read_csv(ICUSTAYS_CSV, low_memory=False)
    icustays.columns = [c.lower() for c in icustays.columns]
    log(f"  {len(icustays)} ICU stays", logfile)

    # 2. Time standardization
    log("Standardizing datetimes...", logfile)
    patients["dob"] = parse_datetime(patients["dob"])
    patients["dod"] = parse_datetime(patients["dod"])

    admissions["admittime"] = parse_datetime(admissions["admittime"])
    admissions["dischtime"] = parse_datetime(admissions["dischtime"])
    admissions["deathtime"] = parse_datetime(admissions["deathtime"])

    icustays["intime"] = parse_datetime(icustays["intime"])
    icustays["outtime"] = parse_datetime(icustays["outtime"])

    # Ensure IDs are strings (only for columns that exist in each table)
    for col in ["subject_id"]:
        patients[col] = patients[col].astype(str)
    for col in ["subject_id", "hadm_id"]:
        admissions[col] = admissions[col].astype(str)
    for col in ["subject_id", "hadm_id", "icustay_id"]:
        icustays[col] = icustays[col].astype(str)

    # 3. Build ICU episodes (Section 3.1)
    log("Building ICU Episodes...", logfile)
    episodes = build_episodes(icustays)
    log(f"  Initial: {len(episodes)} episodes", logfile)

    # 4. Join patient demographics
    episodes = episodes.merge(
        patients[["subject_id", "gender", "dob", "dod"]],
        on="subject_id", how="left"
    )
    # 5. Join admission info
    adm_cols = ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                 "admission_department", "hospital_expire_flag"]
    # Make unique by hadm_id (keep first)
    adm_dedup = admissions[adm_cols].drop_duplicates(subset=["subject_id", "hadm_id"], keep="first")
    episodes = episodes.merge(adm_dedup, on=["subject_id", "hadm_id"], how="left")

    # 6. Compute age (Section 3.2)
    log("Computing ages...", logfile)
    ref_time = episodes["episode_intime"]
    episodes["age_days"] = (ref_time - episodes["dob"]).dt.total_seconds() / 86400.0
    episodes["age_years"] = episodes["age_days"] / 365.25
    episodes["age_group"] = pd.cut(
        episodes["age_years"],
        bins=[0, 28/365.25, 1, 3, 12, 18, float("inf")],
        labels=["neonate", "infant", "toddler", "child", "adolescent", "adult"],
        right=False,
    )

    # 7. Get first_careunit from original icustays
    first_cu = icustays.groupby(["subject_id", "hadm_id"])["first_careunit"].first().reset_index()
    episodes = episodes.merge(first_cu, on=["subject_id", "hadm_id"], how="left")

    # 8. Death time priority (Section 3.3)
    log("Processing death times...", logfile)
    episodes["death_time"] = episodes["deathtime"].fillna(episodes["dod"])
    episodes["death_hour_from_icu"] = hours_between(episodes["death_time"], episodes["episode_intime"])

    # 9. Hospital LOS before ICU
    episodes["hospital_los_before_icu_hours"] = hours_between(
        episodes["episode_intime"], episodes["admittime"]
    )
    episodes["hospital_los_negative_flag"] = (episodes["hospital_los_before_icu_hours"] < 0).astype(int)
    episodes.loc[episodes["hospital_los_before_icu_hours"] < 0, "hospital_los_before_icu_hours"] = 0

    # 10. Apply eligibility criteria (Section 3.2)
    log(f"  Before eligibility: {len(episodes)} episodes", logfile)

    # Age < 18
    mask_age = episodes["age_years"] < 18
    log(f"  Age < 18: {mask_age.sum()}", logfile)

    # ICU stay valid
    mask_valid = episodes["episode_intime"].notna() & episodes["episode_outtime"].notna()
    log(f"  ICU valid: {mask_valid.sum()}", logfile)

    # LOS >= 48h
    mask_los = episodes["episode_los_hours"] >= 48
    log(f"  LOS >= 48h: {mask_los.sum()}", logfile)

    eligible = episodes[mask_age & mask_valid & mask_los].copy()
    log(f"  Eligible: {len(eligible)} episodes", logfile)

    # 11. Select output columns (Section 3.4)
    output_cols = [
        "subject_id", "hadm_id", "episode_sequence", "episode_id",
        "icustay_id_list", "episode_intime", "episode_outtime", "episode_los_hours",
        "gender", "dob", "age_days", "age_years", "age_group", "first_careunit",
        "death_time", "death_hour_from_icu",
        "raw_icu_intervals", "hospital_los_before_icu_hours", "hospital_los_negative_flag",
    ]
    output = eligible[output_cols].reset_index(drop=True)

    # 12. Save
    output.to_csv(ICU_BASE_CSV, index=False, encoding="utf-8")
    log(f"-> Saved {ICU_BASE_CSV}: {len(output):,} rows, {len(output.columns)} cols", logfile)

    # 13. Statistics
    stats = {
        "total_episodes": len(episodes),
        "eligible_episodes": len(output),
        "unique_subjects": int(output["subject_id"].nunique()),
        "episodes_per_subject_mean": round(len(output) / output["subject_id"].nunique(), 4),
        "age_groups": output["age_group"].value_counts().to_dict(),
        "gender": output["gender"].value_counts().to_dict(),
    }
    import json
    with open(os.path.join(OUTPUT_DIR, "step3_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log(f"  Statistics: {json.dumps(stats, indent=2, ensure_ascii=False)}", logfile)
    log("Step 3 complete!", logfile)
    return output


if __name__ == "__main__":
    logfile = os.path.join(OUTPUT_DIR, "processing_log.txt")
    run_step3(logfile)
