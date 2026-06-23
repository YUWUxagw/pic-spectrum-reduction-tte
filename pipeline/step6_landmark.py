"""Step 6: Landmark Eligibility Screening (landmark_eligible_trials.csv)
Implements Section 6 from the processing plan:
- 3 landmark timepoints: 48h, 72h, 96h
- 7 eligibility criteria per timepoint
- Uses raw_icu_intervals for ICU presence check
- Baseline resistant organism exclusion
- Date-only precision handling
"""
import pandas as pd
import numpy as np
import sys, os, ast
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from utils import parse_datetime, hours_between, is_active_at, log


def run_step6(logfile=None, icu_base=None, abx_orders=None, micro_isolates=None,
              icd_diagnoses=None):
    log("=" * 60, logfile)
    log("Step 6: Landmark Eligibility Screening", logfile)
    log("=" * 60, logfile)

    # 1. Load inputs
    if icu_base is None:
        icu_base = pd.read_csv(ICU_BASE_CSV, low_memory=False)
        icu_base["episode_intime"] = parse_datetime(icu_base["episode_intime"])
        icu_base["episode_outtime"] = parse_datetime(icu_base["episode_outtime"])
        if "death_time" in icu_base.columns:
            icu_base["death_time"] = parse_datetime(icu_base["death_time"])
        icu_base["dob"] = parse_datetime(icu_base["dob"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            icu_base[col] = icu_base[col].astype(str)
    log(f"  ICU base: {len(icu_base)} episodes", logfile)

    if abx_orders is None:
        abx_orders = pd.read_csv(ABX_ORDERS_CLEAN_CSV, low_memory=False)
        abx_orders["start_time"] = parse_datetime(abx_orders["start_time"])
        abx_orders["end_time"] = parse_datetime(abx_orders["end_time"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            abx_orders[col] = abx_orders[col].astype(str)
    log(f"  ABX orders: {len(abx_orders)} rows", logfile)

    if micro_isolates is None:
        micro_isolates = pd.read_csv(MICRO_ISOLATES_CLEAN_CSV, low_memory=False)
        micro_isolates["culture_time"] = parse_datetime(micro_isolates["culture_time"])
        micro_isolates["imputed_report_time"] = parse_datetime(
            micro_isolates["imputed_report_time"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            micro_isolates[col] = micro_isolates[col].astype(str)
    log(f"  Micro isolates: {len(micro_isolates)} records", logfile)

    # Load ICD diagnoses for Criterion 7 (infection context)
    if icd_diagnoses is None:
        icd_diagnoses = _load_icd_diagnoses(logfile)
    log(f"  ICD diagnoses: {len(icd_diagnoses) if icd_diagnoses is not None else 0} records", logfile)

    # 2. Build landmark trials (Section 6.1)
    all_trials = []
    for lm_hour in LANDMARK_HOURS:
        trials = icu_base.copy()
        trials["landmark_hour"] = lm_hour
        trials["landmark_time"] = trials["episode_intime"] + pd.Timedelta(hours=lm_hour)
        all_trials.append(trials)

    trials = pd.concat(all_trials, ignore_index=True)
    log(f"  Initial candidate trials: {len(trials):,} ({len(icu_base)} episodes x {len(LANDMARK_HOURS)} landmarks)", logfile)

    # 3. Compute eligibility for each criterion
    log("Evaluating eligibility criteria (Section 6.2)...", logfile)

    # Criterion 1: Age < 18
    trials["age_years_at_landmark"] = (trials["landmark_time"] - trials["dob"]).dt.total_seconds() / (86400 * 365.25)
    trials["c1_age_child"] = trials["age_years_at_landmark"] < 18

    # Criterion 2: Independent ICU episode (all are by construction)

    # Criterion 3: In ICU at landmark (using raw ICU intervals)
    # Parse raw_icu_intervals from string representation
    trials["c3_in_icu"] = False
    for idx, row in trials.iterrows():
        lm_time = row["landmark_time"]
        try:
            intervals = row["raw_icu_intervals"]
            if isinstance(intervals, str):
                intervals = ast.literal_eval(intervals)
            if isinstance(intervals, list):
                for intv in intervals:
                    if isinstance(intv, (list, tuple)) and len(intv) >= 2:
                        intime = parse_datetime(pd.Series([intv[0]])).iloc[0]
                        outtime = parse_datetime(pd.Series([intv[1]])).iloc[0]
                        if pd.notna(intime) and pd.notna(outtime):
                            if intime <= lm_time < outtime:
                                trials.at[idx, "c3_in_icu"] = True
                                break
        except (ValueError, SyntaxError, TypeError):
            # Fall back to episode bounds
            if pd.notna(row["episode_intime"]) and pd.notna(row["episode_outtime"]):
                if row["episode_intime"] <= lm_time < row["episode_outtime"]:
                    trials.at[idx, "c3_in_icu"] = True

    # Criterion 4: Alive at landmark
    if "death_time" in trials.columns:
        trials["c4_alive"] = trials["death_time"].isna() | \
                             (trials["death_time"] > trials["landmark_time"])
    else:
        trials["c4_alive"] = True

    # Criterion 5: No resistant organism before landmark
    # (with imputed_report_time correction)
    micro_flat = micro_isolates[micro_isolates["resistant_organism_flag"] == 1].copy()
    trials["c5_no_resistant"] = True
    trials["pre_landmark_resistant_isolate"] = 0
    trials["unrecognized_baseline_resistant_isolate"] = 0

    if len(micro_flat) > 0:
        for idx, row in trials.iterrows():
            ep = row["episode_id"]
            lm = row["landmark_time"]
            ep_micro = micro_flat[micro_flat["episode_id"] == ep]

            if len(ep_micro) == 0:
                continue

            # culture_time <= landmark AND imputed_report_time <= landmark
            pre_landmark = ep_micro[
                (ep_micro["culture_time"] <= lm) &
                (ep_micro["imputed_report_time"] <= lm)
            ]
            if len(pre_landmark) > 0:
                trials.at[idx, "c5_no_resistant"] = False
                trials.at[idx, "pre_landmark_resistant_isolate"] = 1

            # culture_time <= landmark BUT imputed_report_time > landmark
            unrecognized = ep_micro[
                (ep_micro["culture_time"] <= lm) &
                (ep_micro["imputed_report_time"] > lm)
            ]
            if len(unrecognized) > 0:
                trials.at[idx, "unrecognized_baseline_resistant_isolate"] = 1

    # Criterion 6: On broad-spectrum antibiotic at landmark
    # (using interval overlap: start <= landmark AND end > landmark)
    broad_abx = abx_orders[abx_orders["is_broad_spectrum"] == 1].copy()

    trials["c6_broad_abx"] = False
    trials["current_abx_at_landmark"] = ""
    if len(broad_abx) > 0:
        for (ep, lm_h), grp_lm in trials.groupby(["episode_id", "landmark_hour"]):
            lm_time = grp_lm["landmark_time"].iloc[0]
            ep_broad = broad_abx[broad_abx["episode_id"] == ep]

            # Check interval overlap
            active_mask = (ep_broad["start_time"] <= lm_time) & \
                          (ep_broad["end_time"] > lm_time)
            active_abx = ep_broad[active_mask]

            if len(active_abx) > 0:
                # Check date_only constraint
                date_only_abx = active_abx[active_abx["date_only_flag"] == 1]
                if len(date_only_abx) > 0:
                    trials.loc[grp_lm.index, "date_only_excluded"] = 1

                trials.loc[grp_lm.index, "c6_broad_abx"] = True
                trials.loc[grp_lm.index, "current_abx_at_landmark"] = \
                    "|".join(active_abx["drug_std"].unique())

    # Criterion 7: Infection context (Section 6.2 Criterion 7)
    # Requires either: (a) culture obtained before landmark, OR
    # (b) ICD infection diagnosis for this admission
    # ICD diagnoses lack precise timestamps, so this is flagged for sensitivity analysis.

    # Pre-compute admissions with infection ICD codes
    infection_hadm_ids = set()
    if icd_diagnoses is not None and len(icd_diagnoses) > 0:
        infection_icd = icd_diagnoses[icd_diagnoses["is_infection_code"] == 1]
        infection_hadm_ids = set(infection_icd["hadm_id"].unique())
        log(f"  Admissions with infection ICD codes: {len(infection_hadm_ids)}", logfile)

    trials["c7_infection_context"] = False
    trials["c7_source"] = ""  # Track source: "culture" / "icd" / "both"
    for idx, row in trials.iterrows():
        ep = row["episode_id"]
        lm = row["landmark_time"]
        hadm = row["hadm_id"]
        has_culture = False
        has_icd = False

        # Check (a): culture obtained before landmark
        ep_micro = micro_isolates[micro_isolates["episode_id"] == ep]
        if len(ep_micro) > 0:
            cultures_before = (ep_micro["culture_time"] <= lm).sum()
            if cultures_before > 0:
                has_culture = True

        # Check (b): ICD infection diagnosis for this admission
        if hadm in infection_hadm_ids:
            has_icd = True

        if has_culture or has_icd:
            trials.at[idx, "c7_infection_context"] = True
            if has_culture and has_icd:
                trials.at[idx, "c7_source"] = "both"
            elif has_culture:
                trials.at[idx, "c7_source"] = "culture"
            else:
                trials.at[idx, "c7_source"] = "icd_only"

    # 4. Compile eligibility and reasons (Section 6.3)
    log("Compiling eligibility results...", logfile)

    reason_map = {
        "c1_age_child": "age_not_child",
        "c3_in_icu": "not_in_icu_at_landmark",
        "c4_alive": "dead_before_landmark",
        "c5_no_resistant": "resistant_event_before_landmark",
        "c6_broad_abx": "no_broad_antibiotic_at_landmark",
        "c7_infection_context": "no_infection_context",
    }

    criterion_cols = list(reason_map.keys())
    trials["is_eligible"] = trials[criterion_cols].all(axis=1)

    # Build reason_not_eligible
    def build_reasons(row):
        reasons = []
        for col, reason in reason_map.items():
            if not row[col]:
                reasons.append(reason)
        if not reasons:
            return "eligible"
        return "|".join(reasons)

    trials["reason_not_eligible"] = trials.apply(build_reasons, axis=1)

    # 5. Apply date_only exclusion (not primary eligibility, but flagged)
    if "date_only_excluded" not in trials.columns:
        trials["date_only_excluded"] = 0

    # 6. Statistics by landmark
    for lm_hour in LANDMARK_HOURS:
        lm_trials = trials[trials["landmark_hour"] == lm_hour]
        n_eligible = lm_trials["is_eligible"].sum()
        log(f"  Landmark {lm_hour}h: {n_eligible:,} eligible / {len(lm_trials):,} total "
            f"({n_eligible/len(lm_trials)*100:.1f}%)", logfile)

    # Count reasons
    for reason in reason_map.values():
        count = trials["reason_not_eligible"].str.contains(reason, na=False).sum()
        if count > 0:
            log(f"    {reason}: {count:,}", logfile)

    # 7. Select output columns (Section 6.3)
    output_cols = [
        "subject_id", "hadm_id", "episode_id", "icustay_id_list",
        "landmark_hour", "landmark_time",
        "age_years_at_landmark", "age_group",
        "is_eligible", "reason_not_eligible",
        "c3_in_icu", "c4_alive", "c5_no_resistant",
        "c6_broad_abx", "c7_infection_context", "c7_source",
        "pre_landmark_resistant_isolate",
        "unrecognized_baseline_resistant_isolate",
        "date_only_excluded",
        "current_abx_at_landmark",
    ]
    output = trials[output_cols].reset_index(drop=True)

    # 8. Save
    eligible = output[output["is_eligible"]]
    output.to_csv(LANDMARK_ELIGIBLE_CSV, index=False, encoding="utf-8")
    log(f"-> Saved {LANDMARK_ELIGIBLE_CSV}: {len(output):,} trials, "
        f"{len(eligible):,} eligible", logfile)

    log("Step 6 complete!", logfile)
    return output


def _load_icd_diagnoses(logfile=None):
    """Load ICD diagnoses and flag infection-related codes (Section 6.2 Criterion 7).
    Merges main and update tables if available.
    """
    import os as _os
    rows = []

    # Load main DIAGNOSES_ICD
    if _os.path.exists(DIAGNOSES_ICD_CSV):
        main = pd.read_csv(DIAGNOSES_ICD_CSV, low_memory=False)
        main.columns = [c.lower() for c in main.columns]
        rows.append(main)

    # Load update if exists
    update_path = UPDATE_DIAGNOSES_ICD
    if _os.path.exists(update_path):
        update = pd.read_csv(update_path, low_memory=False)
        update.columns = [c.lower() for c in update.columns]
        rows.append(update)

    if not rows:
        return None

    diag = pd.concat(rows, ignore_index=True)
    for col in ["subject_id", "hadm_id"]:
        if col in diag.columns:
            diag[col] = diag[col].astype(str)

    # ICD infection code ranges (expanded per Section 6.2 Criterion 7)
    # Sepsis, pneumonia, UTI, abdominal infection, skin infection, meningitis, etc.
    infection_icd9_patterns = [
        "038",      # Septicaemia
        "995.91",   # Sepsis
        "995.92",   # Severe sepsis
        "480", "481", "482", "483", "484", "485", "486",  # Pneumonia
        "599",      # UTI
        "540", "541", "542",  # Appendicitis
        "567",      # Peritonitis
        "682",      # Cellulitis/abscess
        "320",      # Meningitis
    ]
    infection_icd10_patterns = [
        "A40", "A41",  # Sepsis
        "R65.20", "R65.21",  # SIRS/sepsis
        "J12", "J13", "J14", "J15", "J16", "J17", "J18",  # Pneumonia
        "N39.0",  # UTI
        "K35", "K36", "K37",  # Appendicitis
        "K65",    # Peritonitis
        "L03",    # Cellulitis
        "G00", "G01", "G02",  # Meningitis
        "K57",    # Diverticulitis
        "J85", "J86",  # Lung abscess/empyema
        "N10", "N11", "N12",  # Pyelonephritis
        "K80", "K81", "K82", "K83",  # Cholecystitis
    ]

    diag["is_infection_code"] = 0

    # Detect ICD code column (PIC uses ICD10_CODE_CN, MIMIC uses icd9_code/icd10_code)
    icd_col = None
    for candidate in ["icd10_code_cn", "icd9_code", "icd10_code", "icd_code"]:
        if candidate in diag.columns:
            icd_col = candidate
            break
    # Also check for any column containing "icd" (case-insensitive)
    if icd_col is None:
        for col in diag.columns:
            if "icd" in col.lower():
                icd_col = col
                break

    if icd_col:
        code_col = diag[icd_col].astype(str)
        log(f"  Using ICD column: {icd_col}", logfile)
        for pattern in infection_icd9_patterns + infection_icd10_patterns:
            diag.loc[code_col.str.startswith(pattern, na=False),
                     "is_infection_code"] = 1
    else:
        log(f"  WARNING: No ICD code column found. Available columns: {list(diag.columns)}", logfile)

    n_infection = diag["is_infection_code"].sum()
    if logfile:
        log(f"  Loaded {len(diag)} ICD diagnoses, {n_infection} infection codes", logfile)
    return diag


if __name__ == "__main__":
    logfile = os.path.join(OUTPUT_DIR, "processing_log.txt")
    run_step6(logfile)
