"""Step 7: Landmark Baseline Covariates (landmark_baseline_covariates.csv)
Implements Section 7 from the processing plan.
v3.1: Added organ support (vasopressor), lab values, comorbidities, weight extraction.
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from utils import parse_datetime, hours_between, log


# ═══════════════════════════════════════════════════════════════
# D_LABITEMS → standard lab name mapping
# ═══════════════════════════════════════════════════════════════
LAB_MAX_ITEMIDS = {
    "wbc":             [5141],                    # WBC Count
    "neutrophil":      [5005, 5095, 5511, 5094], # Neutrophils % + absolute count
    "crp":             [5626, 5821],              # C-Reactive Protein
    "pct":             [6085],                    # Procalcitonin (NOT 5127=plateletcrit)
    "lactate":         [5227],                    # Lactate
    "creatinine":      [5032, 5041, 6954],        # Creatinine (multiple lab sources)
    "bilirubin":       [5075],                    # Bilirubin, Total
    "temperature":     [5253],                    # Temperature (also CHARTEVENTS 1001)
}
LAB_MIN_ITEMIDS = {
    "platelet":        [5129],                    # Platelet Count
    "albumin":         [5024],                    # Albumin
    "hemoglobin":      [5099, 5257],              # Hemoglobin (hematology)
    "ph":              [5237, 5238],              # pH
    "base_excess":     [5211, 5249],              # Base Excess
    "spo2":            [5252],                    # Oxygen Saturation
}

# Physiological plausibility thresholds
LAB_THRESHOLDS = {
    "temperature":     (30, 45),
    "spo2":            (50, 100),
    "ph":              (6.5, 8.0),
    "wbc":             (0.1, 500),
    "platelet":        (1, 3000),
    "hemoglobin":      (2, 25),
    "creatinine":      (0.1, 20),
    "bilirubin":       (0.1, 50),
    "lactate":         (0.1, 30),
    "crp":             (0.1, 500),
    "pct":             (0.01, 200),
    "albumin":         (5, 60),
    "base_excess":     (-30, 30),
    "neutrophil":      (0, 100),
}

# Vasopressor drug name keywords (English)
VASOPRESSOR_KEYWORDS = [
    "dopamine", "dobutamine", "epinephrine", "norepinephrine",
    "noradrenaline", "adrenaline", "vasopressin", "milrinone",
]

# ICD-10-CN comorbidity patterns (prefix match on code like "Q20.01")
# PIC uses ICD-10-CN: letter + 2 digits + . + suffix
COMORBIDITY_ICD10 = {
    "congenital_heart_disease":  ["Q20", "Q21", "Q22", "Q23", "Q24", "Q25", "Q26", "Q27", "Q28"],
    "chronic_lung_disease":      ["J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47"],
    "chronic_kidney_disease":    ["N03", "N04", "N05", "N10", "N11", "N12", "N16", "N18", "N19"],
    "neurologic_disease":        ["G40", "G41", "G80", "G81", "G82", "G83"],
    "malignancy":                ["C"],  # C00-C99
    "hematologic_disease":       ["D50", "D51", "D52", "D53", "D55", "D56", "D57", "D58", "D59",
                                  "D60", "D61", "D62", "D63", "D64", "D65", "D66", "D67", "D68",
                                  "D69", "D70", "D71", "D72", "D73", "D74", "D75"],
    "immunodeficiency":          ["D80", "D81", "D82", "D83", "D84", "D89"],
    "transplant_status":         ["Z94", "T86"],
    "postoperative_status":      ["Z48", "Z98"],
    "prematurity":               ["P07"],
}


# ═══════════════════════════════════════════════════════════════
# Extraction functions
# ═══════════════════════════════════════════════════════════════

def _build_patient_index(raw_df_or_path, eligible, logfile=None, usecols=None, chunksize=200000):
    """Pre-filter raw table to only subject_id+hadm_id in eligible trials.
    Accepts either a DataFrame or a CSV file path. Uses chunked reading for large files.
    """
    pairs = set()
    for _, row in eligible.iterrows():
        pairs.add((str(row["subject_id"]), str(row["hadm_id"])))
    log(f"    Filtering to {len(pairs):,} patient-stays...", logfile)

    if isinstance(raw_df_or_path, str):
        # It's a file path - use chunked reading
        chunks = []
        reader = pd.read_csv(raw_df_or_path, low_memory=False, chunksize=chunksize, usecols=usecols)
        for chunk in reader:
            sid_col = "SUBJECT_ID" if "SUBJECT_ID" in chunk.columns else "subject_id"
            hid_col = "HADM_ID" if "HADM_ID" in chunk.columns else "hadm_id"
            chunk["_sid"] = chunk[sid_col].astype(str)
            chunk["_hid"] = chunk[hid_col].astype(str)
            mask = pd.Series(False, index=chunk.index)
            for sid, hid in pairs:
                mask |= (chunk["_sid"] == sid) & (chunk["_hid"] == hid)
            if mask.any():
                chunks.append(chunk[mask].drop(columns=["_sid", "_hid"]))
        if chunks:
            result = pd.concat(chunks, ignore_index=True)
        else:
            result = pd.DataFrame()
        log(f"    Filtered to {len(result):,} rows", logfile)
        return result
    else:
        # It's a DataFrame
        raw_df = raw_df_or_path
        sid_col = "SUBJECT_ID" if "SUBJECT_ID" in raw_df.columns else "subject_id"
        hid_col = "HADM_ID" if "HADM_ID" in raw_df.columns else "hadm_id"
        raw_df["_sid"] = raw_df[sid_col].astype(str)
        raw_df["_hid"] = raw_df[hid_col].astype(str)
        mask = pd.Series(False, index=raw_df.index)
        for sid, hid in pairs:
            mask |= (raw_df["_sid"] == sid) & (raw_df["_hid"] == hid)
        filtered = raw_df[mask].drop(columns=["_sid", "_hid"])
        log(f"    Filtered to {len(filtered):,} rows", logfile)
        return filtered


def extract_vasopressor(eligible, prescriptions_raw, logfile=None):
    """Extract vasopressor_at_landmark from raw PRESCRIPTIONS."""
    log("  Extracting vasopressor from PRESCRIPTIONS...", logfile)
    rx = _build_patient_index(prescriptions_raw, eligible, logfile,
        usecols=["SUBJECT_ID", "HADM_ID", "STARTDATE", "ENDDATE", "DRUG_NAME_EN"])
    if len(rx) == 0:
        log("    No vasopressor data found for cohort!", logfile)
        eligible["vasopressor_at_landmark"] = 0
        eligible["vasopressor_missing"] = 1
        return eligible
    rx["start_time"] = parse_datetime(rx["STARTDATE"])
    rx["end_time"] = parse_datetime(rx["ENDDATE"])

    drug_col = "DRUG_NAME_EN" if "DRUG_NAME_EN" in rx.columns else "DRUG_NAME"
    pattern = "|".join(VASOPRESSOR_KEYWORDS)
    vaso_rx = rx[rx[drug_col].astype(str).str.lower().str.contains(pattern, case=False, na=False)]
    log(f"    Vasopressor orders for cohort: {len(vaso_rx):,}", logfile)

    # Index by (subject_id, hadm_id)
    vaso_by_patient = {}
    for (sid, hid), group in vaso_rx.groupby(["SUBJECT_ID", "HADM_ID"]):
        vaso_by_patient[(str(sid), str(hid))] = group

    vaso_vals = []
    for _, row in eligible.iterrows():
        key = (str(row["subject_id"]), str(row["hadm_id"]))
        patient_vaso = vaso_by_patient.get(key, pd.DataFrame())
        if len(patient_vaso) == 0:
            vaso_vals.append(0)
            continue
        lm = row["landmark_time"]
        active = patient_vaso[(patient_vaso["start_time"] <= lm) & (patient_vaso["end_time"] > lm)]
        vaso_vals.append(1 if len(active) > 0 else 0)

    eligible["vasopressor_at_landmark"] = vaso_vals
    eligible["vasopressor_missing"] = 0
    n_yes = sum(vaso_vals)
    log(f"    Vasopressor active at landmark: {n_yes}/{len(eligible)} ({100*n_yes/len(eligible):.1f}%)", logfile)
    return eligible


def extract_labs(eligible, labevents, d_labitems, logfile=None):
    """Extract lab values from LABEVENTS within [landmark-24h, landmark] window."""
    log("  Extracting lab values from LABEVENTS...", logfile)

    itemid_to_std = {}
    for std_name, itemids in {**LAB_MAX_ITEMIDS, **LAB_MIN_ITEMIDS}.items():
        for iid in itemids:
            itemid_to_std[iid] = std_name
    relevant_ids = list(itemid_to_std.keys())

    le = _build_patient_index(labevents, eligible, logfile,
        usecols=["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"])
    le["charttime"] = parse_datetime(le["CHARTTIME"])
    le = le[le["ITEMID"].isin(relevant_ids)]
    log(f"    LABEVENTS rows with relevant ITEMIDs: {len(le):,}", logfile)

    # Index by (subject_id, hadm_id)
    le_by_patient = {}
    for (sid, hid), group in le.groupby(["SUBJECT_ID", "HADM_ID"]):
        le_by_patient[(str(sid), str(hid))] = group

    # Initialize result columns
    for v in LAB_MAX_ITEMIDS:
        eligible[f"{v}_max_24h"] = np.nan
        eligible[f"{v}_missing"] = 1
    for v in LAB_MIN_ITEMIDS:
        eligible[f"{v}_min_24h"] = np.nan
        eligible[f"{v}_missing"] = 1
    eligible["systolic_bp_min_24h"] = np.nan
    eligible["systolic_bp_missing"] = 1
    eligible["mean_bp_min_24h"] = np.nan
    eligible["mean_bp_missing"] = 1

    matched_count = 0
    for idx, row in eligible.iterrows():
        key = (str(row["subject_id"]), str(row["hadm_id"]))
        patient_le = le_by_patient.get(key)
        if patient_le is None or len(patient_le) == 0:
            continue

        lm = row["landmark_time"]
        window_start = lm - pd.Timedelta(hours=24)
        window = patient_le[(patient_le["charttime"] >= window_start) & (patient_le["charttime"] <= lm)]
        if len(window) == 0:
            continue

        matched_count += 1
        for itemid, sub in window.groupby("ITEMID"):
            std_name = itemid_to_std.get(itemid)
            if std_name is None:
                continue
            vals = sub["VALUENUM"].dropna()
            if len(vals) == 0:
                continue
            if std_name in LAB_THRESHOLDS:
                lo, hi = LAB_THRESHOLDS[std_name]
                vals = vals[(vals >= lo) & (vals <= hi)]
            if len(vals) == 0:
                continue

            if std_name in LAB_MAX_ITEMIDS:
                eligible.at[idx, f"{std_name}_max_24h"] = vals.max()
                eligible.at[idx, f"{std_name}_missing"] = 0
            elif std_name in LAB_MIN_ITEMIDS:
                eligible.at[idx, f"{std_name}_min_24h"] = vals.min()
                eligible.at[idx, f"{std_name}_missing"] = 0

    log(f"    Trials with >=1 lab value: {matched_count}/{len(eligible)} ({100*matched_count/len(eligible):.1f}%)", logfile)
    for std_name in list(LAB_MAX_ITEMIDS.keys()) + list(LAB_MIN_ITEMIDS.keys()):
        col = f"{std_name}_max_24h" if std_name in LAB_MAX_ITEMIDS else f"{std_name}_min_24h"
        n_avail = eligible[col].notna().sum()
        if n_avail > 0:
            log(f"      {col}: {n_avail}/{len(eligible)} ({100*n_avail/len(eligible):.1f}%)", logfile)
    return eligible


def extract_bp_from_chartevents(eligible, chartevents, logfile=None):
    """Extract systolic BP and compute MAP from CHARTEVENTS (ITEMID 1015/1016)."""
    log("  Extracting BP from CHARTEVENTS...", logfile)
    ce = _build_patient_index(chartevents, eligible, logfile,
        usecols=["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"])
    ce["charttime"] = parse_datetime(ce["CHARTTIME"])
    ce = ce[ce["ITEMID"].isin([1015, 1016])]
    log(f"    CHARTEVENTS BP rows for cohort: {len(ce):,}", logfile)

    ce_by_patient = {}
    for (sid, hid), group in ce.groupby(["SUBJECT_ID", "HADM_ID"]):
        ce_by_patient[(str(sid), str(hid))] = group

    for idx, row in eligible.iterrows():
        key = (str(row["subject_id"]), str(row["hadm_id"]))
        patient_ce = ce_by_patient.get(key)
        if patient_ce is None or len(patient_ce) == 0:
            continue

        lm = row["landmark_time"]
        window_start = lm - pd.Timedelta(hours=24)
        window = patient_ce[(patient_ce["charttime"] >= window_start) & (patient_ce["charttime"] <= lm)]
        if len(window) == 0:
            continue

        sys_vals = window[window["ITEMID"] == 1016]["VALUENUM"].dropna()
        if len(sys_vals) > 0:
            sys_vals = sys_vals[(sys_vals >= 20) & (sys_vals <= 250)]
        if len(sys_vals) > 0:
            eligible.at[idx, "systolic_bp_min_24h"] = sys_vals.min()
            eligible.at[idx, "systolic_bp_missing"] = 0

        dia_vals = window[window["ITEMID"] == 1015]["VALUENUM"].dropna()
        if len(dia_vals) > 0:
            dia_vals = dia_vals[(dia_vals >= 10) & (dia_vals <= 200)]
        if len(sys_vals) > 0 and len(dia_vals) > 0:
            map_vals = (2 * dia_vals.mean() + sys_vals.mean()) / 3
            eligible.at[idx, "mean_bp_min_24h"] = map_vals
            eligible.at[idx, "mean_bp_missing"] = 0

    n_sys = (eligible["systolic_bp_missing"] == 0).sum()
    n_map = (eligible["mean_bp_missing"] == 0).sum()
    log(f"    BP available: systolic={n_sys}/{len(eligible)}, MAP={n_map}/{len(eligible)}", logfile)
    return eligible


def extract_comorbidities(eligible, diagnoses_icd, logfile=None):
    """Extract 10 comorbidity flags from DIAGNOSES_ICD (ICD-10-CN codes)."""
    log("  Extracting comorbidities from DIAGNOSES_ICD...", logfile)
    diag = _build_patient_index(diagnoses_icd, eligible, logfile,
        usecols=["SUBJECT_ID", "HADM_ID", "ICD10_CODE_CN"])

    # Build hadm_id → list of ICD codes
    hadm_diags = {}
    for _, d in diag.iterrows():
        code = str(d.get("ICD10_CODE_CN", "")) if pd.notna(d.get("ICD10_CODE_CN")) else ""
        if not code:
            continue
        hadm_id = str(d["HADM_ID"])
        hadm_diags.setdefault(hadm_id, []).append(code)

    # Initialize
    for flag in COMORBIDITY_ICD10:
        eligible[flag] = 0

    counts = {k: 0 for k in COMORBIDITY_ICD10}
    for idx, row in eligible.iterrows():
        hadm_id = str(row["hadm_id"])
        codes = hadm_diags.get(hadm_id, [])
        if not codes:
            continue
        for cond, prefixes in COMORBIDITY_ICD10.items():
            for code in codes:
                for pf in prefixes:
                    if code.startswith(pf):
                        eligible.at[idx, cond] = 1
                        counts[cond] += 1
                        break
                if eligible.at[idx, cond] == 1:
                    break

    for cond, n in counts.items():
        if n > 0:
            log(f"    {cond}: {n}/{len(eligible)} ({100*n/len(eligible):.1f}%)", logfile)
    return eligible


def extract_weight(eligible, chartevents, logfile=None):
    """Extract weight from CHARTEVENTS (ITEMID 1014)."""
    log("  Extracting weight from CHARTEVENTS...", logfile)
    ce = _build_patient_index(chartevents, eligible, logfile,
        usecols=["SUBJECT_ID", "HADM_ID", "ITEMID", "CHARTTIME", "VALUENUM"])
    ce["charttime"] = parse_datetime(ce["CHARTTIME"])
    wt = ce[ce["ITEMID"] == 1014]
    log(f"    Weight records for cohort: {len(wt):,}", logfile)

    wt_by_patient = {}
    for (sid, hid), group in wt.groupby(["SUBJECT_ID", "HADM_ID"]):
        wt_by_patient[(str(sid), str(hid))] = group

    eligible["admission_weight_kg"] = np.nan
    eligible["admission_weight_missing"] = 1
    eligible["nearest_weight_pre_landmark_kg"] = np.nan
    eligible["nearest_weight_pre_landmark_missing"] = 1

    n_found = 0
    for idx, row in eligible.iterrows():
        key = (str(row["subject_id"]), str(row["hadm_id"]))
        patient_wt = wt_by_patient.get(key)
        if patient_wt is None or len(patient_wt) == 0:
            continue

        lm = row["landmark_time"]
        pre_lm = patient_wt[patient_wt["charttime"] <= lm]
        if len(pre_lm) == 0:
            continue

        nearest = pre_lm.loc[pre_lm["charttime"].idxmax()]
        val = nearest["VALUENUM"] if pd.notna(nearest.get("VALUENUM")) else None
        if val is not None and 0.5 < val < 200:
            eligible.at[idx, "nearest_weight_pre_landmark_kg"] = val
            eligible.at[idx, "nearest_weight_pre_landmark_missing"] = 0
            n_found += 1

            adm_time = row.get("episode_intime")
            if pd.notna(adm_time):
                wt_time = nearest["charttime"]
                if abs((wt_time - adm_time).total_seconds()) < 86400:
                    eligible.at[idx, "admission_weight_kg"] = val
                    eligible.at[idx, "admission_weight_missing"] = 0

    log(f"    Weight found: {n_found}/{len(eligible)} ({100*n_found/len(eligible):.1f}%)", logfile)
    return eligible


def extract_surgery_procedures(eligible, surgery_info, logfile=None):
    """Extract organ support proxies from SURGERY_INFO.
    These are SURGICAL procedure markers (not direct ICU organ support monitoring).
    """
    log("  Extracting surgical procedure markers from SURGERY_INFO...", logfile)
    si = _build_patient_index(surgery_info, eligible, logfile,
        usecols=["SUBJECT_ID", "HADM_ID", "SURGERY_END_TIME", "SURGERY_BEGIN_TIME",
                  "ANES_END_TIME", "SURGERY_NAME", "ANES_METHOD"])
    for col in ["SURGERY_END_TIME", "SURGERY_BEGIN_TIME", "ANES_END_TIME"]:
        if col in si.columns:
            si[col] = parse_datetime(si[col])

    si_by_patient = {}
    for (sid, hid), group in si.groupby(["SUBJECT_ID", "HADM_ID"]):
        si_by_patient[(str(sid), str(hid))] = group

    # Initialize
    eligible["mechanical_ventilation_at_landmark"] = 0
    eligible["mechanical_ventilation_missing"] = 1
    eligible["rrt_at_landmark"] = 0
    eligible["rrt_missing"] = 1
    eligible["central_line_at_landmark"] = 0
    eligible["central_line_missing"] = 1
    eligible["ECMO_at_landmark"] = 0
    eligible["ECMO_missing"] = 1

    counts = {"vent": 0, "rrt": 0, "cvc": 0, "ecmo": 0}
    for idx, row in eligible.iterrows():
        key = (str(row["subject_id"]), str(row["hadm_id"]))
        patient_si = si_by_patient.get(key)
        if patient_si is None or len(patient_si) == 0:
            continue

        lm = row["landmark_time"]
        if "SURGERY_END_TIME" in si.columns:
            pre_lm = patient_si[patient_si["SURGERY_END_TIME"] <= lm]
        elif "SURGERY_BEGIN_TIME" in si.columns:
            pre_lm = patient_si[patient_si["SURGERY_BEGIN_TIME"] <= lm]
        else:
            pre_lm = patient_si

        if len(pre_lm) == 0:
            continue

        for _, surg in pre_lm.iterrows():
            name = str(surg.get("SURGERY_NAME", ""))
            anes = str(surg.get("ANES_METHOD", ""))
            surg_end = surg.get("SURGERY_END_TIME")
            hours_since = (lm - surg_end).total_seconds() / 3600 if pd.notna(surg_end) else 999

            if hours_since <= 48:
                intub_kw = ["intubat", "endotracheal", "laryngeal mask", "ventilat"]
                if any(kw in anes.lower() for kw in intub_kw) or any(kw in name.lower() for kw in intub_kw):
                    eligible.at[idx, "mechanical_ventilation_at_landmark"] = 1
                    eligible.at[idx, "mechanical_ventilation_missing"] = 0
                    counts["vent"] += 1

                rrt_kw = ["dialysis", "hemodialy", "cvvh", "crrt", "hemofilt", "ultrafilt"]
                if any(kw in name.lower() for kw in rrt_kw):
                    eligible.at[idx, "rrt_at_landmark"] = 1
                    eligible.at[idx, "rrt_missing"] = 0
                    counts["rrt"] += 1

                cvc_kw = ["central venous", "central line", "cvc", "deep venous"]
                if any(kw in name.lower() for kw in cvc_kw):
                    eligible.at[idx, "central_line_at_landmark"] = 1
                    eligible.at[idx, "central_line_missing"] = 0
                    counts["cvc"] += 1

                ecmo_kw = ["ecmo", "extracorporeal membrane"]
                if any(kw in name.lower() for kw in ecmo_kw):
                    eligible.at[idx, "ECMO_at_landmark"] = 1
                    eligible.at[idx, "ECMO_missing"] = 0
                    counts["ecmo"] += 1
                break  # one surgery is enough

    log(f"    MV (surgery proxy): {counts['vent']}/{len(eligible)} ({100*counts['vent']/len(eligible):.1f}%)", logfile)
    log(f"    RRT: {counts['rrt']}/{len(eligible)}", logfile)
    log(f"    CVC: {counts['cvc']}/{len(eligible)}", logfile)
    log(f"    ECMO: {counts['ecmo']}/{len(eligible)}", logfile)
    return eligible


# ═══════════════════════════════════════════════════════════════
# Main Step 7
# ═══════════════════════════════════════════════════════════════

def run_step7(logfile=None, eligible_trials=None, icu_base=None,
              abx_orders=None, micro_isolates=None, labevents=None,
              chartevents=None, diagnoses=None, admissions=None,
              prescriptions_raw=None, d_labitems=None, surgery_info=None):
    log("=" * 60, logfile)
    log("Step 7: Baseline Covariates (v3.1 with organ/lab/comorbidity extraction)", logfile)
    log("=" * 60, logfile)

    # 1. Load inputs
    if eligible_trials is None:
        eligible_trials = pd.read_csv(LANDMARK_ELIGIBLE_CSV, low_memory=False)
        eligible_trials["landmark_time"] = parse_datetime(eligible_trials["landmark_time"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            eligible_trials[col] = eligible_trials[col].astype(str)

    eligible = eligible_trials[eligible_trials["is_eligible"]].copy()
    log(f"  Eligible trials: {len(eligible)}", logfile)

    if icu_base is None:
        icu_base = pd.read_csv(ICU_BASE_CSV, low_memory=False)
        icu_base["episode_intime"] = parse_datetime(icu_base["episode_intime"])
        icu_base["dob"] = parse_datetime(icu_base["dob"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            icu_base[col] = icu_base[col].astype(str)

    if abx_orders is None:
        abx_orders = pd.read_csv(ABX_ORDERS_CLEAN_CSV, low_memory=False)
        abx_orders["start_time"] = parse_datetime(abx_orders["start_time"])
        abx_orders["end_time"] = parse_datetime(abx_orders["end_time"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            abx_orders[col] = abx_orders[col].astype(str)

    if micro_isolates is None:
        micro_isolates = pd.read_csv(MICRO_ISOLATES_CLEAN_CSV, low_memory=False)
        micro_isolates["culture_time"] = parse_datetime(micro_isolates["culture_time"])
        micro_isolates["imputed_report_time"] = parse_datetime(micro_isolates["imputed_report_time"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            micro_isolates[col] = micro_isolates[col].astype(str)

    # Merge with ICU base for demographics
    cov = eligible.merge(
        icu_base[["subject_id", "hadm_id", "episode_id", "episode_intime",
                   "gender", "dob", "age_days", "age_years", "age_group",
                   "first_careunit", "hospital_los_before_icu_hours",
                   "hospital_los_negative_flag"]],
        on=["subject_id", "hadm_id", "episode_id"], how="left"
    )

    # Section 7.1: Demographics & ICU
    cov["icu_los_before_landmark_hours"] = cov["landmark_hour"]
    cov["calendar_year"] = cov["landmark_time"].dt.year

    # Merge admission_type from ADMISSIONS if available
    if admissions is not None:
        adm_cols = ["subject_id", "hadm_id", "admission_type"]
        adm_cols = [c for c in adm_cols if c in admissions.columns]
        if len(adm_cols) >= 3:
            cov = cov.merge(
                admissions[adm_cols].drop_duplicates(subset=["subject_id", "hadm_id"]),
                on=["subject_id", "hadm_id"], how="left"
            )
    if "admission_type" not in cov.columns:
        cov["admission_type"] = ""

    # Section 7.2: Antibiotic status at landmark
    abx_cols = [
        "current_abx_list", "current_broad_abx_list", "current_n_abx",
        "current_max_spectrum_score", "current_anti_pseudomonal",
        "current_anti_mrsa", "current_carbapenem", "current_anti_anaerobe",
        "current_combination_therapy", "current_last_resort",
        "cov_gnb", "cov_pseudomonas", "cov_mrsa", "cov_anaerobe",
        "cov_atypical", "cov_fungal", "cov_salvage",
    ]
    for col in abx_cols:
        if "list" in col:
            cov[col] = pd.Series("", index=cov.index, dtype="object")
        elif col in ("current_n_abx",):
            cov[col] = 0
        elif col in ("current_max_spectrum_score",):
            cov[col] = np.nan
        else:
            cov[col] = 0

    for idx, row in cov.iterrows():
        ep = row["episode_id"]
        lm = row["landmark_time"]
        ep_abx = abx_orders[abx_orders["episode_id"] == ep]

        if len(ep_abx) > 0:
            active_mask = (ep_abx["start_time"] <= lm) & (ep_abx["end_time"] > lm)
            active = ep_abx[active_mask]

            if len(active) > 0:
                cov.at[idx, "current_abx_list"] = "|".join(active["drug_std"].unique())
                cov.at[idx, "current_n_abx"] = active["drug_std"].nunique()
                cov.at[idx, "current_max_spectrum_score"] = active["spectrum_score"].max()

                broad = active[active["is_broad_spectrum"] == 1]
                if len(broad) > 0:
                    cov.at[idx, "current_broad_abx_list"] = "|".join(broad["drug_std"].unique())

                cov.at[idx, "current_anti_pseudomonal"] = active["anti_pseudomonal"].max()
                cov.at[idx, "current_anti_mrsa"] = active["anti_mrsa"].max()
                cov.at[idx, "current_carbapenem"] = active["carbapenem"].max()
                cov.at[idx, "current_anti_anaerobe"] = active["anti_anaerobe"].max()
                cov.at[idx, "current_last_resort"] = active["salvage_therapy"].max()
                cov.at[idx, "current_combination_therapy"] = 1 if cov.at[idx, "current_n_abx"] >= 2 else 0

                for cov_dim in ["coverage_gnb", "coverage_pseudomonas", "coverage_mrsa",
                                "coverage_anaerobe", "coverage_atypical", "coverage_fungal"]:
                    col_name = cov_dim.replace("coverage_", "cov_")
                    if cov_dim in active.columns:
                        cov.at[idx, col_name] = active[cov_dim].max()
                cov.at[idx, "cov_salvage"] = active["salvage_therapy"].max()

            # Cumulative exposure before landmark
            pre_landmark_abx = ep_abx[ep_abx["start_time"] < lm]
            if len(pre_landmark_abx) > 0:
                total_hours = 0
                broad_hours = 0
                carb_hours = 0
                mrsa_hours = 0
                for _, abx_row in pre_landmark_abx.iterrows():
                    s = max(abx_row["start_time"], row["episode_intime"])
                    e = min(abx_row["end_time"], lm) if pd.notna(abx_row["end_time"]) else lm
                    if e > s:
                        dur = (e - s).total_seconds() / 3600
                        total_hours += dur
                        if abx_row.get("is_broad_spectrum", 0) == 1:
                            broad_hours += dur
                        if abx_row.get("carbapenem", 0) == 1:
                            carb_hours += dur
                        if abx_row.get("anti_mrsa", 0) == 1:
                            mrsa_hours += dur
                cov.at[idx, "cumulative_systemic_abx_hours"] = total_hours
                cov.at[idx, "cumulative_broad_abx_hours"] = broad_hours
                cov.at[idx, "cumulative_carbapenem_hours"] = carb_hours
                cov.at[idx, "cumulative_anti_mrsa_hours"] = mrsa_hours

    # Section 7.3: Microbiology before landmark
    cov["any_culture_obtained_before_landmark"] = 0
    cov["culture_result_known_before_landmark"] = 0
    cov["number_of_cultures_before_landmark"] = 0
    cov["positive_culture_known_before_landmark"] = 0
    cov["blood_culture_before_landmark"] = 0
    cov["blood_positive_known_before_landmark"] = 0
    cov["baseline_resistant_organism"] = 0
    cov["unrecognized_baseline_resistant_isolate"] = 0

    for idx, row in cov.iterrows():
        ep = row["episode_id"]
        lm = row["landmark_time"]
        ep_micro = micro_isolates[micro_isolates["episode_id"] == ep]

        if len(ep_micro) > 0:
            pre_lm = ep_micro[ep_micro["culture_time"] <= lm]
            cov.at[idx, "any_culture_obtained_before_landmark"] = 1 if len(pre_lm) > 0 else 0
            cov.at[idx, "number_of_cultures_before_landmark"] = len(pre_lm)

            known = pre_lm[pre_lm["imputed_report_time"] <= lm] if len(pre_lm) > 0 else pd.DataFrame()
            cov.at[idx, "culture_result_known_before_landmark"] = 1 if len(known) > 0 else 0
            cov.at[idx, "positive_culture_known_before_landmark"] = \
                1 if len(known) > 0 and known["is_positive_culture"].sum() > 0 else 0

            blood = pre_lm[pre_lm["specimen_group"] == "blood"]
            cov.at[idx, "blood_culture_before_landmark"] = 1 if len(blood) > 0 else 0
            cov.at[idx, "blood_positive_known_before_landmark"] = \
                1 if len(blood) > 0 and blood["is_positive_culture"].sum() > 0 else 0

    if "unrecognized_baseline_resistant_isolate" in eligible.columns:
        cov = cov.merge(
            eligible[["episode_id", "landmark_hour", "unrecognized_baseline_resistant_isolate"]],
            on=["episode_id", "landmark_hour"], how="left"
        )

    cov["respiratory_culture_before_landmark"] = 0
    cov["baseline_pathogen_group"] = ""
    for idx, row in cov.iterrows():
        ep = row["episode_id"]
        lm = row["landmark_time"]
        ep_micro = micro_isolates[micro_isolates["episode_id"] == ep]
        if len(ep_micro) > 0:
            pre_lm = ep_micro[ep_micro["culture_time"] <= lm]
            resp = pre_lm[pre_lm["specimen_group"].isin(["respiratory", "deep_respiratory"])]
            cov.at[idx, "respiratory_culture_before_landmark"] = 1 if len(resp) > 0 else 0
            known_res = pre_lm[(pre_lm["imputed_report_time"] <= lm) &
                               (pre_lm["resistant_organism_flag"] == 1)]
            if len(known_res) > 0 and "highest_risk_pathogen_group" in known_res.columns:
                groups = known_res["highest_risk_pathogen_group"].dropna().unique()
                cov.at[idx, "baseline_pathogen_group"] = "|".join(str(g) for g in groups)

    # Fill NaN defaults for cumulative ABX
    cov["cumulative_systemic_abx_hours"] = cov.get("cumulative_systemic_abx_hours", 0).fillna(0)
    cov["cumulative_broad_abx_hours"] = cov.get("cumulative_broad_abx_hours", 0).fillna(0)
    cov["cumulative_carbapenem_hours"] = cov.get("cumulative_carbapenem_hours", 0).fillna(0)
    cov["cumulative_anti_mrsa_hours"] = cov.get("cumulative_anti_mrsa_hours", 0).fillna(0)

    # ═════════════════════════════════════════════════════
    # NEW in v3.1: Extract previously-missing covariates
    # ═════════════════════════════════════════════════════

    # --- Vasopressor from PRESCRIPTIONS ---
    if prescriptions_raw is None:
        prescriptions_raw = PRESCRIPTIONS_CSV   # pass path for chunked reading
    cov = extract_vasopressor(cov, prescriptions_raw, logfile)

    # --- Lab values from LABEVENTS ---
    if labevents is None:
        labevents = LABEVENTS_CSV              # pass path for chunked reading
    if d_labitems is None:
        d_labitems = pd.read_csv(D_LABITEMS_CSV, low_memory=False)
    cov = extract_labs(cov, labevents, d_labitems, logfile)

    # --- BP from CHARTEVENTS ---
    if chartevents is None:
        chartevents = CHARTEVENTS_CSV           # pass path for chunked reading
    cov = extract_bp_from_chartevents(cov, chartevents, logfile)

    # --- Weight from CHARTEVENTS ---
    cov = extract_weight(cov, chartevents, logfile)

    # --- Comorbidities from DIAGNOSES_ICD ---
    if diagnoses is None:
        diagnoses = DIAGNOSES_ICD_CSV           # pass path for chunked reading
    cov = extract_comorbidities(cov, diagnoses, logfile)

    # --- Surgical procedure markers from SURGERY_INFO ---
    if surgery_info is None:
        surgery_info = SURGERY_INFO_CSV         # pass path for chunked reading
    cov = extract_surgery_procedures(cov, surgery_info, logfile)

    # --- Diagnosis category from ICD + surgery ---
    cov["diagnosis_category"] = ""

    # 4. Select output columns
    output_cols = [
        "subject_id", "hadm_id", "episode_id", "landmark_hour", "landmark_time",
        # Section 7.1: Demographics & ICU
        "age_days", "age_years", "age_group", "gender", "first_careunit",
        "icu_los_before_landmark_hours", "hospital_los_before_icu_hours",
        "calendar_year", "admission_type",
        "admission_weight_kg", "admission_weight_missing",
        "nearest_weight_pre_landmark_kg", "nearest_weight_pre_landmark_missing",
        # Section 7.2: Antibiotic status
        "current_abx_list", "current_broad_abx_list", "current_n_abx",
        "current_max_spectrum_score", "current_anti_pseudomonal",
        "current_anti_mrsa", "current_carbapenem", "current_anti_anaerobe",
        "current_combination_therapy", "current_last_resort",
        "cov_gnb", "cov_pseudomonas", "cov_mrsa", "cov_anaerobe",
        "cov_atypical", "cov_fungal", "cov_salvage",
        "cumulative_systemic_abx_hours", "cumulative_broad_abx_hours",
        "cumulative_carbapenem_hours", "cumulative_anti_mrsa_hours",
        # Section 7.3: Microbiology
        "any_culture_obtained_before_landmark",
        "culture_result_known_before_landmark",
        "number_of_cultures_before_landmark",
        "positive_culture_known_before_landmark",
        "blood_culture_before_landmark",
        "respiratory_culture_before_landmark",
        "blood_positive_known_before_landmark",
        "baseline_resistant_organism",
        "baseline_pathogen_group",
        "unrecognized_baseline_resistant_isolate",
        # Section 7.4: Labs
        "wbc_max_24h", "wbc_missing",
        "neutrophil_max_24h", "neutrophil_missing",
        "crp_max_24h", "crp_missing",
        "pct_max_24h", "pct_missing",
        "lactate_max_24h", "lactate_missing",
        "creatinine_max_24h", "creatinine_missing",
        "bilirubin_max_24h", "bilirubin_missing",
        "temperature_max_24h", "temperature_missing",
        "platelet_min_24h", "platelet_missing",
        "albumin_min_24h", "albumin_missing",
        "hemoglobin_min_24h", "hemoglobin_missing",
        "ph_min_24h", "ph_missing",
        "base_excess_min_24h", "base_excess_missing",
        "spo2_min_24h", "spo2_missing",
        "systolic_bp_min_24h", "systolic_bp_missing",
        "mean_bp_min_24h", "mean_bp_missing",
        # Section 7.5: Organ support
        "mechanical_ventilation_at_landmark", "mechanical_ventilation_missing",
        "vasopressor_at_landmark", "vasopressor_missing",
        "rrt_at_landmark", "rrt_missing",
        "central_line_at_landmark", "central_line_missing",
        "ECMO_at_landmark", "ECMO_missing",
        # Section 7.6: Diagnosis
        "diagnosis_category",
        # Section 7.7: Comorbidities
        "congenital_heart_disease", "chronic_lung_disease",
        "chronic_kidney_disease", "neurologic_disease", "malignancy",
        "hematologic_disease", "immunodeficiency", "transplant_status",
        "postoperative_status", "prematurity",
    ]
    output = cov[[c for c in output_cols if c in cov.columns]].reset_index(drop=True)

    output.to_csv(LANDMARK_BASELINE_CSV, index=False, encoding="utf-8")
    log(f"-> Saved {LANDMARK_BASELINE_CSV}: {len(output):,} rows, {len(output.columns)} cols", logfile)
    log("Step 7 complete!", logfile)
    return output


if __name__ == "__main__":
    logfile = os.path.join(OUTPUT_DIR, "processing_log.txt")
    run_step7(logfile)
