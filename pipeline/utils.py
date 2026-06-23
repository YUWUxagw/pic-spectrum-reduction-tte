"""PIC Pipeline Utilities - Time handling, helpers, constants"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os

# ── Time Helpers ───────────────────────────────────────────

def parse_datetime(series, dayfirst=False):
    """Parse datetime series, coercing errors to NaT."""
    return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)


def hours_between(end_series, start_series):
    """Hours between two datetime series: (end - start) / 3600."""
    return (end_series - start_series).dt.total_seconds() / 3600.0


def has_date_only(series):
    """Check if all non-NaT values in a datetime series are at midnight (date-only)."""
    valid = series.dropna()
    if len(valid) == 0:
        return pd.Series(False, index=series.index)
    return (valid.dt.hour == 0) & (valid.dt.minute == 0) & (valid.dt.second == 0)


def is_active_at(time_point, start_series, end_series):
    """Vectorized: is drug active at a given time T?
    Condition: start <= T AND end > T"""
    return (start_series <= time_point) & (end_series > time_point)


def overlaps_window(window_start, window_end, start_series, end_series):
    """Vectorized: does drug interval overlap [W_start, W_end]?
    Condition: start < W_end AND end > W_start"""
    return (start_series < window_end) & (end_series > window_start)


# ── Age & Demographics ─────────────────────────────────────

def age_group_label(age_years):
    """Assign age_group based on age in years."""
    if age_years < 28/365.25:
        return "neonate"
    elif age_years < 1:
        return "infant"
    elif age_years < 3:
        return "toddler"
    elif age_years < 12:
        return "child"
    else:
        return "adolescent"


def classify_age(df, dob_col, ref_col):
    """Add age_days, age_years, age_group columns.
    Clamps negative ages to 0 (data artifact where dob > reference time).
    """
    age_days = (df[ref_col] - df[dob_col]).dt.total_seconds() / 86400.0
    # Clamp negative ages (dob slightly after reference time due to data precision)
    n_neg = (age_days < 0).sum()
    if n_neg > 0:
        age_days = age_days.clip(lower=0)
    age_years = age_days / 365.25
    groups = pd.cut(
        age_years,
        bins=[-0.001, 28/365.25, 1, 3, 12, 18],
        labels=["neonate", "infant", "toddler", "child", "adolescent"],
        right=False,
    )
    return age_days, age_years, groups


# ── ICU Episode Building ───────────────────────────────────

def build_episodes(icu_df):
    """Build independent ICU episodes from ICUSTAYS.
    Returns DataFrame with episode-level rows.
    """
    df = icu_df.copy()
    df = df.sort_values(["subject_id", "hadm_id", "intime"]).reset_index(drop=True)

    episodes = []
    for (subj, hadm), grp in df.groupby(["subject_id", "hadm_id"], sort=False):
        grp = grp.sort_values("intime")
        episode_seq = 0
        current_stays = []
        current_intime = None
        current_outtime = None

        for _, row in grp.iterrows():
            intime = row["intime"]
            outtime = row["outtime"]
            stay_id = row["icustay_id"]

            if current_intime is None:
                # First stay
                current_stays = [(intime, outtime, stay_id)]
                current_intime = intime
                current_outtime = outtime
            else:
                gap = (intime - current_outtime).total_seconds() / 3600.0
                if gap <= 48:
                    # Merge: contiguous
                    current_stays.append((intime, outtime, stay_id))
                    current_outtime = max(current_outtime, outtime)
                else:
                    # Finalize current episode
                    episode_seq += 1
                    episodes.append(_make_episode(subj, hadm, episode_seq, current_stays))
                    # Start new episode
                    current_stays = [(intime, outtime, stay_id)]
                    current_intime = intime
                    current_outtime = outtime

        # Finalize last episode in group
        if current_stays:
            episode_seq += 1
            episodes.append(_make_episode(subj, hadm, episode_seq, current_stays))

    return pd.DataFrame(episodes)


def _make_episode(subject_id, hadm_id, episode_seq, stays):
    """Build a single episode record from merged ICU stays."""
    intimes = [s[0] for s in stays]
    outtimes = [s[1] for s in stays]
    stay_ids = [s[2] for s in stays]
    raw_intervals = [(str(s[0]), str(s[1])) for s in stays]

    episode_intime = min(intimes)
    episode_outtime = max(outtimes)
    episode_los = (episode_outtime - episode_intime).total_seconds() / 3600.0

    return {
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "episode_sequence": episode_seq,
        "episode_id": f"{subject_id}_{hadm_id}_ep{episode_seq}",
        "icustay_id_list": "|".join(str(s) for s in stay_ids),
        "episode_intime": episode_intime,
        "episode_outtime": episode_outtime,
        "episode_los_hours": episode_los,
        "raw_icu_intervals": raw_intervals,
    }


# ── Specimen Group Mapping ─────────────────────────────────

def map_specimen_group(spec_desc):
    """Map specimen description to specimen_group."""
    if not isinstance(spec_desc, str):
        return "other"
    s = spec_desc.lower()
    if any(kw in s for kw in ["blood", "血培养", "血"]):
        return "blood"
    if any(kw in s for kw in ["csf", "脑脊液", "cerebrospinal"]):
        return "csf"
    if any(kw in s for kw in ["bal", "bronchoalveolar", "支气管肺泡"]):
        return "deep_respiratory"
    if any(kw in s for kw in ["sputum", "痰", "respiratory", "呼吸道", "气管"]):
        return "respiratory"
    if any(kw in s for kw in ["urine", "尿"]):
        return "urine"
    if any(kw in s for kw in ["pleural", "ascitic", "peritoneal", "joint", "胸水", "腹水", "关节",
                               "sterile", "无菌"]):
        return "sterile_fluid"
    if any(kw in s for kw in ["wound", "pus", "abscess", "伤口", "脓", "脓肿"]):
        return "wound"
    if any(kw in s for kw in ["catheter", "导管", "tip"]):
        return "catheter"
    if any(kw in s for kw in ["stool", "粪便", "直肠"]):
        return "stool"
    return "other"


def get_report_lag(specimen_group):
    """Get culture report lag hours for a specimen group."""
    from config import CULTURE_LAG_HOURS
    return CULTURE_LAG_HOURS.get(specimen_group, 48)


# ── Duplicate Detection ────────────────────────────────────

def dedup_identical_cultures(df):
    """Remove identical culture duplicates within 7 days.
    Groups by: subject_id, organism (std), phenotype, specimen_group
    Handles list-type columns by converting to hashable string representation.
    """
    if len(df) == 0:
        return df
    # Convert list-type phenotype to hashable string for grouping
    df = df.copy()
    if "phenotype" in df.columns:
        df["_pheno_str"] = df["phenotype"].apply(
            lambda x: "|".join(sorted([str(p) for p in x if p]))
            if isinstance(x, list) else str(x) if pd.notna(x) else ""
        )
        group_cols = ["subject_id", "org_name_std", "specimen_group", "_pheno_str"]
    else:
        group_cols = ["subject_id", "org_name_std", "specimen_group"]

    df = df.sort_values(["subject_id", "org_name_std", "specimen_group", "culture_time"])
    result = []
    for _, grp in df.groupby(group_cols, sort=False, dropna=False):
        grp = grp.sort_values("culture_time")
        keep = []
        last_time = None
        for _, row in grp.iterrows():
            ct = row["culture_time"]
            if last_time is None or (ct - last_time).total_seconds() / 86400.0 >= 7:
                keep.append(row)
                last_time = ct
        result.extend(keep)
    out = pd.DataFrame(result).reset_index(drop=True) if result else pd.DataFrame()
    if "_pheno_str" in out.columns:
        out = out.drop(columns=["_pheno_str"])
    return out


# ── Logging ────────────────────────────────────────────────

def log(msg, logfile=None):
    """Print and optionally write to logfile."""
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if logfile:
        with open(logfile, "a", encoding="utf-8") as f:
            f.write(line + "\n")
