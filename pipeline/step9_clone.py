"""Step 9: Clone-Censor Dataset (cloned_trials.csv)
Implements Section 9 from the processing plan:
- Create 2 clones per eligible trial (reduction vs continue_broad)
- Artificial censoring rules for each clone (Section 9.2)
- Time Priority Principle for competing events (Section 9.3)
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from utils import parse_datetime, hours_between, log


def run_step9(logfile=None, reduction_events=None, eligible_trials=None,
              micro_isolates=None, icu_base=None):
    log("=" * 60, logfile)
    log("Step 9: Clone-Censor Dataset", logfile)
    log("=" * 60, logfile)

    # 1. Load inputs
    if reduction_events is None:
        reduction_events = pd.read_csv(SPECTRUM_REDUCTION_CSV, low_memory=False)
        reduction_events["landmark_time"] = parse_datetime(reduction_events["landmark_time"])
        reduction_events["grace_end"] = parse_datetime(reduction_events["grace_end"])
        reduction_events["reduction_time"] = parse_datetime(reduction_events["reduction_time"])
        reduction_events["escalation_time"] = parse_datetime(reduction_events["escalation_time"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            reduction_events[col] = reduction_events[col].astype(str)

    if eligible_trials is None:
        eligible_trials = pd.read_csv(LANDMARK_ELIGIBLE_CSV, low_memory=False)
        eligible_trials["landmark_time"] = parse_datetime(eligible_trials["landmark_time"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            eligible_trials[col] = eligible_trials[col].astype(str)

    if icu_base is None:
        icu_base = pd.read_csv(ICU_BASE_CSV, low_memory=False)
        icu_base["episode_intime"] = parse_datetime(icu_base["episode_intime"])
        icu_base["death_time"] = parse_datetime(icu_base["death_time"])
        icu_base["episode_outtime"] = parse_datetime(icu_base["episode_outtime"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            icu_base[col] = icu_base[col].astype(str)

    # Merge reduction events with ICU data for death_time and episode_outtime
    red = reduction_events.merge(
        icu_base[["subject_id", "hadm_id", "episode_id", "episode_intime", "death_time",
                   "episode_outtime"]],
        on=["subject_id", "hadm_id", "episode_id"], how="left"
    )

    # Exclude trials with death in grace period (Section 8.4/9.1)
    n_death_grace = (red["death_in_grace_period"] == 1).sum()
    if n_death_grace > 0:
        log(f"  Excluding {n_death_grace} trials with death in grace period", logfile)
        red = red[red["death_in_grace_period"] != 1].copy()

    log(f"  Trials to clone: {len(red)}", logfile)

    # 2. Build clone pairs (Section 9.1)
    clones = []

    for _, row in red.iterrows():
        ep_id = row["episode_id"]
        lm_hour = row["landmark_hour"]
        lm_time = row["landmark_time"]
        grace_end = row["grace_end"]
        death_time = row.get("death_time")
        episode_outtime = row.get("episode_outtime")

        # Follow-up: landmark to min(discharge, landmark + 28 days)
        followup_start = lm_time
        followup_end_candidates = []
        if pd.notna(episode_outtime):
            followup_end_candidates.append(episode_outtime)
        if pd.notna(followup_start):
            followup_end_candidates.append(followup_start + pd.Timedelta(days=28))
        followup_end = min(followup_end_candidates) if followup_end_candidates else \
            followup_start + pd.Timedelta(days=28)

        base = {
            "subject_id": row["subject_id"],
            "hadm_id": row["hadm_id"],
            "episode_id": ep_id,
            "landmark_hour": lm_hour,
            "landmark_time": lm_time,
            "followup_start_time": followup_start,
            "followup_end_time": followup_end,
            "reduction_occurred_real": row["reduction_occurred"],
            "escalation_occurred_real": row.get("escalation_occurred", 0),
            "continue_broad_48h_real": row.get("continue_broad_48h", 0),
            "reduction_time_real": row.get("reduction_time", pd.NaT),
            "escalation_time_real": row.get("escalation_time", pd.NaT),
            "death_time": death_time,
        }

        # ── Clone A: spectrum_reduction ──
        clone_a = base.copy()
        clone_a["clone_id"] = f"{ep_id}_{lm_hour}_reduction"
        clone_a["assigned_strategy"] = "spectrum_reduction"

        # Check stop_followed_by_death_24h first (Section 9.2 Clone A)
        if row.get("stop_followed_by_death_24h", 0) == 1:
            clone_a["artificial_censored"] = 1
            clone_a["artificial_censor_time"] = clone_a["reduction_time_real"]
            clone_a["artificial_censor_reason"] = "terminal_stop_ambiguous"
        elif row["reduction_occurred"] == 1:
            # Reduction occurred - check for concurrent escalation
            if row.get("reduction_with_concurrent_escalation", 0) == 1:
                clone_a["artificial_censored"] = 1
                clone_a["artificial_censor_time"] = clone_a["reduction_time_real"]
                clone_a["artificial_censor_reason"] = "reduction_with_escalation"
            else:
                clone_a["artificial_censored"] = 0
                clone_a["artificial_censor_time"] = pd.NaT
                clone_a["artificial_censor_reason"] = ""
        elif row.get("escalation_occurred", 0) == 1:
            # No reduction, escalation occurred
            esc_time = row.get("escalation_time")
            clone_a["artificial_censored"] = 1
            clone_a["artificial_censor_time"] = esc_time if pd.notna(esc_time) else grace_end
            clone_a["artificial_censor_reason"] = "treatment_escalation"
        else:
            # No reduction within grace period
            clone_a["artificial_censored"] = 1
            clone_a["artificial_censor_time"] = grace_end
            clone_a["artificial_censor_reason"] = "no_reduction_within_grace"

        clones.append(clone_a)

        # ── Clone B: continue_broad ──
        clone_b = base.copy()
        clone_b["clone_id"] = f"{ep_id}_{lm_hour}_continue"
        clone_b["assigned_strategy"] = "continue_broad"

        if row.get("continue_broad_48h", 0) == 1:
            clone_b["artificial_censored"] = 0
            clone_b["artificial_censor_time"] = pd.NaT
            clone_b["artificial_censor_reason"] = ""
        elif row["reduction_occurred"] == 1:
            # Reduction happened - censor at reduction time
            clone_b["artificial_censored"] = 1
            clone_b["artificial_censor_time"] = clone_b["reduction_time_real"]
            clone_b["artificial_censor_reason"] = "reduction_before_grace_end"
        elif row.get("escalation_occurred", 0) == 1:
            # Escalation happened - censor at escalation time (Section 9.2 Clone B)
            esc_time = row.get("escalation_time")
            clone_b["artificial_censored"] = 1
            clone_b["artificial_censor_time"] = esc_time if pd.notna(esc_time) else grace_end
            clone_b["artificial_censor_reason"] = "treatment_escalation"
        else:
            # Neither reduction, escalation, nor continue_broad - not censored
            # (e.g., death in grace period was excluded in step 8)
            clone_b["artificial_censored"] = 0
            clone_b["artificial_censor_time"] = pd.NaT
            clone_b["artificial_censor_reason"] = ""

        clones.append(clone_b)

    cloned = pd.DataFrame(clones)
    log(f"  Total clones: {len(cloned)}", logfile)

    # Clamp artificial_censor_time to not exceed followup_end_time
    mask = cloned["artificial_censor_time"].notna() & \
           (cloned["artificial_censor_time"] > cloned["followup_end_time"])
    n_clamped = mask.sum()
    if n_clamped > 0:
        cloned.loc[mask, "artificial_censor_time"] = cloned.loc[mask, "followup_end_time"]
        log(f"  Clamped {n_clamped} artificial_censor_time(s) past followup_end", logfile)

    # 3. Event status initialization (Section 9.3)
    # 0 = censored (to be refined by Step 10 with outcome data)
    # Death is pre-computed here: if death occurs before artificial censor, death wins
    # Final Time Priority (Outcome > Death > Artificial Censor) applied in Step 10
    cloned["event_status"] = 0
    cloned["death_competing_same_time"] = 0

    for idx, row in cloned.iterrows():
        death_time = row.get("death_time")

        # Check death within follow-up
        if pd.notna(death_time) and death_time <= row["followup_end_time"]:
            art_time = row["artificial_censor_time"]

            # Time Priority: if artificial_censor_time == death_time, death wins (Section 9.3)
            if pd.isna(art_time) or death_time < art_time:
                cloned.at[idx, "event_status"] = 2
            elif death_time == art_time:
                cloned.at[idx, "event_status"] = 2
                cloned.at[idx, "death_competing_same_time"] = 1
            else:
                # Artificially censored before death - stays censored
                cloned.at[idx, "event_status"] = 0

    # 4. Output (Section 9.4)
    output_cols = [
        "subject_id", "hadm_id", "episode_id", "landmark_hour",
        "clone_id", "assigned_strategy",
        "followup_start_time", "followup_end_time",
        "event_status", "artificial_censored",
        "artificial_censor_time", "artificial_censor_reason",
        "death_competing_same_time", "death_time",
    ]
    output = cloned[[c for c in output_cols if c in cloned.columns]].reset_index(drop=True)

    output.to_csv(CLONED_TRIALS_CSV, index=False, encoding="utf-8")
    log(f"-> Saved {CLONED_TRIALS_CSV}: {len(output):,} rows", logfile)

    # Statistics
    n_a = (output["assigned_strategy"] == "spectrum_reduction").sum()
    n_b = (output["assigned_strategy"] == "continue_broad").sum()
    n_cens_a = ((output["assigned_strategy"] == "spectrum_reduction") &
                 (output["artificial_censored"] == 1)).sum()
    n_cens_b = ((output["assigned_strategy"] == "continue_broad") &
                 (output["artificial_censored"] == 1)).sum()
    log(f"  Clone A (reduction): {n_a:,} ({n_cens_a:,} censored)", logfile)
    log(f"  Clone B (continue):   {n_b:,} ({n_cens_b:,} censored)", logfile)

    # Reason breakdown
    reasons = output[output["artificial_censored"] == 1]["artificial_censor_reason"].value_counts()
    log(f"  Censor reasons: {reasons.to_dict()}", logfile)
    log("Step 9 complete!", logfile)

    return output


if __name__ == "__main__":
    logfile = os.path.join(OUTPUT_DIR, "processing_log.txt")
    run_step9(logfile)
