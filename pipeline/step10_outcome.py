"""Step 10: Primary Outcome Definition and Analysis
Implements Section 10 from the processing plan:
- Incident Resistant Organism Colonization/Infection detection
- Persistent baseline resistance deduplication (Section 5.9.2)
- Incident resistance type classification
- Competing risk handling (death)
"""
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from utils import parse_datetime, hours_between, log

# CONS organisms for sensitivity analysis (common skin contaminants)
CONS_ORGANISMS = [
    "staphylococcus_epidermidis",
    "staphylococcus_haemolyticus",
    "staphylococcus_hominis",
    "staphylococcus_capitis",
    "staphylococcus_warneri",
    "staphylococcus_lugdunensis",
    "staphylococcus_saprophyticus",
    "staphylococcus_pettenkoferi",
    "staphylococcus_sciuri",
    "staphylococcus_xylosus",
    "staphylococcus_pasteuri",
]


def run_step10(logfile=None, cloned_trials=None, micro_isolates=None,
               icu_base=None):
    log("=" * 60, logfile)
    log("Step 10: Primary Outcome Analysis", logfile)
    log("=" * 60, logfile)

    # 1. Load inputs
    if cloned_trials is None:
        cloned_trials = pd.read_csv(CLONED_TRIALS_CSV, low_memory=False)
        cloned_trials["followup_start_time"] = parse_datetime(
            cloned_trials["followup_start_time"])
        cloned_trials["followup_end_time"] = parse_datetime(
            cloned_trials["followup_end_time"])
        cloned_trials["artificial_censor_time"] = parse_datetime(
            cloned_trials["artificial_censor_time"])
        if "death_time" in cloned_trials.columns:
            cloned_trials["death_time"] = parse_datetime(cloned_trials["death_time"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            cloned_trials[col] = cloned_trials[col].astype(str)
    log(f"  Cloned trials: {len(cloned_trials)}", logfile)

    if micro_isolates is None:
        micro_isolates = pd.read_csv(MICRO_ISOLATES_CLEAN_CSV, low_memory=False)
        micro_isolates["culture_time"] = parse_datetime(micro_isolates["culture_time"])
        micro_isolates["imputed_report_time"] = parse_datetime(
            micro_isolates["imputed_report_time"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            micro_isolates[col] = micro_isolates[col].astype(str)

    if icu_base is None:
        icu_base = pd.read_csv(ICU_BASE_CSV, low_memory=False)
        icu_base["episode_intime"] = parse_datetime(icu_base["episode_intime"])
        icu_base["death_time"] = parse_datetime(icu_base["death_time"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            icu_base[col] = icu_base[col].astype(str)

    # 2. Extract resistant isolates
    res_isolates = micro_isolates[
        (micro_isolates["resistant_organism_flag"] == 1) &
        (micro_isolates["is_clinically_relevant"] == 1)
    ].copy()
    log(f"  Clinically relevant resistant isolates: {len(res_isolates):,}", logfile)

    # 3. For each clone, find the first incident resistant outcome
    log("Detecting incident resistant outcomes...", logfile)
    outcome_records = []

    # Diagnostic counters
    diag_persistent_skipped = 0
    diag_new_organism = 0
    diag_resistance_emergence = 0
    diag_persistent_baseline = 0
    diag_no_pre_lm = 0

    for idx, clone in cloned_trials.iterrows():
        ep = clone["episode_id"]
        lm_time = clone["followup_start_time"]
        censor_time = clone["artificial_censor_time"]
        followup_end = clone["followup_end_time"]
        death_time = None  # fetched from icu_base if needed

        # Get effective followup end
        if pd.notna(censor_time) and censor_time < followup_end:
            effective_end = censor_time
        else:
            effective_end = followup_end

        # Post-landmark resistant isolates
        ep_res = res_isolates[res_isolates["episode_id"] == ep]
        post_lm = ep_res[ep_res["imputed_report_time"] > lm_time]
        post_lm = post_lm[post_lm["imputed_report_time"] <= effective_end]

        if len(post_lm) == 0:
            outcome_records.append({
                "clone_id": clone["clone_id"],
                "outcome_occurred": 0,
                "outcome_time": pd.NaT,
                "outcome_hour_from_landmark": np.nan,
                "incident_organism": "",
                "incident_phenotype": "",
                "incident_resistance_type": "",
                "persistent_baseline_resistance": 0,
                "new_organism_flag": 0,
                "resistance_emergence_flag": 0,
                "cons_outcome": 0,
                "mdro_outcome": 0,
            })
            continue

        # Sort by imputed_report_time
        post_lm = post_lm.sort_values("imputed_report_time")

        # Check for persistent baseline resistance (Section 5.9.2)
        pre_lm_res = ep_res[ep_res["imputed_report_time"] <= lm_time]

        for _, outcome_row in post_lm.iterrows():
            org = outcome_row.get("org_name_std", "")
            pheno = str(outcome_row.get("phenotype", ""))
            spec = outcome_row.get("specimen_group", "")
            ct = outcome_row["culture_time"]

            # Check if this is persistent baseline (same org + phenotype within 7 days)
            is_persistent = False
            if len(pre_lm_res) > 0:
                for _, pre_row in pre_lm_res.iterrows():
                    pre_org = pre_row.get("org_name_std", "")
                    pre_pheno = str(pre_row.get("phenotype", ""))
                    pre_spec = pre_row.get("specimen_group", "")
                    pre_ct = pre_row["culture_time"]

                    if pd.notna(ct) and pd.notna(pre_ct):
                        days_diff = abs((ct - pre_ct).total_seconds() / 86400.0)
                        if pre_org == org and pre_pheno == pheno and \
                           pre_spec == spec and days_diff < 7:
                            is_persistent = True
                            break

            if not is_persistent:
                # This is a true incident outcome
                # Classify incident resistance type (Section 5.9.3)
                inc_type = "new_organism"
                if len(pre_lm_res) > 0:
                    for _, pre_row in pre_lm_res.iterrows():
                        pre_org = pre_row.get("org_name_std", "")
                        pre_pheno = str(pre_row.get("phenotype", ""))
                        if pre_org == org:
                            if pre_pheno != pheno:
                                inc_type = "resistance_emergence_same_species"
                            break
                    else:
                        diag_no_pre_lm += 1
                else:
                    diag_no_pre_lm += 1

                if inc_type == "resistance_emergence_same_species":
                    diag_resistance_emergence += 1
                else:
                    diag_new_organism += 1

                # CONS flag for sensitivity analysis
                is_cons = (org.lower() in CONS_ORGANISMS)

                # MDRO flag: has resistant phenotype (MRSA, CRAB, CRPA, CRE, ESBL)
                pheno_str = str(pheno).lower()
                is_mdro = any(kw in pheno_str for kw in
                              ["mrsa", "vre", "cre", "crpa", "crab", "esbl"])

                outcome_records.append({
                    "clone_id": clone["clone_id"],
                    "outcome_occurred": 1,
                    "outcome_time": outcome_row["imputed_report_time"],
                    "outcome_hour_from_landmark": hours_between(
                        pd.Series([outcome_row["imputed_report_time"]]),
                        pd.Series([lm_time])
                    ).iloc[0] if pd.notna(outcome_row["imputed_report_time"]) else np.nan,
                    "incident_organism": org,
                    "incident_phenotype": str(pheno),
                    "incident_resistance_type": inc_type,
                    "persistent_baseline_resistance": 0,
                    "new_organism_flag": 1 if inc_type == "new_organism" else 0,
                    "resistance_emergence_flag": 1 if inc_type == "resistance_emergence_same_species" else 0,
                    "cons_outcome": 1 if is_cons else 0,
                    "mdro_outcome": 1 if is_mdro else 0,
                })
                break
            else:
                diag_persistent_skipped += 1
        else:
            # All post-landmark isolates were persistent baseline
            diag_persistent_baseline += 1
            outcome_records.append({
                "clone_id": clone["clone_id"],
                "outcome_occurred": 0,
                "outcome_time": pd.NaT,
                "outcome_hour_from_landmark": np.nan,
                "incident_organism": "",
                "incident_phenotype": "",
                "incident_resistance_type": "persistent_baseline",
                "persistent_baseline_resistance": 1,
                "new_organism_flag": 0,
                "resistance_emergence_flag": 0,
                "cons_outcome": 0,
                "mdro_outcome": 0,
            })

    outcomes = pd.DataFrame(outcome_records)

    # Diagnostic summary for outcome classification
    log(f"  Outcome classification diagnostics:", logfile)
    log(f"    No pre-LM resistant isolates: {diag_no_pre_lm}", logfile)
    log(f"    New organism (different species): {diag_new_organism}", logfile)
    log(f"    Resistance emergence (same species, new phenotype): {diag_resistance_emergence}", logfile)
    log(f"    Persistent baseline (all post-LM deduplicated): {diag_persistent_baseline}", logfile)
    log(f"    Persistent isolates skipped (continuing loop): {diag_persistent_skipped}", logfile)

    # 4. Merge outcomes back to cloned trials
    final = cloned_trials.merge(outcomes, on="clone_id", how="left")

    # 5. Final event status (Section 10.2 - Competing Risks)
    # Time Priority Principle (Section 9.3): Outcome > Death > Artificial Censor
    # event_status: 0=censored, 1=primary outcome, 2=death (competing risk)
    if "outcome_occurred" in final.columns:
        for idx, row in final.iterrows():
            art_time = row["artificial_censor_time"]
            death_time = None
            # Get death_time from icu_base if merged
            if "death_time" in final.columns:
                death_time = row.get("death_time")
            outcome_time = row.get("outcome_time")

            has_outcome = (row.get("outcome_occurred", 0) == 1)
            has_death = pd.notna(death_time) and death_time <= row["followup_end_time"]
            has_censor = pd.notna(art_time)

            # Collect events with times
            events_list = []
            if has_outcome and pd.notna(outcome_time):
                events_list.append((outcome_time, 1))  # outcome
            if has_death:
                events_list.append((death_time, 2))     # death
            if has_censor:
                events_list.append((art_time, 0))       # artificial censor

            if events_list:
                # Sort by time, then by priority (Outcome > Death > Censor)
                # Lower event_status number = higher priority for outcome (1 > 0)
                # For same time: outcome(1) wins over death(2) over censor(0)
                events_list.sort(key=lambda x: (x[0], [1, 2, 0].index(x[1])))
                final_status = events_list[0][1]
                final.at[idx, "event_status"] = final_status

                # Track death competing at same time
                if final_status == 1 and has_death and outcome_time == death_time:
                    final.at[idx, "death_competing_same_time"] = 1
            elif has_outcome:
                final.at[idx, "event_status"] = 1

    # 6. Update cloned_trials.csv with outcomes
    final.to_csv(CLONED_TRIALS_CSV, index=False, encoding="utf-8")
    log(f"-> Updated {CLONED_TRIALS_CSV} with outcomes", logfile)

    # 7. Summary statistics
    n_clones = len(final)
    n_outcomes = (final["event_status"] == 1).sum()
    n_deaths = (final["event_status"] == 2).sum()
    n_censored = (final["event_status"] == 0).sum()

    log(f"  Total clones: {n_clones:,}", logfile)
    log(f"  Primary outcomes: {n_outcomes:,} ({n_outcomes/n_clones*100:.2f}%)", logfile)
    log(f"  Deaths (competing): {n_deaths:,} ({n_deaths/n_clones*100:.2f}%)", logfile)
    log(f"  Censored: {n_censored:,} ({n_censored/n_clones*100:.2f}%)", logfile)

    # By strategy
    for strat in ["spectrum_reduction", "continue_broad"]:
        s = final[final["assigned_strategy"] == strat]
        n = len(s)
        o = (s["event_status"] == 1).sum()
        log(f"  [{strat}] Outcomes: {o:,}/{n:,} ({o/n*100:.2f}%)" if n > 0
            else f"  [{strat}] No trials", logfile)

    # Incident resistance type breakdown
    outcomes_only = final[final["outcome_occurred"] == 1]
    n_incident = len(outcomes_only)
    if n_incident > 0:
        type_breakdown = outcomes_only["incident_resistance_type"].value_counts()
        log(f"  Outcome types: {type_breakdown.to_dict()}", logfile)

    # CONS and MDRO outcome summary for sensitivity analysis
    n_cons = (final["cons_outcome"] == 1).sum()
    n_mdro = (final["mdro_outcome"] == 1).sum()
    if n_cons > 0:
        log(f"  CONS outcomes (sensitivity flag): {n_cons:,} ({n_cons/n_outcomes*100:.1f}% of outcomes)" if n_outcomes > 0
            else f"  CONS outcomes (sensitivity flag): {n_cons:,}", logfile)
    if n_mdro > 0:
        log(f"  MDRO outcomes: {n_mdro:,} ({n_mdro/n_outcomes*100:.1f}% of outcomes)" if n_outcomes > 0
            else f"  MDRO outcomes: {n_mdro:,}", logfile)
    if n_outcomes > 0:
        n_other = n_outcomes - n_cons - n_mdro
        log(f"  Non-CONS, non-MDRO outcomes: {n_other:,} ({n_other/n_outcomes*100:.1f}% of outcomes) — susceptible pathogens", logfile)

    # Competing risks note (Section 10.2)
    log("  Competing risks: Fine-Gray subdistribution hazard model or", logfile)
    log("  Aalen-Johansen estimator with cluster-robust SE (subject_id level)", logfile)
    log("  recommended for primary analysis.", logfile)

    log("Step 10 complete!", logfile)
    return final


if __name__ == "__main__":
    logfile = os.path.join(OUTPUT_DIR, "processing_log.txt")
    run_step10(logfile)
