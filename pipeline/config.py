"""PIC Pipeline Configuration"""
import os

# Base paths
BASE_DIR = r"F:\test"
DATA_DIR = os.path.join(BASE_DIR, r"paediatric-intensive-care-database-1.1.0\paediatric-intensive-care-database-1.1.0 2")
UPDATE_DIR = os.path.join(BASE_DIR, r"PICv1.1.0_update 2")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# Input tables (main database)
PATIENTS_CSV = os.path.join(DATA_DIR, "PATIENTS.csv")
ADMISSIONS_CSV = os.path.join(DATA_DIR, "ADMISSIONS.csv")
ICUSTAYS_CSV = os.path.join(DATA_DIR, "ICUSTAYS.csv")
PRESCRIPTIONS_CSV = os.path.join(DATA_DIR, "PRESCRIPTIONS.csv")
MICROBIOLOGYEVENTS_CSV = os.path.join(DATA_DIR, "MICROBIOLOGYEVENTS.csv")
LABEVENTS_CSV = os.path.join(DATA_DIR, "LABEVENTS.csv")
D_LABITEMS_CSV = os.path.join(DATA_DIR, "D_LABITEMS.csv")
CHARTEVENTS_CSV = os.path.join(DATA_DIR, "CHARTEVENTS.csv")
D_ITEMS_CSV = os.path.join(DATA_DIR, "D_ITEMS.csv")
INPUTEVENTS_CSV = os.path.join(DATA_DIR, "INPUTEVENTS.csv")
OUTPUTEVENTS_CSV = os.path.join(DATA_DIR, "OUTPUTEVENTS.csv")
DIAGNOSES_ICD_CSV = os.path.join(DATA_DIR, "DIAGNOSES_ICD.csv")
D_ICD_DIAGNOSES_CSV = os.path.join(DATA_DIR, "D_ICD_DIAGNOSES.csv")
EMR_SYMPTOMS_CSV = os.path.join(DATA_DIR, "EMR_SYMPTOMS.csv")
OR_EXAM_REPORTS_CSV = os.path.join(DATA_DIR, "OR_EXAM_REPORTS.csv")
SURGERY_VITAL_SIGNS_CSV = os.path.join(DATA_DIR, "SURGERY_VITAL_SIGNS.csv")
SURGERY_INFO_CSV = os.path.join(DATA_DIR, "SURGERY_INFO.csv")

# Update tables (override if they exist)
UPDATE_DIAGNOSES_ICD = os.path.join(UPDATE_DIR, "DIAGNOSES_ICD.csv")
UPDATE_D_ICD_DIAGNOSES = os.path.join(UPDATE_DIR, "D_ICD_DIAGNOSES.csv")
UPDATE_SURGERY_VITAL_SIGNS = os.path.join(UPDATE_DIR, "SURGERY_VITAL_SIGNS.csv")
UPDATE_D_ITEMS = os.path.join(UPDATE_DIR, "D_ITEMS.csv")
UPDATE_SURGERY_INFO = os.path.join(UPDATE_DIR, "SURGERY_INFO.csv")

# Output files
ICU_BASE_CSV = os.path.join(OUTPUT_DIR, "icu_base.csv")
ABX_DICT_CSV = os.path.join(OUTPUT_DIR, "antibiotic_dictionary.csv")
ABX_ORDERS_CLEAN_CSV = os.path.join(OUTPUT_DIR, "abx_orders_clean.csv")
ORGANISM_DICT_CSV = os.path.join(OUTPUT_DIR, "organism_dictionary.csv")
MICRO_ISOLATES_CLEAN_CSV = os.path.join(OUTPUT_DIR, "micro_isolates_clean.csv")
LANDMARK_ELIGIBLE_CSV = os.path.join(OUTPUT_DIR, "landmark_eligible_trials.csv")
LANDMARK_BASELINE_CSV = os.path.join(OUTPUT_DIR, "landmark_baseline_covariates.csv")
SPECTRUM_REDUCTION_CSV = os.path.join(OUTPUT_DIR, "spectrum_reduction_events.csv")
CLONED_TRIALS_CSV = os.path.join(OUTPUT_DIR, "cloned_trials.csv")

# Landmark timepoints (hours from episode_intime)
LANDMARK_HOURS = [48, 72, 96]
GRACE_PERIOD_HOURS = 48
ASCERTAINMENT_HOURS = 72
ALLOWABLE_GAP_HOURS = 6

# Age group thresholds
AGE_GROUPS = {
    "neonate": (0, 28/365.25),
    "infant": (28/365.25, 1),
    "toddler": (1, 3),
    "child": (3, 12),
    "adolescent": (12, 18),
}

# Culture report lag (hours) by specimen group
CULTURE_LAG_HOURS = {
    "blood": 60,
    "csf": 60,
    "sterile_fluid": 48,
    "respiratory": 36,
    "deep_respiratory": 36,
    "urine": 24,
    "wound": 36,
    "catheter": 48,
    "stool": 48,
    "other": 48,
}

# Vasopressor drug list
VASOPRESSORS = ["dopamine", "dobutamine", "epinephrine", "norepinephrine", "vasopressin", "milrinone"]

# ICD infection codes
INFECTION_ICD9 = ["038", "995.91", "995.92"]
INFECTION_ICD10_PREFIX = ["A40", "A41", "R65.20", "R65.21"]

# Ensure output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)
