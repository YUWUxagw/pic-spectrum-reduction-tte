"""Step 5: Microbiology Processing (micro_isolates_clean.csv)
Implements Sections 5.1-5.9 from the processing plan:
- 4-level data structure: AST row -> Isolate -> Culture-event -> Patient-episode-landmark
- Specimen type mapping and report lag imputation
- Organism classification (pathogen, contaminant, flora, etc.)
- Resistance phenotype detection (MRSA, VRE, CRE, CRPA, CRAB, ESBL_proxy)
- 7-day deduplication rule
- Incident resistant outcome handling
"""
import pandas as pd
import numpy as np
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
from config import *
from utils import (parse_datetime, hours_between, map_specimen_group,
                    get_report_lag, dedup_identical_cultures, log)


# ── Organism Dictionary ────────────────────────────────────
# Maps raw organism names to standardized names with classification

ORGANISM_MAP = {
    # ── Gram-positive cocci ──
    "金黄色葡萄球菌": ("staphylococcus_aureus", "pathogen", "gpac"),
    "人葡萄球菌": ("staphylococcus_hominis", "probable_pathogen", "gpac"),
    "表皮葡萄球菌": ("staphylococcus_epidermidis", "probable_pathogen", "gpac"),
    "溶血葡萄球菌": ("staphylococcus_haemolyticus", "probable_pathogen", "gpac"),
    "头状葡萄球菌": ("staphylococcus_capitis", "probable_pathogen", "gpac"),
    "腐生葡萄球菌": ("staphylococcus_saprophyticus", "probable_pathogen", "gpac"),
    "路邓葡萄球菌": ("staphylococcus_lugdunensis", "pathogen", "gpac"),
    "沃氏葡萄球菌": ("staphylococcus_warneri", "probable_pathogen", "gpac"),
    "木糖葡萄球菌": ("staphylococcus_xylosus", "contaminant", "gpac"),
    "山羊葡萄球菌": ("staphylococcus_caprae", "contaminant", "gpac"),
    "科氏葡萄球菌": ("staphylococcus_cohnii", "contaminant", "gpac"),
    "模仿葡萄球菌": ("staphylococcus_simulans", "contaminant", "gpac"),
    "其他凝固酶阴性葡萄球菌": ("cons_other", "possible_contaminant", "gpac"),
    "凝固酶阴性葡萄球菌": ("cons_unspecified", "possible_contaminant", "gpac"),
    "肺炎链球菌": ("streptococcus_pneumoniae", "pathogen", "gpac"),
    "化脓链球菌": ("streptococcus_pyogenes", "pathogen", "gpac"),
    "无乳链球菌": ("streptococcus_agalactiae", "pathogen", "gpac"),
    "草绿色链球菌": ("viridans_streptococci", "probable_pathogen", "gpac"),
    "咽峡炎链球菌": ("streptococcus_anginosus", "probable_pathogen", "gpac"),
    "血链球菌": ("streptococcus_sanguinis", "probable_pathogen", "gpac"),
    "粪肠球菌": ("enterococcus_faecalis", "pathogen", "gpac"),
    "屎肠球菌": ("enterococcus_faecium", "pathogen", "gpac"),
    "肠球菌属": ("enterococcus_spp", "probable_pathogen", "gpac"),
    "鹑鸡肠球菌": ("enterococcus_gallinarum", "probable_pathogen", "gpac"),
    # ── Gram-negative rods (Enterobacterales) ──
    "大肠埃希菌": ("escherichia_coli", "pathogen", "gnr"),
    "大肠杆菌": ("escherichia_coli", "pathogen", "gnr"),
    "肺炎克雷伯菌": ("klebsiella_pneumoniae", "pathogen", "gnr"),
    "产酸克雷伯菌": ("klebsiella_oxytoca", "pathogen", "gnr"),
    "阴沟肠杆菌": ("enterobacter_cloacae", "pathogen", "gnr"),
    "产气肠杆菌": ("enterobacter_aerogenes", "pathogen", "gnr"),
    "粘质沙雷菌": ("serratia_marcescens", "pathogen", "gnr"),
    "奇异变形杆菌": ("proteus_mirabilis", "pathogen", "gnr"),
    "普通变形杆菌": ("proteus_vulgaris", "pathogen", "gnr"),
    "弗氏柠檬酸杆菌": ("citrobacter_freundii", "pathogen", "gnr"),
    "异型柠檬酸杆菌": ("citrobacter_diversus", "pathogen", "gnr"),
    "摩根摩根菌": ("morganella_morganii", "pathogen", "gnr"),
    # ── Non-fermenting GNR ──
    "铜绿假单胞菌": ("pseudomonas_aeruginosa", "pathogen", "gnr"),
    "鲍曼不动杆菌": ("acinetobacter_baumannii", "pathogen", "gnr"),
    "醋酸钙不动杆菌": ("acinetobacter_calcoaceticus", "pathogen", "gnr"),
    "溶血不动杆菌": ("acinetobacter_haemolyticus", "probable_pathogen", "gnr"),
    "洛菲不动杆菌": ("acinetobacter_lwoffii", "probable_pathogen", "gnr"),
    "嗜麦芽窄食单胞菌": ("stenotrophomonas_maltophilia", "pathogen", "gnr"),
    "洋葱伯克霍尔德菌": ("burkholderia_cepacia", "pathogen", "gnr"),
    "脑膜败血伊丽莎白菌": ("elizabethkingia_meningoseptica", "pathogen", "gnr"),
    # ── Anaerobes ──
    "脆弱拟杆菌": ("bacteroides_fragilis", "pathogen", "anaerobe"),
    "产气荚膜梭菌": ("clostridium_perfringens", "pathogen", "anaerobe"),
    "艰难梭菌": ("clostridium_difficile", "pathogen", "anaerobe"),
    "消化链球菌": ("peptostreptococcus_spp", "probable_pathogen", "anaerobe"),
    # ── Fungi ──
    "白色念珠菌": ("candida_albicans", "pathogen", "fungus"),
    "光滑念珠菌": ("candida_glabrata", "pathogen", "fungus"),
    "热带念珠菌": ("candida_tropicalis", "pathogen", "fungus"),
    "近平滑念珠菌": ("candida_parapsilosis", "pathogen", "fungus"),
    "克柔念珠菌": ("candida_krusei", "pathogen", "fungus"),
    "曲霉菌": ("aspergillus_spp", "pathogen", "fungus"),
    "烟曲霉": ("aspergillus_fumigatus", "pathogen", "fungus"),
    "黄曲霉": ("aspergillus_flavus", "pathogen", "fungus"),
    "新型隐球菌": ("cryptococcus_neoformans", "pathogen", "fungus"),
    # ── Negative / Flora ──
    "无细菌生长": ("no_growth", "negative", "not_applicable"),
    "正常菌群": ("normal_flora", "flora", "not_applicable"),
    "混合菌群": ("mixed_flora", "flora", "not_applicable"),
    "污染": ("contaminated", "contaminant", "not_applicable"),
    "皮肤菌群": ("skin_flora", "flora", "not_applicable"),
    # ── Other ──
    "流感嗜血杆菌": ("haemophilus_influenzae", "pathogen", "gnr"),
    "副流感嗜血杆菌": ("haemophilus_parainfluenzae", "probable_pathogen", "gnr"),
    "卡他莫拉菌": ("moraxella_catarrhalis", "pathogen", "gnr"),
    "淋病奈瑟菌": ("neisseria_gonorrhoeae", "pathogen", "gnc"),
    "脑膜炎奈瑟菌": ("neisseria_meningitidis", "pathogen", "gnc"),
    "单核细胞增生李斯特菌": ("listeria_monocytogenes", "pathogen", "gpr"),
    "星形诺卡菌": ("nocardia_asteroides", "pathogen", "gpr"),
    "结核分枝杆菌": ("mycobacterium_tuberculosis", "pathogen", "afb"),
    "鸟分枝杆菌": ("mycobacterium_avium", "pathogen", "afb"),
    "龟分枝杆菌": ("mycobacterium_chelonae", "pathogen", "afb"),
}

# CoNS species list (for special handling in Section 5.5.3)
CONS_SPECIES = [
    "staphylococcus_epidermidis", "staphylococcus_haemolyticus",
    "staphylococcus_hominis", "staphylococcus_capitis",
    "staphylococcus_saprophyticus", "staphylococcus_warneri",
    "staphylococcus_xylosus", "staphylococcus_caprae",
    "staphylococcus_cohnii", "staphylococcus_simulans",
    "cons_other", "cons_unspecified",
]

# ── Acinetobacter complex mapping (Section 5.4.4) ──
ACINETOBACTER_COMPLEX = {
    "acinetobacter_baumannii", "acinetobacter_calcoaceticus",
    "acinetobacter_nosocomialis", "acinetobacter_pittii",
    "acinetobacter_calcoaceticus_baumannii_complex",
}

# Enterobacterales genera for ESBL/CRE classification
ENTEROBACTERALES_GENERA = [
    "escherichia", "klebsiella", "enterobacter", "serratia",
    "proteus", "citrobacter", "morganella", "providencia",
    "salmonella", "shigella", "yersinia", "edwardsiella",
]

# Resistance phenotype definitions (Section 5.8)
# Based on organism + antibiotic resistance interpretation
def detect_phenotype(org_name_std, ab_name, interpretation, org_group):
    """Detect resistance phenotype from AST result.
    Returns a list of phenotype strings (can have multiple).
    """
    phenotypes = []
    if interpretation not in ("R", "I", "NS"):
        return phenotypes
    interp = interpretation.upper()
    ab_upper = str(ab_name).upper() if pd.notna(ab_name) else ""

    # MRSA: S. aureus + oxacillin/cefoxitin R
    if org_name_std == "staphylococcus_aureus":
        if any(x in ab_upper for x in ["OXACILLIN", "CEFOXITIN", "苯唑西林", "头孢西丁"]):
            if interp == "R":
                phenotypes.append("MRSA")

    # VRE: Enterococcus + vancomycin R
    if "enterococcus" in str(org_name_std):
        if "VANCOMYCIN" in ab_upper or "万古霉素" in str(ab_name):
            if interp == "R":
                phenotypes.append("VRE")

    # CRPA: P. aeruginosa + carbapenem R
    if org_name_std == "pseudomonas_aeruginosa":
        if any(x in ab_upper for x in ["MEROPENEM", "IMIPENEM", "DORIPENEM",
                                         "美罗培南", "亚胺培南"]):
            if interp == "R":
                phenotypes.append("CRPA")

    # CRAB: Acinetobacter spp + carbapenem R
    if "acinetobacter" in str(org_name_std):
        if any(x in ab_upper for x in ["MEROPENEM", "IMIPENEM", "DORIPENEM",
                                         "美罗培南", "亚胺培南"]):
            if interp == "R":
                phenotypes.append("CRAB_complex")

    # CRE: Enterobacterales + carbapenem R
    if any(e in str(org_name_std) for e in ENTEROBACTERALES_GENERA):
        if any(x in ab_upper for x in ["MEROPENEM", "IMIPENEM", "DORIPENEM",
                                         "美罗培南", "亚胺培南"]):
            if interp == "R":
                phenotypes.append("CRE")

    # ESBL_proxy: Enterobacterales + ceftriaxone/cefotaxime/ceftazidime R + carbapenem S (Section 5.8.2)
    # Detected per-isolate after aggregating all AST results; see _detect_esbl_proxy()

    # cefepime_resistant_gnb: Enterobacterales or Pseudomonas + cefepime R (Section 5.8.3)
    # Detected per-isolate after aggregating all AST results; see _detect_cefepime_resistant_gnb()

    return phenotypes


def detect_esbl_proxy(ab_name_list, interp_list):
    """Detect ESBL_proxy from aggregated isolate AST results (Section 5.8.2).
    Enterobacterales + 3rd-gen cephalosporin R + carbapenem S.
    ab_name_list and interp_list are aligned lists from isolate aggregation.
    """
    if not ab_name_list or not interp_list:
        return False
    ab_list = [str(a).upper() for a in ab_name_list if pd.notna(a)]
    interp_list = [str(i).upper() for i in interp_list if pd.notna(i)]

    # Check for 3rd-gen cephalosporin resistance
    ceph_3g_names = ["CEFTRIAXONE", "CEFOTAXIME", "CEFTAZIDIME",
                     "头孢曲松", "头孢噻肟", "头孢他啶"]
    has_ceph_r = False
    carb_s = True  # Assume susceptible unless proven resistant

    for ab, interp in zip(ab_list, interp_list):
        ab_upper = ab.upper()
        # Check cephalosporin resistance
        if any(c in ab_upper for c in ceph_3g_names):
            if interp in ("R", "NS"):
                has_ceph_r = True
        # Check carbapenem susceptibility
        carb_names = ["MEROPENEM", "IMIPENEM", "DORIPENEM", "美罗培南", "亚胺培南"]
        if any(c in ab_upper for c in carb_names):
            if interp in ("R", "NS"):
                carb_s = False

    return has_ceph_r and carb_s


def detect_cefepime_resistant_gnb(ab_name_list, interp_list):
    """Detect cefepime_resistant_gnb (Section 5.8.3).
    Enterobacterales or Pseudomonas + cefepime R.
    """
    if not ab_name_list or not interp_list:
        return False
    ab_list = [str(a).upper() for a in ab_name_list if pd.notna(a)]
    interp_list = [str(i).upper() for i in interp_list if pd.notna(i)]

    cefepime_names = ["CEFEPIME", "头孢吡肟"]
    for ab, interp in zip(ab_list, interp_list):
        ab_upper = ab.upper()
        if any(c in ab_upper for c in cefepime_names):
            if interp in ("R", "NS"):
                return True
    return False


def map_acinetobacter_complex(org_name_std):
    """Map Acinetobacter species to complex group (Section 5.4.4)."""
    if org_name_std in ACINETOBACTER_COMPLEX:
        return "acinetobacter_baumannii_complex_group"
    return org_name_std


def get_highest_risk_pathogen(phenotype_list):
    """Get highest risk pathogen group (Section 5.1.3).
    CRE/CRAB > VRE > MRSA > ESBL_proxy > cefepime_resistant_gnb
    """
    if not phenotype_list:
        return "none"
    plist = [p for p in phenotype_list if p]
    hierarchy = ["CRE", "CRAB_complex", "CRAB_strict", "CRPA",
                 "VRE", "MRSA", "ESBL_proxy", "cefepime_resistant_gnb"]
    for h in hierarchy:
        if h in plist:
            return h
    return "other_resistant" if plist else "none"


def run_step5(logfile=None, icu_base=None, abx_orders=None):
    log("=" * 60, logfile)
    log("Step 5: Microbiology Processing", logfile)
    log("=" * 60, logfile)

    # 1. Load MICROBIOLOGYEVENTS
    log("Loading MICROBIOLOGYEVENTS.csv ...", logfile)
    micro = pd.read_csv(MICROBIOLOGYEVENTS_CSV, low_memory=False)
    micro.columns = [c.lower() for c in micro.columns]
    log(f"  {len(micro):,} AST-level rows", logfile)

    # 2. Load ICU base if not provided
    if icu_base is None:
        icu_base = pd.read_csv(ICU_BASE_CSV, low_memory=False)
        icu_base["episode_intime"] = parse_datetime(icu_base["episode_intime"])
        icu_base["episode_outtime"] = parse_datetime(icu_base["episode_outtime"])
        for col in ["subject_id", "hadm_id", "episode_id"]:
            icu_base[col] = icu_base[col].astype(str)

    # 3. Time standardization (Section 5.2)
    log("Time standardization...", logfile)
    micro["charttime"] = parse_datetime(micro["charttime"])
    micro = micro[micro["charttime"].notna()].copy()
    log(f"  After removing null charttime: {len(micro):,} rows", logfile)

    # Standardize ID types
    for col in ["subject_id", "hadm_id"]:
        micro[col] = micro[col].astype(str)

    # 4. Specimen type mapping (Section 5.7)
    log("Mapping specimen types...", logfile)
    micro["specimen_group"] = micro["spec_type_desc"].apply(map_specimen_group)
    log(f"  Specimen groups: {micro['specimen_group'].value_counts().to_dict()}", logfile)

    # 5. Organism classification (Section 5.4)
    log("Classifying organisms...", logfile)
    micro["org_name_clean"] = micro["org_name"].str.strip()
    org_map_df = pd.DataFrame([
        {"org_name_raw": k, "org_name_std": v[0], "organism_category": v[1],
         "org_group": v[2]}
        for k, v in ORGANISM_MAP.items()
    ])
    micro = micro.merge(org_map_df, left_on="org_name_clean", right_on="org_name_raw",
                        how="left")
    # Fill unmapped
    micro["org_name_std"] = micro["org_name_std"].fillna(micro["org_name_clean"])
    micro["organism_category"] = micro["organism_category"].fillna("unknown")
    micro["org_group"] = micro["org_group"].fillna("unknown")

    n_mapped = (micro["organism_category"] != "unknown").sum()
    log(f"  Organisms mapped: {n_mapped:,} / {len(micro):,} "
        f"({n_mapped/len(micro)*100:.1f}%)", logfile)

    # 6. Culture time relative (Section 5.2.3)
    micro = micro.merge(
        icu_base[["subject_id", "hadm_id", "episode_id", "episode_intime"]],
        on=["subject_id", "hadm_id"], how="inner"
    )
    micro["culture_hour"] = hours_between(micro["charttime"], micro["episode_intime"])

    # 7. Imputed report time (Section 5.2.2)
    micro["culture_report_lag_hours"] = micro["specimen_group"].apply(get_report_lag)
    micro["imputed_report_time"] = micro["charttime"] + pd.to_timedelta(
        micro["culture_report_lag_hours"], unit="h")

    # 8. Positive culture flag (Section 5.5.1)
    negative_keywords = [
        "无细菌生长", "无菌生长", "无细菌", "无菌",
        "无真菌生长", "无真菌",
        "正常菌群", "皮肤菌群", "混合菌群",
        "未培养出", "未检出", "未生长", "未找到",
        "无致病菌", "无致病菌生长",
        "涂片未找到", "涂片未检出",
        "no growth", "no bacterial growth", "sterile", "contaminat",
        "mixed flora", "skin flora",
    ]
    micro["is_positive_culture"] = 1
    for kw in negative_keywords:
        mask = micro["org_name_clean"].str.contains(kw, case=False, na=False)
        micro.loc[mask, "is_positive_culture"] = 0

    # Also flag negative/flora organisms
    micro.loc[micro["organism_category"].isin(["negative", "flora", "contaminant"]),
              "is_positive_culture"] = 0

    # 9. Clinical relevance (Section 5.5.2)
    micro["is_clinically_relevant"] = (
        (micro["is_positive_culture"] == 1) &
        (micro["organism_category"].isin(["pathogen", "probable_pathogen"]))
    ).astype(int)

    # 10. CoNS special handling (Section 5.5.3)
    micro["is_possible_contaminant"] = micro["org_name_std"].isin(CONS_SPECIES).astype(int)
    micro["cons_crbsi_supportive"] = 0  # Requires CVC + anti-MRSA data, set later

    # 11. Resistance phenotype detection (Section 5.8)
    log("Detecting resistance phenotypes...", logfile)
    phenotypes = []
    for _, row in micro.iterrows():
        pheno_list = detect_phenotype(
            row.get("org_name_std"), row.get("ab_name"),
            row.get("interpretation"), row.get("org_group"))
        phenotypes.append(pheno_list)
    micro["phenotype"] = phenotypes

    # Map Acinetobacter complex (Section 5.4.4)
    micro["org_name_std_complex"] = micro["org_name_std"].apply(map_acinetobacter_complex)

    # 12. Build resistance organism flag (Section 5.9.1)
    key_abx_patterns = [
        "CARBAPENEM", "MEROPENEM", "IMIPENEM", "美罗培南", "亚胺培南",
        "CEFTRIAXONE", "CEFTAZIDIME", "CEFEPIME",
        "头孢曲松", "头孢他啶", "头孢吡肟",
        "VANCOMYCIN", "OXACILLIN", "COLISTIN", "LINEZOLID",
        "万古霉素", "苯唑西林", "粘菌素", "利奈唑胺",
    ]
    def is_key_abx(ab_name):
        ab_upper = str(ab_name).upper() if pd.notna(ab_name) else ""
        return any(p in ab_upper for p in ["CARBAPENEM", "MEROPENEM", "IMIPENEM",
                    "CEFTRIAXONE", "CEFTAZIDIME", "CEFEPIME", "VANCOMYCIN",
                    "OXACILLIN", "COLISTIN", "LINEZOLID"]) or \
               any(p in str(ab_name) for p in ["美罗培南", "亚胺培南", "头孢曲松",
                    "头孢他啶", "头孢吡肟", "万古霉素", "苯唑西林", "粘菌素", "利奈唑胺"])

    micro["key_abx_test"] = micro["ab_name"].apply(is_key_abx).astype(int)
    micro["is_resistant"] = micro["interpretation"].isin(["R", "I", "NS"]).astype(int)
    micro["resistant_to_key_abx"] = ((micro["key_abx_test"] == 1) &
                                      (micro["is_resistant"] == 1)).astype(int)

    # Per-organism resistance flag
    micro["resistance_per_org"] = micro.groupby(
        ["subject_id", "hadm_id", "charttime", "org_name_std"]
    )["resistant_to_key_abx"].transform("max")

    micro["resistant_organism_flag"] = (
        (micro["is_clinically_relevant"] == 1) &
        (micro["resistance_per_org"] == 1)
    ).astype(int)

    # Strict MDRO flag (phenotype is a list from detect_phenotype)
    mdro_phenotypes = ["MRSA", "VRE", "CRE", "CRPA", "CRAB_complex", "CRAB_strict"]
    def _has_mdro(pheno_val):
        if not isinstance(pheno_val, list):
            return False
        return any(p in mdro_phenotypes for p in pheno_val)
    micro["strict_mdro_flag"] = micro["phenotype"].apply(_has_mdro).astype(int)

    # 13. Aggregate to Isolate level (Section 5.6)
    log("Aggregating to isolate level...", logfile)
    # Group by: subject_id + hadm_id + charttime + specimen_group + org_name_std
    isolate_cols = [
        "subject_id", "hadm_id", "charttime", "specimen_group", "org_name_std",
        "org_name_std_complex", "org_name_raw", "organism_category", "org_group",
    ]
    agg_dict = {
        "episode_id": "first",
        "culture_hour": "first",
        "imputed_report_time": "first",
        "is_positive_culture": "max",
        "is_clinically_relevant": "max",
        "is_possible_contaminant": "max",
        "cons_crbsi_supportive": "max",
        "resistant_organism_flag": "max",
        "strict_mdro_flag": "max",
        "phenotype": lambda x: list(set([p for sublist in x.dropna() for p in sublist if p])),
        "ab_name": lambda x: list(x.dropna().unique()),
        "interpretation": lambda x: list(x.dropna().unique()),
    }

    isolates = micro.groupby(isolate_cols, as_index=False, dropna=False).agg(agg_dict)
    isolates.rename(columns={
        "charttime": "culture_time",
        "org_name_raw": "org_name_raw",
    }, inplace=True)

    # Build abx_tested_list, resistant_abx_list, susceptible_abx_list
    isolates["resistant_classes_count"] = isolates["phenotype"].apply(
        lambda x: len([p for p in x if p]) if isinstance(x, list) else 0)

    # ESBL_proxy detection at isolate level (Section 5.8.2)
    isolates["esbl_proxy"] = isolates.apply(
        lambda row: detect_esbl_proxy(row["ab_name"], row["interpretation"])
        if row["is_clinically_relevant"] == 1 and any(
            e in str(row.get("org_name_std", "")) for e in ENTEROBACTERALES_GENERA)
        else False, axis=1
    )
    # Add ESBL_proxy to phenotype list
    for idx in isolates[isolates["esbl_proxy"]].index:
        if isinstance(isolates.at[idx, "phenotype"], list):
            isolates.at[idx, "phenotype"] = isolates.at[idx, "phenotype"] + ["ESBL_proxy"]
        else:
            isolates.at[idx, "phenotype"] = ["ESBL_proxy"]
    isolates["resistant_classes_count"] = isolates["phenotype"].apply(
        lambda x: len([p for p in x if p]) if isinstance(x, list) else 0)

    # cefepime_resistant_gnb detection at isolate level (Section 5.8.3)
    gnb_organisms = ENTEROBACTERALES_GENERA + ["pseudomonas"]
    isolates["cefepime_resistant_gnb"] = isolates.apply(
        lambda row: detect_cefepime_resistant_gnb(row["ab_name"], row["interpretation"])
        if row["is_clinically_relevant"] == 1 and any(
            e in str(row.get("org_name_std", "")) for e in gnb_organisms)
        else False, axis=1
    )
    # Add cefepime_resistant_gnb to phenotype list
    for idx in isolates[isolates["cefepime_resistant_gnb"]].index:
        if isinstance(isolates.at[idx, "phenotype"], list):
            isolates.at[idx, "phenotype"] = isolates.at[idx, "phenotype"] + ["cefepime_resistant_gnb"]
        else:
            isolates.at[idx, "phenotype"] = ["cefepime_resistant_gnb"]
    isolates["resistant_classes_count"] = isolates["phenotype"].apply(
        lambda x: len([p for p in x if p]) if isinstance(x, list) else 0)

    # Highest risk pathogen group (Section 5.1.3)
    isolates["highest_risk_pathogen_group"] = isolates["phenotype"].apply(
        get_highest_risk_pathogen)

    # 14. Join with ICU base for episode-level assignment
    isolates = isolates.merge(
        icu_base[["subject_id", "hadm_id", "episode_id", "episode_intime"]],
        on=["subject_id", "hadm_id", "episode_id"], how="left"
    )
    isolates["culture_hour_from_icu"] = hours_between(
        isolates["culture_time"], isolates["episode_intime"])

    # 15. 7-day deduplication (Section 5.1.2)
    log("Applying 7-day deduplication...", logfile)
    before_dedup = len(isolates)
    isolates = dedup_identical_cultures(isolates)
    log(f"  Before: {before_dedup:,} -> After: {len(isolates):,}", logfile)

    # 16. Select output columns (Section 5.6.3)
    output_cols = [
        "subject_id", "hadm_id", "episode_id",
        "culture_time", "culture_hour_from_icu",
        "imputed_report_time", "specimen_group",
        "org_name_raw", "org_name_std", "org_name_std_complex",
        "organism_category", "is_positive_culture",
        "is_clinically_relevant", "is_possible_contaminant",
        "cons_crbsi_supportive",
        "ab_name", "interpretation", "phenotype",
        "esbl_proxy", "cefepime_resistant_gnb",
        "highest_risk_pathogen_group",
        "resistant_classes_count", "resistant_organism_flag",
        "strict_mdro_flag",
    ]
    # Ensure all columns exist
    for col in output_cols:
        if col not in isolates.columns:
            isolates[col] = np.nan
    output = isolates[output_cols].reset_index(drop=True)

    # 17. Save
    output.to_csv(MICRO_ISOLATES_CLEAN_CSV, index=False, encoding="utf-8")
    log(f"-> Saved {MICRO_ISOLATES_CLEAN_CSV}: {len(output):,} isolates", logfile)

    # 18. Statistics
    n_pos = output["is_positive_culture"].sum()
    n_clin = output["is_clinically_relevant"].sum()
    n_res = output["resistant_organism_flag"].sum()
    n_mdro = output["strict_mdro_flag"].sum()
    log(f"  Positive cultures: {n_pos:,}", logfile)
    log(f"  Clinically relevant: {n_clin:,}", logfile)
    log(f"  Resistant organisms: {n_res:,}", logfile)
    log(f"  Strict MDRO: {n_mdro:,}", logfile)
    log(f"  Phenotypes: {output['phenotype'].explode().dropna().value_counts().to_dict()}", logfile)

    # Save organism dictionary (Section 5.4)
    org_dict = pd.DataFrame([
        {"org_name_raw": k, "org_name_std": v[0], "organism_category": v[1],
         "org_group": v[2]}
        for k, v in ORGANISM_MAP.items()
    ])
    org_dict.to_csv(ORGANISM_DICT_CSV, index=False, encoding="utf-8")
    log(f"-> Saved organism dictionary: {len(org_dict)} entries", logfile)
    log("Step 5 complete!", logfile)

    return output


if __name__ == "__main__":
    logfile = os.path.join(OUTPUT_DIR, "processing_log.txt")
    run_step5(logfile)
