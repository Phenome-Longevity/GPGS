#!/usr/bin/env python3
"""
Build a complete, self-contained PAM model bundle.

Run ONCE to extract everything needed for scoring new patients:
  - Module atlas (194 modules, eigenloadings, PGS member IDs)
  - Quality scores per PGS
  - Per-module reference panel stats (regression coefficients, median, MAD)
  - Trained RF classifiers (11 diseases)
  - PGS metadata from LLM classification (trait, family, organs, phenotype type, etc.)
  - Domain lookup tables
  - Feature importances per disease
  - Monitoring recommendations

Output: pam_model.pkl (portable, self-contained)

Usage:
    python build_model.py \
        --final-analysis /path/to/polygen_publish/final_analysis \
        --ref-scores /path/to/ancestry_cache/ref_scores/<hash> \
        --output ./pam_model.pkl
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from collections import Counter

import numpy as np


# ── Domain mapping (LLM organ_system → clinical domain) ────────────────
ORGAN_TO_DOMAIN = {
    'cardiology': 'Cardiology',
    'neurology': 'Neuro_Psychiatry',
    'psychiatry': 'Neuro_Psychiatry',
    'hematology': 'Hematology',
    'immunology': 'Immunology',
    'rheumatology': 'Immunology',
    'dermatology': 'Immunology',
    'endocrinology': 'Endocrinology',
    'oncology': 'Oncology',
    'breast': 'Oncology',
    'gynecology': 'Oncology',
    'nephrology': 'Nephrology_Urology',
    'urology': 'Nephrology_Urology',
    'gastroenterology': 'Gastro_Hepatology',
    'hepatology': 'Gastro_Hepatology',
    'pulmonology': 'Pulmonology',
    'musculoskeletal': 'Musculoskeletal',
    'orthopedics': 'Musculoskeletal',
    'ophthalmology': 'Ophthalmology',
    'dentistry': 'Other',
    'andrology': 'Nephrology_Urology',
}

CATEGORY_TO_DOMAIN = {
    'Cardiovascular disease': 'Cardiology',
    'Cardiovascular measurement': 'Cardiology',
    'Cancer': 'Oncology',
    'Immune system disorder': 'Immunology',
    'Inflammatory measurement': 'Immunology',
    'Neurological disorder': 'Neuro_Psychiatry',
    'Metabolic disorder': 'Endocrinology',
    'Lipid or lipoprotein measurement': 'Endocrinology',
    'Hematological measurement': 'Hematology',
    'Body measurement': 'Endocrinology',
    'Digestive system disorder': 'Gastro_Hepatology',
    'Other measurement': None,
    'Other disease': None,
    'Other trait': None,
    'Biological process': None,
}

FAMILY_TO_DOMAIN = {
    'Coronary_Artery_Disease': 'Cardiology', 'Heart_Failure': 'Cardiology',
    'Atrial_Fibrillation': 'Cardiology', 'Stroke': 'Cardiology',
    'Hypertension': 'Cardiology', 'Peripheral_Artery_Disease': 'Cardiology',
    'Venous_Thromboembolism': 'Cardiology', 'Aortic_Aneurysm': 'Cardiology',
    'Breast_Cancer': 'Oncology', 'Prostate_Cancer': 'Oncology',
    'Colorectal_Cancer': 'Oncology', 'Lung_Cancer': 'Oncology',
    'Ovarian_Cancer': 'Oncology', 'Melanoma': 'Oncology',
    'Bladder_Cancer': 'Oncology', 'Kidney_Cancer': 'Oncology',
    'Pancreatic_Cancer': 'Oncology', 'Endometrial_Cancer': 'Oncology',
    'Thyroid_Cancer': 'Oncology', 'Glioma': 'Oncology',
    'Lymphoma': 'Oncology', 'Testicular_Cancer': 'Oncology',
    'Type_2_Diabetes': 'Endocrinology', 'Type_1_Diabetes': 'Endocrinology',
    'Thyroid_Disease': 'Endocrinology', 'Obesity': 'Endocrinology',
    'Celiac_Disease': 'Immunology', 'Crohns_Disease': 'Immunology',
    'Ulcerative_Colitis': 'Immunology', 'IBD': 'Immunology',
    'Asthma': 'Immunology', 'Atopic_Dermatitis': 'Immunology',
    'Psoriasis': 'Immunology', 'Rheumatoid_Arthritis': 'Immunology',
    'Lupus': 'Immunology', 'Ankylosing_Spondylitis': 'Immunology',
    'Multiple_Sclerosis': 'Immunology', 'Allergy': 'Immunology',
    'Parkinsons_Disease': 'Neuro_Psychiatry', 'Alzheimers_Disease': 'Neuro_Psychiatry',
    'Schizophrenia': 'Neuro_Psychiatry', 'Bipolar_Disorder': 'Neuro_Psychiatry',
    'Depression': 'Neuro_Psychiatry', 'Epilepsy': 'Neuro_Psychiatry',
    'ADHD': 'Neuro_Psychiatry', 'Autism': 'Neuro_Psychiatry',
    'Anxiety': 'Neuro_Psychiatry', 'ALS': 'Neuro_Psychiatry',
    'CKD': 'Nephrology_Urology', 'Kidney_Stones': 'Nephrology_Urology',
    'Gout': 'Musculoskeletal', 'Osteoporosis': 'Musculoskeletal',
    'COPD': 'Pulmonology', 'Sleep_Apnea': 'Pulmonology',
    'NAFLD': 'Gastro_Hepatology', 'Cirrhosis': 'Gastro_Hepatology',
    'Glaucoma': 'Ophthalmology', 'AMD': 'Ophthalmology',
}

# Phenotype landscape subsection mapping
PHENOTYPE_SUBSECTIONS = {
    'biomarker_lab': {
        'Lipid or lipoprotein measurement': 'Lipids',
        'Hematological measurement': 'Blood_Cells',
        'Cardiovascular measurement': 'Cardiac_Markers',
        'Inflammatory measurement': 'Inflammatory',
    },
    'imaging_structure': {
        'neurology': 'Brain_Structure',
        'cardiology': 'Cardiac_Imaging',
        'ophthalmology': 'Eye_Imaging',
        'hepatology': 'Liver_Imaging',
    },
    'quantitative_physiology': {
        'hematology': 'Blood_Function',
        'cardiology': 'Cardiac_Function',
        'orthopedics': 'Bone_Density',
        'pulmonology': 'Lung_Function',
        'ophthalmology': 'Vision',
        'musculoskeletal': 'Bone_Density',
    },
    'behavior_lifestyle': {
        'smok': 'Smoking', 'tobacco': 'Smoking', 'cigarette': 'Smoking',
        'alcohol': 'Alcohol', 'drink': 'Alcohol',
        'sleep': 'Sleep', 'chronotype': 'Sleep', 'insomnia': 'Sleep', 'snor': 'Sleep',
        'education': 'Education_Cognition', 'intelligence': 'Education_Cognition',
        'neurotic': 'Personality', 'depress': 'Personality', 'worry': 'Personality',
        'happiness': 'Personality', 'loneli': 'Personality', 'anxiet': 'Personality',
        'cannabis': 'Substance', 'caffein': 'Substance', 'coffee': 'Substance', 'tea ': 'Substance',
        'menarche': 'Reproductive', 'menopaus': 'Reproductive', 'menstrual': 'Reproductive',
        'children': 'Reproductive', 'sexual': 'Reproductive', 'birth': 'Reproductive',
        'hair': 'Appearance', 'bald': 'Appearance', 'tann': 'Appearance', 'sunburn': 'Appearance',
        'physical': 'Activity', 'walk': 'Activity', 'exercise': 'Activity',
        'diet': 'Diet', 'fruit': 'Diet', 'vegetable': 'Diet', 'cereal': 'Diet',
        'meat': 'Diet', 'fish': 'Diet', 'sugar': 'Diet', 'milk': 'Diet',
    },
    'anthropometric': {
        'bmi': 'BMI_Obesity', 'obes': 'BMI_Obesity', 'body mass': 'BMI_Obesity',
        'height': 'Height', 'sitting height': 'Height',
        'waist': 'Body_Composition', 'hip': 'Body_Composition',
        'fat': 'Body_Composition', 'lean': 'Body_Composition',
        'water': 'Body_Composition', 'ankle': 'Body_Composition',
    },
}

DOMAIN_COLORS = {
    'Cardiology': '#e53935',
    'Neuro_Psychiatry': '#8e24aa',
    'Hematology': '#ec407a',
    'Immunology': '#1e88e5',
    'Endocrinology': '#ff7043',
    'Oncology': '#ffa726',
    'Nephrology_Urology': '#66bb6a',
    'Gastro_Hepatology': '#26a69a',
    'Pulmonology': '#42a5f5',
    'Musculoskeletal': '#ffb74d',
    'Ophthalmology': '#78909c',
    'Other': '#9e9e9e',
}

DOMAIN_ORDER = [
    'Cardiology', 'Neuro_Psychiatry', 'Hematology', 'Immunology',
    'Endocrinology', 'Oncology', 'Nephrology_Urology', 'Gastro_Hepatology',
    'Pulmonology', 'Musculoskeletal', 'Ophthalmology',
]

MONITORS = {
    'Cardiology': ['LDL cholesterol', 'Triglycerides', 'Blood pressure', 'Resting ECG'],
    'Endocrinology': ['HbA1c', 'Fasting glucose', 'TSH', 'BMI'],
    'Immunology': ['CRP', 'tTG-IgA (celiac screen)', 'IgE (allergy)', 'ESR'],
    'Neuro_Psychiatry': ['Cognitive screening if symptomatic', 'PHQ-9 (depression)'],
    'Oncology': ['Age-appropriate cancer screening', 'PSA (males >50)'],
    'Nephrology_Urology': ['eGFR', 'Creatinine', 'Urine albumin'],
    'Gastro_Hepatology': ['ALT', 'AST', 'Bilirubin', 'Stool calprotectin'],
    'Pulmonology': ['Spirometry (FEV1/FVC)', 'Peak flow'],
    'Hematology': ['CBC with differential', 'Ferritin'],
    'Musculoskeletal': ['DEXA bone density', 'Uric acid', 'Vitamin D'],
    'Ophthalmology': ['Intraocular pressure', 'Fundoscopy'],
}

NICE_DISEASE = {
    'Coronary_Artery': 'Coronary Artery Disease',
    'Type2_Diabetes': 'Type 2 Diabetes',
    'Celiac': 'Celiac Disease',
    'Parkinsons': "Parkinson's Disease",
    'Breast_Cancer': 'Breast Cancer',
    'Asthma': 'Asthma',
    'Atopic_Dermatitis': 'Atopic Dermatitis',
    'Bipolar': 'Bipolar Disorder',
    'Colorectal_Cancer': 'Colorectal Cancer',
    'Longevity': 'Longevity',
    'Ankylosing_Spondylitis': 'Ankylosing Spondylitis',
}


def classify_domain(row):
    """Classify a PGS into a clinical domain using LLM fields. Priority: family > organ > category."""
    fam = row.get('family_primary')
    if fam and fam in FAMILY_TO_DOMAIN:
        return FAMILY_TO_DOMAIN[fam]
    organs = json.loads(row['organ_system_tags_json']) if row.get('organ_system_tags_json') else []
    for o in organs:
        if o in ORGAN_TO_DOMAIN and ORGAN_TO_DOMAIN[o] != 'Other':
            return ORGAN_TO_DOMAIN[o]
    cats = json.loads(row['categories_json']) if row.get('categories_json') else []
    for c in cats:
        if c in CATEGORY_TO_DOMAIN and CATEGORY_TO_DOMAIN[c]:
            return CATEGORY_TO_DOMAIN[c]
    pheno = row.get('phenotype_type', '')
    if pheno == 'imaging_structure':
        return 'Neuro_Psychiatry'
    if pheno == 'anthropometric':
        return 'Endocrinology'
    return None


def classify_phenotype_subsection(row):
    """Classify a PGS into a phenotype landscape subsection."""
    ptype = row.get('phenotype_type', '')
    trait = (row.get('short_report_label') or '').lower()

    if ptype == 'biomarker_lab':
        cats = json.loads(row['categories_json']) if row.get('categories_json') else []
        for c in cats:
            if c in PHENOTYPE_SUBSECTIONS['biomarker_lab']:
                return PHENOTYPE_SUBSECTIONS['biomarker_lab'][c]
        return 'Other_Lab'

    if ptype == 'imaging_structure':
        organs = json.loads(row['organ_system_tags_json']) if row.get('organ_system_tags_json') else []
        for o in organs:
            if o in PHENOTYPE_SUBSECTIONS['imaging_structure']:
                return PHENOTYPE_SUBSECTIONS['imaging_structure'][o]
        return 'Brain_Structure'

    if ptype == 'quantitative_physiology':
        organs = json.loads(row['organ_system_tags_json']) if row.get('organ_system_tags_json') else []
        for o in organs:
            if o in PHENOTYPE_SUBSECTIONS['quantitative_physiology']:
                return PHENOTYPE_SUBSECTIONS['quantitative_physiology'][o]
        return 'Other_Physiology'

    if ptype == 'behavior_lifestyle':
        for kw, subsec in PHENOTYPE_SUBSECTIONS['behavior_lifestyle'].items():
            if kw in trait:
                return subsec
        return 'Other_Behavioral'

    if ptype == 'anthropometric':
        for kw, subsec in PHENOTYPE_SUBSECTIONS['anthropometric'].items():
            if kw in trait:
                return subsec
        return 'Other_Body'

    return None


def main():
    parser = argparse.ArgumentParser(description="Build PAM model bundle")
    parser.add_argument("--final-analysis", required=True, help="Path to final_analysis directory")
    parser.add_argument("--ref-scores", required=True, help="Path to ref_scores/<hash> directory")
    parser.add_argument("--output", default="pam_model.pkl", help="Output model file")
    args = parser.parse_args()

    fa = Path(args.final_analysis)
    ref_dir = Path(args.ref_scores)
    out_path = Path(args.output)

    # ── 1. Load atlas modules ────────────────────────────────────────
    print("[1/8] Loading atlas modules...")
    with open(fa / "atlas_modules.json") as f:
        atlas = json.load(f)
    print(f"  {len(atlas)} modules")

    # ── 2. Load trained models ───────────────────────────────────────
    print("[2/8] Loading trained models...")
    with open(fa / "models.pkl", "rb") as f:
        bundle = pickle.load(f)
    models = bundle["models"]
    diseases = bundle["diseases"]
    age_mean = bundle["age_mean"]
    age_std = bundle["age_std"]
    good_modules = bundle["good_modules"]
    print(f"  {len(diseases)} classifiers: {diseases}")

    # ── 3. Load quality scores ───────────────────────────────────────
    print("[3/8] Loading quality scores...")
    quality = {}
    quality_labels = {}
    with open(fa / "quality_scores.tsv") as f:
        header = f.readline().strip().split("\t")
        total_col = header.index("total")
        label_col = header.index("label")
        for line in f:
            parts = line.strip().split("\t")
            pgs_id = parts[0]
            try:
                score = float(parts[total_col])
            except (ValueError, IndexError):
                score = 0.0
            quality[pgs_id] = score
            quality_labels[pgs_id] = parts[label_col] if label_col < len(parts) else "unknown"
    print(f"  {len(quality)} PGS scored")

    # ── 4. Load LLM parquet metadata ─────────────────────────────────
    print("[4/8] Loading LLM classification metadata...")
    import polars as pl
    df = pl.read_parquet(fa / "assets" / "live5232_payloads_2026_03_11.parquet")
    pgs_metadata = {}
    for row in df.iter_rows(named=True):
        pid = row['pgs_id']
        domain = classify_domain(row)
        ptype = row.get('phenotype_type', '')
        subsection = classify_phenotype_subsection(row)
        flags = json.loads(row['quality_flags_json']) if row.get('quality_flags_json') else []
        pgs_metadata[pid] = {
            'trait': row.get('short_report_label', ''),
            'family': row.get('family_primary'),
            'organ_systems': json.loads(row['organ_system_tags_json']) if row.get('organ_system_tags_json') else [],
            'categories': json.loads(row['categories_json']) if row.get('categories_json') else [],
            'phenotype_type': ptype,
            'variants': row.get('variants_number'),
            'method': row.get('method_name', ''),
            'plain_language': row.get('plain_language_summary', ''),
            'quality_score': quality.get(pid, 0),
            'quality_label': quality_labels.get(pid, 'unknown'),
            'domain': domain,
            'phenotype_subsection': subsection,
            'reportability_score': row.get('reportability_score', 0) or 0,
            'catalog_evidence_score': row.get('catalog_evidence_score', 0) or 0,
            'proxy_trait_flag': bool(row.get('proxy_trait_flag', False)),
            'technical_trait_flag': bool(row.get('technical_trait_flag', False)),
            'quality_flags': flags,
        }
    print(f"  {len(pgs_metadata)} PGS with metadata")

    # Domain distribution
    domain_counts = Counter(m['domain'] for m in pgs_metadata.values() if m['domain'])
    for d in DOMAIN_ORDER:
        print(f"    {d}: {domain_counts.get(d, 0)}")
    n_unclassified = sum(1 for m in pgs_metadata.values() if not m['domain'])
    print(f"    Unclassified: {n_unclassified}")

    # ── 5. Load feature importances ──────────────────────────────────
    print("[5/8] Loading feature importances...")
    with open(fa / "feature_importances.json") as f:
        feat_imp = json.load(f)
    feature_names = feat_imp['feature_names']
    disease_importances = {}
    for disease in diseases:
        if disease in feat_imp:
            disease_importances[disease] = feat_imp[disease].get('top_20', [])
    print(f"  {len(disease_importances)} diseases with importances")

    # ── 6. Load reference PCs ────────────────────────────────────────
    print("[6/8] Loading reference PCs...")
    ref_pcs = np.load(fa / "assets" / "ref_pcs.npy")
    print(f"  Shape: {ref_pcs.shape}")

    # ── 7. Compute per-module reference stats ────────────────────────
    print("[7/8] Loading reference scores and computing module stats...")
    ref_pgs_files = sorted(ref_dir.glob("*.npy"))
    ref_pgs_ids = [f.stem for f in ref_pgs_files]
    pgs_to_idx = {pid: i for i, pid in enumerate(ref_pgs_ids)}
    n_ref = ref_pcs.shape[0]
    n_pgs = len(ref_pgs_ids)
    print(f"  {n_pgs} PGS in reference panel")

    ref_scores = np.zeros((n_ref, n_pgs), dtype=np.float32)
    for i, f in enumerate(ref_pgs_files):
        ref_scores[:, i] = np.load(f).astype(np.float32)
        if (i + 1) % 1000 == 0:
            print(f"    loaded {i+1}/{n_pgs}")

    X_design = np.column_stack([np.ones(n_ref), ref_pcs[:, :10]]).astype(np.float64)
    ref_resid = np.zeros_like(ref_scores, dtype=np.float32)
    for j in range(n_pgs):
        y = ref_scores[:, j].astype(np.float64)
        beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
        ref_resid[:, j] = (y - X_design @ beta).astype(np.float32)

    module_keys = sorted(atlas.keys(), key=lambda k: int(k))
    module_ref_stats = {}

    for mod_id in module_keys:
        mod = atlas[mod_id]
        member_pgs = mod["pgs_ids"]
        loadings = np.array(mod["loadings"], dtype=np.float32)

        col_idx = [pgs_to_idx[p] for p in member_pgs if p in pgs_to_idx]
        present = [p for p in member_pgs if p in pgs_to_idx]
        load_idx = [i for i, p in enumerate(member_pgs) if p in pgs_to_idx]
        if load_idx:
            loadings = loadings[load_idx]
        member_pgs = present

        q = np.array([quality.get(p, 0.0) for p in member_pgs], dtype=np.float32)
        w = loadings * q

        R = ref_resid[:, col_idx]
        mask = np.isfinite(R)
        num = np.nansum(R * w[None, :], axis=1)
        denom = np.sqrt(np.nansum((mask * w[None, :]) ** 2, axis=1))
        denom[denom == 0] = 1.0
        raw_scores = num / denom

        beta_mod, _, _, _ = np.linalg.lstsq(X_design, raw_scores.astype(np.float64), rcond=None)
        pred = X_design @ beta_mod
        resid = raw_scores - pred.astype(np.float32)

        vals = resid[np.isfinite(resid)]
        center = float(np.median(vals))
        mad = float(np.median(np.abs(vals - center)))
        scale = 1.4826 * mad
        if scale < 1e-8:
            scale = float(np.std(vals, ddof=1)) if len(vals) > 1 else 1.0

        # Determine module's primary trait/domain
        traits = [pgs_metadata.get(p, {}).get('trait', '') for p in member_pgs]
        domains = [pgs_metadata.get(p, {}).get('domain') for p in member_pgs]
        domain_counts_mod = Counter(d for d in domains if d)
        primary_domain = domain_counts_mod.most_common(1)[0][0] if domain_counts_mod else None
        trait_counts = Counter(traits)
        primary_trait = trait_counts.most_common(1)[0][0] if trait_counts else ''

        module_ref_stats[mod_id] = {
            "pgs_ids": member_pgs,
            "loadings": loadings.tolist(),
            "quality_weights": w.tolist(),
            "regression_intercept": float(beta_mod[0]),
            "regression_coefs": beta_mod[1:].tolist(),
            "center": center,
            "scale": scale,
            "primary_domain": primary_domain,
            "primary_trait": primary_trait,
            "size": len(member_pgs),
            "mean_jaccard": mod.get("mean_jaccard", 0),
        }

    print(f"  {len(module_ref_stats)} module stats computed")

    # ── 8. Save complete bundle ──────────────────────────────────────
    print("[8/8] Saving model bundle...")
    pam_model = {
        "module_ref_stats": module_ref_stats,
        "module_keys": module_keys,
        "good_modules": good_modules,
        "pgs_to_idx": pgs_to_idx,
        "ref_pgs_ids": ref_pgs_ids,
        "quality": quality,
        "quality_labels": quality_labels,
        "pgs_metadata": pgs_metadata,
        "models": models,
        "diseases": diseases,
        "disease_importances": disease_importances,
        "feature_names": feature_names,
        "age_mean": float(age_mean),
        "age_std": float(age_std),
        "n_ref_samples": n_ref,
        "n_modules": len(module_keys),
        "n_modules_clean": len(good_modules),
        "domain_order": DOMAIN_ORDER,
        "domain_colors": DOMAIN_COLORS,
        "monitors": MONITORS,
        "nice_disease": NICE_DISEASE,
    }

    with open(out_path, "wb") as f:
        pickle.dump(pam_model, f)

    sz = out_path.stat().st_size / 1024 / 1024
    print(f"\nSaved: {out_path} ({sz:.1f} MB)")
    print(f"  {len(module_keys)} modules, {len(diseases)} classifiers")
    print(f"  {len(pgs_metadata)} PGS with full metadata")
    print(f"  Ready for analyze.py")


if __name__ == "__main__":
    main()
