#!/usr/bin/env python3
"""
Score a patient through the frozen PolyGen Atlas Model.

Input job folder is READ-ONLY. All output goes to --out.

Usage:
    python analyze.py --model pam_model.pkl --job /path/to/job --sex M --age 50 --out ./output
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


# ══════════════════════════════════════════════════════════════════════════
# TIERING: Full composite + penalties + convergence + IQR-adaptive thresholds
# ══════════════════════════════════════════════════════════════════════════

def compute_base_importance(z, meta):
    """Full composite importance using all metadata fields."""
    score = abs(z)
    qs = meta.get("quality_score", 0) / 100.0
    ev = meta.get("catalog_evidence_score", 0.5)
    rp = meta.get("reportability_score", 0.5)
    flags = meta.get("quality_flags", [])
    variants = meta.get("variants") or 0

    # Composite multipliers
    score *= max(qs, 0.1)          # quality: 0.1 to 1.0
    score *= (0.3 + ev)            # evidence: 0.3 to 1.3
    score *= (0.5 + rp * 0.7)     # reportability: 0.5 to 1.2

    # Penalties
    if meta.get("proxy_trait_flag"):
        score *= 0.5
    if meta.get("technical_trait_flag"):
        score *= 0.3
    if "limited_evidence" in flags:
        score *= 0.6
    if "no_performance_metrics" in flags:
        score *= 0.5
    if variants and variants < 500:
        score *= 0.6

    return score


def compute_convergence(z_scores, pgs_meta):
    """Count agreeing peers per PGS (same family or normalized trait, same direction)."""
    family_groups = defaultdict(list)
    trait_groups = defaultdict(list)

    for pid, z in z_scores.items():
        meta = pgs_meta.get(pid, {})
        direction = "up" if z > 0 else "down"
        fam = meta.get("family")
        if fam:
            family_groups[(fam, direction)].append(pid)
        trait = (meta.get("trait") or "").lower().strip()
        for suffix in [" susceptibility", " level", " variation", " tendency"]:
            trait = trait.replace(suffix, "")
        if trait:
            trait_groups[(trait.strip(), direction)].append(pid)

    convergence = {}
    for pid, z in z_scores.items():
        meta = pgs_meta.get(pid, {})
        direction = "up" if z > 0 else "down"
        fam = meta.get("family")
        trait = (meta.get("trait") or "").lower().strip()
        for suffix in [" susceptibility", " level", " variation", " tendency"]:
            trait = trait.replace(suffix, "")
        n_fam = len(family_groups.get((fam, direction), [])) - 1 if fam else 0
        n_trait = len(trait_groups.get((trait.strip(), direction), [])) - 1 if trait else 0
        convergence[pid] = max(n_fam, n_trait)

    return convergence


def compute_all_tiers(z_scores, pgs_meta):
    """Compute importance + tier for all PGS using full recommended method."""
    convergence = compute_convergence(z_scores, pgs_meta)

    # Compute boosted importance for all PGS
    scores = {}
    for pid, z in z_scores.items():
        meta = pgs_meta.get(pid, {})
        imp = compute_base_importance(z, meta)
        n_agree = convergence.get(pid, 0)
        boost = 1.0 + 0.15 * min(n_agree, 8)
        scores[pid] = imp * boost

    # IQR-adaptive thresholds
    all_imp = np.array(list(scores.values()))
    if len(all_imp) == 0:
        return {}, 0, 0
    q25, q50, q75 = np.percentile(all_imp, [25, 50, 75])
    iqr = q75 - q25
    t1_thresh = q75 + 4.0 * iqr
    t2_thresh = q75 + 2.0 * iqr

    results = {}
    for pid, imp in scores.items():
        tier = "T1" if imp >= t1_thresh else ("T2" if imp >= t2_thresh else "T3")
        results[pid] = {
            "importance": round(imp, 2),
            "tier": tier,
            "convergence": convergence.get(pid, 0),
        }

    return results, t1_thresh, t2_thresh


def make_key(pgs_id, tier, direction):
    return f"{pgs_id}.{tier}.{direction}"


def build_pgs_entry(pgs_id, z, meta, tier_info):
    """Build a tiered PGS entry. Data included depends on tier."""
    direction = "elevated" if z > 0 else "reduced"
    tier = tier_info["tier"]
    importance = tier_info["importance"]
    convergence = tier_info.get("convergence", 0)
    key = make_key(pgs_id, tier, direction)

    if tier == "T1":
        return {
            "pgs_id": pgs_id,
            "key": key,
            "tier": tier,
            "z_score": round(z, 4),
            "direction": direction,
            "importance": importance,
            "convergence": convergence,
            "trait": meta.get("trait", ""),
            "plain_language": meta.get("plain_language", ""),
            "catalog_url": f"https://www.pgscatalog.org/score/{pgs_id}/",
            "family": meta.get("family"),
            "quality_score": meta.get("quality_score"),
            "quality_label": meta.get("quality_label"),
            "variants": meta.get("variants"),
            "method": meta.get("method"),
            "organ_systems": meta.get("organ_systems"),
            "categories": meta.get("categories"),
            "phenotype_type": meta.get("phenotype_type"),
            "domain": meta.get("domain"),
            "reportability_score": meta.get("reportability_score"),
            "catalog_evidence_score": meta.get("catalog_evidence_score"),
        }
    elif tier == "T2":
        return {
            "pgs_id": pgs_id,
            "key": key,
            "tier": tier,
            "z_score": round(z, 4),
            "direction": direction,
            "importance": importance,
            "convergence": convergence,
            "trait": meta.get("trait", ""),
            "quality_score": meta.get("quality_score"),
            "quality_label": meta.get("quality_label"),
            "variants": meta.get("variants"),
            "method": meta.get("method"),
        }
    else:
        return {
            "pgs_id": pgs_id,
            "key": key,
            "tier": tier,
            "z_score": round(z, 4),
            "direction": direction,
            "importance": importance,
        }


# ══════════════════════════════════════════════════════════════════════════
# Patient loading + module projection (unchanged)
# ══════════════════════════════════════════════════════════════════════════

def load_patient_job(job_dir):
    job_dir = Path(job_dir)
    results_dir = job_dir / "results"
    if not results_dir.exists():
        results_dir = job_dir

    scores_file = results_dir / "scores_adjusted.tsv"
    ancestry_file = results_dir / "ancestry.tsv"

    if not scores_file.exists():
        print(f"ERROR: {scores_file} not found")
        sys.exit(1)

    z_scores = {}
    with open(scores_file) as f:
        header = f.readline().strip().split("\t")
        z_col = None
        for pref in ["Z_norm1", "Z_norm2", "Z_MostSimilarPop", "Z"]:
            if pref in header:
                z_col = header.index(pref)
                break
        if z_col is None:
            z_col = len(header) - 1
        pgs_col = None
        for pref in ["pgs_id", "PGS", "pgs"]:
            if pref in header:
                pgs_col = header.index(pref)
                break
        if pgs_col is None:
            pgs_col = 0
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) <= max(pgs_col, z_col):
                continue
            try:
                z_scores[parts[pgs_col]] = float(parts[z_col])
            except ValueError:
                pass

    ancestry = {}
    if ancestry_file.exists():
        with open(ancestry_file) as f:
            header = f.readline().strip().split("\t")
            vals = f.readline().strip().split("\t")
            for h, v in zip(header, vals):
                try:
                    ancestry[h] = float(v)
                except ValueError:
                    ancestry[h] = v

    return {"z_scores": z_scores, "ancestry": ancestry}


def project_patient(z_scores, ancestry, model):
    module_keys = model["good_modules"]
    ref_stats = model["module_ref_stats"]

    pcs = np.zeros(10, dtype=np.float64)
    for i in range(10):
        key = f"PC{i+1}"
        if key in ancestry:
            pcs[i] = ancestry[key]
    design = np.concatenate([[1.0], pcs])

    n_mod = len(module_keys)
    z_out = np.full(n_mod, np.nan, dtype=np.float64)
    rel_out = np.full(n_mod, np.nan, dtype=np.float64)

    for mi, mod_id in enumerate(module_keys):
        stats = ref_stats[str(mod_id)]
        member_pgs = stats["pgs_ids"]
        weights = np.array(stats["quality_weights"], dtype=np.float64)

        vals = np.array([z_scores.get(p, float("nan")) for p in member_pgs], dtype=np.float64)
        mask = np.isfinite(vals)
        if mask.sum() == 0:
            continue

        num = np.nansum(vals * weights)
        denom = np.sqrt(np.nansum((mask * weights) ** 2))
        if denom == 0:
            continue
        raw_score = num / denom

        total_abs_w = np.abs(weights).sum()
        rel = np.nansum(mask * np.abs(weights)) / total_abs_w if total_abs_w > 0 else 0.0

        intercept = stats["regression_intercept"]
        coefs = np.array(stats["regression_coefs"], dtype=np.float64)
        pred = intercept + pcs @ coefs
        resid = raw_score - pred

        center = stats["center"]
        scale = stats["scale"]
        z_out[mi] = (resid - center) / scale if scale > 1e-8 else 0.0
        rel_out[mi] = rel

    return z_out, rel_out


def strength_labels(ranking):
    """Positional cap + absolute thresholds.
    #1 can be anything. #2-4 max STRONG. #5-11 max MODERATE.
    Thresholds: STRONGEST>=0.65, STRONG>=0.50, MODERATE>=0.30."""
    if not ranking:
        return ranking

    for d in ranking:
        p = d["probability"]
        rank = d["rank"]

        # Absolute threshold
        if p >= 0.65:
            s = "STRONGEST"
        elif p >= 0.50:
            s = "STRONG"
        elif p >= 0.30:
            s = "MODERATE"
        else:
            s = "WEAK"

        # Positional cap
        if rank == 1:
            pass  # no cap
        elif rank <= 4:
            if s == "STRONGEST":
                s = "STRONG"
        else:
            if s in ("STRONGEST", "STRONG"):
                s = "MODERATE"

        d["strength"] = s

    return ranking


def domain_risk_level(mean_z, n_elevated, n_total):
    if n_total == 0:
        return "INSUFFICIENT_DATA"
    frac = n_elevated / n_total
    # Require both elevated fraction AND meaningful mean_z
    # Prevents small domains with near-zero mean from being flagged
    if mean_z >= 1.5 or (frac >= 0.40 and abs(mean_z) >= 0.3):
        return "ELEVATED"
    if mean_z >= 0.8 or (frac >= 0.33 and abs(mean_z) >= 0.3):
        return "MILDLY_ELEVATED"
    return "TYPICAL"


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="PolyGen Atlas Model Analysis")
    parser.add_argument("--model", required=True, help="Path to pam_model.pkl")
    parser.add_argument("--job", required=True, help="Path to PolyGen job output folder (read-only)")
    parser.add_argument("--sex", required=True, choices=["M", "F"], help="Patient sex")
    parser.add_argument("--age", required=True, type=float, help="Patient age")
    parser.add_argument("--out", default="./output", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────
    print(f"Loading model: {args.model}")
    with open(args.model, "rb") as f:
        model = pickle.load(f)

    diseases = model["diseases"]
    module_keys = model["good_modules"]
    pgs_meta = model["pgs_metadata"]
    domain_order = model["domain_order"]
    domain_colors = model["domain_colors"]
    monitors = model["monitors"]
    nice_disease = model["nice_disease"]
    n_mod = len(module_keys)
    print(f"  {n_mod} modules, {len(diseases)} diseases, {len(pgs_meta)} PGS metadata")

    # ── Load patient ─────────────────────────────────────────────────
    print(f"Loading patient: {args.job}")
    data = load_patient_job(args.job)
    z_scores = data["z_scores"]
    ancestry = data["ancestry"]
    barcode = Path(args.job).name
    print(f"  {len(z_scores)} PGS scored")

    anc_keys = ["RF_P_AFR", "RF_P_AMR", "RF_P_CSA", "RF_P_EAS", "RF_P_EUR", "RF_P_MID"]
    anc_probs = {k: round(ancestry.get(k, 0.0), 4) for k in anc_keys}
    predicted_pop = max(anc_probs, key=anc_probs.get).replace("RF_P_", "")
    pcs_out = {f"PC{i+1}": round(ancestry.get(f"PC{i+1}", 0.0), 6) for i in range(10)}

    # ── Module projection ────────────────────────────────────────────
    print("Projecting onto module atlas...")
    z_mod, rel_mod = project_patient(z_scores, ancestry, model)

    # ── Disease classification ───────────────────────────────────────
    age_z = (args.age - model["age_mean"]) / model["age_std"] if model["age_std"] > 0 else 0.0
    sex_val = 1.0 if args.sex == "M" else 0.0
    anc_feat = [ancestry.get(k, 0.0) for k in anc_keys]

    feat = np.concatenate([
        np.nan_to_num(z_mod, nan=0.0),
        np.nan_to_num(rel_mod, nan=0.0),
        np.array(anc_feat),
        [sex_val],
        [age_z],
    ])

    print("Running disease classifiers...")
    probs = {}
    for disease in diseases:
        rf = model["models"][disease]
        p = rf.predict_proba(feat.reshape(1, -1))[0, 1]
        probs[disease] = round(float(p), 6)

    ranked = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    # ── Filter Z-scores and compute tiers ────────────────────────────
    print("Computing tiers...")
    filtered_z = {}
    for pid, z in z_scores.items():
        meta = pgs_meta.get(pid)
        if not meta:
            continue
        if not np.isfinite(z) or abs(z) > 10 or meta.get("quality_label") == "discard":
            continue
        filtered_z[pid] = z

    all_tiers, t1_thresh, t2_thresh = compute_all_tiers(filtered_z, pgs_meta)
    n_t1 = sum(1 for t in all_tiers.values() if t["tier"] == "T1")
    n_t2 = sum(1 for t in all_tiers.values() if t["tier"] == "T2")
    print(f"  T1={n_t1}, T2={n_t2}, thresholds: T1>={t1_thresh:.2f}, T2>={t2_thresh:.2f}")

    # ══════════════════════════════════════════════════════════════════
    # SECTION 1: Patient Profile
    # ══════════════════════════════════════════════════════════════════
    patient_profile = {
        "barcode": barcode,
        "sex": args.sex,
        "age": args.age,
        "ancestry": {
            "predicted_population": predicted_pop,
            "probabilities": anc_probs,
            "principal_components": pcs_out,
        },
        "pgs_scored": len(z_scores),
        "pgs_after_filter": len(filtered_z),
        "modules_available": int(np.isfinite(z_mod).sum()),
        "mean_reliability": round(float(np.nanmean(rel_mod)), 4) if np.any(np.isfinite(rel_mod)) else 0.0,
        "tier_thresholds": {"T1": round(t1_thresh, 2), "T2": round(t2_thresh, 2)},
        "tier_counts": {"T1": n_t1, "T2": n_t2, "T3": len(filtered_z) - n_t1 - n_t2},
    }

    # ══════════════════════════════════════════════════════════════════
    # SECTION 2: Clinical Circos + Domain Deep Dives (tiered)
    # ══════════════════════════════════════════════════════════════════
    print("Building clinical circos + domain deep dives...")
    domain_pgs = defaultdict(list)
    unclassified_pgs = []

    for pid, z in filtered_z.items():
        meta = pgs_meta.get(pid, {})
        tier_info = all_tiers.get(pid, {"importance": 0, "tier": "T3", "convergence": 0})
        domain = meta.get("domain")
        subsection = domain or "unclassified"
        entry = build_pgs_entry(pid, z, meta, tier_info)

        if domain and domain in domain_colors:
            domain_pgs[domain].append(entry)
        else:
            unclassified_pgs.append(entry)

    clinical_circos = {"domains": {}}
    elevated_domains = []
    for domain in domain_order:
        pgs_list = domain_pgs.get(domain, [])
        pgs_list.sort(key=lambda x: x.get("importance", 0), reverse=True)

        zvals = [p["z_score"] for p in pgs_list]
        n_elevated = sum(1 for z in zvals if abs(z) >= 1.5)
        mean_z = float(np.mean(zvals)) if zvals else 0.0
        risk = domain_risk_level(mean_z, n_elevated, len(zvals))

        if risk in ("ELEVATED", "MILDLY_ELEVATED"):
            elevated_domains.append(domain)

        t1 = [p for p in pgs_list if p["tier"] == "T1"]
        t2 = [p for p in pgs_list if p["tier"] == "T2"]
        t3 = [p for p in pgs_list if p["tier"] == "T3"]

        clinical_circos["domains"][domain] = {
            "n_pgs": len(pgs_list),
            "n_elevated": n_elevated,
            "mean_z": round(mean_z, 3),
            "risk_level": risk,
            "color": domain_colors.get(domain, "#9e9e9e"),
            "tier_counts": {"T1": len(t1), "T2": len(t2), "T3": len(t3)},
            "pgs": pgs_list,
        }

    if unclassified_pgs:
        clinical_circos["unclassified"] = {
            "n_pgs": len(unclassified_pgs),
            "pgs": sorted(unclassified_pgs, key=lambda x: x.get("importance", 0), reverse=True),
        }

    # ══════════════════════════════════════════════════════════════════
    # SECTION 3: Phenotype Landscapes (tiered)
    # ══════════════════════════════════════════════════════════════════
    print("Building phenotype landscapes...")
    landscape_types = {
        "biomarker_lab": "biomarkers",
        "imaging_structure": "brain_imaging",
        "quantitative_physiology": "physiology",
        "behavior_lifestyle": "behavioral",
        "anthropometric": "anthropometric",
        "longevity_resilience": "longevity",
    }

    phenotype_landscapes = {}
    for ptype, landscape_name in landscape_types.items():
        subsections = defaultdict(list)
        for pid, z in filtered_z.items():
            meta = pgs_meta.get(pid, {})
            if meta.get("phenotype_type") != ptype:
                continue
            tier_info = all_tiers.get(pid, {"importance": 0, "tier": "T3", "convergence": 0})
            subsec = meta.get("phenotype_subsection") or "other"
            entry = build_pgs_entry(pid, z, meta, tier_info)
            subsections[subsec].append(entry)

        if not subsections:
            continue

        landscape = {"n_pgs": 0, "subsections": {}}
        total_t1 = total_t2 = total_t3 = 0
        for subsec_name, pgs_list in sorted(subsections.items()):
            pgs_list.sort(key=lambda x: x.get("importance", 0), reverse=True)
            zvals = [p["z_score"] for p in pgs_list]
            t1 = [p for p in pgs_list if p["tier"] == "T1"]
            t2 = [p for p in pgs_list if p["tier"] == "T2"]
            t3 = [p for p in pgs_list if p["tier"] == "T3"]
            total_t1 += len(t1)
            total_t2 += len(t2)
            total_t3 += len(t3)

            landscape["subsections"][subsec_name] = {
                "n_pgs": len(pgs_list),
                "mean_z": round(float(np.mean(zvals)), 3) if zvals else 0.0,
                "n_elevated": sum(1 for z in zvals if abs(z) >= 1.5),
                "tier_counts": {"T1": len(t1), "T2": len(t2), "T3": len(t3)},
                "pgs": pgs_list,
            }
            landscape["n_pgs"] += len(pgs_list)

        landscape["tier_counts"] = {"T1": total_t1, "T2": total_t2, "T3": total_t3}
        phenotype_landscapes[landscape_name] = landscape

    # ══════════════════════════════════════════════════════════════════
    # SECTION 4: Disease Similarity
    # ══════════════════════════════════════════════════════════════════
    print("Building disease similarity data...")
    ref_stats = model["module_ref_stats"]
    disease_importances = model.get("disease_importances", {})

    disease_ranking = []
    for rank_idx, (disease, prob) in enumerate(ranked):
        imp_list = disease_importances.get(disease, [])
        driving_modules = []
        for feat_name, imp_val in imp_list[:10]:
            if not feat_name.startswith("z_"):
                continue
            mod_id_str = feat_name[2:]
            if mod_id_str not in ref_stats:
                continue
            stats = ref_stats[mod_id_str]
            mod_idx = None
            for mi, mk in enumerate(module_keys):
                if str(mk) == mod_id_str:
                    mod_idx = mi
                    break
            patient_z = float(z_mod[mod_idx]) if mod_idx is not None and np.isfinite(z_mod[mod_idx]) else None

            member_detail = []
            for j, pid in enumerate(stats["pgs_ids"]):
                pz = z_scores.get(pid, float("nan"))
                pmeta = pgs_meta.get(pid, {})
                member_detail.append({
                    "pgs_id": pid,
                    "z_score": round(pz, 4) if np.isfinite(pz) else None,
                    "trait": pmeta.get("trait", ""),
                    "family": pmeta.get("family"),
                    "quality_label": pmeta.get("quality_label", ""),
                    "variants": pmeta.get("variants"),
                    "loading": round(stats["loadings"][j], 4) if j < len(stats["loadings"]) else None,
                })

            driving_modules.append({
                "module_id": int(mod_id_str),
                "patient_z": round(patient_z, 4) if patient_z is not None else None,
                "importance": round(imp_val, 4),
                "direction": "elevated" if (patient_z or 0) > 0 else "reduced",
                "n_member_pgs": stats["size"],
                "primary_domain": stats["primary_domain"],
                "primary_trait": stats["primary_trait"],
                "mean_jaccard": stats["mean_jaccard"],
                "member_pgs": member_detail,
            })

        disease_ranking.append({
            "rank": rank_idx + 1,
            "disease": disease,
            "disease_name": nice_disease.get(disease, disease),
            "probability": prob,
            "strength": "WEAK",  # placeholder, set by relative labeling below
            "driving_modules": driving_modules,
        })

    # Apply relative strength labeling
    disease_ranking = strength_labels(disease_ranking)

    # Module profile
    module_profile = []
    for mi, mod_id in enumerate(module_keys):
        stats = ref_stats[str(mod_id)]
        driven = []
        for disease in diseases:
            for feat_name, imp_val in disease_importances.get(disease, [])[:20]:
                if feat_name == f"z_{mod_id}":
                    driven.append(disease)
                    break

        module_profile.append({
            "module_id": int(mod_id),
            "z_score": round(float(z_mod[mi]), 4) if np.isfinite(z_mod[mi]) else None,
            "reliability": round(float(rel_mod[mi]), 4) if np.isfinite(rel_mod[mi]) else None,
            "n_pgs": stats["size"],
            "primary_trait": stats["primary_trait"],
            "primary_domain": stats["primary_domain"],
            "mean_jaccard": stats["mean_jaccard"],
            "diseases_driven": driven,
        })

    disease_similarity = {
        "framing": "Your genomic signature most closely resembles...",
        "ranking": disease_ranking,
        "module_profile": module_profile,
    }

    # ══════════════════════════════════════════════════════════════════
    # SECTION 5: Suggested Monitoring
    # ══════════════════════════════════════════════════════════════════
    suggested_monitoring = {
        "elevated_domains": elevated_domains,
        "recommendations": {d: monitors.get(d, []) for d in elevated_domains if d in monitors},
    }

    # ══════════════════════════════════════════════════════════════════
    # Assemble and write
    # ══════════════════════════════════════════════════════════════════
    report = {
        "patient": patient_profile,
        "clinical_circos": clinical_circos,
        "phenotype_landscapes": phenotype_landscapes,
        "disease_similarity": disease_similarity,
        "suggested_monitoring": suggested_monitoring,
    }

    out_file = out_dir / f"{barcode}_report.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    sz = out_file.stat().st_size / 1024

    # Count T1 across all sections
    all_t1_count = sum(1 for d in clinical_circos["domains"].values()
                       for p in d["pgs"] if p["tier"] == "T1")
    all_t1_count += sum(1 for l in phenotype_landscapes.values()
                        for s in l["subsections"].values()
                        for p in s["pgs"] if p["tier"] == "T1")

    print(f"\nDone: {out_file} ({sz:.0f} KB)")
    print(f"  Patient: {barcode}, {args.sex}, age {args.age}, {predicted_pop}")
    print(f"  PGS scored: {len(z_scores)}, after filter: {len(filtered_z)}")
    print(f"  Modules: {patient_profile['modules_available']}/{n_mod}")
    print(f"  Tier 1 findings: {all_t1_count}")
    top_strength = disease_ranking[0]["strength"] if disease_ranking else "?"
    print(f"  Top disease: {ranked[0][0]} ({ranked[0][1]:.3f}, {top_strength})")
    print(f"  Elevated domains: {elevated_domains}")


if __name__ == "__main__":
    main()
