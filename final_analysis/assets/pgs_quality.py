#!/usr/bin/env python3
"""
PGS Quality & Confidence Scoring System
=========================================
Three-axis evaluation for each PGS result:

  1. PERFORMANCE — actual predictive power from published evaluations
     Uses real AUROC, R², OR from the PGS Catalog performance API.
     For PGS without evaluations, estimates from method + GWAS power proxies.

  2. APPLICABILITY — how relevant is this PGS for this specific patient
     Ancestry match between GWAS/eval populations and patient's population.
     Evaluated in patient's ancestry group = much higher confidence.

  3. MATCH CONFIDENCE — how trustworthy is this specific computation
     Variant match rate + Z-score stability.
     Low coverage or extreme Z = unreliable result regardless of PGS quality.

Final label: confident / likely / uncertain / discard

Usage:
  scorer = PGSQualityScorer("pgs_catalog_all.json", "performance_summary.json")
  scorer.score_all(patient_pop="MID", match_rates={...}, z_scores={...})
  scorer.save("quality_scores.tsv")
"""

import json
import math
import re
import numpy as np
from pathlib import Path
from datetime import datetime


# ---------------------------------------------------------------------------
# Ancestry mapping
# ---------------------------------------------------------------------------
POPULATION_TO_ANCESTRY = {
    "EUR": {"codes": ["EUR", "European"], "multi": ["MAE"]},
    "AFR": {"codes": ["AFR", "African"], "multi": ["MAO"]},
    "EAS": {"codes": ["EAS", "East Asian", "ASN", "Asian unspecified"], "multi": ["MAO"]},
    "SAS": {"codes": ["SAS", "South Asian"], "multi": ["MAO"]},
    "AMR": {"codes": ["AMR", "Hispanic or Latin American"], "multi": ["MAO"]},
    "MID": {"codes": ["GME", "Greater Middle Eastern"], "multi": []},
    "OTH": {"codes": [], "multi": []},
}

# All multi-ancestry codes
MULTI_CODES = {"MAE", "MAO"}

# ---------------------------------------------------------------------------
# Method tier classification (for fallback when no performance data)
# ---------------------------------------------------------------------------
# S=20, A=17, B=14, C=10, D=7, E=4
METHOD_PATTERNS = [
    # Tier S: Bayesian shrinkage
    (20, [r"(?i)PRS[-_]?CS", r"(?i)LDpred", r"(?i)SBayes", r"(?i)DBSLMM",
          r"(?i)PROSPER", r"(?i)SDPR", r"(?i)AnnoPred", r"(?i)PolyFun",
          r"(?i)SBLUP", r"(?i)BOLT[-_]?LMM", r"(?i)shaPRS"]),
    # Tier A: Penalized regression
    (17, [r"(?i)snpnet", r"(?i)lassosum", r"(?i)LASSO", r"(?i)elastic\s*net",
          r"(?i)penalized\s*regression", r"(?i)bigstatsr", r"(?i)GenoBoost",
          r"(?i)SparSNP", r"(?i)Random\s*(?:survival\s*)?forest",
          r"(?i)stepwise", r"(?i)Select\s*and\s*Shrink", r"(?i)(?:^|[^a-zA-Z])S4(?:[^a-zA-Z]|$)"]),
    # Tier B: Clumping + thresholding
    (14, [r"(?i)PRSice", r"(?i)PRSmix", r"(?i)P\s*\+\s*T", r"(?i)C\s*\+\s*T",
          r"(?i)Pruning\s*(?:and|&)\s*Threshold", r"(?i)Clumping\s*(?:and|&)\s*Threshold",
          r"(?i)CT[-_]?SLEB", r"(?i)PLINK", r"(?i)LD[-_\s]*clump", r"(?i)LD[-_\s]*prun",
          r"(?i)clump(?:ing|ed)", r"(?i)pt_clump", r"(?i)megaprs", r"(?i)RFDiseasemetaPRS",
          r"(?i)mJam", r"(?i)GCTA[-_]?COJO"]),
    # Tier C: Meta / ensemble
    (10, [r"(?i)metaGRS", r"(?i)MetaPRS", r"(?i)PRSsum", r"(?i)MultiPRS",
          r"(?i)integrative\s*P[GR]S", r"(?i)composite", r"(?i)summation"]),
    # Tier D: GWAS hits / curated
    (7,  [r"(?i)genome[-_\s]*wide\s*signif", r"(?i)Genomewide[-\s]*signif",
          r"(?i)GWAS", r"(?i)susceptibility", r"(?i)curated", r"(?i)fine[-_]?mapping",
          r"(?i)significant\s*(?:SNP|variant)", r"(?i)Variants?\s*(?:associated|significantly)",
          r"(?i)risk\s*allele", r"(?i)variants?\s*from\s*\w+\s*et\s*al",
          r"(?i)HLA", r"(?i)independent\s*(?:genome|SNP|variant)", r"(?i)Weighted\s*sum"]),
]


def _classify_method(method_name: str) -> int:
    """Return method score (4-20) by pattern matching."""
    if not method_name or method_name.strip() == "":
        return 4
    best = 4
    for score, patterns in METHOD_PATTERNS:
        for pat in patterns:
            if re.search(pat, method_name):
                best = max(best, score)
                break
    return best


class PGSQualityScorer:
    """Three-axis PGS quality scorer using real performance data."""

    def __init__(self, catalog_path: str, performance_path: str = None):
        with open(catalog_path) as f:
            catalog_list = json.load(f)
        self.catalog = {e["id"]: e for e in catalog_list}

        self.performance = {}
        if performance_path:
            with open(performance_path) as f:
                self.performance = json.load(f)
            print(f"Loaded {len(self.catalog)} PGS, {len(self.performance)} with performance data")
        else:
            print(f"Loaded {len(self.catalog)} PGS (no performance data)")

        self.scores = {}

    # -------------------------------------------------------------------
    # Axis 1: Performance (0-40)
    # -------------------------------------------------------------------
    def _score_performance(self, pgs_id: str, entry: dict) -> tuple:
        """
        Score predictive performance from real evaluation data.
        Uses: AUROC > R² > OR > method proxy (in priority order).
        """
        perf = self.performance.get(pgs_id)

        if perf:
            auroc = perf.get("auroc_median")
            r2 = perf.get("r2_max")
            or_val = perf.get("or_median")
            n_evals = perf.get("n_evals", 0)

            # AUROC is the gold standard for binary outcomes
            if auroc is not None and 0.5 <= auroc <= 1.0:
                # Map AUROC to 0-40 score
                # 0.50 = random (0), 0.60 = poor (12), 0.70 = fair (22),
                # 0.80 = good (32), 0.90 = excellent (38), 1.0 = perfect (40)
                s = int(round(max(0, min(40, (auroc - 0.5) * 80))))
                source = f"AUROC={auroc:.3f} ({perf.get('auroc_n', '?')} studies)"

                # Bonus for many evaluations
                if n_evals >= 10:
                    s = min(40, s + 2)
                    source += f", {n_evals} evals"

                return (s, source, "measured")

            # R² for continuous traits
            if r2 is not None and 0 <= r2 <= 1.0:
                # Map R² to 0-40. PGS R² is typically 0.01-0.30
                # 0.01 = marginal (8), 0.05 = modest (16), 0.10 = decent (22),
                # 0.20 = good (30), 0.30+ = excellent (36)
                if r2 >= 0.30:
                    s = 36
                elif r2 >= 0.20:
                    s = 30
                elif r2 >= 0.10:
                    s = 22
                elif r2 >= 0.05:
                    s = 16
                elif r2 >= 0.02:
                    s = 12
                elif r2 >= 0.01:
                    s = 8
                else:
                    s = 4
                source = f"R²={r2:.4f} ({perf.get('r2_n', '?')} studies)"

                if n_evals >= 10:
                    s = min(40, s + 2)
                    source += f", {n_evals} evals"

                return (s, source, "measured")

            # OR for case/control without AUROC
            if or_val is not None and or_val > 0:
                # Map OR to score. OR=1 means no effect.
                # OR 1.1-1.3 = weak, 1.3-1.5 = moderate, 1.5-2.0 = strong, 2.0+ = very strong
                if or_val >= 3.0 or or_val <= 0.33:
                    s = 32
                elif or_val >= 2.0 or or_val <= 0.5:
                    s = 26
                elif or_val >= 1.5 or or_val <= 0.67:
                    s = 20
                elif or_val >= 1.3 or or_val <= 0.77:
                    s = 14
                elif or_val >= 1.1 or or_val <= 0.91:
                    s = 8
                else:
                    s = 4
                source = f"OR={or_val:.2f} ({perf.get('or_n', '?')} studies)"
                return (s, source, "measured")

            # Has evaluations but no usable metrics
            if n_evals > 0:
                return (10, f"{n_evals} evals (no AUROC/R²/OR)", "partial")

        # No performance data at all — fall back to proxy estimation
        method_score = _classify_method(entry.get("method_name", ""))

        # Scale method (4-20) to performance (0-40) with penalty for being estimated
        proxy_score = int(round(method_score * 1.2))  # max ~24 from proxy

        # Boost for large GWAS
        ad = entry.get("ancestry_distribution", {})
        gwas_n = ad.get("gwas", {}).get("count", 0)
        if gwas_n and gwas_n >= 500_000:
            proxy_score = min(28, proxy_score + 4)
        elif gwas_n and gwas_n >= 100_000:
            proxy_score = min(26, proxy_score + 2)

        method_name = entry.get("method_name", "?")[:40]
        return (proxy_score, f"est. from method ({method_name})", "estimated")

    # -------------------------------------------------------------------
    # Axis 2: Applicability (0-30)
    # -------------------------------------------------------------------
    def _score_applicability(self, pgs_id: str, entry: dict, patient_pop: str) -> tuple:
        """
        How applicable is this PGS for this patient's ancestry?
        Uses GWAS ancestry, eval ancestry, and whether eval was done
        in patient's population.
        """
        pop_info = POPULATION_TO_ANCESTRY.get(patient_pop, {"codes": [], "multi": []})
        target_codes = pop_info["codes"]
        target_multi = pop_info["multi"]

        ad = entry.get("ancestry_distribution", {})
        gwas_dist = ad.get("gwas", {}).get("dist", {})
        eval_dist = ad.get("eval", {}).get("dist", {})

        details = []

        # --- GWAS ancestry match (0-12) ---
        gwas_s = 0
        if gwas_dist:
            match_pct = sum(gwas_dist.get(c, 0) for c in target_codes if isinstance(c, str) and len(c) <= 5)
            multi_pct = sum(gwas_dist.get(c, 0) for c in MULTI_CODES)

            if match_pct >= 50:
                gwas_s = 12
            elif match_pct >= 20:
                gwas_s = 9
            elif match_pct > 0 or multi_pct > 0:
                gwas_s = 6
            else:
                gwas_s = 3
        else:
            gwas_s = 5  # unknown = middle

        # --- Eval ancestry match (0-10) ---
        eval_s = 0
        if eval_dist:
            eval_match = sum(eval_dist.get(c, 0) for c in target_codes if isinstance(c, str) and len(c) <= 5)
            eval_multi = sum(eval_dist.get(c, 0) for c in MULTI_CODES)

            if eval_match > 0:
                eval_s = 10
            elif eval_multi > 0:
                eval_s = 6
            else:
                eval_s = 2
        else:
            eval_s = 4

        # --- Evaluated in patient's population? (0-8) ---
        # Check performance data for actual evaluation cohorts in patient's ancestry
        perf = self.performance.get(pgs_id, {})
        eval_anc = perf.get("eval_ancestries", [])
        pop_evaluated = 0

        # Check if any eval ancestry matches patient
        for anc in eval_anc:
            for code in target_codes:
                if code.lower() in anc.lower():
                    pop_evaluated = 8
                    break
            if pop_evaluated:
                break

        if not pop_evaluated:
            # Check for "Middle Eastern" specifically for MID
            if patient_pop == "MID":
                for anc in eval_anc:
                    if "middle eastern" in anc.lower() or "persian" in anc.lower():
                        pop_evaluated = 8
                        break
            if not pop_evaluated and any("multi" in a.lower() for a in eval_anc):
                pop_evaluated = 4

        total = gwas_s + eval_s + pop_evaluated
        detail = f"gwas={gwas_s}/12, eval={eval_s}/10, pop_eval={pop_evaluated}/8"

        return (total, detail)

    # -------------------------------------------------------------------
    # Axis 3: Match Confidence (0-30)
    # -------------------------------------------------------------------
    def _score_match_confidence(self, match_rate: float, z_score: float) -> tuple:
        """
        How trustworthy is this specific computation for this patient?
        Based on variant coverage and Z-score stability.
        """
        details = []

        # --- Variant coverage (0-18) ---
        if match_rate is None:
            cov_s = 9  # unknown
            details.append("cov=?")
        elif match_rate < 0.02:
            cov_s = 0
            details.append(f"cov={match_rate:.1%} NONE")
        elif match_rate < 0.05:
            cov_s = 1
            details.append(f"cov={match_rate:.1%} negligible")
        elif match_rate < 0.15:
            cov_s = 4
            details.append(f"cov={match_rate:.0%} very low")
        elif match_rate < 0.25:
            cov_s = 7
            details.append(f"cov={match_rate:.0%} low")
        elif match_rate < 0.35:
            cov_s = 10
            details.append(f"cov={match_rate:.0%} fair")
        elif match_rate < 0.50:
            cov_s = 13
            details.append(f"cov={match_rate:.0%} adequate")
        elif match_rate < 0.65:
            cov_s = 16
            details.append(f"cov={match_rate:.0%} good")
        else:
            cov_s = 18
            details.append(f"cov={match_rate:.0%} excellent")

        # --- Z-score stability (0-12) ---
        if z_score is None:
            z_s = 6  # unknown
            details.append("Z=?")
        elif match_rate is not None and match_rate < 0.02:
            z_s = 0  # Z is meaningless with no coverage
            details.append("Z=N/A")
        elif abs(z_score) <= 4:
            z_s = 12
            details.append(f"Z={z_score:.1f}")
        elif abs(z_score) <= 7:
            z_s = 8
            details.append(f"Z={z_score:.1f} elevated")
        elif abs(z_score) <= 10:
            z_s = 4
            details.append(f"|Z|={abs(z_score):.0f} high")
        else:
            z_s = 0
            details.append(f"|Z|={abs(z_score):.0f} extreme")

        total = cov_s + z_s
        return (total, "; ".join(details))

    # -------------------------------------------------------------------
    # Composite
    # -------------------------------------------------------------------
    def score_pgs(self, pgs_id: str, patient_pop: str = "EUR",
                  match_rate: float = None, z_score: float = None) -> dict:
        entry = self.catalog.get(pgs_id)
        if entry is None:
            return {"pgs_id": pgs_id, "label": "discard", "error": "not in catalog"}

        perf_score, perf_detail, perf_source = self._score_performance(pgs_id, entry)
        app_score, app_detail = self._score_applicability(pgs_id, entry, patient_pop)
        match_score, match_detail = self._score_match_confidence(match_rate, z_score)

        total = perf_score + app_score + match_score  # max 100

        # Labeling — hard gates override the total score
        if match_rate is not None and match_rate < 0.02:
            label = "discard"
        elif z_score is not None and abs(z_score) > 10:
            label = "discard"
        elif match_rate is not None and match_rate < 0.05:
            label = "discard"
        elif total >= 60:
            label = "confident"
        elif total >= 40:
            label = "likely"
        elif total >= 25:
            label = "uncertain"
        else:
            label = "discard"

        return {
            "pgs_id": pgs_id,
            "total": total,
            "label": label,
            "performance": perf_score,
            "performance_detail": perf_detail,
            "performance_source": perf_source,
            "applicability": app_score,
            "applicability_detail": app_detail,
            "match_confidence": match_score,
            "match_detail": match_detail,
            "trait": entry.get("trait_reported", ""),
            "method_name": entry.get("method_name", ""),
            "variants_number": entry.get("variants_number", 0),
        }

    def score_all(self, patient_pop: str = "EUR",
                  match_rates: dict = None,
                  z_scores: dict = None) -> dict:
        if match_rates is None:
            match_rates = {}
        if z_scores is None:
            z_scores = {}

        self.scores = {}
        for pgs_id in self.catalog:
            self.scores[pgs_id] = self.score_pgs(
                pgs_id, patient_pop,
                match_rates.get(pgs_id),
                z_scores.get(pgs_id),
            )

        # Summary
        scored = [s for s in self.scores.values() if "error" not in s]
        labels = {}
        sources = {}
        for s in scored:
            labels[s["label"]] = labels.get(s["label"], 0) + 1
            sources[s["performance_source"]] = sources.get(s["performance_source"], 0) + 1

        totals = [s["total"] for s in scored]
        print(f"\nScored {len(scored)} PGS (patient: {patient_pop}):")
        print(f"  Total score: mean={np.mean(totals):.1f}, median={np.median(totals):.1f}, range=[{min(totals)},{max(totals)}]")
        print(f"  Labels: {dict(sorted(labels.items()))}")
        print(f"  Performance source: {dict(sorted(sources.items()))}")

        return self.scores

    def save(self, output_path: str):
        if not self.scores:
            raise ValueError("No scores — call score_all() first")

        header = [
            "pgs_id", "total", "label",
            "performance", "applicability", "match_confidence",
            "performance_source",
            "trait", "method_name", "variants_number",
            "performance_detail", "applicability_detail", "match_detail",
        ]

        with open(output_path, "w") as f:
            f.write("\t".join(header) + "\n")
            for pgs_id in sorted(self.scores.keys()):
                s = self.scores[pgs_id]
                if "error" in s:
                    continue
                row = [str(s.get(h, "")) for h in header]
                f.write("\t".join(row) + "\n")

        print(f"Saved to {output_path}")

    def save_json(self, output_path: str):
        if not self.scores:
            raise ValueError("No scores — call score_all() first")
        with open(output_path, "w") as f:
            json.dump(self.scores, f, indent=2)
        print(f"Saved JSON to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="PGS Quality Scorer")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--performance", help="Path to performance_summary.json")
    parser.add_argument("--population", default="EUR")
    parser.add_argument("--match-rates", help="match_rates.tsv from pipeline")
    parser.add_argument("--z-scores", help="scores_adjusted.tsv from pipeline")
    parser.add_argument("--output", default="quality_scores.tsv")
    parser.add_argument("--output-json")
    parser.add_argument("--inspect", nargs="*")
    args = parser.parse_args()

    scorer = PGSQualityScorer(args.catalog, args.performance)

    match_rates = {}
    if args.match_rates:
        with open(args.match_rates) as f:
            next(f)
            for line in f:
                p = line.strip().split("\t")
                if len(p) >= 5:
                    match_rates[p[0]] = float(p[4])
        print(f"Loaded {len(match_rates)} match rates")

    z_scores = {}
    if args.z_scores:
        import csv
        with open(args.z_scores) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                z_scores[row["pgs_id"]] = float(row["Z_MostSimilarPop"])
        print(f"Loaded {len(z_scores)} Z-scores")

    scorer.score_all(patient_pop=args.population, match_rates=match_rates, z_scores=z_scores)
    scorer.save(args.output)
    if args.output_json:
        scorer.save_json(args.output_json)

    if args.inspect:
        for pgs_id in args.inspect:
            s = scorer.scores.get(pgs_id)
            if not s:
                print(f"\n{pgs_id}: NOT FOUND")
                continue
            print(f"\n{'='*65}")
            print(f" {pgs_id}: {s['trait']}")
            print(f"{'='*65}")
            print(f"  Performance:   {s['performance']:2d}/40  {s['performance_detail']}")
            print(f"  Applicability: {s['applicability']:2d}/30  {s['applicability_detail']}")
            print(f"  Match Conf:    {s['match_confidence']:2d}/30  {s['match_detail']}")
            print(f"  {'─'*55}")
            print(f"  TOTAL:         {s['total']:2d}/100")
            print(f"  LABEL:         {s['label'].upper()}")


if __name__ == "__main__":
    main()
