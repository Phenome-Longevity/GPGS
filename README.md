# PolyGen

GPU-accelerated polygenic score pipeline that scores patients against the entire PGS Catalog (5,232 scores) in ~1 hour per patient.

## Pipeline

Four-phase pipeline: VCF preprocessing, PGS Catalog sync, GPU scoring, ancestry-adjusted Z-normalization.

```
VCF → Binary Store (20s) → PGS Sync (7s) → GPU Scoring (8 min) → Ancestry Adjustment (70 min)
```

### Benchmarks

| PGS Count | PolyGen | pgsc_calc (est) | Speedup |
|-----------|---------|-----------------|---------|
| 50 | 2.6 min | 36 min | 14x |
| 150 | 2.5 min | ~96 min | 38x |
| 300 | 4.5 min | ~192 min | 43x |
| 500 | 5.8 min | ~320 min | 55x |
| 5,232 (full catalog) | 78 min | N/A (days) | -- |

Fresh VCF, no cache, single patient, NVIDIA GB10 + 119GB RAM.

### Accuracy vs pgsc_calc

Validated against pgsc_calc across 6 patients and 1,054 PGS comparisons:

| Metric | Result |
|--------|--------|
| Within 0.1 Z | 95.2% |
| Within 0.5 Z | 99.5% |
| Within 1.0 Z | 100% |
| Mean |delta Z| | 0.016 |
| Median |delta Z| | 0.000002 |

### Concurrency

| Mode | Wall time | Throughput |
|------|----------|------------|
| 1 patient | 1h 18m | 0.77/hr |
| 2 parallel | 1h 27m | 1.37/hr |
| 3 parallel | 1h 49m | 1.65/hr |

### Storage

- Per patient job: ~860 MB (vs pgsc_calc ~18 GB per batch of 11 PGS)
- Full catalog (5,232 PGS) in one job vs pgsc_calc requiring ~75 separate batch jobs

### Pipeline Detail

1. **VCF to Binary Store** (~20s) -- converts VCF to FNV1a-hashed binary format (uint64 key = chrom|pos|FNV1a(REF)|FNV1a(ALT), float32 dosage)
2. **PGS Catalog Sync** (~7s) -- downloads and caches scoring files with 8-strategy variant matching + ambiguous allele exclusion
3. **GPU Scoring** (~8 min) -- custom CUDA v2 kernel: binary search + atomicCAS dedup + flip + model + weighted sum, all on GPU. Vectorized rsID fallback for non-GPU matches
4. **Ancestry Adjustment** (~70 min) -- FRAPOSA online PCA projection (37,997 LD-pruned variants) + Random Forest population classifier (6 superpopulations) + per-population Z-normalization against 3,330 HGDP+1kGP reference samples. Batched plink2 `--score-list` for reference scoring

### Architecture

- **Binary store**: FNV1a uint64 keys + float32 dosages
- **GPU kernel**: CUDA v2 -- binary search + atomicCAS dedup + flip + model + multiply, all on GPU
- **Reference scoring**: plink2 `--score-list` on compressed reference panel (13 GB pgen), 10x faster than GPU memmap
- **Ancestry**: FRAPOSA PCA + RF + batched plink2 → Z-normalization
- **Caching**: Per-PGS binary cache (~861 GB for full catalog), per-patient reference score cache (keyed by variant set hash)

## System Requirements

**Minimum:**
- Python 3.9+
- plink2
- bcftools
- 32 GB RAM
- 30 GB disk (reference panel + ancestry cache)

**Recommended:**
- NVIDIA GPU with CUDA (CuPy) -- 7x faster patient scoring
- 64+ GB RAM -- enables concurrent patients
- NVMe SSD

## Setup

```bash
# 1. Install dependencies
./setup.sh --check

# 2. Install reference panel (one-time, ~15 GB)
# Download pgsc_HGDP+1kGP_v1.tar.zst from PGS Catalog
./setup.sh --ref /path/to/pgsc_HGDP+1kGP_v1.tar.zst
```

## Usage

```bash
# Score against full catalog
./run.sh --input patient.vcf.gz --output ./jobs --pgsid $(cat pgs_ids.txt)

# Score specific PGS
./run.sh --input patient.vcf.gz --output ./jobs --pgsid PGS000001,PGS000802

# Skip ancestry adjustment (raw scores only)
./run.sh --input patient.vcf.gz --output ./jobs --pgsid PGS000001 --no-ancestry
```

### Output per patient

- `results/scores.tsv` -- raw PGS scores
- `results/scores_adjusted.tsv` -- ancestry-adjusted Z-scores, percentiles
- `results/ancestry.tsv` -- predicted population, PCA coordinates
- `results/match_rates.tsv` -- variant matching statistics per PGS
- `benchmark.json` -- timing breakdown

---

## Downstream Analysis

Scoring the full PGS Catalog per patient enables a new form of genomic profiling: correlation-based PGS modules that capture polygenic signal across thousands of scores simultaneously.

### Module Atlas

- 5,221 PGS scored across 3,330 reference genomes (HGDP+1kGP)
- Correlation matrix + reciprocal kNN graph (K=15, min |r| = 0.18)
- Louvain community detection: 584 communities
- Bootstrap stability filtering (25 iterations, Jaccard >= 0.70): **194 retained modules** covering 4,646 PGS
- Patient scoring: quality-weighted eigenloading projection + ancestry residualization + MAD robust Z-standardization

### Results

934 patients across 11 disease cohorts + 201 healthy controls, all scored on 5,232 PGS. 5-fold stratified cross-validation throughout.

**Module-based scoring vs published disease-specific PGS:**

| Disease | n | Module AUC | 95% CI | Best Published PGS | Margin | PGS Beaten |
|---------|---|-----------|--------|-------------------|--------|------------|
| Longevity | 150 | **1.000** | [1.000, 1.000] | 0.534 | +0.466 | 2/2 |
| Atopic Dermatitis | 62 | **0.992** | [0.980, 0.997] | 0.591 | +0.400 | 16/16 |
| Asthma | 41 | **0.968** | [0.943, 0.984] | 0.638 | +0.330 | 78/78 |
| Type 2 Diabetes | 34 | **0.933** | [0.891, 0.959] | 0.689 | +0.245 | 183/183 |
| Coronary Artery Disease | 121 | **0.953** | [0.927, 0.970] | 0.867 | +0.086 | 125/125 |
| Breast Cancer | 60 | **0.935** | [0.894, 0.967] | 0.739 | +0.196 | 163/163 |
| Celiac Disease | 35 | **0.987** | [0.925, 1.000] | 0.812 | +0.175 | 11/11 |
| Bipolar Disorder | 34 | **0.900** | [0.822, 0.954] | 0.721 | +0.178 | 3/3 |
| Parkinson's Disease | 114 | **1.000** | [1.000, 1.000] | 0.910 | +0.090 | 11/11 |
| Colorectal Cancer | 48 | **0.977** | [0.949, 0.992] | 0.939 | +0.038 | 74/74 |
| Ankylosing Spondylitis | 34 | 0.673 | [0.569, 0.753] | 0.732 | -0.059 | 8/10 |

**Total: 674/676 (99.7%) disease-specific PGS beaten across 11 diseases.**

**Multi-class disease ranking (genome only, no clinical data):**

| Metric | Result |
|--------|--------|
| Top-1 accuracy | 52.3% (chance = 9.1%) |
| Top-3 accuracy | 82.5% |
| Permutation significance | 19.1 sigma (p < 0.01, 100 shuffles) |
| Null mean | 18.4% |

**Robustness checks:**
- Clean comparator (no proxy/technical PGS): 635/637 (99.7%) beaten
- Stress test vs ALL 5,232 PGS (not just disease-matched): module wins 6/11 diseases
- Atlas sensitivity: stable across K=10 (157 modules), K=15 (194 modules), K=20 (141 modules)
- Median AUC margin vs best published PGS: +0.358

### Reproducing the Analysis

```bash
cd final_analysis
python3 robustness.py
```

Rebuilds everything from scratch: quality scores from PGS Catalog data, reference matrix from cached scores, patient aggregation from raw job outputs, module atlas, ML training, cross-validation, permutation test, stress test. All results are deterministic (fixed random seeds). Run time ~30 minutes.

### Output Files

| File | Description |
|------|-------------|
| `results.json` | All summary statistics |
| `patient_predictions.csv` | Per-patient probabilities and rankings (934 patients) |
| `patient_module_scores.csv` | Per-patient module Z-scores (934 x 388 features) |
| `per_pgs_comparison.csv` | Module AUC vs each of 676 published PGS |
| `stress_test_all5232.csv` | Module vs all 5,232 PGS per disease |
| `feature_importances.json` | RF feature importances per disease |
| `confusion_matrix.json` | Binary + multi-class confusion matrices |
| `per_fold_aucs.json` | Per-fold cross-validation AUCs |
| `permutation_null.json` | 100 null distribution accuracies |
| `atlas_modules.json` | 194 module definitions with PGS members |
| `models.pkl` | Trained models + atlas for scoring new patients |

## Repository Structure

```
modules/              # Pipeline source
  vcf_to_bin.py       # Phase 1: VCF to FNV1a-hashed binary store
  pgs_catalog.py      # Phase 2: PGS Catalog sync + 8-strategy matching
  score.py            # Phase 3: GPU/CPU scoring (CUDA v2 kernel)
  ancestry.py         # Phase 4: FRAPOSA PCA + Z-normalization
data/ancestry_cache/  # Bundled FRAPOSA reference (3,330 HGDP+1kGP)
final_analysis/       # Downstream analysis (self-contained)
  robustness.py       # End-to-end reproduction + robustness script
  assets/             # Input data (PGS Catalog, quality scoring, LLM classifications)
run.sh                # Pipeline entry point
setup.sh              # Dependency checker + reference installer
```

## License

Apache 2.0
