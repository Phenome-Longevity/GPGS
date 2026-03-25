#!/usr/bin/env python3
"""
PolyGen Analysis — FULL END-TO-END REPRODUCTION + ROBUSTNESS CHECKS.

EXTERNAL INPUTS (only two):
  1. JOB_DIR: path to cohort job folders (957 PolyGen patient outputs)
  2. POLYGEN_REF: path to PolyGen reference assets (ships with tool, derived from 1000 Genomes)

SELF-CONTAINED:
  - assets/live5232_payloads_2026_03_11.parquet (LLM PGS classification, in this folder)

Steps:
  0. Aggregate raw job outputs → wide Z-score matrix + ancestry
  1. Rebuild module atlas from reference panel
  2. Score patients through atlas (quality-weighted projection + ancestry residualization + MAD robust_z)
  3. Build feature matrices
  4. Per-disease evaluation (vs ALL and CLEAN family-matched PGS)
  5. Multi-class ranking
  6. Permutation test (100 shuffles)
  7. Bootstrap CIs
  8. Stress test: modules vs ALL 5,232 PGS
  9. Per-PGS margin analysis
  10. Atlas sensitivity (K=10, K=20)
  11. Save everything (11 output files)

Run: python3 robustness.py
"""
import csv, json, os, time, pickle
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
import community as community_louvain
import networkx as nx
import polars as pl

def log(msg): print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)

# ═══════════════════════════════════════
# PATHS — only external inputs
# ═══════════════════════════════════════
JOB_DIR = Path('/tmp/cohort_drop_2026_03_07')           # patient job outputs (only external input)
OUT = Path(__file__).parent
ASSETS = OUT / 'assets'                                 # everything else is self-contained
POLYGEN_REF = ASSETS                                    # reference panel files
LLM_PARQUET = ASSETS / 'live5232_payloads_2026_03_11.parquet'

# Atlas parameters
K_NEIGHBORS = 15
MIN_ABS_CORR = 0.18
WINSOR_Q = 0.005
N_BOOT = 25
BOOT_FRACTION = 0.8
RETAIN_MIN_SIZE = 3
RETAIN_MIN_JACCARD = 0.70

log('=' * 70)
log('POLYGEN — FULL END-TO-END REPRODUCTION + ROBUSTNESS')
log('=' * 70)

# ═══════════════════════════════════════
# REBUILD: generate everything from scratch
# ═══════════════════════════════════════
import subprocess
log('\n[REBUILD] Running pgs_quality.py → quality_scores.tsv')
subprocess.check_call([
    'python3', str(ASSETS / 'pgs_quality.py'),
    '--catalog', str(ASSETS / 'pgs_catalog_all.json'),
    '--performance', str(ASSETS / 'performance_summary.json'),
    '--output', str(OUT / 'quality_scores.tsv'),
])
log(f'  Generated quality_scores.tsv')

log('\n[REBUILD] Building ref_matrix from cached per-PGS reference scores')
REF_SCORES_DIR = Path('/home/vigil/Research/PolyGen/accelerated/data/ancestry_cache/ref_scores/8ed2c873533b')
score_files = sorted(REF_SCORES_DIR.glob('*.npy'))
log(f'  Loading {len(score_files)} cached .npy files...')
ref_pgs_ids_rebuild = [f.stem for f in score_files]
ref_scores_raw = np.stack([np.load(f) for f in score_files], axis=1)  # (3330, 5221)
ref_pcs_rebuild = np.load(POLYGEN_REF / 'ref_pcs.npy').astype(np.float64)[:, :10]
ref_sample_ids_rebuild = json.loads((POLYGEN_REF / 'ref_sample_ids.json').read_text())
X_design = np.column_stack([np.ones(ref_pcs_rebuild.shape[0]), ref_pcs_rebuild])
X_ref_rebuilt = np.zeros_like(ref_scores_raw, dtype=np.float32)
for j in range(ref_scores_raw.shape[1]):
    y = ref_scores_raw[:, j].astype(np.float64)
    beta, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)
    X_ref_rebuilt[:, j] = (y - X_design @ beta).astype(np.float32)
log(f'  Rebuilt: {X_ref_rebuilt.shape[0]} samples × {X_ref_rebuilt.shape[1]} PGS (PC1-10 residualized)')
del ref_scores_raw, X_design

# ═══════════════════════════════════════
# STEP 0: AGGREGATE RAW JOB OUTPUTS
# ═══════════════════════════════════════
log('\n[0/11] Aggregating raw job outputs into wide matrix')

manifest_by_bc = {}
manifest_by_jid = {}
with open(JOB_DIR / 'manifest.csv') as f:
    for row in csv.DictReader(f):
        manifest_by_bc[row['barcode']] = row
        manifest_by_jid[row['job_id']] = row
manifest = manifest_by_bc  # used later for sex/age lookup

jobs_path = JOB_DIR / 'jobs'
job_dirs = sorted([d for d in jobs_path.iterdir() if d.is_dir()])
log(f'  Found {len(job_dirs)} job folders, {len(manifest_by_jid)} in manifest')

all_scores = []   # list of (barcode, pgs_id, z)
all_ancestry = [] # list of dicts per patient
skipped = 0

for jdir in job_dirs:
    scores_file = jdir / 'results' / 'scores_adjusted.tsv'
    ancestry_file = jdir / 'results' / 'ancestry.tsv'
    if not scores_file.exists():
        skipped += 1
        continue

    # Match job_id from dir name: source_JOBID_accelerated
    parts = jdir.name.split('_')
    job_id = parts[1] if len(parts) >= 2 else ''
    meta = manifest_by_jid.get(job_id)
    if meta is None:
        skipped += 1
        continue
    barcode = meta['barcode']

    # Read scores
    with open(scores_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            z = row.get('Z_norm1', row.get('Z_MostSimilarPop', '0'))
            try:
                z = float(z)
                if not np.isfinite(z): z = 0.0
            except: z = 0.0
            all_scores.append((barcode, row['pgs_id'], z))

    # Read ancestry
    if ancestry_file.exists():
        with open(ancestry_file) as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                anc_row = {
                    'sample_id': row.get('sample_id', barcode),
                    'barcode': barcode,
                    'job_id': job_id,
                    'label': meta.get('label', ''),
                    'source': meta.get('source', ''),
                    'MostSimilarPop': row.get('MostSimilarPop', ''),
                    'LowConfidence': row.get('LowConfidence', ''),
                }
                for i in range(1, 11):
                    try: anc_row[f'PC{i}'] = float(row.get(f'PC{i}', 0))
                    except: anc_row[f'PC{i}'] = 0.0
                for col in ['RF_P_AFR','RF_P_AMR','RF_P_CSA','RF_P_EAS','RF_P_EUR','RF_P_MID']:
                    try: anc_row[col] = float(row.get(col, 0))
                    except: anc_row[col] = 0.0
                all_ancestry.append(anc_row)

log(f'  Read {len(all_scores)} score rows from {len(job_dirs) - skipped} patients (skipped {skipped})')

# Pivot to wide matrix (deduplicate: keep last per barcode+pgs_id)
scores_df = pl.DataFrame({
    'barcode': [s[0] for s in all_scores],
    'pgs_id': [s[1] for s in all_scores],
    'z': [s[2] for s in all_scores],
}).group_by(['barcode', 'pgs_id']).last()
pgs_wide = scores_df.pivot(on='pgs_id', index='barcode', values='z').sort('barcode')

# Add metadata columns from manifest
meta_cols = {'barcode': [], 'job_id': [], 'sample_id': [], 'label': [], 'disease': [], 'source': [], 'sex': [], 'age': []}
for bc in pgs_wide['barcode'].to_list():
    m = manifest[bc]
    meta_cols['barcode'].append(bc)
    meta_cols['job_id'].append(m.get('job_id', ''))
    meta_cols['sample_id'].append(bc)
    meta_cols['label'].append(m.get('label', ''))
    meta_cols['disease'].append(m.get('disease', ''))
    meta_cols['source'].append(m.get('source', ''))
    meta_cols['sex'].append(m.get('sex', ''))
    meta_cols['age'].append(m.get('age', ''))

meta_df = pl.DataFrame(meta_cols)
pgs_cols_only = [c for c in pgs_wide.columns if c.startswith('PGS')]
pgs_wide = meta_df.join(pgs_wide.select(['barcode'] + pgs_cols_only), on='barcode', how='left')

anc_df = pl.DataFrame(all_ancestry).group_by('barcode').last()  # deduplicate

# Save aggregated data
pgs_wide.write_parquet(OUT / 'patient_pgs_z_wide.parquet')
anc_df.write_csv(OUT / 'patient_ancestry.tsv', separator='\t')

log(f'  Wide matrix: {pgs_wide.shape[0]} patients × {len(pgs_cols_only)} PGS')
log(f'  Ancestry: {anc_df.shape[0]} patients × {anc_df.shape[1]} columns')
log(f'  Saved: patient_pgs_z_wide.parquet, patient_ancestry.tsv')

# ═══════════════════════════════════════
# STEP 1: REBUILD MODULE ATLAS
# (exact same algorithm as build_module_atlas.py)
# ═══════════════════════════════════════
log('\n[1/11] Rebuilding module atlas from reference panel')

# --- Atlas helper functions (copied from build_module_atlas.py) ---

def winsorize_and_standardize(X):
    lo = np.quantile(X, WINSOR_Q, axis=0)
    hi = np.quantile(X, 1.0 - WINSOR_Q, axis=0)
    Xw = np.clip(X, lo, hi)
    mean = Xw.mean(axis=0)
    std = Xw.std(axis=0, ddof=1)
    std[std == 0] = 1.0
    Xz = (Xw - mean) / std
    return Xz.astype(np.float32)

def correlation_matrix(Xz):
    n = Xz.shape[0]
    corr = (Xz.T @ Xz) / float(n - 1)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 0.0)
    return corr.astype(np.float32)

def sparse_edges_from_corr(corr):
    n = corr.shape[0]
    abs_corr = np.abs(corr)
    nbrs = []
    for i in range(n):
        row = abs_corr[i]
        idx = np.argpartition(row, -K_NEIGHBORS)[-K_NEIGHBORS:]
        idx = idx[np.argsort(row[idx])[::-1]]
        idx = [j for j in idx if j != i and row[j] >= MIN_ABS_CORR]
        nbrs.append(idx)
    edges = []
    seen = set()
    for i in range(n):
        for j in nbrs[i]:
            if i not in nbrs[j]: continue
            a, b = sorted((i, j))
            if (a, b) in seen: continue
            seen.add((a, b))
            edges.append((a, b, float(corr[a, b]), float(abs_corr[a, b])))
    return edges

def _sparse_edges_k(corr, k):
    """Same as sparse_edges_from_corr but with explicit K parameter."""
    n = corr.shape[0]
    abs_corr = np.abs(corr)
    nbrs = []
    for i in range(n):
        row = abs_corr[i]
        idx = np.argpartition(row, -k)[-k:]
        idx = idx[np.argsort(row[idx])[::-1]]
        idx = [j for j in idx if j != i and row[j] >= MIN_ABS_CORR]
        nbrs.append(idx)
    edges = []
    seen = set()
    for i in range(n):
        for j in nbrs[i]:
            if i not in nbrs[j]: continue
            a, b = sorted((i, j))
            if (a, b) in seen: continue
            seen.add((a, b))
            edges.append((a, b, float(corr[a, b]), float(abs_corr[a, b])))
    return edges

def build_partition(edges, n_nodes):
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    for a, b, corr_val, abs_corr in edges:
        graph.add_edge(a, b, weight=abs_corr, corr=corr_val)
    part = community_louvain.best_partition(graph, weight="weight", random_state=42)
    module_members = defaultdict(list)
    for node, module_id in part.items():
        module_members[module_id].append(node)
    return graph, part, module_members

def stability_scores(Xz, base_modules):
    rng = np.random.default_rng(42)
    base_sets = {m: set(nodes) for m, nodes in base_modules.items()}
    n_samples = Xz.shape[0]
    out = defaultdict(list)
    for boot_idx in range(N_BOOT):
        take = rng.choice(n_samples, size=int(n_samples * BOOT_FRACTION), replace=False)
        corr = correlation_matrix(Xz[take])
        edges = sparse_edges_from_corr(corr)
        _, _, boot_modules = build_partition(edges, Xz.shape[1])
        boot_sets = [set(nodes) for nodes in boot_modules.values()]
        for module_id, base_set in base_sets.items():
            if not base_set: continue
            best = 0.0
            for boot_set in boot_sets:
                inter = len(base_set & boot_set)
                union = len(base_set | boot_set)
                if union: best = max(best, inter / union)
            out[module_id].append(best)
    return out

def eig_module_scores(Xz, member_indices):
    Xm = Xz[:, member_indices]
    u, s, vt = np.linalg.svd(Xm, full_matrices=False)
    loadings = vt[0].astype(np.float32)
    scores = Xm @ loadings
    if np.nanmean(loadings) < 0:
        loadings *= -1.0
        scores *= -1.0
    return scores.astype(np.float32), loadings

# --- Run atlas construction ---

# Use the rebuilt ref_matrix (from cached .npy files, not shipped .npz)
ref_pgs_ids = ref_pgs_ids_rebuild
X_ref = X_ref_rebuilt
log(f'  Reference panel: {X_ref.shape[0]} samples × {X_ref.shape[1]} PGS (REBUILT from cache)')

Xz = winsorize_and_standardize(X_ref)
log(f'  Winsorized + standardized')

corr = correlation_matrix(Xz)
log(f'  Correlation matrix: {corr.shape}')

edges = sparse_edges_from_corr(corr)
log(f'  Edges: {len(edges)}')

graph, part, module_members = build_partition(edges, len(ref_pgs_ids))
log(f'  Louvain communities: {len(module_members)}')

stability = stability_scores(Xz, module_members)
log(f'  Bootstrap stability done ({N_BOOT} iterations)')

# Retain stable modules
retained = {}
for mod_id, members in module_members.items():
    stab = stability.get(mod_id, [])
    mean_j = np.mean(stab) if stab else 0
    if len(members) >= RETAIN_MIN_SIZE and mean_j >= RETAIN_MIN_JACCARD:
        _, loadings = eig_module_scores(Xz, members)
        retained[mod_id] = {
            'members': members,
            'pgs_ids': [ref_pgs_ids[i] for i in members],
            'loadings': loadings.tolist(),
            'size': len(members),
            'mean_jaccard': round(float(mean_j), 3),
        }

log(f'  Retained modules: {len(retained)}')
log(f'  PGS covered: {sum(m["size"] for m in retained.values())}')

# ═══════════════════════════════════════
# STEP 2: SCORE PATIENTS WITH ATLAS
# ═══════════════════════════════════════
log('\n[2/7] Scoring patients with rebuilt atlas')
log('  (exact same pipeline as bundle_runtime.py:compute_submodule_scores)')

# manifest already loaded in step 0

# --- Helper functions (copied from bundle_runtime.py) ---

def weighted_projection(matrix, weights):
    """L2-normalized quality-weighted projection with NaN masking."""
    mask = np.isfinite(matrix)
    numerator = np.nansum(matrix * weights[None, :], axis=1)
    denom = np.sqrt(np.nansum((mask * weights[None, :]) ** 2, axis=1))
    score = np.divide(numerator, denom,
        out=np.full(matrix.shape[0], np.nan, dtype=np.float32), where=denom > 0)
    abs_w = np.abs(weights)
    total_abs_w = float(abs_w.sum())
    reliability = np.divide(np.nansum(mask * abs_w[None, :], axis=1), total_abs_w,
        out=np.zeros(matrix.shape[0], dtype=np.float32), where=total_abs_w > 0)
    available = mask.sum(axis=1).astype(np.int32)
    return score.astype(np.float32), reliability.astype(np.float32), available

def fit_linear_residualizer(y, pcs10):
    x = np.column_stack([np.ones(pcs10.shape[0], dtype=np.float32), pcs10])
    beta, _, _, _ = np.linalg.lstsq(x, y.astype(np.float64), rcond=None)
    pred = x @ beta
    resid = y - pred
    return {'intercept': float(beta[0]), 'coef': [float(v) for v in beta[1:]],
            'pred': pred.astype(np.float32), 'resid': resid.astype(np.float32)}

def predict_residualized(y, pcs10, fit):
    coef = np.asarray(fit['coef'], dtype=np.float32)
    pred = fit['intercept'] + pcs10 @ coef
    return (y - pred).astype(np.float32)

def robust_center_scale(values):
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0: return 0.0, 1.0
    center = float(np.median(vals))
    mad = float(np.median(np.abs(vals - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-8:
        scale = float(np.std(vals, ddof=1)) if vals.size > 1 else 1.0
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    return center, scale

# --- Load quality scores (same as bundle_runtime.build_quality_scale_map) ---
quality_scale = {}
with open(OUT / 'quality_scores.tsv') as f:  # generated by pgs_quality.py in REBUILD step
    qrows = list(csv.DictReader(f, delimiter='\t'))
max_q = max(float(r['total']) for r in qrows if r.get('total'))
for r in qrows:
    t = float(r['total']) if r.get('total') else max_q * 0.5
    quality_scale[r['pgs_id']] = t / max_q

log(f'  Quality scores: {len(quality_scale)} PGS (max={max_q:.1f})')

# --- Build module weights: eig_loading * quality_scale (same as bundle_runtime) ---
module_weights = {}
for mod_id, mod in retained.items():
    weights = []
    for pgs_id, loading in zip(mod['pgs_ids'], mod['loadings']):
        q = quality_scale.get(pgs_id, 0.5)
        weights.append(loading * q)
    module_weights[mod_id] = {
        'pgs_ids': mod['pgs_ids'],
        'weights': np.array(weights, dtype=np.float32),
    }

# --- Load patient wide matrix ---
pgs_wide = pl.read_parquet(OUT / 'patient_pgs_z_wide.parquet')  # generated in step 0
pgs_cols = [c for c in pgs_wide.columns if c.startswith('PGS')]
pgs_wide = pgs_wide.filter(pl.col('barcode').is_in(list(manifest.keys())))
BARCODES = pgs_wide['barcode'].to_list()
LABELS = np.array(pgs_wide['label'].to_list())
patient_matrix = pgs_wide.select(pgs_cols).to_numpy().astype(np.float32)  # NaN preserved for weighted_projection
patient_matrix_clean = np.nan_to_num(patient_matrix, nan=0.0)  # for per-PGS AUC comparison
patient_matrix_clean = np.clip(patient_matrix_clean, -10, 10)
PGS_IDX = {pid: j for j, pid in enumerate(pgs_cols)}
N = len(BARCODES)

# --- Load reference PCs + patient PCs ---
import json as _json
ref_pcs_all = np.load(POLYGEN_REF / 'ref_pcs.npy').astype(np.float32)
ref_sample_order = _json.loads((POLYGEN_REF / 'ref_sample_ids.json').read_text())
# Align ref PCs to match ref_matrix sample order
ref_matrix_ids = ref_sample_ids_rebuild
ref_pc_idx = {sid: i for i, sid in enumerate(ref_sample_order)}
ref_pcs_take = np.array([ref_pc_idx[sid] for sid in ref_matrix_ids])
ref_pcs10 = ref_pcs_all[ref_pcs_take, :10]

anc_df = pl.read_csv(OUT / 'patient_ancestry.tsv', separator='\t')  # generated in step 0
anc_bc_map = {row['barcode']: row for row in anc_df.select(['barcode'] + [f'PC{i}' for i in range(1,11)]).to_dicts()}
patient_pcs10 = np.zeros((N, 10), dtype=np.float32)
for i, bc in enumerate(BARCODES):
    row = anc_bc_map.get(bc)
    if row:
        for j in range(10):
            try: patient_pcs10[i, j] = float(row[f'PC{j+1}'])
            except: pass

log(f'  Ref PCs: {ref_pcs10.shape}, Patient PCs: {patient_pcs10.shape}')

# --- Reference matrix index ---
ref_pgs_idx = {pid: i for i, pid in enumerate(ref_pgs_ids)}

# --- Score all modules (exact same loop as bundle_runtime.compute_submodule_scores) ---
module_keys = sorted(retained.keys())
N_modules = len(module_keys)

X_z = np.full((N, N_modules), np.nan, dtype=np.float32)
X_rel = np.zeros((N, N_modules), dtype=np.float32)

for mi, mod_id in enumerate(module_keys):
    mw = module_weights[mod_id]
    # Find PGS present in both patient and reference
    pgs_ids = [pid for pid in mw['pgs_ids'] if pid in PGS_IDX and pid in ref_pgs_idx]
    if not pgs_ids:
        continue
    # Rebuild aligned weights
    orig_idx = {pid: i for i, pid in enumerate(mw['pgs_ids'])}
    weights = np.array([mw['weights'][orig_idx[pid]] for pid in pgs_ids], dtype=np.float32)
    ref_take = [ref_pgs_idx[pid] for pid in pgs_ids]
    pat_take = [PGS_IDX[pid] for pid in pgs_ids]

    # 1. Weighted projection on reference (from already-residualized ref matrix)
    raw_ref, _, _ = weighted_projection(X_ref[:, ref_take], weights)

    # 2. Fit ancestry residualizer on reference projections
    fit = fit_linear_residualizer(raw_ref, ref_pcs10)
    resid_ref = fit['resid']

    # 3. Robust center + scale from reference residuals
    center, scale = robust_center_scale(resid_ref)

    # 4. Patient weighted projection + residualize + robust_z
    raw_pat, pat_rel, _ = weighted_projection(patient_matrix[:, pat_take], weights)
    resid_pat = predict_residualized(raw_pat, patient_pcs10, fit)
    z_pat = (resid_pat - center) / scale

    X_z[:, mi] = z_pat
    X_rel[:, mi] = pat_rel

X_z = np.nan_to_num(X_z, nan=0.0)
X_z = np.clip(X_z, -10, 10)

# Filter pathological modules: |robust_z| > 10 for >30% of patients (same as all scripts)
frac_extreme = (np.abs(X_z) > 10).mean(axis=0)
good_mask = frac_extreme < 0.3
good_idx = np.where(good_mask)[0]
GOOD_MODULES = [module_keys[i] for i in good_idx]
X_z = X_z[:, good_idx]
X_rel = X_rel[:, good_idx]
M = len(GOOD_MODULES)

log(f'  Patient scores: {N} × {N_modules} → {M} clean modules')

# ═══════════════════════════════════════
# STEP 3: BUILD FEATURES
# ═══════════════════════════════════════
log('\n[3/7] Building feature matrices')

anc_df = pl.read_csv(OUT / 'patient_ancestry.tsv', separator='\t')  # generated in step 0
anc_cols = ['RF_P_AFR','RF_P_AMR','RF_P_CSA','RF_P_EAS','RF_P_EUR','RF_P_MID']
anc_bc_idx = {b: i for i, b in enumerate(anc_df['barcode'].to_list())}
X_anc = np.zeros((N, 6), dtype=np.float32)
for i, bc in enumerate(BARCODES):
    ai = anc_bc_idx.get(bc)
    if ai is not None:
        for j, col in enumerate(anc_cols):
            try: X_anc[i, j] = float(anc_df[ai, col])
            except: pass

X_sex = np.array([1.0 if manifest[bc]['sex'] == 'M' else 0.0 for bc in BARCODES], dtype=np.float32).reshape(-1, 1)
X_age_raw = np.array([float(manifest[bc]['age']) for bc in BARCODES], dtype=np.float32)
age_mean, age_std = float(X_age_raw.mean()), float(X_age_raw.std())
X_age = ((X_age_raw - age_mean) / age_std).reshape(-1, 1)

X_full = np.hstack([X_z, X_rel, X_anc, X_sex, X_age])
X_genome = np.hstack([X_z, X_rel])
log(f'  Full: {X_full.shape}, Genome: {X_genome.shape}')

# ═══════════════════════════════════════
# STEP 4: EVALUATE
# ═══════════════════════════════════════
log('\n[4/7] Per-disease evaluation')

llm = pl.read_parquet(LLM_PARQUET)

# --- Build 3 PGS sets: ALL family-matched, CLEAN family-matched, ALL 5232 ---
FAMILY_PGS_ALL = defaultdict(list)    # original (all family-matched)
FAMILY_PGS_CLEAN = defaultdict(list)  # no proxy, no technical
ALL_PGS = [r['pgs_id'] for r in llm.select(['pgs_id']).to_dicts()]  # all 5232

clean_set = set()
for row in llm.select(['pgs_id','family_primary','proxy_trait_flag','technical_trait_flag']).to_dicts():
    if row['family_primary']:
        FAMILY_PGS_ALL[row['family_primary']].append(row['pgs_id'])
        if not row.get('proxy_trait_flag', False) and not row.get('technical_trait_flag', False):
            FAMILY_PGS_CLEAN[row['family_primary']].append(row['pgs_id'])
            clean_set.add(row['pgs_id'])

n_all = sum(len(v) for v in FAMILY_PGS_ALL.values())
n_clean = sum(len(v) for v in FAMILY_PGS_CLEAN.values())
log(f'  PGS sets: ALL family-matched={n_all}, CLEAN (no proxy/tech)={n_clean}, ALL catalog={len(ALL_PGS)}')

DISEASE_FAMILY = {'Coronary_Artery':'Coronary_Artery_Disease','Type2_Diabetes':'Type_2_Diabetes','Celiac':'Celiac_Disease','Ankylosing_Spondylitis':'Ankylosing_Spondylitis','Asthma':'Asthma','Atopic_Dermatitis':'Eczema','Parkinsons':'Parkinsons_Disease','Bipolar':'Bipolar_Disorder','Breast_Cancer':'Breast_Cancer','Colorectal_Cancer':'Colorectal_Cancer','Longevity':'Longevity'}

DISEASES = sorted([d for d in set(LABELS) if d != 'Healthy'])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = {}
models_final = {}
cv_probs = {d: np.zeros(N) for d in DISEASES}
cv_probs_genome = {d: np.zeros(N) for d in DISEASES}  # genome-only sensitivity
per_fold_aucs = {}
feature_importances = {}

log(f'\n{"Disease":25s} {"n":>4s} {"#All":>5s} {"#Cln":>5s} {"BstAll":>7s} {"BstCln":>7s} {"ModFull":>7s} {"BeatAll":>8s} {"BeatCln":>8s}')
log('-' * 95)

for disease in DISEASES:
    mask = (LABELS == disease) | (LABELS == 'Healthy')
    y = (LABELS[mask] == disease).astype(int)
    mask_idx = np.where(mask)[0]

    clf = RandomForestClassifier(500, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    aucs_full = cross_val_score(clf, X_full[mask], y, cv=cv, scoring='roc_auc')
    aucs_genome = cross_val_score(clf, X_genome[mask], y, cv=cv, scoring='roc_auc')
    mod_full = float(aucs_full.mean())
    mod_genome = float(aucs_genome.mean())
    per_fold_aucs[disease] = {'full': aucs_full.tolist(), 'genome': aucs_genome.tolist()}

    fold_importances = []
    for train_idx, test_idx in cv.split(X_full[mask], y):
        rf = RandomForestClassifier(500, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
        rf.fit(X_full[mask][train_idx], y[train_idx])
        probs = rf.predict_proba(X_full[mask][test_idx])[:, 1]
        for i, ti in enumerate(test_idx): cv_probs[disease][mask_idx[ti]] = probs[i]
        all_p = rf.predict_proba(X_full)[:, 1]
        for i in range(N):
            if not mask[i] and cv_probs[disease][i] == 0: cv_probs[disease][i] = all_p[i]
        fold_importances.append(rf.feature_importances_)

    # Genome-only CV probabilities (sensitivity analysis)
    for train_idx, test_idx in cv.split(X_genome[mask], y):
        rf_g = RandomForestClassifier(500, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
        rf_g.fit(X_genome[mask][train_idx], y[train_idx])
        probs_g = rf_g.predict_proba(X_genome[mask][test_idx])[:, 1]
        for i, ti in enumerate(test_idx): cv_probs_genome[disease][mask_idx[ti]] = probs_g[i]
        all_pg = rf_g.predict_proba(X_genome)[:, 1]
        for i in range(N):
            if not mask[i] and cv_probs_genome[disease][i] == 0: cv_probs_genome[disease][i] = all_pg[i]

    rf_final = RandomForestClassifier(500, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
    rf_final.fit(X_full[mask], y)
    models_final[disease] = rf_final
    feature_importances[disease] = {
        'final': rf_final.feature_importances_.tolist(),
        'per_fold': [fi.tolist() for fi in fold_importances],
        'mean_across_folds': np.mean(fold_importances, axis=0).tolist(),
    }

    fam = DISEASE_FAMILY.get(disease)

    def eval_pgs_set(pgs_list):
        res = []; best = 0.0
        for pid in pgs_list:
            j = PGS_IDX.get(pid)
            if j is None: continue
            x = patient_matrix_clean[mask, j]
            if np.std(x) < 1e-8: continue
            try:
                a = roc_auc_score(y, x); a = max(a, 1-a)
                res.append((pid, float(a)))
                if a > best: best = a
            except: pass
        beaten = sum(1 for _, a in res if mod_full > a)
        return res, best, beaten

    all_res, best_all, beaten_all = eval_pgs_set(FAMILY_PGS_ALL.get(fam, []))
    cln_res, best_cln, beaten_cln = eval_pgs_set([pid for pid in FAMILY_PGS_CLEAN.get(fam, []) if pid in PGS_IDX])

    results[disease] = {
        'n': int(y.sum()), 'mod_full': round(mod_full, 4), 'mod_genome': round(mod_genome, 4),
        'n_all': len(all_res), 'best_all': round(best_all, 4), 'beaten_all': beaten_all,
        'n_clean': len(cln_res), 'best_clean': round(best_cln, 4), 'beaten_clean': beaten_cln,
    }
    log(f'{disease:25s} {y.sum():4d} {len(all_res):5d} {len(cln_res):5d} {best_all:7.4f} {best_cln:7.4f} {mod_full:7.4f} {beaten_all}/{len(all_res)} {beaten_cln}/{len(cln_res)}')

tb_all = sum(r['beaten_all'] for r in results.values())
tt_all = sum(r['n_all'] for r in results.values())
tb_cln = sum(r['beaten_clean'] for r in results.values())
tt_cln = sum(r['n_clean'] for r in results.values())
log(f'\n  ALL family-matched: {tb_all}/{tt_all} ({100*tb_all/tt_all:.1f}%)')
log(f'  CLEAN (no proxy/tech): {tb_cln}/{tt_cln} ({100*tb_cln/tt_cln:.1f}%)')

# ═══════════════════════════════════════
# STEP 5: MULTI-CLASS
# ═══════════════════════════════════════
log('\n[5/7] Multi-class ranking')
c1 = c3 = td = 0
for bi in range(N):
    if LABELS[bi] == 'Healthy': continue
    probs = {d: cv_probs[d][bi] for d in DISEASES}
    ranked = sorted(probs, key=probs.get, reverse=True)
    c1 += int(ranked[0] == LABELS[bi]); c3 += int(LABELS[bi] in ranked[:3]); td += 1
log(f'  Full model:    Top-1: {c1}/{td} = {c1/td:.1%}, Top-3: {c3}/{td} = {c3/td:.1%}')

# Genome-only multi-class
g1 = g3 = gd = 0
for bi in range(N):
    if LABELS[bi] == 'Healthy': continue
    probs_g = {d: cv_probs_genome[d][bi] for d in DISEASES}
    ranked_g = sorted(probs_g, key=probs_g.get, reverse=True)
    g1 += int(ranked_g[0] == LABELS[bi]); g3 += int(LABELS[bi] in ranked_g[:3]); gd += 1
log(f'  Genome only:   Top-1: {g1}/{gd} = {g1/gd:.1%}, Top-3: {g3}/{gd} = {g3/gd:.1%}')

# ═══════════════════════════════════════
# STEP 6: PERMUTATION
# ═══════════════════════════════════════
N_PERM = 100
log(f'\n[6/9] Permutation test ({N_PERM} shuffles)')
perm_accs = []
for pi in range(N_PERM):
    pl2 = LABELS.copy(); np.random.seed(pi+100)
    dm = pl2 != 'Healthy'; dl = pl2[dm].copy(); np.random.shuffle(dl); pl2[dm] = dl
    pp = {}
    for disease in DISEASES:
        mask = (pl2==disease)|(pl2=='Healthy'); y2=(pl2[mask]==disease).astype(int); mi=np.where(mask)[0]
        p = np.zeros(N)
        for tri, tei in cv.split(X_full[mask], y2):
            rf = RandomForestClassifier(200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)
            rf.fit(X_full[mask][tri], y2[tri])
            pr = rf.predict_proba(X_full[mask][tei])[:, 1]
            for i, ti in enumerate(tei): p[mi[ti]] = pr[i]
            ap = rf.predict_proba(X_full)[:, 1]
            for i in range(N):
                if not mask[i] and p[i]==0: p[i]=ap[i]
        pp[disease] = p
    pc=pt=0
    for bi in range(N):
        if pl2[bi]=='Healthy': continue
        if max(pp,key=lambda d:pp[d][bi])==pl2[bi]: pc+=1
        pt+=1
    perm_accs.append(pc/pt)
    if (pi+1) % 10 == 0:
        log(f'  Shuffle {pi+1}/{N_PERM}: {pc}/{pt} = {pc/pt:.3f}  (running null mean={np.mean(perm_accs):.3f})')

real_acc=c1/td; perm_mean=np.mean(perm_accs); perm_std=max(np.std(perm_accs),0.001)
sigma=(real_acc-perm_mean)/perm_std
p_value = np.mean([pa >= real_acc for pa in perm_accs])
log(f'  Null distribution: mean={perm_mean:.4f}, std={perm_std:.4f}, range=[{min(perm_accs):.3f}, {max(perm_accs):.3f}]')
log(f'  Real: {real_acc:.4f}, Sigma: {sigma:.1f}, p < {max(p_value, 1/N_PERM):.4f}')

# ═══════════════════════════════════════
# STEP 7: PER-DISEASE AUC CONFIDENCE INTERVALS (bootstrap)
# ═══════════════════════════════════════
log('\n[7/9] Bootstrap 95% CIs for per-disease AUC (1000 resamples)')
N_BOOT_CI = 1000
rng_ci = np.random.default_rng(42)

for disease in DISEASES:
    mask_d = (LABELS == disease) | (LABELS == 'Healthy')
    y_d = (LABELS[mask_d] == disease).astype(int)
    probs_d = cv_probs[disease][mask_d]
    boot_aucs = []
    for _ in range(N_BOOT_CI):
        idx = rng_ci.choice(len(y_d), size=len(y_d), replace=True)
        if len(set(y_d[idx])) < 2: continue
        try: boot_aucs.append(roc_auc_score(y_d[idx], probs_d[idx]))
        except: pass
    lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
    results[disease]['auc_ci_lo'] = round(lo, 4)
    results[disease]['auc_ci_hi'] = round(hi, 4)
    log(f'  {disease:25s} AUC={results[disease]["mod_full"]:.4f} [{lo:.4f}, {hi:.4f}]')

# ═══════════════════════════════════════
# STEP 8: STRESS TEST — modules vs ALL 5,232 PGS (not just family-matched)
# ═══════════════════════════════════════
log('\n[8/11] STRESS TEST: modules vs ALL 5,232 PGS per disease')

pgs_to_trait = {r['pgs_id']: r.get('reported_trait','') for r in llm.select(['pgs_id','reported_trait']).to_dicts()}

stress_rows = []
for disease in DISEASES:
    mask_d = (LABELS == disease) | (LABELS == 'Healthy')
    y_d = (LABELS[mask_d] == disease).astype(int)
    mod_auc = results[disease]['mod_full']
    n_beat = 0; n_total = 0; best_any = 0.0; best_any_id = ''
    for pid in ALL_PGS:
        j = PGS_IDX.get(pid)
        if j is None: continue
        x = patient_matrix_clean[mask_d, j]
        if np.std(x) < 1e-8: continue
        try:
            a = roc_auc_score(y_d, x); a = max(a, 1-a)
            n_total += 1
            if mod_auc > a: n_beat += 1
            if a > best_any: best_any = a; best_any_id = pid
            stress_rows.append({'disease': disease, 'pgs_id': pid,
                'trait': pgs_to_trait.get(pid,''), 'pgs_auc': round(a,4),
                'module_auc': round(mod_auc,4), 'margin': round(mod_auc-a,4)})
        except: pass
    results[disease]['stress_n'] = n_total
    results[disease]['stress_beaten'] = n_beat
    results[disease]['stress_best_pgs'] = round(best_any, 4)
    results[disease]['stress_best_id'] = best_any_id
    beat_pct = 100*n_beat/n_total if n_total else 0
    log(f'  {disease:25s} beat {n_beat}/{n_total} ({beat_pct:.1f}%)  best_any={best_any:.4f} ({best_any_id})  module={mod_auc:.4f}  {"WIN" if mod_auc > best_any else "LOSS"}')

stress_wins = sum(1 for d in DISEASES if results[d]['mod_full'] > results[d]['stress_best_pgs'])
log(f'\n  Module beats BEST of ALL 5,232 PGS: {stress_wins}/{len(DISEASES)} diseases')

# Save stress test CSV
stress_rows.sort(key=lambda r: (r['disease'], -r['margin']))
with open(OUT / 'stress_test_all5232.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['disease','pgs_id','trait','pgs_auc','module_auc','margin'])
    w.writeheader(); w.writerows(stress_rows)

# ═══════════════════════════════════════
# STEP 9: PER-PGS MARGINS (clean comparator)
# ═══════════════════════════════════════
log('\n[9/11] Per-PGS margin analysis (CLEAN comparator)')

all_pgs_rows = []
cln_pgs_rows = []
for disease in DISEASES:
    fam = DISEASE_FAMILY.get(disease)
    mask_d = (LABELS == disease) | (LABELS == 'Healthy')
    y_d = (LABELS[mask_d] == disease).astype(int)
    mod_auc = results[disease]['mod_full']
    for pid in FAMILY_PGS_ALL.get(fam, []):
        j = PGS_IDX.get(pid)
        if j is None: continue
        x = patient_matrix_clean[mask_d, j]
        if np.std(x) < 1e-8: continue
        try:
            a = roc_auc_score(y_d, x); a = max(a, 1-a)
            row = {'disease': disease, 'pgs_id': pid, 'family': fam,
                'trait': pgs_to_trait.get(pid,''), 'pgs_auc': round(a,4),
                'module_auc': round(mod_auc,4), 'margin': round(mod_auc-a,4),
                'is_clean': pid in clean_set}
            all_pgs_rows.append(row)
            if pid in clean_set: cln_pgs_rows.append(row)
        except: pass

all_margins = [r['margin'] for r in all_pgs_rows]
cln_margins = [r['margin'] for r in cln_pgs_rows]
log(f'  ALL family-matched: {sum(1 for m in all_margins if m>0)}/{len(all_margins)} beaten, median margin {np.median(all_margins):+.4f}')
log(f'  CLEAN (no proxy/tech): {sum(1 for m in cln_margins if m>0)}/{len(cln_margins)} beaten, median margin {np.median(cln_margins):+.4f}')

log(f'\n  {"Disease":25s} {"ModAUC":>7s} {"CI":>17s} {"BstAll":>7s} {"MgnAll":>7s} {"BstCln":>7s} {"MgnCln":>7s}')
log(f'  {"-"*90}')
for disease in DISEASES:
    r = results[disease]
    ci = f'[{r.get("auc_ci_lo",0):.3f},{r.get("auc_ci_hi",0):.3f}]'
    log(f'  {disease:25s} {r["mod_full"]:7.4f} {ci:>17s} {r["best_all"]:7.4f} {r["mod_full"]-r["best_all"]:+7.4f} {r["best_clean"]:7.4f} {r["mod_full"]-r["best_clean"]:+7.4f}')

# Save CSVs
all_pgs_rows.sort(key=lambda r: (r['disease'], -r['margin']))
with open(OUT / 'per_pgs_comparison.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['disease','pgs_id','family','trait','pgs_auc','module_auc','margin','is_clean'])
    w.writeheader(); w.writerows(all_pgs_rows)

# ═══════════════════════════════════════
# STEP 10: ATLAS SENSITIVITY (K=10, K=20)
# ═══════════════════════════════════════
log('\n[10/11] Atlas sensitivity analysis')

for K_TEST in [10, 20]:
    edges_t = _sparse_edges_k(corr, K_TEST)
    _, _, mod_t = build_partition(edges_t, len(ref_pgs_ids))
    stab_t = stability_scores(Xz, mod_t)
    n_ret = sum(1 for mid, members in mod_t.items()
                if len(members) >= RETAIN_MIN_SIZE and np.mean(stab_t.get(mid, [0])) >= RETAIN_MIN_JACCARD)
    log(f'  K={K_TEST}: {len(mod_t)} communities, {n_ret} retained (vs K=15: {len(module_members)} communities, {len(retained)} retained)')

# ═══════════════════════════════════════
# STEP 11: SAVE
# ═══════════════════════════════════════
log('\n[11/11] Saving everything')

# --- 1. models.pkl (atlas + trained models) ---
bundle = {
    'atlas': {k: {kk: vv for kk, vv in v.items() if kk != 'members'} for k, v in retained.items()},
    'atlas_params': {'K': K_NEIGHBORS, 'MIN_CORR': MIN_ABS_CORR, 'WINSOR_Q': WINSOR_Q,
                     'N_BOOT': N_BOOT, 'BOOT_FRACTION': BOOT_FRACTION},
    'good_modules': GOOD_MODULES, 'n_retained': len(retained), 'n_clean': M,
    'models': models_final, 'diseases': DISEASES,
    'cv_probs': {d: cv_probs[d].tolist() for d in DISEASES},
    'barcodes': BARCODES, 'labels': LABELS.tolist(),
    'age_mean': age_mean, 'age_std': age_std,
}
with open(OUT / 'models.pkl', 'wb') as f: pickle.dump(bundle, f)
log(f'  models.pkl')

# --- 2. results.json (high-level summary) ---
results['multiclass'] = {'top1': round(c1/td,4), 'top3': round(c3/td,4)}
results['multiclass_genome'] = {'top1': round(g1/gd,4), 'top3': round(g3/gd,4)}
results['permutation'] = {'n_perm': N_PERM, 'sigma': round(sigma,1), 'p_value': round(max(p_value, 1/N_PERM),4),
                          'real': round(real_acc,4), 'null_mean': round(perm_mean,4), 'null_std': round(perm_std,4),
                          'all_null_accs': [round(a,4) for a in perm_accs]}
results['atlas'] = {'n_ref_samples': int(X_ref.shape[0]), 'n_pgs': int(X_ref.shape[1]),
                    'n_communities': len(module_members), 'n_retained': len(retained), 'n_clean': M,
                    'n_boot': N_BOOT}
results['metadata'] = {'n_patients': N, 'n_disease': td, 'n_healthy': N - td,
                       'features_full': int(X_full.shape[1]), 'features_genome': int(X_genome.shape[1]),
                       'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
results['margins_all'] = {'n_pgs': len(all_pgs_rows), 'beaten': sum(1 for m in all_margins if m>0),
                          'median': round(float(np.median(all_margins)),4)}
results['margins_clean'] = {'n_pgs': len(cln_pgs_rows), 'beaten': sum(1 for m in cln_margins if m>0),
                            'median': round(float(np.median(cln_margins)),4)}
results['stress_test'] = {'wins': stress_wins, 'total': len(DISEASES)}
with open(OUT / 'results.json', 'w') as f: json.dump(results, f, indent=2, default=str)
log(f'  results.json')

# --- 3. per_fold_aucs.json (per-disease, per-fold AUC scores) ---
with open(OUT / 'per_fold_aucs.json', 'w') as f: json.dump(per_fold_aucs, f, indent=2)
log(f'  per_fold_aucs.json')

# --- 4. feature_importances.json (per-disease RF importances) ---
feat_names = [f'z_{m}' for m in GOOD_MODULES] + [f'rel_{m}' for m in GOOD_MODULES] + \
             ['RF_P_AFR','RF_P_AMR','RF_P_CSA','RF_P_EAS','RF_P_EUR','RF_P_MID','sex','age']
fi_out = {'feature_names': feat_names}
for disease in DISEASES:
    fi = feature_importances[disease]
    fi_out[disease] = {
        'final': fi['final'],
        'mean_across_folds': fi['mean_across_folds'],
        'top_20': sorted([(feat_names[i], round(fi['final'][i],5)) for i in range(len(feat_names))],
                         key=lambda x: -x[1])[:20],
    }
with open(OUT / 'feature_importances.json', 'w') as f: json.dump(fi_out, f, indent=2)
log(f'  feature_importances.json')

# --- 5. patient_predictions.csv (per-patient, per-disease CV probabilities + ranking) ---
pred_rows = []
for bi in range(N):
    probs = {d: cv_probs[d][bi] for d in DISEASES}
    ranked = sorted(probs, key=probs.get, reverse=True)
    row = {'barcode': BARCODES[bi], 'true_label': LABELS[bi]}
    for d in DISEASES:
        row[f'prob_{d}'] = round(probs[d], 4)
    row['predicted_rank1'] = ranked[0]
    row['predicted_rank2'] = ranked[1]
    row['predicted_rank3'] = ranked[2]
    row['correct_rank1'] = int(ranked[0] == LABELS[bi]) if LABELS[bi] != 'Healthy' else ''
    row['correct_top3'] = int(LABELS[bi] in ranked[:3]) if LABELS[bi] != 'Healthy' else ''
    # Genome-only probabilities
    probs_g = {d: cv_probs_genome[d][bi] for d in DISEASES}
    ranked_g = sorted(probs_g, key=probs_g.get, reverse=True)
    for d in DISEASES:
        row[f'genome_prob_{d}'] = round(probs_g[d], 4)
    row['genome_rank1'] = ranked_g[0]
    row['genome_correct_rank1'] = int(ranked_g[0] == LABELS[bi]) if LABELS[bi] != 'Healthy' else ''
    row['genome_correct_top3'] = int(LABELS[bi] in ranked_g[:3]) if LABELS[bi] != 'Healthy' else ''
    pred_rows.append(row)
pred_fields = ['barcode','true_label'] + [f'prob_{d}' for d in DISEASES] + \
              ['predicted_rank1','predicted_rank2','predicted_rank3','correct_rank1','correct_top3'] + \
              [f'genome_prob_{d}' for d in DISEASES] + ['genome_rank1','genome_correct_rank1','genome_correct_top3']
with open(OUT / 'patient_predictions.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=pred_fields); w.writeheader(); w.writerows(pred_rows)
log(f'  patient_predictions.csv ({len(pred_rows)} patients)')

# --- 6. patient_module_scores.csv (per-patient module Z-scores + reliability) ---
mod_rows = []
for bi in range(N):
    row = {'barcode': BARCODES[bi], 'label': LABELS[bi]}
    for mi, mod_id in enumerate(GOOD_MODULES):
        row[f'z_{mod_id}'] = round(float(X_z[bi, mi]), 4)
        row[f'rel_{mod_id}'] = round(float(X_rel[bi, mi]), 4)
    mod_rows.append(row)
mod_fields = ['barcode','label'] + [f'{t}_{m}' for m in GOOD_MODULES for t in ['z','rel']]
with open(OUT / 'patient_module_scores.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=mod_fields); w.writeheader(); w.writerows(mod_rows)
log(f'  patient_module_scores.csv ({N} × {M*2} features)')

# --- 7. confusion_matrix.json (per-disease + multi-class) ---
confusion = {}
for disease in DISEASES:
    mask_d = (LABELS == disease) | (LABELS == 'Healthy')
    y_d = (LABELS[mask_d] == disease).astype(int)
    probs_d = cv_probs[disease][mask_d]
    preds_d = (probs_d >= 0.5).astype(int)
    tp = int(((preds_d == 1) & (y_d == 1)).sum())
    fp = int(((preds_d == 1) & (y_d == 0)).sum())
    tn = int(((preds_d == 0) & (y_d == 0)).sum())
    fn = int(((preds_d == 0) & (y_d == 1)).sum())
    confusion[disease] = {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
                          'sensitivity': round(tp/(tp+fn),4) if tp+fn else 0,
                          'specificity': round(tn/(tn+fp),4) if tn+fp else 0,
                          'ppv': round(tp/(tp+fp),4) if tp+fp else 0}
# Multi-class confusion
mc_matrix = {d1: {d2: 0 for d2 in DISEASES} for d1 in DISEASES}
for bi in range(N):
    if LABELS[bi] == 'Healthy': continue
    probs = {d: cv_probs[d][bi] for d in DISEASES}
    pred = max(probs, key=probs.get)
    mc_matrix[LABELS[bi]][pred] += 1
confusion['multiclass_matrix'] = mc_matrix
with open(OUT / 'confusion_matrix.json', 'w') as f: json.dump(confusion, f, indent=2)
log(f'  confusion_matrix.json')

# --- 8. atlas_modules.json (module definitions with PGS members) ---
atlas_out = {}
for mod_id, mod in retained.items():
    atlas_out[str(mod_id)] = {
        'size': mod['size'],
        'mean_jaccard': mod['mean_jaccard'],
        'pgs_ids': mod['pgs_ids'],
        'loadings': [round(l, 5) for l in mod['loadings']],
        'is_clean': mod_id in GOOD_MODULES,
    }
with open(OUT / 'atlas_modules.json', 'w') as f: json.dump(atlas_out, f, indent=2)
log(f'  atlas_modules.json ({len(retained)} modules)')

# --- 9. permutation_null.json (all null accuracies for plotting) ---
with open(OUT / 'permutation_null.json', 'w') as f:
    json.dump({'real_acc': round(real_acc, 4), 'null_accs': [round(a,4) for a in perm_accs],
               'sigma': round(sigma,1), 'n_perm': N_PERM}, f, indent=2)
log(f'  permutation_null.json ({N_PERM} shuffles)')

log(f'\n{"="*70}')
log(f'ROBUSTNESS SUMMARY')
log(f'{"="*70}')
log(f'  Atlas (N_BOOT={N_BOOT}): {len(retained)} retained → {M} clean')
log(f'  ALL family-matched PGS beaten: {tb_all}/{tt_all} ({100*tb_all/tt_all:.1f}%)')
log(f'  CLEAN PGS beaten (no proxy/tech): {tb_cln}/{tt_cln} ({100*tb_cln/tt_cln:.1f}%)')
log(f'  Stress test (vs ALL 5,232): module wins {stress_wins}/{len(DISEASES)} diseases')
log(f'  Multi-class: top-1={c1/td:.1%}, top-3={c3/td:.1%}')
log(f'  Permutation ({N_PERM} shuffles): {sigma:.1f}σ')
log('DONE')
