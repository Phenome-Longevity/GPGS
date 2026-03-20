#!/usr/bin/env python3
"""
GPU PRS Scorer
Loads binary genotype store + binary weight files → computes PRS via GPU dot product.

Outputs:
  - scores.tsv: raw PRS per score
  - match_rates.tsv: variant match stats per score
"""

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path

# ============================================================
# GPU DETECTION
# ============================================================

try:
    import cupy as xp
    GPU_AVAILABLE = True
except ImportError:
    import numpy as xp
    GPU_AVAILABLE = False

# ============================================================
# GPU KERNELS
# ============================================================

if GPU_AVAILABLE:
    prs_score_kernel = xp.RawKernel(r'''
    extern "C" __global__
    void prs_score(
        const unsigned long long* sample_keys,
        const float* sample_dosage,
        const int sample_size,
        const unsigned long long* weight_keys,
        const float* weights,
        const int weight_size,
        float* result_dosage,   // matched dosages (for debugging/QC)
        int* result_matched     // 1 if matched, 0 if not
    ) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= weight_size) return;

        unsigned long long query = weight_keys[tid];

        // Binary search in sample_keys
        int lo = 0;
        int hi = sample_size;

        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (sample_keys[mid] < query) {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        if (lo < sample_size && sample_keys[lo] == query) {
            float d = sample_dosage[lo];
            // Check for NaN (missing genotype)
            if (d == d) {  // NaN != NaN
                result_dosage[tid] = d;
                result_matched[tid] = 1;
            } else {
                result_dosage[tid] = 0.0f;
                result_matched[tid] = 0;
            }
        } else {
            result_dosage[tid] = 0.0f;
            result_matched[tid] = 0;
        }
    }
    ''', 'prs_score')

    # ── V2 kernel: search + dedup + flip + model + multiply all on GPU ──
    prs_score_kernel_v2 = xp.RawKernel(r'''
    extern "C" __global__
    void prs_score_v2(
        const unsigned long long* sample_keys,
        const float* sample_dosage,
        const int sample_size,
        const unsigned long long* weight_keys,
        const float* weights,
        const signed char* flip,
        const signed char* model,
        const int weight_size,
        int* claimed,           // size = sample_size, init to 0
        float* contributions,   // output: per-weight score contribution
        int* result_matched     // output: 1 if matched and not dup
    ) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        if (tid >= weight_size) return;

        unsigned long long query = weight_keys[tid];

        // Binary search in sample_keys (sorted)
        int lo = 0;
        int hi = sample_size;
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (sample_keys[mid] < query) lo = mid + 1;
            else hi = mid;
        }

        if (lo < sample_size && sample_keys[lo] == query) {
            float d = sample_dosage[lo];

            // NaN check (missing genotype)
            if (d != d) {
                contributions[tid] = 0.0f;
                result_matched[tid] = 0;
                return;
            }

            // Dedup: first thread to claim this store position wins
            int old = atomicCAS(&claimed[lo], 0, 1);
            if (old != 0) {
                contributions[tid] = 0.0f;
                result_matched[tid] = 0;
                return;
            }

            // Flip: effect allele orientation
            if (flip[tid] == 1) d = 2.0f - d;

            // Scoring model
            signed char m = model[tid];
            if (m == 1) d = fminf(d, 1.0f);                    // dominant
            else if (m == 2) d = (d >= 2.0f) ? 1.0f : 0.0f;   // recessive

            contributions[tid] = d * weights[tid];
            result_matched[tid] = 1;
        } else {
            contributions[tid] = 0.0f;
            result_matched[tid] = 0;
        }
    }
    ''', 'prs_score_v2')


# ============================================================
# SCORING ENGINE
# ============================================================

class GenotypeStore:
    """Memory-mapped binary genotype store."""

    def __init__(self, store_dir: str):
        store = Path(store_dir)
        self.keys = np.fromfile(str(store / 'keys.bin'), dtype=np.uint64)
        self.dosage = np.fromfile(str(store / 'dosage.bin'), dtype=np.float32)

        with open(store / 'qc.json') as f:
            self.qc = json.load(f)

        self.sample_id = self.qc.get('sample_id', 'unknown')
        self.n_variants = len(self.keys)

        # Load rsID index for fallback matching
        rsid_hash_path = store / 'rsid_hash.bin'
        rsid_idx_path = store / 'rsid_idx.bin'
        if rsid_hash_path.exists() and rsid_idx_path.exists():
            self.rsid_hashes = np.fromfile(str(rsid_hash_path), dtype=np.uint64)
            self.rsid_indices = np.fromfile(str(rsid_idx_path), dtype=np.uint32)
            self.has_rsid = True
        else:
            self.rsid_hashes = None
            self.rsid_indices = None
            self.has_rsid = False

        # Pre-load to GPU if available
        if GPU_AVAILABLE:
            self.keys_gpu = xp.asarray(self.keys)
            self.dosage_gpu = xp.asarray(self.dosage)
        else:
            self.keys_gpu = self.keys
            self.dosage_gpu = self.dosage


class WeightFile:
    """Binary PGS weight file."""

    def __init__(self, pgs_id: str, cache_dir: str):
        cache = Path(cache_dir)
        self.pgs_id = pgs_id
        self.keys = np.fromfile(str(cache / f"{pgs_id}.keys.bin"), dtype=np.uint64)
        self.weights = np.fromfile(str(cache / f"{pgs_id}.weights.bin"), dtype=np.float32)

        flip_path = cache / f"{pgs_id}.flip.bin"
        if flip_path.exists():
            self.flip = np.fromfile(str(flip_path), dtype=np.int8)
        else:
            self.flip = np.zeros(len(self.keys), dtype=np.int8)

        model_path = cache / f"{pgs_id}.model.bin"
        if model_path.exists():
            self.model = np.fromfile(str(model_path), dtype=np.int8)
        else:
            self.model = np.zeros(len(self.keys), dtype=np.int8)

        # rsID fallback lookup
        rsid_path = cache / f"{pgs_id}.rsid_hash.bin"
        if rsid_path.exists():
            self.rsid_hashes = np.fromfile(str(rsid_path), dtype=np.uint64)
            self.rsid_weights = np.fromfile(str(cache / f"{pgs_id}.rsid_weight.bin"), dtype=np.float32)
            self.rsid_eff14 = np.fromfile(str(cache / f"{pgs_id}.rsid_eff14.bin"), dtype=np.uint16)
            self.rsid_oth14 = np.fromfile(str(cache / f"{pgs_id}.rsid_oth14.bin"), dtype=np.uint16)
            self.rsid_models = np.fromfile(str(cache / f"{pgs_id}.rsid_model.bin"), dtype=np.int8)
            self.has_rsid = True
        else:
            self.has_rsid = False

        with open(cache / f"{pgs_id}.meta.json") as f:
            self.meta = json.load(f)

        self.n_variants = len(self.keys)


def apply_dosage_model(dosages, flip, model):
    """Apply flip + dominant/recessive model to raw dosages.
    
    flip: 0=normal (effect=ALT), 1=flipped (effect=REF, use 2-dosage)
    model: 0=additive, 1=dominant (min(d,1)), 2=recessive (1 if d==2 else 0)
    """
    # First flip
    d = np.where(flip == 1, 2.0 - dosages, dosages)
    # Then apply model
    # Dominant: any copy of effect allele counts as 1
    d = np.where(model == 1, np.minimum(d, 1.0), d)
    # Recessive: only homozygous effect allele counts
    d = np.where(model == 2, np.where(d >= 2.0, 1.0, 0.0), d)
    return d


def rsid_fallback(store: GenotypeStore, wf: WeightFile, already_matched_keys: set) -> tuple:
    """
    rsID-based fallback matching for variants that didn't match by position+allele.
    Fully vectorized — no Python loops.

    Returns: (additional_score, additional_matched_count)
    """
    if not store.has_rsid or not wf.has_rsid:
        return 0.0, 0

    n = len(wf.rsid_hashes)
    if n == 0 or len(store.rsid_hashes) == 0:
        return 0.0, 0

    # Vectorized rsID lookup in store
    idx = np.searchsorted(store.rsid_hashes, wf.rsid_hashes)
    safe_idx = np.minimum(idx, len(store.rsid_hashes) - 1)
    found = (idx < len(store.rsid_hashes)) & (store.rsid_hashes[safe_idx] == wf.rsid_hashes)

    if not np.any(found):
        return 0.0, 0

    found_pos = np.where(found)[0]
    store_variant_idx = store.rsid_indices[safe_idx[found_pos]].astype(np.intp)
    store_key_vals = store.keys[store_variant_idx]

    # Filter out already-matched keys
    if already_matched_keys:
        already_arr = np.array(sorted(already_matched_keys), dtype=np.uint64)
        check_idx = np.searchsorted(already_arr, store_key_vals)
        safe_check = np.minimum(check_idx, len(already_arr) - 1)
        not_already = ~((check_idx < len(already_arr)) & (already_arr[safe_check] == store_key_vals))
    else:
        not_already = np.ones(len(found_pos), dtype=bool)

    if not np.any(not_already):
        return 0.0, 0

    # Apply not-already filter
    fp = found_pos[not_already]
    svi = store_variant_idx[not_already]
    skv = store_key_vals[not_already]

    # Get dosages, filter NaN
    dosages = store.dosage[svi]
    valid = ~np.isnan(dosages)
    if not np.any(valid):
        return 0.0, 0

    fp = fp[valid]
    svi = svi[valid]
    dosages = dosages[valid]

    # Extract allele hashes from store keys
    store_alt14 = (store.keys[svi].astype(np.int64)) & 0x3FFF
    store_ref14 = (store.keys[svi].astype(np.int64) >> 14) & 0x3FFF
    eff14 = wf.rsid_eff14[fp].astype(np.int64)
    oth14 = wf.rsid_oth14[fp].astype(np.int64)

    # Vectorized flip determination
    flip = np.zeros(len(fp), dtype=np.int8)
    flip = np.where(eff14 == store_alt14, 0, flip)       # effect = ALT
    flip = np.where(eff14 == store_ref14, 1, flip)       # effect = REF
    flip = np.where((eff14 != store_alt14) & (eff14 != store_ref14) & (oth14 == store_ref14), 0, flip)
    flip = np.where((eff14 != store_alt14) & (eff14 != store_ref14) & (oth14 == store_alt14), 1, flip)

    # Apply flip + model
    d = np.where(flip == 1, 2.0 - dosages, dosages)
    models = wf.rsid_models[fp]
    d = np.where(models == 1, np.minimum(d, 1.0), d)
    d = np.where(models == 2, np.where(d >= 2.0, 1.0, 0.0), d)

    weights = wf.rsid_weights[fp]
    extra_score = float(np.sum(d * weights))
    extra_matched = len(fp)

    return extra_score, extra_matched


def score_gpu(store: GenotypeStore, wf: WeightFile, _profile=None) -> dict:
    """Score a single PGS using GPU binary search + dot product."""
    t0 = time.time()

    t_upload_start = time.time()
    n_weights = wf.n_variants
    weight_keys_gpu = xp.asarray(wf.keys)
    weights_gpu = xp.asarray(wf.weights)

    result_dosage = xp.zeros(n_weights, dtype=xp.float32)
    result_matched = xp.zeros(n_weights, dtype=xp.int32)
    t_upload = time.time() - t_upload_start

    block_size = 256
    grid_size = (n_weights + block_size - 1) // block_size

    t_kernel_start = time.time()
    prs_score_kernel(
        (grid_size,), (block_size,),
        (store.keys_gpu, store.dosage_gpu, store.n_variants,
         weight_keys_gpu, weights_gpu, n_weights,
         result_dosage, result_matched)
    )
    xp.cuda.Device(0).synchronize()
    t_kernel = time.time() - t_kernel_start

    t_download_start = time.time()
    matched = xp.asnumpy(result_matched)
    dosages = xp.asnumpy(result_dosage)
    t_download = time.time() - t_download_start
# Deduplicate by store variant (CPU post-processing)
    if np.any(matched):
        store_idx = np.searchsorted(store.keys, wf.keys)
        safe_idx = np.minimum(store_idx, store.n_variants - 1)
        matched_wf_idx = np.where(matched)[0]
        matched_si = safe_idx[matched_wf_idx]
        _, unique_pos = np.unique(matched_si, return_index=True)
        dup_mask = np.ones(len(matched_wf_idx), dtype=bool)
        dup_mask[unique_pos] = False
        if np.any(dup_mask):
            matched[matched_wf_idx[dup_mask]] = False
    # Apply flip + dominant/recessive model
    adjusted = apply_dosage_model(dosages, wf.flip, wf.model)
    # Zero out unmatched
    adjusted = np.where(matched, adjusted, 0.0)

    raw_score = float(np.sum(adjusted * wf.weights))
    n_matched = int(np.sum(matched))

    # rsID fallback for unmatched variants
    n_rsid_matched = 0
    if store.has_rsid and wf.has_rsid:
        # Build set of store keys that were already matched
        matched_indices = np.where(matched)[0]
        matched_store_keys = set()
        for mi in matched_indices:
            # Find the store index for this weight key
            wk = wf.keys[mi]
            si = np.searchsorted(store.keys, wk)
            if si < store.n_variants and store.keys[si] == wk:
                matched_store_keys.add(int(store.keys[si]))

        extra_score, n_rsid_matched = rsid_fallback(store, wf, matched_store_keys)
        raw_score += extra_score
        n_matched += n_rsid_matched

    t_cpu = time.time() - t0 - t_upload - t_kernel - t_download
    n_total = wf.meta.get('num_original', n_weights)
    match_rate = n_matched / n_total if n_total > 0 else 0

    t_elapsed = time.time() - t0

    if _profile is not None:
        _profile['upload'] = _profile.get('upload', 0) + t_upload
        _profile['kernel'] = _profile.get('kernel', 0) + t_kernel
        _profile['download'] = _profile.get('download', 0) + t_download
        _profile['cpu_post'] = _profile.get('cpu_post', 0) + t_cpu
        _profile['n_calls'] = _profile.get('n_calls', 0) + 1
        _profile['total_weights'] = _profile.get('total_weights', 0) + n_weights

    return {
        'pgs_id': wf.pgs_id,
        'sample_id': store.sample_id,
        'raw_score': raw_score,
        'variants_total': n_total,
        'variants_matched': n_matched,
        'variants_missing': n_total - n_matched,
        'match_rate': round(match_rate, 6),
        'time_seconds': round(t_elapsed, 4),
    }


def score_gpu_v2(store: GenotypeStore, wf: WeightFile, _profile=None) -> dict:
    """Score using GPU kernel v2 — dedup + flip + model + multiply all on GPU.

    Only the final sum and rsID fallback remain on CPU.
    """
    t0 = time.time()

    n_weights = wf.n_variants

    # ── Upload weight data to GPU ──
    t_upload_start = time.time()
    weight_keys_gpu = xp.asarray(wf.keys)
    weights_gpu = xp.asarray(wf.weights)
    flip_gpu = xp.asarray(wf.flip)
    model_gpu = xp.asarray(wf.model)
    # Dedup array: one int per store variant, zeroed each call
    claimed = xp.zeros(store.n_variants, dtype=xp.int32)
    # Output arrays
    contributions = xp.zeros(n_weights, dtype=xp.float32)
    result_matched = xp.zeros(n_weights, dtype=xp.int32)
    t_upload = time.time() - t_upload_start

    # ── Launch kernel ──
    block_size = 256
    grid_size = (n_weights + block_size - 1) // block_size

    t_kernel_start = time.time()
    prs_score_kernel_v2(
        (grid_size,), (block_size,),
        (store.keys_gpu, store.dosage_gpu, store.n_variants,
         weight_keys_gpu, weights_gpu, flip_gpu, model_gpu, n_weights,
         claimed, contributions, result_matched)
    )

    # ── Reduce on GPU (no download needed!) ──
    raw_score = float(xp.sum(contributions))
    n_matched = int(xp.sum(result_matched))
    xp.cuda.Device(0).synchronize()
    t_kernel = time.time() - t_kernel_start

    # ── rsID fallback (CPU — vectorized where possible) ──
    t_rsid_start = time.time()
    n_rsid_matched = 0
    if store.has_rsid and wf.has_rsid:
        # Build matched_store_keys set (vectorized, no Python loop)
        matched_cpu = xp.asnumpy(result_matched)
        matched_wf_idx = np.where(matched_cpu)[0]
        if len(matched_wf_idx) > 0:
            matched_wkeys = wf.keys[matched_wf_idx]
            si = np.searchsorted(store.keys, matched_wkeys)
            safe_si = np.minimum(si, store.n_variants - 1)
            valid = (si < store.n_variants) & (store.keys[safe_si] == matched_wkeys)
            matched_store_keys = set(store.keys[safe_si[valid]].tolist())
        else:
            matched_store_keys = set()

        extra_score, n_rsid_matched = rsid_fallback(store, wf, matched_store_keys)
        raw_score += extra_score
        n_matched += n_rsid_matched
    t_rsid = time.time() - t_rsid_start

    n_total = wf.meta.get('num_original', n_weights)
    match_rate = n_matched / n_total if n_total > 0 else 0
    t_elapsed = time.time() - t0

    if _profile is not None:
        _profile['upload'] = _profile.get('upload', 0) + t_upload
        _profile['kernel+reduce'] = _profile.get('kernel+reduce', 0) + t_kernel
        _profile['rsid_fallback'] = _profile.get('rsid_fallback', 0) + t_rsid
        _profile['n_calls'] = _profile.get('n_calls', 0) + 1
        _profile['total_weights'] = _profile.get('total_weights', 0) + n_weights

    return {
        'pgs_id': wf.pgs_id,
        'sample_id': store.sample_id,
        'raw_score': raw_score,
        'variants_total': n_total,
        'variants_matched': n_matched,
        'variants_missing': n_total - n_matched,
        'match_rate': round(match_rate, 6),
        'time_seconds': round(t_elapsed, 4),
    }


def score_cpu(store: GenotypeStore, wf: WeightFile) -> dict:
    """Score a single PGS using vectorized CPU binary search."""
    t0 = time.time()

    # Vectorized searchsorted — single C call, no Python loop
    indices = np.searchsorted(store.keys, wf.keys)

    # Clamp indices to valid range for safe indexing
    safe_indices = np.minimum(indices, store.n_variants - 1)

    # Match: index in bounds AND key actually matches
    matched = (indices < store.n_variants) & (store.keys[safe_indices] == wf.keys)

    # Extract dosages for matched variants
    dosages = np.where(matched, store.dosage[safe_indices], 0.0).astype(np.float32)

    # Handle NaN dosages (missing genotypes)
    nan_mask = np.isnan(dosages)
    dosages = np.where(nan_mask, 0.0, dosages)
    matched = matched & ~nan_mask
    if np.any(matched):
        matched_wf_idx = np.where(matched)[0]
        matched_si = safe_indices[matched_wf_idx]
        _, unique_pos = np.unique(matched_si, return_index=True)
        dup_mask = np.ones(len(matched_wf_idx), dtype=bool)
        dup_mask[unique_pos] = False
        if np.any(dup_mask):
            matched[matched_wf_idx[dup_mask]] = False


    # Handle flipped alleles and dominant/recessive models
    adjusted = apply_dosage_model(dosages, wf.flip, wf.model)
    # Zero out unmatched
    adjusted = np.where(matched, adjusted, 0.0)

    raw_score = float(np.sum(adjusted * wf.weights))
    n_matched = int(np.sum(matched))

    # rsID fallback for unmatched variants
    n_rsid_matched = 0
    if store.has_rsid and wf.has_rsid:
        matched_store_keys = set()
        matched_wf_indices = np.where(matched)[0]
        for mi in matched_wf_indices:
            si = safe_indices[mi]
            if matched[mi]:
                matched_store_keys.add(int(store.keys[si]))

        extra_score, n_rsid_matched = rsid_fallback(store, wf, matched_store_keys)
        raw_score += extra_score
        n_matched += n_rsid_matched

    n_total = wf.meta.get('num_original', wf.n_variants)
    match_rate = n_matched / n_total if n_total > 0 else 0

    t_elapsed = time.time() - t0

    return {
        'pgs_id': wf.pgs_id,
        'sample_id': store.sample_id,
        'raw_score': raw_score,
        'variants_total': n_total,
        'variants_matched': n_matched,
        'variants_missing': n_total - n_matched,
        'match_rate': round(match_rate, 6),
        'time_seconds': round(t_elapsed, 4),
    }


def score_pgs(store: GenotypeStore, wf: WeightFile) -> dict:
    """Score using best available backend."""
    if GPU_AVAILABLE:
        return score_gpu(store, wf)
    else:
        return score_cpu(store, wf)


# ============================================================
# BATCH SCORING
# ============================================================

def score_batch(store_dir: str, cache_dir: str, pgs_ids: list, output_dir: str) -> list:
    """Score multiple PGS IDs against a genotype store."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  PolyGen GPU Scorer")
    print(f"{'='*50}")
    print(f"  GPU:     {'Yes' if GPU_AVAILABLE else 'No (CPU fallback)'}")
    print(f"  Store:   {store_dir}")
    print(f"  Scores:  {len(pgs_ids)}")

    # Load genotype store
    print(f"\n  Loading genotype store...")
    t0 = time.time()
    store = GenotypeStore(store_dir)
    t_load = time.time() - t0
    print(f"  Loaded {store.n_variants:,} variants in {t_load:.2f}s")
    print(f"  Sample: {store.sample_id}")

    # Score each PGS
    results = []
    t_total_score = 0
    t_total_io = 0
    profile = {} if GPU_AVAILABLE else None

    for pgs_id in pgs_ids:
        pgs_id = pgs_id.strip().upper()
        try:
            t_io_start = time.time()
            wf = WeightFile(pgs_id, cache_dir)
            t_io = time.time() - t_io_start
            t_total_io += t_io
        except FileNotFoundError:
            print(f"  {pgs_id}: weight files not found, skipping")
            continue

        if GPU_AVAILABLE:
            result = score_gpu_v2(store, wf, _profile=profile)
        else:
            result = score_cpu(store, wf)
        results.append(result)
        t_total_score += result['time_seconds']

        status = "✓" if result['match_rate'] > 0.5 else "⚠"
        print(f"  {status} {pgs_id}: score={result['raw_score']:.6f} "
              f"matched={result['variants_matched']}/{result['variants_total']} "
              f"({result['match_rate']*100:.1f}%) "
              f"in {result['time_seconds']:.4f}s")

    # Write scores.tsv
    scores_path = out / 'scores.tsv'
    with open(scores_path, 'w') as f:
        f.write("sample_id\tpgs_id\traw_score\tvariants_matched\tvariants_total\tmatch_rate\ttime_seconds\n")
        for r in results:
            f.write(f"{r['sample_id']}\t{r['pgs_id']}\t{r['raw_score']:.8f}\t"
                    f"{r['variants_matched']}\t{r['variants_total']}\t"
                    f"{r['match_rate']:.6f}\t{r['time_seconds']:.4f}\n")

    # Write match_rates.tsv
    match_path = out / 'match_rates.tsv'
    with open(match_path, 'w') as f:
        f.write("pgs_id\tvariants_total\tvariants_matched\tvariants_missing\tmatch_rate\n")
        for r in results:
            f.write(f"{r['pgs_id']}\t{r['variants_total']}\t{r['variants_matched']}\t"
                    f"{r['variants_missing']}\t{r['match_rate']:.6f}\n")

    print(f"\n  Total scoring time: {t_total_score:.4f}s")
    print(f"  --- Timing breakdown ---")
    print(f"  File I/O (WeightFile load): {t_total_io:.4f}s")
    if profile:
        print(f"  GPU upload (CPU→GPU):       {profile.get('upload',0):.4f}s")
        print(f"  GPU kernel+reduce:          {profile.get('kernel+reduce',0):.4f}s")
        print(f"  rsID fallback (CPU):        {profile.get('rsid_fallback',0):.4f}s")
        print(f"  Total weight keys:          {profile.get('total_weights',0):,}")
        print(f"  Kernel calls:               {profile.get('n_calls',0)}")
    print(f"  Results: {scores_path}")

    return results


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PolyGen GPU PRS Scorer')
    parser.add_argument('--store', '-s', required=True, help='Path to binary genotype store directory')
    parser.add_argument('--cache', '-c', required=True, help='Path to PGS weight cache directory')
    parser.add_argument('--pgsid', required=True, help='Comma-separated PGS IDs')
    parser.add_argument('--output', '-o', required=True, help='Output directory for results')
    args = parser.parse_args()

    pgs_ids = [x.strip() for x in args.pgsid.split(',')]
    score_batch(args.store, args.cache, pgs_ids, args.output)