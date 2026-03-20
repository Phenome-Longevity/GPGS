#!/usr/bin/env python3
"""
Ancestry Runtime Module
Projects target sample into reference PCA space, classifies population,
and normalizes PGS scores by genetic ancestry.

Uses pre-computed FRAPOSA artifacts and cached reference PGS scores
for fast runtime execution (~3s vs ~130s with runtime PCA + scoring).

Usage (standalone):
    python ancestry.py \
        --store <genotype_store_dir> \
        --scores <raw_scores.tsv> \
        --cache ~/.polygen/ancestry_cache \
        --output <output_dir>

Called by run.sh as Phase 4 (optional, if ancestry cache exists).

Outputs:
    - ancestry.tsv:  population assignment + PCs
    - scores_adjusted.tsv:  raw + normalized PGS scores
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from scipy.stats import percentileofscore
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Increment when ref scoring logic changes (e.g., variant intersection fix)
# Stale per-sample ref caches auto-rebuild on version mismatch
REF_CACHE_VERSION = 3  # v3: ref store rebuilt from full reference (84M variants)

N_PCS = 10           # Total PCs to compute
N_PCS_CLASSIFY = 5   # PCs used for population classification (matches pgsc_calc)
N_PCS_NORMALIZE = 4  # PCs used for PGS normalization (matches pgsc_calc)
DIM_ONLINE = 40      # SVD dimensions for OADP

SCRIPT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ANCESTRY_CACHE = SCRIPT_DIR / 'data' / 'ancestry_cache'
DEFAULT_REF_PFILE = Path.home() / '.polygen' / 'reference' / 'GRCh38_HGDP+1kGP_ALL'


def flip_keys(keys):
    """Swap ref_hash and alt_hash in variant keys to match allele-swapped variants.

    Key layout: (chrom:8 | pos:28 | ref_hash:14 | alt_hash:14)
    Flipped:    (chrom:8 | pos:28 | alt_hash:14 | ref_hash:14)
    """
    ref_hash = (keys >> np.uint64(14)) & np.uint64(0x3FFF)
    alt_hash = keys & np.uint64(0x3FFF)
    chrom_pos = keys >> np.uint64(28)
    return (chrom_pos << np.uint64(28)) | (alt_hash << np.uint64(14)) | ref_hash


# ============================================================
# REFERENCE STORE (GPU-based reference scoring)
# ============================================================

# FNV1a constants (must match vcf_to_bin.py)
_FNV_OFFSET = 0x811c9dc5
_FNV_PRIME  = 0x01000193
_CHROM_MAP = {
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15,
    '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '22': 22,
    'X': 23, 'Y': 24, 'MT': 25, 'M': 25,
}


def _fnv1a(s: str) -> int:
    h = _FNV_OFFSET
    for c in s.encode():
        h ^= c
        h = (h * _FNV_PRIME) & 0xFFFFFFFF
    return h


def _encode_key(chrom: str, pos: int, ref: str, alt: str) -> int:
    chrom_clean = chrom[3:] if chrom.startswith('chr') else chrom
    ci = _CHROM_MAP.get(chrom_clean, 0)
    if ci == 0:
        return 0
    rh = _fnv1a(ref) & 0x3FFF
    ah = _fnv1a(alt) & 0x3FFF
    pos = min(pos, 0x0FFFFFFF)
    return (ci << 56) | (pos << 28) | (rh << 14) | ah


class ReferenceStore:
    """Lightweight index for GPU-based reference panel scoring.

    Stores sorted FNV1a keys mapped to pvar indices, enabling fast binary
    search + pgenlib random-access reads at scoring time. ~96MB on disk
    vs 106GB for a full dosage matrix.
    """

    REQUIRED = ['ref_keys.bin', 'ref_afreq_sorted.bin',
                'ref_dosage.uint8', 'ref_store_meta.json']

    def __init__(self, store_dir):
        d = Path(store_dir)
        self.store_dir = d
        self.keys = np.fromfile(str(d / 'ref_keys.bin'), dtype=np.uint64)
        self.n_variants = len(self.keys)
        with open(d / 'ref_store_meta.json') as f:
            self.meta = json.load(f)
        self.n_samples = self.meta['n_samples']

        # Memory-mapped dosage matrix: (n_variants, n_samples) uint8, sorted by key
        # Values: 0/1/2 = dosage, 255 = missing
        self.dosage_mmap = np.memmap(
            str(d / 'ref_dosage.uint8'), dtype=np.uint8, mode='r',
            shape=(self.n_variants, self.n_samples))

        # Alt allele frequencies in sorted key order (for mean imputation)
        self.afreq = np.fromfile(str(d / 'ref_afreq_sorted.bin'), dtype=np.float32)

    def close(self):
        pass  # memmap is cleaned up by garbage collector

    @staticmethod
    def exists(store_dir):
        d = Path(store_dir)
        return all((d / f).exists() for f in ReferenceStore.REQUIRED)

    @staticmethod
    def build(ref_pfile, store_dir, unrelated_ids):
        """One-time build of reference store from a plink2 pfile.

        Creates sorted keys, allele freqs, and a uint8 dosage matrix
        (n_variants × n_samples) stored as a memory-mapped file (~26GB
        for 8M variants × 3,330 samples).

        Args:
            ref_pfile: path prefix for .pgen/.pvar/.psam (may have .pvar.zst)
            store_dir: directory to write store files
            unrelated_ids: list of sample IDs used for scoring
        """
        import pgenlib

        d = Path(store_dir)
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        # --- Parse psam: map unrelated IDs to pfile sample indices ---
        psam_path = f'{ref_pfile}.psam'
        psam_ids = []
        with open(psam_path) as f:
            header = f.readline()
            has_fid = header.startswith('#FID')
            for line in f:
                parts = line.split('\t')
                iid = parts[1] if has_fid else parts[0]
                psam_ids.append(iid.strip())

        id_to_idx = {s: i for i, s in enumerate(psam_ids)}
        sample_idx = np.array([id_to_idx[s] for s in unrelated_ids
                               if s in id_to_idx], dtype=np.uint32)
        n_unrelated = len(sample_idx)
        print(f"    Mapped {n_unrelated}/{len(unrelated_ids)} "
              f"unrelated samples [{time.time()-t0:.1f}s]")

        # --- Parse pvar: encode variant keys ---
        pvar_path = f'{ref_pfile}.pvar'
        pvar_zst = f'{ref_pfile}.pvar.zst'
        if os.path.exists(pvar_zst) and not os.path.exists(pvar_path):
            print(f"    Decompressing {pvar_zst}...")
            subprocess.run(['plink2', '--zst-decompress', pvar_zst, pvar_path],
                           check=True, capture_output=True)

        print(f"    Encoding pvar keys...")
        t_pvar = time.time()
        keys = []
        n_lines = 0
        with open(pvar_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.split('\t', 5)
                chrom, pos = parts[0], parts[1]
                ref = parts[3]
                alt = parts[4].split('\t')[0].rstrip('\n')
                k = _encode_key(chrom, int(pos), ref, alt)
                keys.append(k)
                n_lines += 1
                if n_lines % 2_000_000 == 0:
                    print(f"      {n_lines/1e6:.0f}M variants...")

        keys = np.array(keys, dtype=np.uint64)
        n_variants = len(keys)
        print(f"    Encoded {n_variants:,} variant keys [{time.time()-t_pvar:.1f}s]")

        # --- Parse afreq ---
        afreq_path = f'{ref_pfile}.afreq'
        t_af = time.time()
        afreq = np.full(n_variants, 0.0, dtype=np.float32)
        if os.path.exists(afreq_path):
            i = 0
            with open(afreq_path) as f:
                f.readline()  # skip header
                for line in f:
                    afreq[i] = float(line.split('\t')[4])
                    i += 1
            print(f"    Loaded {i:,} allele frequencies [{time.time()-t_af:.1f}s]")

        # --- Sort by key ---
        sort_idx = np.argsort(keys)
        sorted_keys = keys[sort_idx]
        sorted_afreq = afreq[sort_idx]

        # --- Save keys and afreq (both in sorted order) ---
        sorted_keys.tofile(str(d / 'ref_keys.bin'))
        sorted_afreq.tofile(str(d / 'ref_afreq_sorted.bin'))

        # --- Build dosage matrix: read from pgenlib, write as uint8 memmap ---
        print(f"    Building dosage matrix ({n_variants:,} × {n_unrelated})...")
        matrix_size_gb = n_variants * n_unrelated / 1e9
        print(f"    Matrix size: {matrix_size_gb:.1f} GB")

        pgen_path = f'{ref_pfile}.pgen'
        pr = pgenlib.PgenReader(pgen_path.encode(),
                                sample_subset=sample_idx)

        # Create memory-mapped output file
        dosage_path = str(d / 'ref_dosage.uint8')
        mmap = np.memmap(dosage_path, dtype=np.uint8, mode='w+',
                         shape=(n_variants, n_unrelated))

        row_buf = np.empty(n_unrelated, dtype=np.float64)
        t_read = time.time()
        CHUNK = 100_000
        for chunk_start in range(0, n_variants, CHUNK):
            chunk_end = min(chunk_start + CHUNK, n_variants)
            for i in range(chunk_start, chunk_end):
                pvar_i = int(sort_idx[i])  # original pvar index for sorted position i
                pr.read_dosages(pvar_i, row_buf)
                # Encode: 0/1/2 → uint8, -9.0 → 255
                row_u8 = np.where(row_buf == -9.0, 255,
                                  np.clip(np.round(row_buf), 0, 2)).astype(np.uint8)
                mmap[i] = row_u8
            mmap.flush()
            elapsed_r = time.time() - t_read
            rate = (chunk_end) / elapsed_r if elapsed_r > 0 else 0
            print(f"      {chunk_end/1e6:.1f}M / {n_variants/1e6:.1f}M "
                  f"[{elapsed_r:.0f}s, {rate:.0f} var/s]")

        pr.close()
        del mmap
        print(f"    Dosage matrix built [{time.time()-t_read:.0f}s]")

        # --- Save metadata ---
        meta = {
            'n_variants': int(n_variants),
            'n_samples': int(n_unrelated),
            'n_samples_total': int(len(psam_ids)),
            'pgen_path': pgen_path,
            'ref_pfile': str(ref_pfile),
            'built': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        with open(d / 'ref_store_meta.json', 'w') as f:
            json.dump(meta, f, indent=2)

        elapsed = time.time() - t0
        print(f"    Reference store built: {n_variants:,} variants, "
              f"{n_unrelated} samples [{elapsed:.0f}s]")
        print(f"    Saved to {d}")
        return ReferenceStore(str(d))


def _score_ref_gpu(ref_store, weight_keys, weights, flip, model,
                   matched_target_keys=None):
    """Score one PGS across all reference samples using GPU.

    Args:
        ref_store: ReferenceStore instance
        weight_keys: uint64 array of PGS weight keys
        weights: float32 array of PGS weights
        flip: int8 array of flip flags
        model: int8 array of scoring model (0=add, 1=dom, 2=rec)
        matched_target_keys: set of uint64 keys that matched the target store.
            If provided, only score variants that matched both target and ref.

    Returns:
        np.ndarray of shape (n_samples,) with float64 scores
    """
    # Filter to target-matched keys if provided (searchsorted faster than np.isin)
    if matched_target_keys is not None:
        mtk = np.array(sorted(matched_target_keys), dtype=np.uint64)
        if len(mtk) == 0:
            return np.zeros(ref_store.n_samples, dtype=np.float64)
        idx = np.searchsorted(mtk, weight_keys)
        safe = np.minimum(idx, len(mtk) - 1)
        mask = (idx < len(mtk)) & (mtk[safe] == weight_keys)
        weight_keys = weight_keys[mask]
        weights = weights[mask]
        flip = flip[mask]
        model = model[mask]

    if len(weight_keys) == 0:
        return np.zeros(ref_store.n_samples, dtype=np.float64)

    # Binary search weight keys in ref store (sorted)
    indices = np.searchsorted(ref_store.keys, weight_keys)
    safe_indices = np.minimum(indices, ref_store.n_variants - 1)
    matched = (indices < ref_store.n_variants) & \
              (ref_store.keys[safe_indices] == weight_keys)

    # Dedup by store index (same as score.py)
    if np.any(matched):
        matched_wf_idx = np.where(matched)[0]
        matched_si = safe_indices[matched_wf_idx]
        _, unique_pos = np.unique(matched_si, return_index=True)
        dup_mask = np.ones(len(matched_wf_idx), dtype=bool)
        dup_mask[unique_pos] = False
        if np.any(dup_mask):
            matched[matched_wf_idx[dup_mask]] = False

    n_matched = matched.sum()
    if n_matched == 0:
        return np.zeros(ref_store.n_samples, dtype=np.float64)

    # Get matched data
    m_idx = np.where(matched)[0]
    m_weights = weights[m_idx].astype(np.float32)
    m_flip = flip[m_idx]
    m_model = model[m_idx]
    ref_sorted_idx = safe_indices[m_idx]  # indices into sorted ref store

    # Chunked scoring — process CHUNK variants at a time to bound memory.
    # For 2.57M matched × 3,330 samples, unchunked would need ~34GB float32.
    # With CHUNK=100k: peak ~330MB uint8 read + ~1.3GB float32 on GPU per chunk.
    CHUNK = 100_000
    n_samples = ref_store.n_samples

    try:
        import cupy as xp
        scores_acc = xp.zeros(n_samples, dtype=xp.float64)

        for start in range(0, n_matched, CHUNK):
            end = min(start + CHUNK, n_matched)
            chunk_ref_idx = ref_sorted_idx[start:end]

            # Read uint8 from memmap (4x less CPU memory than float32)
            chunk_uint8 = ref_store.dosage_mmap[chunk_ref_idx]

            # Transfer uint8 to GPU, convert to float32 there (fast)
            dos_gpu = xp.asarray(chunk_uint8).astype(xp.float32)
            del chunk_uint8

            # Mean impute missing (255 → 2*afreq) on GPU
            missing = dos_gpu == 255.0
            if xp.any(missing):
                afreqs = xp.asarray(ref_store.afreq[chunk_ref_idx])
                impute_vals = (2.0 * afreqs).reshape(-1, 1)
                dos_gpu = xp.where(missing, impute_vals, dos_gpu)

            w_gpu = xp.asarray(m_weights[start:end])
            fl = xp.asarray(m_flip[start:end])
            mo = xp.asarray(m_model[start:end])

            # Flip
            flip_mask = (fl == 1).reshape(-1, 1)
            dos_gpu = xp.where(flip_mask, 2.0 - dos_gpu, dos_gpu)

            # Dominant model
            dom_mask = (mo == 1).reshape(-1, 1)
            dos_gpu = xp.where(dom_mask, xp.minimum(dos_gpu, 1.0), dos_gpu)

            # Recessive model
            rec_mask = (mo == 2).reshape(-1, 1)
            dos_gpu = xp.where(rec_mask, xp.where(dos_gpu >= 2.0, 1.0, 0.0), dos_gpu)

            # Weighted sum per sample, accumulate
            scores_acc += (w_gpu[:, None] * dos_gpu).sum(axis=0)

        return xp.asnumpy(scores_acc)

    except ImportError:
        # CPU fallback (also chunked)
        scores_acc = np.zeros(n_samples, dtype=np.float64)

        for start in range(0, n_matched, CHUNK):
            end = min(start + CHUNK, n_matched)
            chunk_ref_idx = ref_sorted_idx[start:end]
            chunk_dos = ref_store.dosage_mmap[chunk_ref_idx].astype(np.float32)

            missing = chunk_dos == 255.0
            if np.any(missing):
                afreqs = ref_store.afreq[chunk_ref_idx]
                impute_vals = (2.0 * afreqs).reshape(-1, 1)
                chunk_dos = np.where(missing, impute_vals, chunk_dos)

            fl_np = m_flip[start:end].reshape(-1, 1)
            chunk_dos = np.where(fl_np == 1, 2.0 - chunk_dos, chunk_dos)
            mo_np = m_model[start:end].reshape(-1, 1)
            chunk_dos = np.where(mo_np == 1, np.minimum(chunk_dos, 1.0), chunk_dos)
            chunk_dos = np.where(mo_np == 2, np.where(chunk_dos >= 2.0, 1.0, 0.0), chunk_dos)

            scores_acc += (m_weights[start:end, None] * chunk_dos).sum(axis=0)

        return scores_acc


# ============================================================
# ANCESTRY CACHE
# ============================================================

class AncestryCache:
    """Loads pre-computed ancestry reference data: FRAPOSA SVD, RF model, ref scores."""

    REQUIRED_FILES = [
        'pca_keys.bin', 'ref_sample_ids.json',
        'pop_labels.json', 'meta.json',
        'fraposa_U.npy', 'fraposa_s.npy', 'fraposa_V.npy',
        'fraposa_mean.npy', 'fraposa_std.npy',
        'rf_model.joblib', 'ref_pcs.npy',
    ]

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self._validate()
        self._load()

    def _validate(self):
        missing = [f for f in self.REQUIRED_FILES
                   if not (self.cache_dir / f).exists()]
        if missing:
            print(f"ERROR: Ancestry cache incomplete. Missing: {missing}")
            print(f"Run precompute to build cache.")
            sys.exit(1)

    def _load(self):
        t0 = time.time()
        c = self.cache_dir

        # FRAPOSA SVD artifacts
        self.fraposa_U = np.load(str(c / 'fraposa_U.npy'))    # (n_variants, dim)
        self.fraposa_s = np.load(str(c / 'fraposa_s.npy'))    # (dim,)
        self.fraposa_V = np.load(str(c / 'fraposa_V.npy'))    # (n_samples, dim)
        self.fraposa_mean = np.load(str(c / 'fraposa_mean.npy'))  # (n_variants,)
        self.fraposa_std = np.load(str(c / 'fraposa_std.npy'))    # (n_variants,)

        # Reference PCs = V * s
        self.ref_pcs = np.load(str(c / 'ref_pcs.npy'))  # (n_samples, dim)

        # PCA variant keys for matching target genotypes
        self.keys = np.fromfile(str(c / 'pca_keys.bin'), dtype=np.uint64)

        with open(c / 'ref_sample_ids.json') as f:
            self.sample_ids = json.load(f)
        with open(c / 'pop_labels.json') as f:
            self.pop_labels = json.load(f)
        with open(c / 'meta.json') as f:
            self.meta = json.load(f)

        # RF model
        import joblib
        self.rf_model = joblib.load(c / 'rf_model.joblib')

        # Derived indices
        self.n_variants = len(self.keys)
        self.n_samples = len(self.sample_ids)
        self.populations = sorted(set(
            v for v in self.pop_labels.values() if v))
        self.pop_per_sample = np.array(
            [self.pop_labels.get(s, '') for s in self.sample_ids])
        # All PCA reference samples are unrelated (FRAPOSA convention)
        self.unrelated_mask = np.ones(self.n_samples, dtype=bool)

        # Reference score cache base directory (per-sample subdirs created later)
        self.ref_scores_base = c / 'ref_scores'
        self.ref_scores_base.mkdir(exist_ok=True)
        self.ref_scores_dir = self.ref_scores_base  # default; set_store_hash overrides

        print(f"  Ancestry cache: {self.n_variants:,} PCA variants, "
              f"{self.n_samples:,} ref samples ({time.time()-t0:.2f}s)")

    def set_store_hash(self, keys_md5: str):
        """Set per-sample ref score cache using the store's keys_md5.

        Different samples have different variant sets, producing different
        variant intersections with each PGS. Reference scores must be
        cached per-sample to ensure correct Z-normalization.
        """
        self.ref_scores_dir = self.ref_scores_base / keys_md5[:12]
        self.ref_scores_dir.mkdir(exist_ok=True)

        # Check cache version — invalidate stale ref scores
        version_file = self.ref_scores_dir / '_version'
        if version_file.exists():
            try:
                cached_ver = int(version_file.read_text().strip())
            except (ValueError, OSError):
                cached_ver = 0
        else:
            cached_ver = 0

        if cached_ver != REF_CACHE_VERSION:
            stale = list(self.ref_scores_dir.glob('*.npy'))
            if stale:
                print(f"  Ref cache v{cached_ver} stale (need v{REF_CACHE_VERSION}), "
                      f"clearing {len(stale)} cached scores")
                for f in stale:
                    f.unlink()
            version_file.write_text(str(REF_CACHE_VERSION))

    def get_cached_ref_scores(self, pgs_id: str):
        """Load cached reference scores for a PGS, or return None."""
        path = self.ref_scores_dir / f'{pgs_id}.npy'
        if path.exists():
            return np.load(str(path))
        return None

    def save_ref_scores(self, pgs_id: str, scores: np.ndarray):
        """Cache reference scores for a PGS."""
        path = self.ref_scores_dir / f'{pgs_id}.npy'
        np.save(str(path), scores)


# ============================================================
# FRAPOSA OADP MATH (Zhang et al. 2020)
# ============================================================

def _svd_online(U1, d1, V1, b):
    """Online SVD update: augment decomposition with new sample vector b."""
    n, k = V1.shape
    p = U1.shape[0]
    b = b.reshape((p, 1))
    b_tilde = b - U1 @ (U1.T @ b)
    b_tilde_norm = np.sqrt(np.sum(b_tilde ** 2))
    if b_tilde_norm > 1e-10:
        b_tilde = b_tilde / b_tilde_norm
    else:
        b_tilde = np.zeros_like(b_tilde)
        b_tilde[0] = 1.0

    R = np.concatenate([np.diag(d1), U1.T @ b], axis=1)
    R_tail = np.concatenate([np.zeros((1, k)), b_tilde.T @ b], axis=1)
    R = np.concatenate([R, R_tail], axis=0)

    d2, R_Vt = np.linalg.svd(R, full_matrices=False)[1:]
    V_new = np.zeros((k + 1, n + 1))
    V_new[:k, :n] = V1.T
    V_new[k, n] = 1
    V2 = (R_Vt @ V_new).T
    return d2, V2


def _procrustes(Y, X):
    """Find best rotation+scale from X to Y (same dimension)."""
    X = np.array(X, dtype=np.double, copy=True)
    Y = np.array(Y, dtype=np.double, copy=True)
    X_mean = X.mean(0)
    Y_mean = Y.mean(0)
    X -= X_mean
    Y -= Y_mean
    C = Y.T @ X
    U, s, VT = np.linalg.svd(C, full_matrices=False)
    trXX = np.sum(X ** 2)
    trS = np.sum(s)
    R = VT.T @ U.T
    rho = trS / trXX
    c = Y_mean - rho * X_mean @ R
    return R, rho, c


def _procrustes_diffdim(Y, X, n_iter_max=10000, epsilon_min=1e-6):
    """Procrustes alignment when X has more columns than Y."""
    X = np.array(X, dtype=np.double, copy=True)
    Y = np.array(Y, dtype=np.double, copy=True)
    n_X, p_X = X.shape
    n_Y, p_Y = Y.shape
    assert n_X == n_Y and p_X >= p_Y
    if p_X == p_Y:
        return _procrustes(Y, X)
    Z = np.zeros((n_X, p_X - p_Y))
    for _ in range(n_iter_max):
        W = np.hstack((Y, Z))
        R, rho, c = _procrustes(W, X)
        X_new = X @ R * rho + c
        Z_new = X_new[:, p_Y:]
        Z_new_centered = Z_new - Z_new.mean(0)
        Z_diff = Z_new - Z
        denom = np.sum(Z_new_centered ** 2)
        epsilon = np.sum(Z_diff ** 2) / denom if denom > 0 else 0
        if epsilon < epsilon_min:
            break
        Z = Z_new
    return R, rho, c


def _oadp(U, s, V, w, dim_ref=N_PCS, dim_stu=N_PCS*2, dim_online=DIM_ONLINE):
    """OADP: Online Augmentation, Decomposition, and Procrustes."""
    pcs_ref = V[:, :dim_ref] * s[:dim_ref]
    s_aug, V_aug = _svd_online(
        U[:, :dim_online], s[:dim_online], V[:, :dim_online], w)
    s_aug = s_aug[:dim_stu]
    V_aug = V_aug[:, :dim_stu]
    pcs_aug = V_aug * s_aug
    pcs_aug_head = pcs_aug[:-1, :]
    pcs_aug_tail = pcs_aug[-1:, :]
    R, rho, c = _procrustes_diffdim(pcs_ref, pcs_aug_head)
    pcs_new = pcs_aug_tail @ R * rho + c
    return pcs_new.flatten()[:dim_ref]


# ============================================================
# PCA PROJECTION — Cached FRAPOSA + OADP
# ============================================================

def project_target(store_dir: str, cache: AncestryCache):
    """
    Project target into reference PCA space using cached FRAPOSA artifacts + OADP.

    Loads target dosages from binary store, matches against cached PCA variant
    keys (exact + allele-flip), centers/scales with cached mean/std, and
    projects via OADP into the pre-computed reference PCA space.

    Returns: target_pcs (1, N_PCS)
    """
    store = Path(store_dir)
    target_keys = np.fromfile(str(store / 'keys.bin'), dtype=np.uint64)
    target_dosage = np.fromfile(str(store / 'dosage.bin'), dtype=np.float32)

    pca_keys = cache.keys
    n_pca = len(pca_keys)

    # --- Find matching variants: exact + allele-flip ---
    # Binary store keys are sorted, use searchsorted for fast lookup
    sort_idx = np.argsort(target_keys)
    sorted_target_keys = target_keys[sort_idx]

    # Exact match
    indices = np.searchsorted(sorted_target_keys, pca_keys)
    safe_idx = np.minimum(indices, len(sorted_target_keys) - 1)
    exact_mask = (indices < len(sorted_target_keys)) & \
                 (sorted_target_keys[safe_idx] == pca_keys)

    # Flip match (swap ref_hash and alt_hash)
    flipped_keys = flip_keys(pca_keys)
    flip_indices = np.searchsorted(sorted_target_keys, flipped_keys)
    flip_safe_idx = np.minimum(flip_indices, len(sorted_target_keys) - 1)
    flip_mask = (~exact_mask) & \
                (flip_indices < len(sorted_target_keys)) & \
                (sorted_target_keys[flip_safe_idx] == flipped_keys)

    n_exact = int(np.sum(exact_mask))
    n_flip = int(np.sum(flip_mask))
    n_found = n_exact + n_flip
    n_miss = n_pca - n_found
    overlap_pct = 100 * n_found / n_pca

    print(f"  PCA variants: {n_found:,}/{n_pca:,} ({overlap_pct:.1f}%) "
          f"[exact={n_exact:,} flip={n_flip:,} miss={n_miss:,}]")

    if n_found < 1000:
        print(f"  ERROR: Too few PCA variants matched (<1000). Cannot project.")
        return None

    # --- Build target dosage vector (all n_pca variants) ---
    # Missing variants default to 0 (hom ref), which after centering
    # becomes -mean/std (appropriate for WGS where absent = reference)
    w_target = np.zeros(n_pca, dtype=np.float64)

    # Exact matches: use dosage directly
    if n_exact > 0:
        orig_positions = sort_idx[safe_idx[exact_mask]]
        w_target[exact_mask] = target_dosage[orig_positions].astype(np.float64)

    # Flip matches: use 2 - dosage (alleles swapped)
    if n_flip > 0:
        flip_positions = sort_idx[flip_safe_idx[flip_mask]]
        w_target[flip_mask] = 2.0 - target_dosage[flip_positions].astype(np.float64)

    # --- Center and scale using cached reference mean/std ---
    w_std = (w_target - cache.fraposa_mean) / cache.fraposa_std
    w_std[~np.isfinite(w_std)] = 0.0

    # --- Project via OADP ---
    target_pcs = _oadp(cache.fraposa_U, cache.fraposa_s, cache.fraposa_V, w_std)

    print(f"  Target PCs: {' '.join(f'{v:.2f}' for v in target_pcs[:5])} ...")
    return target_pcs.reshape(1, -1)


# ============================================================
# POPULATION CLASSIFICATION — Cached RF model
# ============================================================

def classify_population(target_pcs: np.ndarray, cache: AncestryCache) -> dict:
    """
    Classify target population using cached RF model.

    Returns dict with pop_probs, most_similar, low_confidence.
    """
    pcs_for_rf = target_pcs[:, :N_PCS_CLASSIFY]
    probs = cache.rf_model.predict_proba(pcs_for_rf)[0]
    classes = cache.rf_model.classes_

    pop_probs = {cls: float(prob) for cls, prob in zip(classes, probs)}
    most_similar = classes[np.argmax(probs)]
    max_prob = float(np.max(probs))
    low_confidence = max_prob < 0.5

    print(f"  Population: {most_similar} (P={max_prob:.2f})")
    for pop in sorted(pop_probs.keys()):
        print(f"    {pop}: {pop_probs[pop]:.2f}")
    if low_confidence:
        print(f"  WARNING: Low confidence assignment")

    return {
        'pop_probs': pop_probs,
        'most_similar': most_similar,
        'low_confidence': low_confidence,
    }


# ============================================================
# REFERENCE PGS SCORING (target-filtered, parallel plink2)
# ============================================================

# --- Multiprocessing worker for batched target scoring ---
_mp_store_keys = None
_mp_ref_keys = None

def _mp_init_worker(store_keys_path, ref_keys_path=None):
    """Process pool initializer: load store keys (and ref keys) once per worker."""
    global _mp_store_keys, _mp_ref_keys
    _mp_store_keys = np.fromfile(store_keys_path, dtype=np.uint64)
    if ref_keys_path and os.path.exists(ref_keys_path):
        _mp_ref_keys = np.fromfile(ref_keys_path, dtype=np.uint64)
    else:
        _mp_ref_keys = None


def _mp_prepare_target_files(args):
    """Prepare filtered scoring files for one PGS (multiprocessing worker).

    Uses module-level _mp_store_keys and _mp_ref_keys loaded by _mp_init_worker.
    Filters to variants present in BOTH patient store AND reference panel,
    ensuring target and reference use identical variant sets for Z normalization.
    Returns list of (pgs_id, mode, filepath) tuples.
    """
    pgs_id, pgs_cache_dir, scoring_dir, tmpdir = args
    store_keys = _mp_store_keys
    ref_keys = _mp_ref_keys
    cache_path = Path(pgs_cache_dir)
    wf_path = cache_path / f'{pgs_id}.keys.bin'
    if not wf_path.exists():
        return []
    weight_keys = np.fromfile(str(wf_path), dtype=np.uint64)

    # Compute matched mask (vectorized numpy) — patient store intersection
    indices = np.searchsorted(store_keys, weight_keys)
    safe_indices = np.minimum(indices, len(store_keys) - 1)
    matched = (indices < len(store_keys)) & \
              (store_keys[safe_indices] == weight_keys)
    if np.any(matched):
        m_idx = np.where(matched)[0]
        m_si = safe_indices[m_idx]
        _, unique_pos = np.unique(m_si, return_index=True)
        dup_mask = np.ones(len(m_idx), dtype=bool)
        dup_mask[unique_pos] = False
        if np.any(dup_mask):
            matched[m_idx[dup_mask]] = False

    matched_uint64 = weight_keys[matched]

    # Intersect with reference store keys — only keep variants present in both
    # patient AND reference, matching how _score_ref_gpu computes ref scores
    if ref_keys is not None and len(matched_uint64) > 0:
        ri = np.searchsorted(ref_keys, matched_uint64)
        rs = np.minimum(ri, len(ref_keys) - 1)
        ref_hit = (ri < len(ref_keys)) & (ref_keys[rs] == matched_uint64)
        matched_uint64 = matched_uint64[ref_hit]

    # Build string set of matched key values for TSV filtering
    matched_keys_str = set(str(k) for k in matched_uint64.tolist())
    if not matched_keys_str:
        return []

    jobs = []
    scoring_path = Path(scoring_dir)
    for mode in ('additive', 'dominant', 'recessive'):
        suffix = '' if mode == 'additive' else f'.{mode}'
        src = scoring_path / f'{pgs_id}{suffix}.tsv'
        if not src.exists():
            continue
        dst = os.path.join(tmpdir, f'{pgs_id}_{mode}_score.tsv')
        n_kept = 0
        with open(src) as fin, open(dst, 'w') as fout:
            fin.readline()
            fout.write(f"ID\tA1\t{pgs_id}\n")
            for line in fin:
                key_str = line[line.rfind('\t') + 1:].rstrip('\n')
                if key_str in matched_keys_str:
                    fout.write(line[:line.rfind('\t')] + '\n')
                    n_kept += 1
        if n_kept > 0:
            jobs.append((pgs_id, mode, dst))
    return jobs


def _get_matched_weight_keys(store_dir: str, pgs_id: str, pgs_cache_dir: str,
                              store_keys: np.ndarray = None) -> set:
    """Determine which weight file keys actually matched the target store.

    Replicates score.py's matching: binary search weight keys against store keys,
    then deduplicates by store index. Returns the set of matched weight keys
    (uint64) so reference scoring uses the identical variant set.

    If store_keys is provided, uses it directly (avoids repeated disk reads).
    """
    if store_keys is None:
        store_keys = np.fromfile(str(Path(store_dir) / 'keys.bin'), dtype=np.uint64)
    cache = Path(pgs_cache_dir)
    wf_path = cache / f'{pgs_id}.keys.bin'
    if not wf_path.exists():
        return set()
    weight_keys = np.fromfile(str(wf_path), dtype=np.uint64)

    # Binary search (same as score.py score_cpu)
    indices = np.searchsorted(store_keys, weight_keys)
    safe_indices = np.minimum(indices, len(store_keys) - 1)
    matched = (indices < len(store_keys)) & (store_keys[safe_indices] == weight_keys)

    # Deduplicate by store index (same as score.py)
    if np.any(matched):
        matched_wf_idx = np.where(matched)[0]
        matched_si = safe_indices[matched_wf_idx]
        _, unique_pos = np.unique(matched_si, return_index=True)
        dup_mask = np.ones(len(matched_wf_idx), dtype=bool)
        dup_mask[unique_pos] = False
        if np.any(dup_mask):
            matched[matched_wf_idx[dup_mask]] = False

    return set(weight_keys[matched].tolist())


def score_reference(pgs_ids: list, ref_pfile: str, store_dir: str,
                    cache: AncestryCache, pgs_cache_dir: str = None,
                    target_pfile: str = None) -> tuple:
    """
    Score reference panel for each PGS, with caching.
    Also scores the target with plink2 using matched-weight-key filtered files,
    ensuring target and reference use the identical variant set for Z normalization.

    Returns (ref_scores, target_plink2_scores):
        ref_scores: {pgs_id: np.ndarray aligned to cache.sample_ids}
        target_plink2_scores: {pgs_id: float} or {} if no target_pfile
    """
    MAX_WORKERS = 4
    THREADS_PER = max(1, (os.cpu_count() or 8) // MAX_WORKERS)

    # Reference allele frequencies for mean imputation (matches pgsc_calc behavior)
    afreq_path = f'{ref_pfile}.afreq'
    has_afreq = os.path.exists(afreq_path)
    if not has_afreq:
        print(f"  WARNING: No afreq file at {afreq_path} — using no-mean-imputation")

    scoring_dir = None
    if pgs_cache_dir:
        scoring_dir = Path(pgs_cache_dir) / 'scoring_files'
    if not scoring_dir or not scoring_dir.exists():
        print(f"  WARNING: No pre-parsed scoring files found")
        return {}, {}

    t0 = time.time()
    results = {}
    target_plink2_scores = {}

    # Check cache first — load any pre-computed reference scores
    uncached_pgs = []
    for pgs_id in pgs_ids:
        cached = cache.get_cached_ref_scores(pgs_id)
        if cached is not None:
            results[pgs_id] = cached
        else:
            uncached_pgs.append(pgs_id)

    n_cached = len(pgs_ids) - len(uncached_pgs)
    if n_cached:
        print(f"  Loaded {n_cached} cached reference scores [{time.time()-t0:.1f}s]")

    # Score uncached PGS via GPU (with pgenlib) or plink2 fallback
    if uncached_pgs:
        print(f"  Scoring {len(uncached_pgs)} uncached PGS on reference panel...")

        # Check for GPU reference store — auto-build if missing
        ref_store_dir = str(cache.cache_dir / 'ref_store')
        if not ReferenceStore.exists(ref_store_dir):
            print(f"  Building GPU reference store (one-time, ~2 min)...")
            try:
                ReferenceStore.build(ref_pfile, ref_store_dir, cache.sample_ids)
            except Exception as e:
                print(f"  WARNING: GPU ref store build failed: {e}")
                print(f"  Falling back to plink2 --score-list...")

        use_gpu = False  # plink2 --score-list is 10x faster than GPU memmap for ref scoring

        if use_gpu:
            # --- GPU path: pgenlib + CuPy ---
            t_gpu = time.time()
            ref_store = ReferenceStore(ref_store_dir)
            print(f"  GPU ref store: {ref_store.n_variants:,} variants, "
                  f"{ref_store.n_samples} samples [{time.time()-t_gpu:.1f}s]",
                  flush=True)

            store_keys = np.fromfile(str(Path(store_dir) / 'keys.bin'), dtype=np.uint64)

            n_scored = 0
            cache_p = Path(pgs_cache_dir)
            for pgs_id in uncached_pgs:
                wf_path = cache_p / f'{pgs_id}.keys.bin'
                if not wf_path.exists():
                    continue
                wf_keys = np.fromfile(str(wf_path), dtype=np.uint64)
                wf_weights = np.fromfile(str(cache_p / f'{pgs_id}.weights.bin'), dtype=np.float32)
                flip_path = cache_p / f'{pgs_id}.flip.bin'
                wf_flip = np.fromfile(str(flip_path), dtype=np.int8) \
                    if flip_path.exists() else np.zeros(len(wf_keys), dtype=np.int8)
                model_path = cache_p / f'{pgs_id}.model.bin'
                wf_model = np.fromfile(str(model_path), dtype=np.int8) \
                    if model_path.exists() else np.zeros(len(wf_keys), dtype=np.int8)
                matched_target_keys = _get_matched_weight_keys(
                    store_dir, pgs_id, pgs_cache_dir, store_keys)
                scores = _score_ref_gpu(
                    ref_store, wf_keys, wf_weights, wf_flip, wf_model,
                    matched_target_keys)
                results[pgs_id] = scores
                n_scored += 1
            ref_store.close()
            print(f"  GPU ref scoring: {n_scored} PGS [{time.time()-t_gpu:.1f}s]",
                  flush=True)

        else:
            # --- plink2 --score-list path (reads compressed ref pgen sequentially) ---
            print(f"  Using plink2 --score-list for ref scoring...")

            tmpdir = tempfile.mkdtemp(prefix='polygen_refscore_')
            t_prep = time.time()
            store_keys_path = str(Path(store_dir) / 'keys.bin')
            mp_args = [(pid, pgs_cache_dir, str(scoring_dir), tmpdir)
                        for pid in uncached_pgs]

            ref_jobs = []
            n_workers = min(8, len(uncached_pgs))
            with ProcessPoolExecutor(max_workers=n_workers,
                                      initializer=_mp_init_worker,
                                      initargs=(store_keys_path,)) as pool:
                for result in pool.map(_mp_prepare_target_files, mp_args):
                    ref_jobs.extend(result)

            print(f"  Ref file prep: {len(ref_jobs)} files [{time.time()-t_prep:.1f}s]")

            by_mode = {}
            for pgs_id, mode, path in ref_jobs:
                by_mode.setdefault(mode, []).append((pgs_id, path))

            t_plink = time.time()
            n_cpus = os.cpu_count() or 8
            REF_WORKERS = min(4, n_cpus // 4)
            threads_per = max(2, n_cpus // REF_WORKERS)

            def _run_ref_chunk(chunk_args):
                chunk_id, mode, items, out_dir = chunk_args
                list_path = os.path.join(out_dir, f'ref_list_{mode}_{chunk_id}.txt')
                with open(list_path, 'w') as f:
                    for _, path in items:
                        f.write(f'{path}\n')

                out_prefix = os.path.join(out_dir, f'ref_out_{mode}_{chunk_id}')
                cmd = ['plink2', '--pfile', ref_pfile, 'vzs',
                       '--score-list', list_path, '1', '2', '3',
                       'header-read', 'cols=scoresums',
                       '--out', out_prefix,
                       '--threads', str(threads_per), '--memory', '8000']
                if has_afreq:
                    cmd += ['--read-freq', afreq_path]
                else:
                    cmd.insert(cmd.index('cols=scoresums'), 'no-mean-imputation')
                if mode == 'dominant':
                    cmd.insert(cmd.index('cols=scoresums') + 1, 'dominant')
                elif mode == 'recessive':
                    cmd.insert(cmd.index('cols=scoresums') + 1, 'recessive')

                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode != 0:
                    return mode, chunk_id, None, proc.stderr[:300]

                sscore = f'{out_prefix}.sscore'
                if not os.path.exists(sscore):
                    return mode, chunk_id, None, "no .sscore"

                chunk_results = {}
                with open(sscore) as f:
                    header = f.readline().strip().split('\t')
                    h0 = header[0].lstrip('#')
                    iid_col = 1 if (h0 == 'FID' and len(header) > 1
                                    and header[1] == 'IID') else 0
                    sum_cols = {}
                    for ci, h in enumerate(header):
                        h_clean = h.lstrip('#')
                        if h_clean.endswith('_SUM'):
                            sum_cols[h_clean[:-4]] = ci

                    score_maps = {pid: {} for pid in sum_cols}
                    for line in f:
                        fields = line.strip().split('\t')
                        sid = fields[iid_col].lstrip('#')
                        for pid, col_idx in sum_cols.items():
                            score_maps[pid][sid] = float(fields[col_idx])

                    for pid, smap in score_maps.items():
                        chunk_results[pid] = np.array(
                            [smap.get(sid, 0.0) for sid in cache.sample_ids],
                            dtype=np.float64)

                return mode, chunk_id, chunk_results, None

            chunk_args = []
            for mode, items in by_mode.items():
                chunk_size = max(1, (len(items) + REF_WORKERS - 1) // REF_WORKERS)
                for ci in range(0, len(items), chunk_size):
                    chunk = items[ci:ci + chunk_size]
                    chunk_args.append((ci // chunk_size, mode, chunk, tmpdir))

            with ThreadPoolExecutor(max_workers=REF_WORKERS) as executor:
                futures = [executor.submit(_run_ref_chunk, args) for args in chunk_args]
                for future in as_completed(futures):
                    mode, chunk_id, chunk_results, error = future.result()
                    if error:
                        print(f"  WARNING: ref chunk {mode}/{chunk_id} failed: {error}")
                    elif chunk_results:
                        for pid, scores in chunk_results.items():
                            if pid in results:
                                results[pid] += scores
                            else:
                                results[pid] = scores

            print(f"  Ref --score-list: {len(chunk_args)} chunks across "
                  f"{REF_WORKERS} workers [{time.time()-t_plink:.1f}s]")

            import shutil
            shutil.rmtree(tmpdir, ignore_errors=True)

        # Cache the newly computed reference scores
        n_cached_new = 0
        for pgs_id in uncached_pgs:
            if pgs_id in results:
                cache.save_ref_scores(pgs_id, results[pgs_id])
                n_cached_new += 1

        print(f"  Computed + cached {n_cached_new} reference scores "
              f"[{time.time()-t0:.1f}s]", flush=True)

    # --- Batched target scoring using plink2 --score-list ---
    # Instead of N sequential plink2 calls, we run one call per mode (additive/
    # dominant/recessive). plink2 loads the target pfile once and scores all PGS.
    use_plink2_target = target_pfile and os.path.exists(f'{target_pfile}.pgen')
    pgs_need_target = [p for p in pgs_ids
                       if p not in target_plink2_scores and p in results
                       and use_plink2_target]

    if pgs_need_target:
        t_tgt = time.time()
        tmpdir_tgt = tempfile.mkdtemp(prefix='polygen_batch_tgt_')

        # Parallel file prep using multiprocessing (avoids GIL for CPU-bound TSV filtering)
        store_keys_path = str(Path(store_dir) / 'keys.bin')
        ref_keys_path = os.path.join(str(cache.cache_dir / 'ref_store'), 'ref_keys.bin')
        if not os.path.exists(ref_keys_path):
            ref_keys_path = None
        mp_args = [(pid, pgs_cache_dir, str(scoring_dir), tmpdir_tgt)
                    for pid in pgs_need_target]

        tgt_jobs = []
        n_workers = min(8, len(pgs_need_target))
        with ProcessPoolExecutor(max_workers=n_workers,
                                  initializer=_mp_init_worker,
                                  initargs=(store_keys_path, ref_keys_path)) as pool:
            for result in pool.map(_mp_prepare_target_files, mp_args):
                tgt_jobs.extend(result)

        t_prep = time.time() - t_tgt
        print(f"  Target file prep: {len(tgt_jobs)} files [{t_prep:.1f}s]")

        # Group by mode, run one --score-list call per mode
        by_mode = {}
        for pgs_id, mode, path in tgt_jobs:
            by_mode.setdefault(mode, []).append((pgs_id, path))

        t_plink = time.time()
        for mode, items in by_mode.items():
            list_path = os.path.join(tmpdir_tgt, f'scorelist_{mode}.txt')
            with open(list_path, 'w') as f:
                for _, path in items:
                    f.write(f'{path}\n')

            out_prefix = os.path.join(tmpdir_tgt, f'batch_{mode}')
            cmd = ['plink2', '--pfile', target_pfile,
                   '--score-list', list_path, '1', '2', '3',
                   'header-read', 'no-mean-imputation', 'cols=scoresums',
                   '--out', out_prefix,
                   '--threads', str(os.cpu_count() or 8), '--memory', '8000']
            if mode == 'dominant':
                cmd.insert(cmd.index('cols=scoresums') + 1, 'dominant')
            elif mode == 'recessive':
                cmd.insert(cmd.index('cols=scoresums') + 1, 'recessive')

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  WARNING: batch {mode} target scoring failed: {result.stderr[:300]}")
                continue

            sscore = f'{out_prefix}.sscore'
            if not os.path.exists(sscore):
                continue

            with open(sscore) as f:
                header = f.readline().strip().split('\t')
                sum_cols = {}
                for i, h in enumerate(header):
                    h_clean = h.lstrip('#')
                    if h_clean.endswith('_SUM'):
                        sum_cols[h_clean[:-4]] = i
                line = f.readline().strip().split('\t')
                for pgs_id_col, col_idx in sum_cols.items():
                    score = float(line[col_idx])
                    target_plink2_scores[pgs_id_col] = \
                        target_plink2_scores.get(pgs_id_col, 0.0) + score

        print(f"  Batch target scoring: {len(target_plink2_scores)}/{len(pgs_ids)} PGS "
              f"[{time.time()-t_tgt:.1f}s]")

        import shutil
        shutil.rmtree(tmpdir_tgt, ignore_errors=True)

    if target_plink2_scores:
        print(f"  Target plink2 scores: {len(target_plink2_scores)}/{len(pgs_ids)} PGS")

    elapsed = time.time() - t0
    print(f"  Reference scoring: {len(results)}/{len(pgs_ids)} PGS [{elapsed:.1f}s]")
    return results, target_plink2_scores


# ============================================================
# PGS NORMALIZATION
# ============================================================

def normalize_pgs(target_score: float, target_pcs: np.ndarray,
                  ref_scores: np.ndarray, ref_pcs: np.ndarray,
                  cache: AncestryCache,
                  most_similar_pop: str) -> dict:
    """
    Normalize PGS score using ancestry, matching pgsc_calc methods:
        1. Z_MostSimilarPop — z-score vs population distribution
        2. percentile_MostSimilarPop — percentile in population
        3. Z_norm1 — PCA regression residual (Khera et al.)
        4. Z_norm2 — variance-normalized (Khan et al.)
    """
    results = {}

    pop_mask = cache.pop_per_sample == most_similar_pop
    train_mask = cache.unrelated_mask

    pop_scores = ref_scores[pop_mask & train_mask]

    # --- Method 1: Empirical Z-score vs population ---
    pop_mean = np.mean(pop_scores)
    pop_std = np.std(pop_scores, ddof=0)

    if pop_std > 0:
        results['Z_MostSimilarPop'] = float((target_score - pop_mean) / pop_std)
    else:
        results['Z_MostSimilarPop'] = 0.0

    # --- Method 2: Percentile vs population ---
    results['percentile_MostSimilarPop'] = float(
        percentileofscore(pop_scores, target_score))

    # --- Method 3: Z_norm1 (Khera) — regress PGS ~ PCs, normalize residual ---
    n_pcs_norm = min(N_PCS_NORMALIZE, ref_pcs.shape[1])
    ref_pcs_train = ref_pcs[train_mask, :n_pcs_norm]
    ref_scores_train = ref_scores[train_mask]

    pgs_mean = np.mean(ref_scores_train)
    ref_scores_centered = ref_scores_train - pgs_mean
    target_score_centered = target_score - pgs_mean

    pc_means = np.mean(ref_pcs_train, axis=0)
    pc_stds = np.std(ref_pcs_train, axis=0, ddof=0)
    pc_stds[pc_stds == 0] = 1.0

    ref_pcs_std = (ref_pcs_train - pc_means) / pc_stds
    target_pcs_std = (target_pcs[0, :n_pcs_norm] - pc_means) / pc_stds

    reg = LinearRegression()
    reg.fit(ref_pcs_std, ref_scores_centered)

    train_pred = reg.predict(ref_pcs_std)
    train_resid = ref_scores_centered - train_pred
    resid_std = np.std(train_resid, ddof=0)

    target_pred = reg.predict(target_pcs_std.reshape(1, -1))[0]
    target_resid = target_score_centered - target_pred

    if resid_std > 0:
        results['Z_norm1'] = float(target_resid / resid_std)
    else:
        results['Z_norm1'] = 0.0

    # --- Method 4: Z_norm2 (Khan) — variance normalization ---
    try:
        from sklearn.linear_model import GammaRegressor
        resid_mean = np.mean(train_resid)
        resid_sq = (train_resid - resid_mean) ** 2

        gamma_reg = GammaRegressor(max_iter=1000)
        gamma_reg.fit(ref_pcs_std, resid_sq)

        target_var_pred = gamma_reg.predict(target_pcs_std.reshape(1, -1))[0]

        if target_var_pred > 0:
            results['Z_norm2'] = float(target_resid / np.sqrt(target_var_pred))
        else:
            results['Z_norm2'] = results['Z_norm1']

    except Exception:
        results['Z_norm2'] = results['Z_norm1']

    return results


# ============================================================
# OUTPUT WRITERS
# ============================================================

def write_ancestry_tsv(output_dir: Path, sample_id: str, target_pcs: np.ndarray,
                       pop_result: dict):
    """Write ancestry.tsv — population assignment + PCs."""
    out_path = output_dir / 'ancestry.tsv'

    with open(out_path, 'w') as f:
        pc_cols = '\t'.join(f'PC{i+1}' for i in range(target_pcs.shape[1]))
        prob_cols = '\t'.join(f'RF_P_{p}' for p in sorted(pop_result['pop_probs'].keys()))
        f.write(f"sample_id\t{pc_cols}\t{prob_cols}\tMostSimilarPop\tLowConfidence\n")

        pc_vals = '\t'.join(f'{v:.4f}' for v in target_pcs[0])
        prob_vals = '\t'.join(f'{pop_result["pop_probs"][p]:.4f}'
                              for p in sorted(pop_result['pop_probs'].keys()))
        f.write(f"{sample_id}\t{pc_vals}\t{prob_vals}\t"
                f"{pop_result['most_similar']}\t{pop_result['low_confidence']}\n")

    print(f"  Ancestry: {out_path}")


def write_adjusted_scores(output_dir: Path, sample_id: str, pgs_results: list,
                          normalizations: dict):
    """Write scores_adjusted.tsv — raw + normalized scores."""
    out_path = output_dir / 'scores_adjusted.tsv'

    with open(out_path, 'w') as f:
        f.write("sample_id\tpgs_id\tSUM\tZ_MostSimilarPop\tZ_norm1\tZ_norm2\t"
                "percentile_MostSimilarPop\tMostSimilarPop\n")

        for result in pgs_results:
            pgs_id = result['pgs_id']
            norm = normalizations.get(pgs_id, {})

            def fmt(val, spec):
                if isinstance(val, (int, float)):
                    return f"{val:{spec}}"
                return str(val)

            f.write(f"{sample_id}\t{pgs_id}\t{result['raw_score']:.8f}\t"
                    f"{fmt(norm.get('Z_MostSimilarPop', 'NA'), '.6f')}\t"
                    f"{fmt(norm.get('Z_norm1', 'NA'), '.6f')}\t"
                    f"{fmt(norm.get('Z_norm2', 'NA'), '.6f')}\t"
                    f"{fmt(norm.get('percentile_MostSimilarPop', 'NA'), '.4f')}\t"
                    f"{norm.get('most_similar', 'NA')}\n")

    print(f"  Adjusted scores: {out_path}")


# ============================================================
# MAIN ANCESTRY PIPELINE
# ============================================================

def run_ancestry(store_dir: str, pgs_results: list, ancestry_cache_dir: str,
                 output_dir: str, ref_pfile: str = None,
                 pgs_cache_dir: str = None, target_pfile: str = None, **kwargs):
    """
    Full ancestry adjustment pipeline.

    Args:
        store_dir: target genotype binary store
        pgs_results: list of dicts from score.py (pgs_id, raw_score, ...)
        ancestry_cache_dir: pre-computed ancestry cache
        output_dir: output directory
        ref_pfile: path to reference pfile (for runtime scoring)
        pgs_cache_dir: PGS cache dir containing scoring_files/ for ref scoring
        target_pfile: path to target pfile prefix (for plink2-based normalization)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if ref_pfile is None:
        ref_pfile = str(DEFAULT_REF_PFILE)

    t_start = time.time()
    print(f"\n{'='*50}")
    print(f"  PolyGen Ancestry Adjustment")
    print(f"{'='*50}")

    # Load ancestry cache
    print(f"\n  Loading ancestry cache...")
    cache = AncestryCache(ancestry_cache_dir)

    # Get sample ID and store hash from store
    qc_path = Path(store_dir) / 'qc.json'
    with open(qc_path) as f:
        qc = json.load(f)
    sample_id = qc.get('sample_id', 'unknown')

    # Per-sample ref score cache (different samples have different variant sets)
    keys_md5 = qc.get('keys_md5', '')
    if keys_md5:
        cache.set_store_hash(keys_md5)
        print(f"  Ref score cache: {cache.ref_scores_dir}")

    # Step 1: FRAPOSA OADP projection (cached artifacts)
    print(f"\n  [1/4] PCA projection (FRAPOSA OADP)...")
    t0 = time.time()
    target_pcs = project_target(store_dir, cache)
    if target_pcs is None:
        print(f"  ERROR: PCA projection failed")
        return {}
    print(f"  Projection: {time.time()-t0:.2f}s")

    # Step 2: Population classification (cached RF)
    print(f"\n  [2/4] Population classification...")
    t0 = time.time()
    pop_result = classify_population(target_pcs, cache)
    print(f"  Classification: {time.time()-t0:.2f}s")

    write_ancestry_tsv(out, sample_id, target_pcs, pop_result)

    # Step 3: Score reference panel (filtered to target variants)
    print(f"\n  [3/4] Reference panel scoring (target-filtered)...")
    t0 = time.time()
    pgs_ids = [r['pgs_id'] for r in pgs_results]
    all_ref_scores, target_plink2_scores = score_reference(
        pgs_ids, ref_pfile, store_dir, cache,
        pgs_cache_dir=pgs_cache_dir, target_pfile=target_pfile)

    # Reference PCs for normalization
    ref_pcs = cache.ref_pcs

    # Step 4: Normalize each PGS
    print(f"\n  [4/4] PGS normalization...")
    normalizations = {}

    for result in pgs_results:
        pgs_id = result['pgs_id']

        ref_scores = all_ref_scores.get(pgs_id)
        if ref_scores is None:
            normalizations[pgs_id] = {
                'most_similar': pop_result['most_similar'],
            }
            continue

        # Use plink2 target score (ensures same variant set as reference)
        norm_score = target_plink2_scores.get(pgs_id, result['raw_score'])

        norm = normalize_pgs(
            target_score=norm_score,
            target_pcs=target_pcs,
            ref_scores=ref_scores,
            ref_pcs=ref_pcs,
            cache=cache,
            most_similar_pop=pop_result['most_similar'],
        )
        norm['most_similar'] = pop_result['most_similar']
        normalizations[pgs_id] = norm

        print(f"  {pgs_id}: Z_pop={norm['Z_MostSimilarPop']:.4f} "
              f"pctile={norm['percentile_MostSimilarPop']:.1f}% "
              f"Z1={norm['Z_norm1']:.4f} Z2={norm['Z_norm2']:.4f}")

    write_adjusted_scores(out, sample_id, pgs_results, normalizations)

    t_total = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"  Ancestry complete in {t_total:.1f}s")
    print(f"  Population: {pop_result['most_similar']}")
    print(f"  PGS normalized: {len(normalizations)}")
    print(f"{'='*50}")

    return normalizations


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PolyGen Ancestry Adjustment')
    parser.add_argument('--store', required=True,
                        help='Path to binary genotype store')
    parser.add_argument('--scores', required=True,
                        help='Path to raw scores.tsv from scorer')
    parser.add_argument('--cache', default=str(DEFAULT_ANCESTRY_CACHE),
                        help='Path to ancestry cache directory')
    parser.add_argument('--ref-pfile', default=str(DEFAULT_REF_PFILE),
                        help='Path to reference pfile prefix (for scoring)')
    parser.add_argument('--pgs-cache', default=None,
                        help='PGS cache dir (contains scoring_files/ for ref scoring)')
    parser.add_argument('--target-pfile', default=None,
                        help='Path to target pfile prefix (for plink2-based normalization)')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory')
    args = parser.parse_args()

    # Parse raw scores into result dicts
    pgs_results = []
    with open(args.scores) as f:
        header = f.readline().strip().split('\t')
        for line in f:
            fields = line.strip().split('\t')
            row = dict(zip(header, fields))
            pgs_results.append({
                'pgs_id': row['pgs_id'],
                'sample_id': row['sample_id'],
                'raw_score': float(row['raw_score']),
                'variants_matched': int(row['variants_matched']),
                'variants_total': int(row['variants_total']),
            })

    run_ancestry(
        store_dir=args.store,
        pgs_results=pgs_results,
        ancestry_cache_dir=args.cache,
        output_dir=args.output,
        ref_pfile=args.ref_pfile,
        pgs_cache_dir=args.pgs_cache,
        target_pfile=args.target_pfile,
    )
