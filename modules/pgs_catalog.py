#!/usr/bin/env python3
"""
PGS Catalog Sync
Downloads scoring files from PGS Catalog and converts to binary weight format:
  - <pgs_id>.weights.bin — array of (variant_key uint64, weight float32)
  - <pgs_id>.meta.json   — score metadata
"""

import os
import sys
import json
import gzip
import time
import struct
import hashlib
import argparse
import urllib.request
import numpy as np
from pathlib import Path

# Increment when matching logic changes to invalidate stale binary caches
CACHE_VERSION = 2  # v2: fix multi-valued hm_inferOtherAllele (set to None, not first value)

# Reuse the same key encoding as vcf_to_bin
CHROM_MAP = {
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15,
    '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '22': 22,
    'X': 23, 'Y': 24, 'MT': 25, 'M': 25,
}

COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
BASES = {'A', 'C', 'G', 'T'}


def load_target_pvar(pvar_path: str) -> dict:
    """Load target pvar into a lookup: {(chrom, pos): [(ref, alt), ...]}.

    Used for no_oa matching: instead of wildcard allele generation, look up
    actual REF/ALT at each position in the target genotypes (like pgsc_calc).
    """
    lookup = {}
    with open(pvar_path) as f:
        for line in f:
            if line.startswith('#'):
                continue
            fields = line.split('\t', 5)
            if len(fields) < 5:
                continue
            chrom = fields[0].lstrip('chr')
            try:
                pos = int(fields[1])
            except ValueError:
                continue
            ref = fields[3].upper()
            alt = fields[4].upper().split(',')[0]  # First ALT for multiallelic
            lookup.setdefault((chrom, pos), []).append((ref, alt))
    return lookup


def complement_allele(allele: str) -> str:
    """Return the strand complement of an allele."""
    return ''.join(COMPLEMENT.get(b, b) for b in allele.upper())


def fnv1a_hash(s: str) -> int:
    h = 0x811c9dc5
    for c in s.encode():
        h ^= c
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def encode_variant_key(chrom: str, pos: int, ref: str, alt: str) -> np.uint64:
    chrom_clean = chrom[3:] if chrom.startswith('chr') else chrom
    chrom_int = CHROM_MAP.get(chrom_clean, 0)
    if chrom_int == 0 or chrom_int > 22:  # Autosomes only (matches pgsc_calc --chr 1-22)
        return np.uint64(0)

    ref_hash = fnv1a_hash(ref) & 0x3FFF
    alt_hash = fnv1a_hash(alt) & 0x3FFF
    pos = min(pos, 0x0FFFFFFF)

    key = (int(chrom_int) << 56) | (int(pos) << 28) | (int(ref_hash) << 14) | int(alt_hash)
    return np.uint64(key)


# ============================================================
# PGS CATALOG API
# ============================================================

PGS_API_BASE = "https://www.pgscatalog.org/rest"


def fetch_pgs_metadata(pgs_id: str) -> dict:
    """Fetch score metadata from PGS Catalog REST API."""
    url = f"{PGS_API_BASE}/score/{pgs_id}"
    try:
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR fetching metadata for {pgs_id}: {e}")
        return None


def download_scoring_file(pgs_id: str, output_dir: Path) -> str:
    """Download scoring file from PGS Catalog FTP."""
    # Harmonized files (GRCh38)
    url = f"https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/{pgs_id}/ScoringFiles/Harmonized/{pgs_id}_hmPOS_GRCh38.txt.gz"

    local_path = output_dir / f"{pgs_id}_scoring.txt.gz"

    if local_path.exists():
        print(f"  Using cached: {local_path.name}")
        return str(local_path)

    print(f"  Downloading: {url}")
    try:
        urllib.request.urlretrieve(url, str(local_path))
        return str(local_path)
    except Exception as e:
        # Try non-harmonized
        url_fallback = f"https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/{pgs_id}/ScoringFiles/{pgs_id}.txt.gz"
        print(f"  Harmonized not found, trying: {url_fallback}")
        try:
            urllib.request.urlretrieve(url_fallback, str(local_path))
            return str(local_path)
        except Exception as e2:
            print(f"  ERROR downloading {pgs_id}: {e2}")
            return None


# ============================================================
# SCORING FILE PARSER
# ============================================================

def parse_scoring_file(scoring_path: str) -> list:
    """
    Parse PGS Catalog scoring file into list of (chrom, pos, effect_allele, other_allele, weight).
    Handles both harmonized and non-harmonized formats.
    """
    variants = []

    opener = gzip.open if scoring_path.endswith('.gz') else open

    with opener(scoring_path, 'rt') as f:
        header_cols = None
        col_map = {}

        for line in f:
            # Skip comment headers
            if line.startswith('#'):
                continue

            line = line.rstrip('\n\r')
            if not line.strip():
                continue

            # Parse column header
            if header_cols is None:
                header_cols = line.split('\t')
                if len(header_cols) == 1:
                    header_cols = line.split()

                for i, col in enumerate(header_cols):
                    col_map[col.lower()] = i

                print(f"  Columns ({len(header_cols)}): {header_cols[:8]}...")
                print(f"  Mapped: {list(col_map.keys())[:8]}...")
                continue

            # Parse data row
            fields = line.split('\t')
            if len(fields) == 1:
                fields = line.split()

            # Be lenient — don't require all columns, just the ones we need
            if len(fields) < 4:
                continue

            # Extract fields — try harmonized columns first, fall back to original
            chrom = None
            pos = None
            effect_allele = None
            other_allele = None
            weight = None

            def safe_get(idx):
                """Safely get field by index, return None if out of bounds."""
                if idx < len(fields):
                    val = fields[idx].strip()
                    return val if val and val not in ('.', '', 'NA', 'None') else None
                return None

            # Chromosome — prefer harmonized
            for col_name in ['hm_chr', 'chr_name', 'chromosome']:
                if col_name in col_map:
                    chrom = safe_get(col_map[col_name])
                    if chrom:
                        break

            # Position — prefer harmonized
            for col_name in ['hm_pos', 'chr_position', 'bp']:
                if col_name in col_map:
                    val = safe_get(col_map[col_name])
                    if val:
                        try:
                            pos = int(val)
                            break
                        except ValueError:
                            pos = None

            # Effect allele
            for col_name in ['effect_allele']:
                if col_name in col_map:
                    effect_allele = safe_get(col_map[col_name])
                    if effect_allele:
                        effect_allele = effect_allele.upper()

            # Other allele (reference context)
            for col_name in ['other_allele', 'reference_allele', 'hm_inferotherallele']:
                if col_name in col_map:
                    other_allele = safe_get(col_map[col_name])
                    if other_allele:
                        other_allele = other_allele.upper()
                        # Multi-valued inferred alleles (e.g. "A/C") mean it's ambiguous —
                        # treat as no other_allele so wildcard matching handles it
                        # (matches pgsc_calc's no_oa matching strategy)
                        if '/' in other_allele:
                            other_allele = None
                            continue
                        break

            # Fallback: parse other_allele from variant_description (format: chr:pos:REF:ALT)
            if not other_allele and 'variant_description' in col_map:
                vd = safe_get(col_map['variant_description'])
                if vd:
                    parts = vd.split(':')
                    if len(parts) >= 4:
                        vd_ref = parts[2].upper()
                        vd_alt = parts[3].upper()
                        # effect_allele should be one of REF or ALT
                        if effect_allele == vd_alt:
                            other_allele = vd_ref
                        elif effect_allele == vd_ref:
                            other_allele = vd_alt

            # Weight
            for col_name in ['effect_weight', 'weight', 'beta', 'or']:
                if col_name in col_map:
                    val = safe_get(col_map[col_name])
                    if val:
                        try:
                            weight = float(val)
                            if col_name == 'or' and weight > 0:
                                import math
                                weight = math.log(weight)
                            break
                        except ValueError:
                            weight = None

            # Dosage model (dominant / recessive / additive)
            is_dominant = False
            is_recessive = False
            if 'is_dominant' in col_map:
                dom_val = safe_get(col_map['is_dominant'])
                if dom_val and dom_val.lower() == 'true':
                    is_dominant = True
            if 'is_recessive' in col_map:
                rec_val = safe_get(col_map['is_recessive'])
                if rec_val and rec_val.lower() == 'true':
                    is_recessive = True

            if chrom and pos and effect_allele and weight is not None:
                # Get rsID if available
                rsid = None
                if 'rsid' in col_map:
                    rsid = safe_get(col_map['rsid'])
                if not rsid and 'hm_rsid' in col_map:
                    rsid = safe_get(col_map['hm_rsid'])

                variants.append((chrom, pos, effect_allele, other_allele, weight, is_dominant, is_recessive, rsid))

    return variants


# ============================================================
# CONVERT TO BINARY WEIGHTS
# ============================================================

def convert_to_binary(pgs_id: str, variants: list, output_dir: Path, metadata: dict = None,
                      target_pvar: dict = None):
    """
    Convert parsed scoring file to binary weight format.

    Produces:
      - <pgs_id>.keys.bin:    uint64 sorted variant keys
      - <pgs_id>.weights.bin: float32 weights (same order)
    """
    keys = []
    weights = []
    flip_flags = []
    model_flags = []
    # TSV lines parallel to keys/weights — same variants, same order
    tsv_lines = []  # (vid_str, effect_allele, weight, model, key_int)
    # rsID fallback: for variants that don't match by position+allele
    rsid_hashes = []     # uint64 FNV1a of rsID
    rsid_weights = []    # float32
    rsid_eff_h14 = []    # uint16: 14-bit hash of effect_allele (for flip detection)
    rsid_oth_h14 = []    # uint16: 14-bit hash of other_allele
    rsid_models = []     # int8
    n_skipped = 0
    n_no_other = 0
    n_no_other_matched = 0

    for entry in variants:
        chrom, pos, effect_allele, other_allele, weight = entry[:5]
        is_dominant = entry[5] if len(entry) > 5 else False
        is_recessive = entry[6] if len(entry) > 6 else False
        rsid = entry[7] if len(entry) > 7 else None

        # Determine model
        if is_dominant:
            model = 1
        elif is_recessive:
            model = 2
        else:
            model = 0

        # Store rsID fallback entry
        if rsid and rsid.startswith('rs'):
            rsid_hashes.append(np.uint64(fnv1a_hash(rsid.lower())))
            rsid_weights.append(weight)
            rsid_eff_h14.append(fnv1a_hash(effect_allele) & 0x3FFF)
            rsid_oth_h14.append(fnv1a_hash(other_allele) & 0x3FFF if other_allele else 0)
            rsid_models.append(model)

        if other_allele:
            # Check for ambiguous allele pairs (A/T, T/A, C/G, G/C)
            # These cannot be resolved by strand complement — pgsc_calc excludes them
            AMBIGUOUS_PAIRS = {('A','T'), ('T','A'), ('C','G'), ('G','C')}
            is_ambiguous = (effect_allele, other_allele) in AMBIGUOUS_PAIRS

            if is_ambiguous:
                # Skip ambiguous variants — matches pgsc_calc default behavior
                n_skipped += 1
                continue

            eff_comp = complement_allele(effect_allele)
            oth_comp = complement_allele(other_allele)

            # Generate up to 4 orientations:
            # 1. normal: REF=other, ALT=effect
            # 2. allele flip: REF=effect, ALT=other
            # 3. strand flip: REF=comp(other), ALT=comp(effect)
            # 4. both: REF=comp(effect), ALT=comp(other)
            orientations = [
                (other_allele, effect_allele, 0),       # normal, effect=ALT
                (effect_allele, other_allele, 1),       # flipped, effect=REF
                (oth_comp, eff_comp, 0),                # strand flip, effect=ALT
                (eff_comp, oth_comp, 1),                # both, effect=REF
            ]

            seen_keys = set()
            for ref_a, alt_a, flip in orientations:
                k = encode_variant_key(chrom, pos, ref_a, alt_a)
                k_int = int(k)
                if k == np.uint64(0) or k_int in seen_keys:
                    continue
                seen_keys.add(k_int)
                keys.append(k_int)
                weights.append(weight)
                flip_flags.append(flip)
                model_flags.append(model)
                # TSV A1 must match REF or ALT for plink2 scoring:
                # flip=0 → effect=ALT → A1=alt_a; flip=1 → effect=REF → A1=ref_a
                tsv_a1 = ref_a if flip == 1 else alt_a
                tsv_lines.append((f"{chrom}:{pos}:{ref_a}:{alt_a}",
                                  tsv_a1, weight, model, k_int))
        else:
            # No other_allele — match against target pvar if available (like pgsc_calc),
            # otherwise fall back to wildcard matching.
            n_no_other += 1
            eff = effect_allele.upper()
            matched_no_oa = False

            if target_pvar and len(eff) >= 1:
                # pgsc_calc-style: look up position in target pvar, match effect_allele
                # against REF/ALT (and their complements)
                target_alleles = target_pvar.get((chrom, int(pos)), [])
                eff_comp = complement_allele(eff)
                seen_keys = set()

                for ref, alt in target_alleles:
                    # Skip ambiguous target positions (A/T, C/G) — pgsc_calc excludes these
                    if len(ref) == 1 and len(alt) == 1 and COMPLEMENT.get(ref) == alt:
                        continue
                    # no_oa_ref: effect_allele == REF → flip=1
                    if eff == ref:
                        k = encode_variant_key(chrom, pos, ref, alt)
                        k_int = int(k)
                        if k_int not in seen_keys and k != np.uint64(0):
                            seen_keys.add(k_int)
                            keys.append(k_int); weights.append(weight)
                            flip_flags.append(1); model_flags.append(model)
                            tsv_lines.append((f"{chrom}:{pos}:{ref}:{alt}",
                                              eff, weight, model, k_int))
                            matched_no_oa = True
                    # no_oa_alt: effect_allele == ALT → flip=0
                    if eff == alt:
                        k = encode_variant_key(chrom, pos, ref, alt)
                        k_int = int(k)
                        if k_int not in seen_keys and k != np.uint64(0):
                            seen_keys.add(k_int)
                            keys.append(k_int); weights.append(weight)
                            flip_flags.append(0); model_flags.append(model)
                            tsv_lines.append((f"{chrom}:{pos}:{ref}:{alt}",
                                              eff, weight, model, k_int))
                            matched_no_oa = True
                    # no_oa_ref_flip: complement(effect) == REF → flip=1
                    if eff_comp == ref:
                        k = encode_variant_key(chrom, pos, ref, alt)
                        k_int = int(k)
                        if k_int not in seen_keys and k != np.uint64(0):
                            seen_keys.add(k_int)
                            keys.append(k_int); weights.append(weight)
                            flip_flags.append(1); model_flags.append(model)
                            tsv_lines.append((f"{chrom}:{pos}:{ref}:{alt}",
                                              eff_comp, weight, model, k_int))
                            matched_no_oa = True
                    # no_oa_alt_flip: complement(effect) == ALT → flip=0
                    if eff_comp == alt:
                        k = encode_variant_key(chrom, pos, ref, alt)
                        k_int = int(k)
                        if k_int not in seen_keys and k != np.uint64(0):
                            seen_keys.add(k_int)
                            keys.append(k_int); weights.append(weight)
                            flip_flags.append(0); model_flags.append(model)
                            tsv_lines.append((f"{chrom}:{pos}:{ref}:{alt}",
                                              eff_comp, weight, model, k_int))
                            matched_no_oa = True

            if not matched_no_oa and len(eff) == 1 and eff in BASES:
                # Fallback: wildcard matching (no target pvar available)
                seen_keys = set()
                eff_comp = complement_allele(eff)
                for allele in [eff, eff_comp]:
                    other_bases = BASES - {allele}
                    for other_base in other_bases:
                        for ref_a, alt_a, flip in [(other_base, allele, 0), (allele, other_base, 1)]:
                            # Skip ambiguous pairs (A/T, C/G)
                            if COMPLEMENT.get(ref_a) == alt_a:
                                continue
                            k = encode_variant_key(chrom, pos, ref_a, alt_a)
                            k_int = int(k)
                            if k_int not in seen_keys and k != np.uint64(0):
                                seen_keys.add(k_int)
                                keys.append(k_int); weights.append(weight)
                                flip_flags.append(flip); model_flags.append(model)
                                tsv_lines.append((f"{chrom}:{pos}:{ref_a}:{alt_a}",
                                                  allele, weight, model, k_int))
                matched_no_oa = True

            if matched_no_oa:
                n_no_other_matched += 1
            else:
                n_skipped += 1
                continue

    if not keys:
        print(f"  WARNING: No usable variants for {pgs_id}")
        return None

    # Sort by key
    keys_arr = np.array(keys, dtype=np.uint64)
    weights_arr = np.array(weights, dtype=np.float32)
    flip_arr = np.array(flip_flags, dtype=np.int8)
    model_arr = np.array(model_flags, dtype=np.int8)
    sort_idx = np.argsort(keys_arr)
    keys_arr = keys_arr[sort_idx]
    weights_arr = weights_arr[sort_idx]
    flip_arr = flip_arr[sort_idx]
    model_arr = model_arr[sort_idx]

    # Deduplicate (keep first)
    unique_mask = np.concatenate([[True], keys_arr[1:] != keys_arr[:-1]])
    keys_arr = keys_arr[unique_mask]
    weights_arr = weights_arr[unique_mask]
    flip_arr = flip_arr[unique_mask]
    model_arr = model_arr[unique_mask]

    # Write
    keys_path = output_dir / f"{pgs_id}.keys.bin"
    weights_path = output_dir / f"{pgs_id}.weights.bin"
    flip_path = output_dir / f"{pgs_id}.flip.bin"
    model_path = output_dir / f"{pgs_id}.model.bin"
    meta_path = output_dir / f"{pgs_id}.meta.json"

    keys_arr.tofile(str(keys_path))
    weights_arr.tofile(str(weights_path))
    flip_arr.tofile(str(flip_path))
    model_arr.tofile(str(model_path))

    # Write plink2 scoring TSV from the same data (guarantees binary == TSV)
    # Apply same sort+dedup as binary
    tsv_sorted = [tsv_lines[i] for i in sort_idx]
    tsv_deduped = [tsv_sorted[i] for i in range(len(tsv_sorted)) if unique_mask[i]]
    scoring_dir = output_dir / 'scoring_files'
    scoring_dir.mkdir(exist_ok=True)
    # Group by model: 0=additive, 1=dominant, 2=recessive
    mode_map = {0: 'additive', 1: 'dominant', 2: 'recessive'}
    mode_lines = {'additive': [], 'dominant': [], 'recessive': []}
    for vid, eff, w, m, k_int in tsv_deduped:
        mode_lines[mode_map.get(m, 'additive')].append(f"{vid}\t{eff}\t{w}\t{k_int}")
    for mode, lines in mode_lines.items():
        if not lines:
            continue
        suffix = '' if mode == 'additive' else f'.{mode}'
        tsv_path = scoring_dir / f'{pgs_id}{suffix}.tsv'
        with open(tsv_path, 'w') as f:
            f.write("ID\tA1\tWEIGHT\tKEY\n")
            for line in lines:
                f.write(line + '\n')

    # Write rsID fallback lookup (sorted by rsid_hash)
    if rsid_hashes:
        rh = np.array(rsid_hashes, dtype=np.uint64)
        rw = np.array(rsid_weights, dtype=np.float32)
        re = np.array(rsid_eff_h14, dtype=np.uint16)
        ro = np.array(rsid_oth_h14, dtype=np.uint16)
        rm = np.array(rsid_models, dtype=np.int8)
        rsort = np.argsort(rh)
        rh, rw, re, ro, rm = rh[rsort], rw[rsort], re[rsort], ro[rsort], rm[rsort]

        (output_dir / f"{pgs_id}.rsid_hash.bin").write_bytes(rh.tobytes())
        (output_dir / f"{pgs_id}.rsid_weight.bin").write_bytes(rw.tobytes())
        (output_dir / f"{pgs_id}.rsid_eff14.bin").write_bytes(re.tobytes())
        (output_dir / f"{pgs_id}.rsid_oth14.bin").write_bytes(ro.tobytes())
        (output_dir / f"{pgs_id}.rsid_model.bin").write_bytes(rm.tobytes())

    # Metadata
    meta = {
        'cache_version': CACHE_VERSION,
        'pgs_id': pgs_id,
        'num_variants': len(keys_arr),
        'num_original': len(variants),
        'num_skipped': n_skipped,
        'num_no_other_allele': n_no_other,
        'num_no_other_matched': n_no_other_matched,
        'num_duplicates_removed': int(np.sum(~unique_mask)),
        'weight_sum': float(np.sum(weights_arr)),
        'weight_mean': float(np.mean(weights_arr)),
        'weight_min': float(np.min(weights_arr)),
        'weight_max': float(np.max(weights_arr)),
        'keys_md5': hashlib.md5(keys_arr.tobytes()).hexdigest(),
        'weights_md5': hashlib.md5(weights_arr.tobytes()).hexdigest(),
        'keys_bytes': int(keys_arr.nbytes),
        'weights_bytes': int(weights_arr.nbytes),
    }

    # Add catalog metadata if available
    if metadata:
        meta['name'] = metadata.get('name', '')
        meta['trait_reported'] = metadata.get('trait_reported', '')
        meta['trait_efo'] = [t.get('id', '') for t in metadata.get('trait_efo', [])]
        meta['variants_number'] = metadata.get('variants_number', 0)
        meta['publication'] = metadata.get('ftp_scoring_file', '')

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


# ============================================================
# SYNC PIPELINE
# ============================================================

def sync_pgs_scores(pgs_ids: list, cache_dir: str, target_pvar_path: str = None):
    """
    Download and convert multiple PGS scores to binary format.
    If target_pvar_path is provided, uses target-aware no_oa matching.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    downloads_dir = cache / 'downloads'
    downloads_dir.mkdir(exist_ok=True)

    # Load target pvar for no_oa matching (if available)
    target_pvar = None
    if target_pvar_path and os.path.exists(target_pvar_path):
        target_pvar = load_target_pvar(target_pvar_path)
        print(f"  Target pvar loaded: {len(target_pvar):,} positions")

    results = {}

    for pgs_id in pgs_ids:
        pgs_id = pgs_id.strip().upper()
        print(f"\n{'='*50}")
        print(f"  Processing: {pgs_id}")
        print(f"{'='*50}")

        # Check if already converted (and cache version matches)
        if (cache / f"{pgs_id}.keys.bin").exists() and (cache / f"{pgs_id}.meta.json").exists():
            with open(cache / f"{pgs_id}.meta.json") as f:
                meta = json.load(f)
            if meta.get('cache_version', 0) >= CACHE_VERSION:
                print(f"  Cached: {meta['num_variants']:,} variants")
                results[pgs_id] = meta
                continue
            else:
                print(f"  Cache stale (v{meta.get('cache_version', 0)} < v{CACHE_VERSION}), rebuilding...")

        # Fetch metadata
        print(f"  Fetching metadata...")
        metadata = fetch_pgs_metadata(pgs_id)

        # Download scoring file
        scoring_path = download_scoring_file(pgs_id, downloads_dir)
        if not scoring_path:
            results[pgs_id] = {'error': 'download_failed'}
            continue

        # Parse
        print(f"  Parsing scoring file...")
        t0 = time.time()
        variants = parse_scoring_file(scoring_path)
        t_parse = time.time() - t0
        print(f"  Parsed {len(variants):,} variants in {t_parse:.1f}s")

        if not variants:
            print(f"  WARNING: No variants parsed from scoring file")
            results[pgs_id] = {'error': 'no_variants'}
            continue

        # Convert to binary
        print(f"  Converting to binary...")
        meta = convert_to_binary(pgs_id, variants, cache, metadata, target_pvar=target_pvar)

        if meta:
            print(f"  Stored: {meta['num_variants']:,} variants ({meta['keys_bytes'] + meta['weights_bytes']:.0f} bytes)")
            results[pgs_id] = meta
        else:
            results[pgs_id] = {'error': 'conversion_failed'}

    # Build plink2-ready scoring files for any PGS that don't have them yet
    # (e.g. old cached binaries from before TSV co-generation was added)
    missing_tsv = [p for p in pgs_ids if not (cache / 'scoring_files' / f'{p.strip().upper()}.tsv').exists()]
    if missing_tsv:
        _build_plink2_scoring_files(missing_tsv, cache, target_pvar=target_pvar)

    return results


# ============================================================
# PLINK2-READY SCORING FILES (for ancestry reference scoring)
# ============================================================

def _build_plink2_scoring_files(pgs_ids: list, cache_dir: Path, target_pvar: dict = None):
    """
    Write pre-parsed TSV scoring files for plink2 --score.

    Generates TWO files per PGS:
      <PGS_ID>.tsv         — additive variants (default)
      <PGS_ID>.dominant.tsv — dominant variants (if any)
      <PGS_ID>.recessive.tsv — recessive variants (if any)

    Format: ID<tab>A1<tab>WEIGHT<tab>KEY
    - ID: chr:pos:other:effect AND chr:pos:effect:other (both orientations)
    - A1: always the effect_allele
    - KEY: uint64 variant key (decimal) for exact matching against binary store
    """
    scoring_dir = cache_dir / 'scoring_files'
    scoring_dir.mkdir(exist_ok=True)

    downloads_dir = cache_dir / 'downloads'
    n_built = 0

    for pgs_id in pgs_ids:
        pgs_id = pgs_id.strip().upper()
        out_path = scoring_dir / f'{pgs_id}.tsv'
        if out_path.exists():
            continue

        sf = downloads_dir / f'{pgs_id}_scoring.txt.gz'
        if not sf.exists():
            continue

        # {mode: [(vid, effect, weight, key_str), ...]}
        mode_variants = {'additive': [], 'dominant': [], 'recessive': []}
        seen = set()
        opener = gzip.open if str(sf).endswith('.gz') else open

        with opener(sf, 'rt') as f:
            header = None
            col_map = {}
            for line in f:
                if line.startswith('#'):
                    continue
                line = line.rstrip()
                if not line:
                    continue
                fields = line.split('\t')
                if len(fields) == 1:
                    fields = line.split()
                if header is None:
                    header = fields
                    for i, col in enumerate(header):
                        col_map[col.lower()] = i
                    continue

                def sg(c):
                    idx = col_map.get(c)
                    if idx is not None and idx < len(fields):
                        v = fields[idx].strip()
                        return v if v and v not in ('.', 'NA') else None
                    return None

                chrom = sg('hm_chr') or sg('chr_name')
                pos = sg('hm_pos') or sg('chr_position')
                effect = sg('effect_allele')
                other = sg('other_allele') or sg('hm_inferotherallele')
                weight = sg('effect_weight')

                if other and '/' in other:
                    other = None  # multi-valued OA: match by position+EA only (matches pgscatalog-match)
                if not all([chrom, pos, effect, weight]):
                    continue
                chrom = chrom.lstrip('chr')

                # Determine scoring mode from PGS Catalog columns
                is_dom = (sg('is_dominant') or '').lower() == 'true'
                is_rec = (sg('is_recessive') or '').lower() == 'true'
                if is_dom:
                    mode = 'dominant'
                elif is_rec:
                    mode = 'recessive'
                else:
                    mode = 'additive'

                if other:
                    allele_pairs = [(other, effect)]
                elif target_pvar:
                    # pgsc_calc-style: look up target pvar for actual REF/ALT
                    target_alleles = target_pvar.get((chrom, int(pos)), [])
                    allele_pairs = []
                    eff_up = effect.upper()
                    eff_comp = complement_allele(eff_up)
                    for ref, alt in target_alleles:
                        if eff_up == alt or eff_comp == alt:
                            allele_pairs.append((ref, alt))
                        elif eff_up == ref or eff_comp == ref:
                            allele_pairs.append((ref, alt))
                    if not allele_pairs:
                        continue  # No match at this position
                else:
                    nucs = {'A', 'C', 'G', 'T'}
                    others = sorted(nucs - {effect.upper()}) if len(effect) == 1 else []
                    allele_pairs = [(o, effect) for o in others]

                # Strand complement: A↔T, C↔G (matches pgsc_calc refalt_flip/altref_flip)
                _comp = str.maketrans('ACGTacgt', 'TGCAtgca')
                def complement(seq):
                    return seq.translate(_comp)

                for a1, a2 in allele_pairs:
                    # Skip strand-ambiguous SNPs (A/T, C/G) — matches pgsc_calc keep_ambiguous=false
                    if len(a1) == 1 and len(a2) == 1:
                        pair = frozenset({a1.upper(), a2.upper()})
                        if pair in (frozenset({'A', 'T'}), frozenset({'C', 'G'})):
                            continue
                    # Generate orientations: normal + allele-swap + strand-flip + both
                    orientations = [
                        (a1, a2),                           # refalt
                        (a2, a1),                           # altref
                        (complement(a1), complement(a2)),   # refalt_flip
                        (complement(a2), complement(a1)),   # altref_flip
                    ]
                    for ref_a, alt_a in orientations:
                        vid = f"{chrom}:{pos}:{ref_a}:{alt_a}"
                        if vid not in seen:
                            seen.add(vid)
                            key = encode_variant_key(chrom, int(pos), ref_a, alt_a)
                            mode_variants[mode].append(
                                f"{vid}\t{effect}\t{weight}\t{int(key)}")

        # Write scoring files per mode
        for mode, variants in mode_variants.items():
            if not variants:
                continue
            suffix = '' if mode == 'additive' else f'.{mode}'
            path = scoring_dir / f'{pgs_id}{suffix}.tsv'
            with open(path, 'w') as f:
                f.write("ID\tA1\tWEIGHT\tKEY\n")
                for v in variants:
                    f.write(v + '\n')

        if any(mode_variants.values()):
            n_built += 1

    if n_built:
        print(f"  Built {n_built} plink2 scoring files in {scoring_dir}")


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PGS Catalog Sync')
    parser.add_argument('--pgsid', required=True, help='Comma-separated PGS IDs (e.g. PGS000802,PGS000001)')
    parser.add_argument('--cache-dir', required=True, help='Cache directory for binary weight files')
    parser.add_argument('--target-pvar', default=None, help='Target pvar file for no_oa matching')
    args = parser.parse_args()

    pgs_ids = [x.strip() for x in args.pgsid.split(',')]
    results = sync_pgs_scores(pgs_ids, args.cache_dir, target_pvar_path=args.target_pvar)

    print(f"\n{'='*50}")
    print(f"  Summary")
    print(f"{'='*50}")
    for pgs_id, meta in results.items():
        if 'error' in meta:
            print(f"  {pgs_id}: FAILED ({meta['error']})")
        else:
            print(f"  {pgs_id}: {meta['num_variants']:,} variants")