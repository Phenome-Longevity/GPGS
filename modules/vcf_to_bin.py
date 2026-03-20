#!/usr/bin/env python3
"""
VCF → Binary Genotype Store
Converts a bgzipped VCF into:
  - keys.bin   (uint64, sorted) — variant keys
  - dosage.bin (float32)        — alt allele dosage per variant
  - qc.json                     — metadata and stats
"""

import sys
import gzip
import json
import time
import struct
import hashlib
import argparse
import numpy as np
from pathlib import Path

# ============================================================
# VARIANT KEY ENCODING (matches G-VEP v4)
# ============================================================

CHROM_MAP = {
    '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
    '9': 9, '10': 10, '11': 11, '12': 12, '13': 13, '14': 14, '15': 15,
    '16': 16, '17': 17, '18': 18, '19': 19, '20': 20, '21': 21, '22': 22,
    'X': 23, 'Y': 24, 'MT': 25, 'M': 25,
    'chr1': 1, 'chr2': 2, 'chr3': 3, 'chr4': 4, 'chr5': 5, 'chr6': 6,
    'chr7': 7, 'chr8': 8, 'chr9': 9, 'chr10': 10, 'chr11': 11, 'chr12': 12,
    'chr13': 13, 'chr14': 14, 'chr15': 15, 'chr16': 16, 'chr17': 17,
    'chr18': 18, 'chr19': 19, 'chr20': 20, 'chr21': 21, 'chr22': 22,
    'chrX': 23, 'chrY': 24, 'chrMT': 25, 'chrM': 25,
}


def fnv1a_hash(s: str) -> int:
    h = 0x811c9dc5
    for c in s.encode():
        h ^= c
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def encode_variant_key(chrom: str, pos: int, ref: str, alt: str) -> np.uint64:
    """Encode variant into uint64 key: chrom(8) | pos(28) | ref_hash(14) | alt_hash(14)"""
    chrom_clean = chrom[3:] if chrom.startswith('chr') else chrom
    chrom_int = CHROM_MAP.get(chrom_clean, 0)
    if chrom_int == 0:
        return np.uint64(0)

    ref_hash = fnv1a_hash(ref) & 0x3FFF
    alt_hash = fnv1a_hash(alt) & 0x3FFF
    pos = min(pos, 0x0FFFFFFF)

    key = (int(chrom_int) << 56) | (int(pos) << 28) | (int(ref_hash) << 14) | int(alt_hash)
    return np.uint64(key)


# ============================================================
# GENOTYPE → DOSAGE
# ============================================================

def gt_to_dosage(gt_str: str) -> float:
    """Convert GT field to alt allele dosage (0.0, 1.0, 2.0, or NaN)."""
    gt = gt_str.split(':')[0]  # take only GT field
    sep = '/' if '/' in gt else '|'
    alleles = gt.split(sep)

    if '.' in alleles:
        return float('nan')

    try:
        return float(sum(int(a) > 0 for a in alleles))
    except ValueError:
        return float('nan')


# ============================================================
# VCF PARSER
# ============================================================

def parse_vcf_to_binary(vcf_path: str, output_dir: str, sample_idx: int = 0):
    """
    Stream-parse VCF and produce binary genotype store.

    Args:
        vcf_path: path to .vcf.gz / .gz / .vcf
        output_dir: directory for output files
        sample_idx: which sample column to extract (0 = first sample)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    keys = []
    dosages = []
    rsid_hashes = []  # FNV1a hash of rsID for each variant
    sample_id = None
    build = None
    n_skipped = 0
    n_multiallelic_expanded = 0
    n_missing_gt = 0
    n_duplicate = 0
    seen_keys = set()

    t0 = time.time()

    opener = gzip.open if vcf_path.endswith('.gz') else open

    with opener(vcf_path, 'rt') as f:
        for line in f:
            # --- Header lines ---
            if line.startswith('##'):
                # Try to detect build from header
                ll = line.lower()
                if 'grch38' in ll or 'hg38' in ll:
                    build = 'GRCh38'
                elif 'grch37' in ll or 'hg19' in ll:
                    build = 'GRCh37'
                continue

            if line.startswith('#CHROM'):
                cols = line.strip().split('\t')
                if len(cols) < 10:
                    print(f"ERROR: VCF has no sample columns")
                    sys.exit(1)
                sample_id = cols[9 + sample_idx]
                print(f"  Sample: {sample_id}")
                continue

            # --- Data lines ---
            cols = line.strip().split('\t')
            if len(cols) < 10:
                n_skipped += 1
                continue

            chrom = cols[0]
            try:
                pos = int(cols[1])
            except ValueError:
                n_skipped += 1
                continue

            rsid_raw = cols[2]  # ID column — typically rsID
            ref = cols[3].upper()
            alt_field = cols[4].upper()
            gt_raw = cols[9 + sample_idx]

            # Skip missing ALT
            if alt_field == '.' or alt_field == '*':
                n_skipped += 1
                continue

            # Handle multiallelic (split into biallelic)
            alts = alt_field.split(',')
            gt = gt_raw.split(':')[0]
            sep = '/' if '/' in gt else '|'
            allele_indices = gt.split(sep)

            for alt_idx, alt in enumerate(alts, start=1):
                if alt == '*' or alt == '.':
                    continue

                key = encode_variant_key(chrom, pos, ref, alt)
                if key == np.uint64(0):
                    n_skipped += 1
                    continue

                # Deduplicate
                key_int = int(key)
                if key_int in seen_keys:
                    n_duplicate += 1
                    continue
                seen_keys.add(key_int)

                # Compute dosage for this specific alt allele
                try:
                    dosage = sum(1 for a in allele_indices if a != '.' and int(a) == alt_idx)
                    dosage = float(dosage)
                except (ValueError, IndexError):
                    dosage = float('nan')
                    n_missing_gt += 1

                if len(alts) > 1:
                    n_multiallelic_expanded += 1

                keys.append(key)
                dosages.append(dosage)

                # Store rsID hash (0 if no rsID)
                if rsid_raw and rsid_raw != '.' and rsid_raw.startswith('rs'):
                    rsid_hashes.append(np.uint64(fnv1a_hash(rsid_raw.lower())))
                else:
                    rsid_hashes.append(np.uint64(0))

    t_parse = time.time() - t0

    if not keys:
        print("ERROR: No variants extracted from VCF")
        sys.exit(1)

    # --- Sort by key ---
    print(f"  Sorting {len(keys):,} variants...")
    t1 = time.time()
    keys_arr = np.array(keys, dtype=np.uint64)
    dosage_arr = np.array(dosages, dtype=np.float32)
    rsid_arr = np.array(rsid_hashes, dtype=np.uint64)

    sort_idx = np.argsort(keys_arr)
    keys_arr = keys_arr[sort_idx]
    dosage_arr = dosage_arr[sort_idx]
    rsid_arr = rsid_arr[sort_idx]
    t_sort = time.time() - t1

    # --- Build rsID lookup (sorted rsid_hash → index in keys_arr) ---
    # Only for variants that have valid rsIDs
    has_rsid = rsid_arr > 0
    rsid_valid = rsid_arr[has_rsid]
    rsid_indices = np.where(has_rsid)[0].astype(np.uint32)

    # Sort by rsID hash for binary search
    rsid_sort = np.argsort(rsid_valid)
    rsid_sorted_hashes = rsid_valid[rsid_sort]
    rsid_sorted_indices = rsid_indices[rsid_sort]

    # --- Write binary files ---
    print(f"  Writing binary store...")
    t2 = time.time()

    keys_path = out / 'keys.bin'
    dosage_path = out / 'dosage.bin'
    rsid_hash_path = out / 'rsid_hash.bin'
    rsid_idx_path = out / 'rsid_idx.bin'
    qc_path = out / 'qc.json'

    keys_arr.tofile(str(keys_path))
    dosage_arr.tofile(str(dosage_path))
    rsid_sorted_hashes.tofile(str(rsid_hash_path))
    rsid_sorted_indices.tofile(str(rsid_idx_path))

    # Compute checksum
    keys_md5 = hashlib.md5(keys_arr.tobytes()).hexdigest()
    dosage_md5 = hashlib.md5(dosage_arr.tobytes()).hexdigest()

    # Non-missing dosage stats
    valid_mask = ~np.isnan(dosage_arr)
    n_valid = int(np.sum(valid_mask))
    n_het = int(np.sum(dosage_arr[valid_mask] == 1.0))
    n_homalt = int(np.sum(dosage_arr[valid_mask] == 2.0))
    n_homref = int(np.sum(dosage_arr[valid_mask] == 0.0))

    qc = {
        'sample_id': sample_id,
        'build': build or 'unknown',
        'total_variants': len(keys_arr),
        'valid_genotypes': n_valid,
        'missing_genotypes': n_missing_gt,
        'het': n_het,
        'hom_alt': n_homalt,
        'hom_ref': n_homref,
        'rsid_count': int(np.sum(has_rsid)),
        'skipped': n_skipped,
        'duplicates_removed': n_duplicate,
        'multiallelic_expanded': n_multiallelic_expanded,
        'missingness_rate': round(1.0 - n_valid / len(keys_arr), 6) if keys_arr.size > 0 else 0,
        'keys_md5': keys_md5,
        'dosage_md5': dosage_md5,
        'keys_bytes': int(keys_arr.nbytes),
        'dosage_bytes': int(dosage_arr.nbytes),
        'total_bytes': int(keys_arr.nbytes + dosage_arr.nbytes),
        'time_parse_seconds': round(t_parse, 2),
        'time_sort_seconds': round(t_sort, 2),
        'time_write_seconds': round(time.time() - t2, 2),
        'time_total_seconds': round(time.time() - t0, 2),
    }

    with open(qc_path, 'w') as f:
        json.dump(qc, f, indent=2)

    t_total = time.time() - t0

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"  Binary Genotype Store")
    print(f"{'='*50}")
    print(f"  Sample:     {sample_id}")
    print(f"  Build:      {qc['build']}")
    print(f"  Variants:   {qc['total_variants']:,}")
    print(f"  Valid GT:   {n_valid:,} ({100*n_valid/len(keys_arr):.1f}%)")
    print(f"  Het/HomAlt: {n_het:,} / {n_homalt:,}")
    print(f"  Missing GT: {n_missing_gt:,}")
    print(f"  Duplicates: {n_duplicate:,}")
    print(f"  Store size: {qc['total_bytes']/1e6:.1f} MB")
    print(f"  Parse:      {t_parse:.1f}s")
    print(f"  Sort:       {t_sort:.1f}s")
    print(f"  Total:      {t_total:.1f}s")
    print(f"{'='*50}")
    print(f"  Output: {out}")

    return qc


# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='VCF → Binary Genotype Store')
    parser.add_argument('--input', '-i', required=True, help='Input VCF (.vcf.gz, .gz, or .vcf)')
    parser.add_argument('--output', '-o', required=True, help='Output directory for .bin files')
    parser.add_argument('--sample-index', type=int, default=0, help='Sample column index (default: 0 = first)')
    args = parser.parse_args()

    parse_vcf_to_binary(args.input, args.output, args.sample_index)