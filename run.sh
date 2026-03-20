#!/bin/bash
# PolyGen — Accelerated Pipeline
# Usage:
#   Full:    ./run.sh --input <vcf> --output <base_dir> --pgsid PGS000802,PGS000001
#   Prepped: ./run.sh --input <store_dir> --output <base_dir> --pgsid PGS000802 --prepped
#   No ancestry: ./run.sh --input <vcf> --output <dir> --pgsid PGS000329 --no-ancestry
#
# Creates a job folder: <base_dir>/<uuid>_accelerated/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODULES_DIR="${SCRIPT_DIR}/modules"
PGS_CACHE="${HOME}/.polygen/pgs_cache"
POLYGEN_DIR="${HOME}/.polygen"

# ─── Defaults ────────────────────────────────────────────────
INPUT=""
OUTPUT_BASE=""
PGSID=""
PREPPED=false
THREADS=$(nproc)
NO_ANCESTRY=false
ANCESTRY_CACHE="${SCRIPT_DIR}/data/ancestry_cache"
REF_PFILE="${POLYGEN_DIR}/reference/GRCh38_HGDP+1kGP_ALL"

# ─── Parse args ──────────────────────────────────────────────
usage() {
    cat << EOF
PolyGen — Accelerated GPU Pipeline

Usage: $0 [OPTIONS]

OPTIONS:
    --input PATH          Input VCF (.vcf.gz/.gz) or prepped store directory
    --output DIR          Base output directory (job folder created inside)
    --pgsid IDS           Comma-separated PGS Catalog IDs (e.g. PGS000802,PGS000001)
    --prepped             Input is an existing binary store (skip VCF conversion)
    --threads N           Number of threads (default: all cores)
    --no-ancestry         Skip ancestry adjustment (output raw scores only)
    --ancestry-cache DIR  Custom ancestry cache path (default: ~/.polygen/ancestry_cache)
    -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)           INPUT="$2"; shift 2 ;;
        --output)          OUTPUT_BASE="$2"; shift 2 ;;
        --pgsid)           PGSID="$2"; shift 2 ;;
        --prepped)         PREPPED=true; shift ;;
        --threads)         THREADS="$2"; shift 2 ;;
        --no-ancestry)     NO_ANCESTRY=true; shift ;;
        --ancestry-cache)  ANCESTRY_CACHE="$2"; shift 2 ;;
        -h|--help)         usage; exit 0 ;;
        *)                 echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# ─── Validate ────────────────────────────────────────────────
[[ -z "$INPUT" ]]       && { echo "Error: --input required"; usage; exit 1; }
[[ -z "$OUTPUT_BASE" ]] && { echo "Error: --output required"; usage; exit 1; }
[[ -z "$PGSID" ]]       && { echo "Error: --pgsid required"; usage; exit 1; }

if $PREPPED; then
    [[ ! -f "${INPUT}/keys.bin" ]] && { echo "Error: ${INPUT}/keys.bin not found — not a valid store"; exit 1; }
    [[ ! -f "${INPUT}/dosage.bin" ]] && { echo "Error: ${INPUT}/dosage.bin not found"; exit 1; }
    [[ ! -f "${INPUT}/qc.json" ]] && { echo "Error: ${INPUT}/qc.json not found"; exit 1; }
else
    [[ ! -f "$INPUT" ]] && { echo "Error: VCF not found: ${INPUT}"; exit 1; }
fi

# Check ancestry availability (needs both cache + reference panel)
ANCESTRY_AVAILABLE=false
if ! $NO_ANCESTRY && [[ -f "${ANCESTRY_CACHE}/meta.json" ]] && [[ -f "${REF_PFILE}.pgen" ]]; then
    ANCESTRY_AVAILABLE=true
elif ! $NO_ANCESTRY && [[ -f "${ANCESTRY_CACHE}/meta.json" ]] && [[ ! -f "${REF_PFILE}.pgen" ]]; then
    echo "WARNING: Ancestry cache found but reference panel missing at ${REF_PFILE}"
    echo "         Run setup.sh --ref <pgsc_HGDP+1kGP_v1.tar.zst> to install"
    echo "         Skipping ancestry adjustment"
fi

# ─── Create job folder ──────────────────────────────────────
JOB_ID=$(cat /proc/sys/kernel/random/uuid | cut -d'-' -f1)
JOB_DIR="${OUTPUT_BASE}/${JOB_ID}_accelerated"
mkdir -p "${JOB_DIR}"/{intermediate,results}

LOG_FILE="${JOB_DIR}/pipeline.log"
> "$LOG_FILE"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

PIPELINE_START=$(date +%s%N)

log "============================================"
log "  PolyGen — Accelerated Pipeline"
log "============================================"
log "Job ID:    ${JOB_ID}"
log "Input:     ${INPUT}"
log "Output:    ${JOB_DIR}"
log "PGS IDs:   ${PGSID}"
log "Prepped:   ${PREPPED}"
log "Ancestry:  ${ANCESTRY_AVAILABLE}"
log "Threads:   ${THREADS}"
log "============================================"

# ─── Save config ─────────────────────────────────────────────
cat > "${JOB_DIR}/config.json" << EOF
{
    "pipeline": "polygen_accelerated",
    "input": "${INPUT}",
    "pgs_ids": "${PGSID}",
    "prepped": ${PREPPED},
    "ancestry": ${ANCESTRY_AVAILABLE},
    "threads": ${THREADS},
    "created": "$(date -Iseconds)"
}
EOF

# ─── Phase 1: VCF → Binary Store ────────────────────────────
STORE_DIR="${JOB_DIR}/intermediate"

if $PREPPED; then
    log "PHASE 1: SKIPPED (using prepped store at ${INPUT})"
    STORE_DIR="$INPUT"
    PREP_TIME="0"
else
    log "PHASE 1: VCF → Binary Genotype Store"
    PREP_START=$(date +%s%N)

    python3 "${MODULES_DIR}/vcf_to_bin.py" \
        --input "$INPUT" \
        --output "${JOB_DIR}/intermediate" \
        2>&1 | tee -a "$LOG_FILE"

    PREP_END=$(date +%s%N)
    PREP_TIME=$(echo "scale=2; ($PREP_END - $PREP_START) / 1000000000" | bc)
    log "Phase 1 complete: ${PREP_TIME}s"
fi

# Validate store
if [[ ! -f "${STORE_DIR}/keys.bin" || ! -f "${STORE_DIR}/dosage.bin" ]]; then
    log "ERROR: Binary store not created"
    exit 1
fi

# Create target pfile for plink2-based normalization.
# Applies same preprocessing as pgsc_calc: split multiallelics, remove dups,
# strip chr prefix, set IDs to chr:pos:ref:alt. This ensures Z-score parity.
TARGET_PFILE="${STORE_DIR}/target_plink"
if [[ ! -f "${TARGET_PFILE}.pgen" ]]; then
    INPUT_VCF=""
    if ! $PREPPED; then
        INPUT_VCF="$INPUT"
    elif [[ -f "${STORE_DIR}/../config.json" ]]; then
        INPUT_VCF=$(python3 -c "import json; print(json.load(open('${STORE_DIR}/../config.json')).get('input',''))" 2>/dev/null || echo "")
    fi
    if [[ -z "$INPUT_VCF" || ! -f "$INPUT_VCF" ]]; then
        INPUT_VCF=$(python3 -c "import json; print(json.load(open('${STORE_DIR}/qc.json')).get('input_path',''))" 2>/dev/null || echo "")
    fi
    if [[ -n "$INPUT_VCF" && -f "$INPUT_VCF" ]]; then
        log "Creating target pfile (pgsc_calc-compatible preprocessing)..."
        TMPVCF=$(mktemp -d "${STORE_DIR}/pfile_prep_XXXXXX")
        RENAME_CHR="${SCRIPT_DIR}/../data/remove_chrs.txt"

        # Remove missing ALT + split multiallelics + remove duplicates
        bcftools view -e 'ALT = "."' "$INPUT_VCF" --threads "$THREADS" 2>/dev/null \
            | bcftools norm -m -both --threads "$THREADS" 2>/dev/null \
            | bcftools norm -d exact --threads "$THREADS" -Oz -o "${TMPVCF}/norm.vcf.gz" 2>/dev/null

        # Strip chr prefix (if rename file exists) + set variant IDs
        if [[ -f "$RENAME_CHR" ]]; then
            bcftools annotate --rename-chrs "$RENAME_CHR" "${TMPVCF}/norm.vcf.gz" \
                --threads "$THREADS" 2>/dev/null \
                | bcftools annotate --set-id '%CHROM:%POS:%REF:%ALT' --threads "$THREADS" \
                    -Oz -o "${TMPVCF}/final.vcf.gz" 2>/dev/null
        else
            bcftools annotate --set-id '%CHROM:%POS:%REF:%ALT' "${TMPVCF}/norm.vcf.gz" \
                --threads "$THREADS" -Oz -o "${TMPVCF}/final.vcf.gz" 2>/dev/null
        fi

        # Convert to pfile (autosomes only)
        plink2 --vcf "${TMPVCF}/final.vcf.gz" --make-pgen --chr 1-22 \
            --allow-extra-chr --new-id-max-allele-len 10000 \
            --out "$TARGET_PFILE" --threads "$THREADS" --memory 8000 \
            2>&1 | tail -3 | tee -a "$LOG_FILE"

        rm -rf "$TMPVCF"
        log "Target pfile ready"
    fi
fi

# ─── Phase 2: PGS Catalog Sync ──────────────────────────────
log "PHASE 2: PGS Catalog Sync"
mkdir -p "$PGS_CACHE"
SYNC_START=$(date +%s%N)

TARGET_PVAR_ARG=""
if [[ -f "${STORE_DIR}/target_plink.pvar" ]]; then
    TARGET_PVAR_ARG="--target-pvar ${STORE_DIR}/target_plink.pvar"
fi

python3 "${MODULES_DIR}/pgs_catalog.py" \
    --pgsid "$PGSID" \
    --cache-dir "$PGS_CACHE" \
    $TARGET_PVAR_ARG \
    2>&1 | tee -a "$LOG_FILE"

SYNC_END=$(date +%s%N)
SYNC_TIME=$(echo "scale=2; ($SYNC_END - $SYNC_START) / 1000000000" | bc)
log "Phase 2 complete: ${SYNC_TIME}s"

# ─── Phase 3: GPU Scoring ───────────────────────────────────
log "PHASE 3: GPU Scoring"
SCORE_START=$(date +%s%N)

python3 "${MODULES_DIR}/score.py" \
    --store "$STORE_DIR" \
    --cache "$PGS_CACHE" \
    --pgsid "$PGSID" \
    --output "${JOB_DIR}/results" \
    2>&1 | tee -a "$LOG_FILE"

SCORE_END=$(date +%s%N)
SCORE_TIME=$(echo "scale=2; ($SCORE_END - $SCORE_START) / 1000000000" | bc)
log "Phase 3 complete: ${SCORE_TIME}s"

# ─── Phase 4: Ancestry Adjustment (optional) ────────────────
ANCESTRY_TIME="0"

if $ANCESTRY_AVAILABLE; then
    log "PHASE 4: Ancestry Adjustment"
    ANCESTRY_START=$(date +%s%N)

    # Pass target pfile if available (for plink2-based normalization)
    TARGET_PFILE_ARG=""
    if [[ -f "${STORE_DIR}/target_plink.pgen" ]]; then
        TARGET_PFILE_ARG="--target-pfile ${STORE_DIR}/target_plink"
    fi

    python3 "${MODULES_DIR}/ancestry.py" \
        --store "$STORE_DIR" \
        --scores "${JOB_DIR}/results/scores.tsv" \
        --cache "$ANCESTRY_CACHE" \
        --ref-pfile "$REF_PFILE" \
        --pgs-cache "$PGS_CACHE" \
        --output "${JOB_DIR}/results" \
        $TARGET_PFILE_ARG \
        2>&1 | tee -a "$LOG_FILE"

    ANCESTRY_END=$(date +%s%N)
    ANCESTRY_TIME=$(echo "scale=2; ($ANCESTRY_END - $ANCESTRY_START) / 1000000000" | bc)
    log "Phase 4 complete: ${ANCESTRY_TIME}s"
else
    if $NO_ANCESTRY; then
        log "PHASE 4: SKIPPED (--no-ancestry)"
    else
        log "PHASE 4: SKIPPED (no ancestry cache — run setup.sh to enable)"
    fi
fi

# ─── Finalize ────────────────────────────────────────────────
PIPELINE_END=$(date +%s%N)
TOTAL_TIME=$(echo "scale=2; ($PIPELINE_END - $PIPELINE_START) / 1000000000" | bc)

# Read variant count from QC
VARIANT_COUNT=$(python3 -c "import json; print(json.load(open('${STORE_DIR}/qc.json'))['total_variants'])" 2>/dev/null || echo 0)

# Write benchmark
cat > "${JOB_DIR}/benchmark.json" << EOF
{
    "pipeline": "polygen_accelerated",
    "timestamp": "$(date -Iseconds)",
    "input": "${INPUT}",
    "variant_count": ${VARIANT_COUNT},
    "pgs_ids": "${PGSID}",
    "prepped": ${PREPPED},
    "ancestry": ${ANCESTRY_AVAILABLE},
    "gpu_available": $(python3 -c "try:
    import cupy; print('true')
except: print('false')" 2>/dev/null),
    "total_time_seconds": ${TOTAL_TIME},
    "steps": {
        "vcf_to_bin": ${PREP_TIME},
        "pgs_sync": ${SYNC_TIME},
        "scoring": ${SCORE_TIME},
        "ancestry": ${ANCESTRY_TIME}
    },
    "hostname": "$(hostname)",
    "threads": ${THREADS}
}
EOF

log "============================================"
log "Pipeline complete in ${TOTAL_TIME}s"
log "  VCF→bin:   ${PREP_TIME}s"
log "  PGS sync:  ${SYNC_TIME}s"
log "  Scoring:   ${SCORE_TIME}s"
log "  Ancestry:  ${ANCESTRY_TIME}s"
log "Results:     ${JOB_DIR}/results/"
log "Benchmark:   ${JOB_DIR}/benchmark.json"
log "============================================"