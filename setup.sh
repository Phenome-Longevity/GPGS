#!/bin/bash
# PolyGen — Setup
# Checks dependencies and builds ancestry cache from reference panel.
#
# Usage:
#   ./setup.sh --ref /path/to/pgsc_HGDP+1kGP_v1.tar.zst
#   ./setup.sh --check                # just check dependencies
#   ./setup.sh --ref <tar> --cache /custom/path

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODULES_DIR="${SCRIPT_DIR}/modules"
DEFAULT_CACHE="${HOME}/.polygen"

# ─── Colors ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; }

# ─── Parse args ──────────────────────────────────────────────
REF_TAR=""
CACHE_DIR="$DEFAULT_CACHE"
CHECK_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --ref)      REF_TAR="$2"; shift 2 ;;
        --cache)    CACHE_DIR="$2"; shift 2 ;;
        --check)    CHECK_ONLY=true; shift ;;
        -h|--help)
            echo "Usage: $0 --ref <pgsc_HGDP+1kGP_v1.tar.zst> [--cache ~/.polygen]"
            echo "       $0 --check"
            exit 0 ;;
        *)  echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo ""
echo "============================================"
echo "  PolyGen — Setup"
echo "============================================"
echo ""

ERRORS=0

# ─── Check Python ────────────────────────────────────────────
echo "Python:"
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version 2>&1 | awk '{print $2}')
    PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
    PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)
    if [[ "$PY_MAJOR" -ge 3 && "$PY_MINOR" -ge 9 ]]; then
        ok "Python ${PY_VER}"
    else
        fail "Python ${PY_VER} (need ≥3.9)"
        ERRORS=$((ERRORS + 1))
    fi
else
    fail "Python3 not found"
    ERRORS=$((ERRORS + 1))
fi

# ─── Check Python packages ──────────────────────────────────
echo ""
echo "Python packages:"

check_pkg() {
    local pkg="$1"
    local import_name="${2:-$1}"
    if python3 -c "import ${import_name}" 2>/dev/null; then
        local ver=$(python3 -c "import ${import_name}; print(getattr(${import_name}, '__version__', 'ok'))" 2>/dev/null)
        ok "${pkg} (${ver})"
    else
        fail "${pkg} — install with: pip install ${pkg}"
        ERRORS=$((ERRORS + 1))
    fi
}

check_pkg "numpy" "numpy"
check_pkg "scipy" "scipy"
check_pkg "scikit-learn" "sklearn"
check_pkg "joblib" "joblib"

# CuPy (optional but recommended)
if python3 -c "import cupy" 2>/dev/null; then
    CUPY_VER=$(python3 -c "import cupy; print(cupy.__version__)" 2>/dev/null)
    ok "cupy (${CUPY_VER}) — GPU acceleration enabled"
else
    warn "cupy not found — CPU fallback will be used"
    echo "       Install for GPU: pip install cupy-cuda12x"
fi

# ─── Check system tools ─────────────────────────────────────
echo ""
echo "System tools:"

if command -v plink2 &>/dev/null; then
    PLINK_VER=$(plink2 --version 2>&1 | head -1 | awk '{print $2}' || echo "unknown")
    ok "plink2 (${PLINK_VER})"
else
    fail "plink2 not found — required for ancestry analysis"
    ERRORS=$((ERRORS + 1))
fi

if command -v bcftools &>/dev/null; then
    BCF_VER=$(bcftools --version 2>&1 | head -1 | awk '{print $2}' || echo "unknown")
    ok "bcftools (${BCF_VER})"
else
    fail "bcftools not found — required for VCF preprocessing"
    ERRORS=$((ERRORS + 1))
fi

if command -v bc &>/dev/null; then
    ok "bc"
else
    fail "bc not found — install with: apt install bc"
    ERRORS=$((ERRORS + 1))
fi

if command -v zstd &>/dev/null || command -v unzstd &>/dev/null; then
    ok "zstd decompressor"
else
    fail "zstd/unzstd not found — required to extract reference panel"
    ERRORS=$((ERRORS + 1))
fi

# ─── Check GPU ───────────────────────────────────────────────
echo ""
echo "GPU:"

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    ok "${GPU_NAME} (${GPU_MEM})"
else
    warn "No NVIDIA GPU detected — CPU fallback will be used"
fi

# ─── Summary ─────────────────────────────────────────────────
echo ""
if [[ $ERRORS -gt 0 ]]; then
    echo -e "${RED}Setup has ${ERRORS} error(s). Fix them before running PolyGen.${NC}"
    exit 1
fi

if $CHECK_ONLY; then
    ok "All dependencies satisfied"
    exit 0
fi

# ─── Ancestry cache ──────────────────────────────────────────
echo "============================================"
echo "  Ancestry Cache"
echo "============================================"
echo ""

BUNDLED_CACHE="${SCRIPT_DIR}/data/ancestry_cache"
if [[ -f "${BUNDLED_CACHE}/meta.json" ]]; then
    ok "Ancestry cache: ${BUNDLED_CACHE}/ (bundled)"
else
    fail "Bundled ancestry cache missing — reinstall PolyGen"
    ERRORS=$((ERRORS + 1))
fi

# ─── Reference panel ────────────────────────────────────────
echo ""
echo "============================================"
echo "  Reference Panel"
echo "============================================"
echo ""

if [[ -f "${CACHE_DIR}/reference/GRCh38_HGDP+1kGP_ALL.pgen" ]]; then
    REF_SIZE=$(du -sh "${CACHE_DIR}/reference/" 2>/dev/null | awk '{print $1}')
    ok "Reference panel: ${CACHE_DIR}/reference/ (${REF_SIZE})"
elif [[ -n "$REF_TAR" ]]; then
    if [[ ! -f "$REF_TAR" ]]; then
        fail "Reference panel tar not found: ${REF_TAR}"
        exit 1
    fi

    echo "  Extracting reference panel from ${REF_TAR}..."
    REF_DIR="${CACHE_DIR}/reference"
    mkdir -p "$REF_DIR"

    for fname in GRCh38_HGDP+1kGP_ALL.pgen GRCh38_HGDP+1kGP_ALL.psam GRCh38_HGDP+1kGP_ALL.pvar.zst; do
        echo "    Extracting ${fname}..."
        tar --use-compress-program=unzstd -xf "$REF_TAR" -C "$REF_DIR" "$fname" 2>/dev/null \
            || tar --use-compress-program=unzstd --warning=no-unknown-keyword -xf "$REF_TAR" -C "$REF_DIR" "$fname"
    done

    ok "Reference panel extracted to ${REF_DIR}"
else
    warn "No reference panel found"
    echo "       To install: $0 --ref /path/to/pgsc_HGDP+1kGP_v1.tar.zst"
    echo "       Required for ancestry-adjusted scoring"
fi

echo ""
echo "============================================"
echo "  Setup Complete"
echo "============================================"
echo ""
ok "Dependencies: all satisfied"
echo ""
echo "  Run PolyGen:"
echo "    ./run.sh --input <vcf.gz> --output <dir> --pgsid PGS000329"
echo ""