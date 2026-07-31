#!/usr/bin/env bash
#
# setup_env.sh -- create/refresh the uv-managed virtual environment for SAJAX.
#
# The environment is described by pyproject.toml and pinned by uv.lock, so every
# machine that runs this script gets byte-identical package versions.
#
# Usage
# -----
#   ./setup_env.sh              # auto-detect: GPU if nvidia-smi sees a device
#   ./setup_env.sh --cpu        # force the CPU-only environment (~150 MB)
#   ./setup_env.sh --gpu        # force the CUDA environment (~3.1 GB)
#   ./setup_env.sh --docs       # also install the sphinx docs dependencies
#   ./setup_env.sh --check      # don't install; just report what's in .venv
#   ./setup_env.sh --locked     # fail if uv.lock is stale (used in CI)
#
# On HPC, provision on a GPU-less login node with --gpu (or SAJAX_ACCEL=cuda),
# since auto-detection would otherwise build a CPU environment for a GPU job.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

ACCEL="${SAJAX_ACCEL:-auto}"
CHECK_ONLY=0
WITH_DOCS=0
LOCK_MODE="${SAJAX_LOCK_MODE:-}"   # set to "--locked" to fail if uv.lock is stale

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu)          ACCEL=cpu ;;
        --gpu|--cuda)   ACCEL=cuda ;;
        --auto)         ACCEL=auto ;;
        --docs)         WITH_DOCS=1 ;;
        --check)        CHECK_ONLY=1 ;;
        --locked)       LOCK_MODE="--locked" ;;
        # Print the header block, stopping at the first non-comment line.
        -h|--help)      sed -n '2,${/^#/!q; s/^# \?//; p;}' "${BASH_SOURCE[0]}"; exit 0 ;;
        *)              echo "setup_env.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
    shift
done

if ! command -v uv >/dev/null 2>&1; then
    echo "setup_env.sh: uv not found on PATH." >&2
    echo "  Install it with:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

# ---- Resolve the accelerator ------------------------------------------------
detect_accel() {
    # A GPU counts as present only if nvidia-smi exists AND reports a device.
    # Login nodes typically have neither; some have the binary but no hardware.
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "^GPU 0:"; then
        echo cuda
    else
        echo cpu
    fi
}

case "$ACCEL" in
    auto)       ACCEL="$(detect_accel)"; DETECTED=" (auto-detected)" ;;
    cpu|cuda)   DETECTED="" ;;
    *)          echo "setup_env.sh: SAJAX_ACCEL must be auto, cpu or cuda (got '$ACCEL')" >&2; exit 2 ;;
esac

# The cuda extra is Linux x86_64 only (see pyproject.toml). Refuse early
# elsewhere rather than silently installing a CPU environment under a GPU flag.
if [[ "$ACCEL" == cuda ]]; then
    if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
        echo "setup_env.sh: the cuda extra requires Linux x86_64 (this is $(uname -s)/$(uname -m))." >&2
        echo "  Re-run with --cpu." >&2
        exit 1
    fi
fi

# ---- Report what is installed ----------------------------------------------
report() {
    echo
    echo "--- environment report ---"
    uv run --no-sync python - <<'PY'
import importlib.metadata as md
import sys

import jax

print(f"python      : {sys.version.split()[0]}  ({sys.executable})")
print(f"jax         : {jax.__version__}")
try:
    print(f"jaxlib      : {md.version('jaxlib')}")
except md.PackageNotFoundError:
    print("jaxlib      : MISSING")
try:
    md.version("jax-cuda12-plugin")
    plugin = "installed"
except md.PackageNotFoundError:
    plugin = "not installed"
print(f"cuda plugin : {plugin}")

devices = jax.devices()
print(f"devices     : {devices}")
print(f"backend     : {jax.default_backend()}")

import sajax
print(f"sajax       : {md.version('sajax')}  ({sajax.__file__})")
PY
}

if [[ "$CHECK_ONLY" == 1 ]]; then
    if [[ ! -d .venv ]]; then
        echo "setup_env.sh: no .venv here. Run ./setup_env.sh first." >&2
        exit 1
    fi
    report
    exit 0
fi

# ---- Sync -------------------------------------------------------------------
EXTRAS=(--extra dev)
if [[ "$ACCEL" == cuda ]]; then
    EXTRAS+=(--extra cuda)
fi
if [[ "$WITH_DOCS" == 1 ]]; then
    EXTRAS+=(--extra docs)
fi

echo "setup_env.sh: syncing ${ACCEL} environment${DETECTED} into .venv"
if [[ "$ACCEL" == cuda ]]; then
    echo "  note: the CUDA runtime is ~3.1 GB on first install."
fi

# uv sync is exact: switching cpu<->gpu prunes and reinstalls the CUDA stack.
uv sync $LOCK_MODE "${EXTRAS[@]}"

report

echo
echo "Done. Use the environment with 'uv run <cmd>' (e.g. 'uv run pytest'),"
echo "or activate it directly with 'source .venv/bin/activate'."
