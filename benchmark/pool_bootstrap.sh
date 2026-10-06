#!/usr/bin/env bash
# Dependency path must be present before Python imports site/scientific modules.
set -euo pipefail
source_root=$1
staged_site=$2
python=$3
shift 3
export PYTHONPATH="$source_root/src:$staged_site"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
cd "$source_root"
exec "$python" "$source_root/benchmark/pool_campaign.py" "$@"
