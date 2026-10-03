#!/usr/bin/env bash
# Resource settings, arrays and dependencies are supplied by queue_phases.py.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen checkout}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Keep new campaign caches beside its lab-resident checkout. Credentials are
# neither moved nor copied; an existing lab Hugging Face cache can be reused.
export XDG_CACHE_HOME="${SHAPIQ_CACHE_ROOT:-$PWD/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export TORCH_HOME="$XDG_CACHE_HOME/torch" TABPFN_MODEL_CACHE_DIR="$XDG_CACHE_HOME/tabpfn"
export TMPDIR="$XDG_CACHE_HOME/tmp" JOBLIB_TEMP_FOLDER="$XDG_CACHE_HOME/tmp"
export NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba" TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton"
export CUDA_CACHE_PATH="$XDG_CACHE_HOME/cuda"
mkdir -p "$TMPDIR"
srun --cpu-bind=cores .venv/bin/python benchmark/phase_batch.py "$@" --index "${SLURM_ARRAY_TASK_ID:?}"
