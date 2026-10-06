#!/usr/bin/env bash
# Submit held from the frozen checkout; record the CPU-hour reservation before release.
# Resource limits are explicit sbatch arguments, not a whole-node allocation.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen checkout}"
export PYTHONPATH="$PWD/src" PYTHONDONTWRITEBYTECODE=1 UV_NO_SYNC=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=''
export XDG_CACHE_HOME="${SHAPIQ_CACHE_ROOT:?Set the lab cache directory}"
export HF_HOME="${HF_HOME:?Set the prewarmed lab Hugging Face cache}"
export TABPFN_MODEL_CACHE_DIR="${TABPFN_MODEL_CACHE_DIR:?Set the prewarmed TabPFN cache}"
export TORCH_HOME="$XDG_CACHE_HOME/torch" NUMBA_CACHE_DIR="$XDG_CACHE_HOME/numba"
export TMPDIR="$XDG_CACHE_HOME/tmp" JOBLIB_TEMP_FOLDER="$XDG_CACHE_HOME/tmp"
mkdir -p "$TMPDIR"
srun --cpu-bind=cores "${SHAPIQ_PYTHON:?Set the pinned interpreter}" -u "$@"
