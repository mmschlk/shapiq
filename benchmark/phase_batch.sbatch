#!/usr/bin/env bash
# Resource settings, arrays and dependencies are supplied by queue_phases.py.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen checkout}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun --cpu-bind=cores .venv/bin/python benchmark/phase_batch.py "$@" --index "${SLURM_ARRAY_TASK_ID:?}"
