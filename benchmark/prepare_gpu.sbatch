#!/usr/bin/env bash
#SBATCH --job-name=shapiq-gpu-prep
#SBATCH --partition=main
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

# Explicit CUDA TabPFN recipes only. Each job owns one GPU and one preparation stage.
# Submit from frozen source: sbatch benchmark/prepare_gpu.sbatch SUITE STAGE SNAPSHOT
# Keep Slurm's CUDA_VISIBLE_DEVICES; do not reuse CPU payoff caches for this recipe.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen repository root}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun --cpu-bind=cores .venv/bin/python -u benchmark/prepare_matrix.py "$@" --workers 1
