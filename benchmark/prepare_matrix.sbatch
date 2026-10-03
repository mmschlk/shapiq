#!/usr/bin/env bash
#SBATCH --job-name=shapiq-matrix-prep
#SBATCH --partition=main
#SBATCH --nodelist=himem02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=384G
#SBATCH --time=3-00:00:00

# Submit from the frozen checkout: sbatch benchmark/prepare_matrix.sbatch SUITE STAGE SNAPSHOT
# Rerun unchanged to resume. Fewer active workers leave memory headroom per game.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the frozen repository root}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=''
srun --cpu-bind=cores .venv/bin/python -u benchmark/prepare_matrix.py "$@"
