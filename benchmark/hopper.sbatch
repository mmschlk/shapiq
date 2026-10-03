#!/usr/bin/env bash
#SBATCH --job-name=shapiq-benchmark
#SBATCH --partition=debug
#SBATCH --nodelist=himem02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=16G
#SBATCH --time=00:10:00

# Submit from the repository root; override the log path on the sbatch command.
# The whole CPU-only node is reserved, but the estimator uses one physical core.
set -euo pipefail
if [ "$#" -lt 2 ]; then
  echo 'Usage: sbatch --output=benchmark/results/timing-%j.log benchmark/hopper.sbatch SNAPSHOT OUTPUT [runner options]' >&2
  exit 2
fi
snapshot_path=$1
result_path=$2
shift 2
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository root}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
srun --cpu-bind=cores uv run python -m shapiq_benchmark.runner \
  --snapshot "$snapshot_path" --output "$result_path" \
  --timing-profile hopper-epyc9754-1c-v1 --timeout 60 --memory-gb 8 "$@"
