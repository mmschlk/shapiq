#!/usr/bin/env bash
#SBATCH --job-name=shapiq-sweep
#SBATCH --partition=debug
#SBATCH --nodelist=himem02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=384G
#SBATCH --time=00:30:00

# Accuracy sweep: concurrent workers make these timings diagnostic.
# Submit from the repository root: sbatch benchmark/sweep.sbatch SNAPSHOT OUTPUT
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
# A shared editable environment can otherwise import a different checkout.
.venv/bin/python - "$1" <<'PY'
import sys
from pathlib import Path
import shapiq
import shapiq_benchmark
from shapiq_benchmark.runner import load_snapshot, provenance

root = Path.cwd() / "src"
if any(not Path(module.__file__).resolve().is_relative_to(root) for module in (shapiq, shapiq_benchmark)):
    raise SystemExit("Sweep imported a package outside the selected checkout.")
snapshot, _ = load_snapshot(Path(sys.argv[1]))
if provenance() != snapshot["provenance"]:
    raise SystemExit("Sweep source/environment differs from the prepared snapshot.")
PY
srun --cpu-bind=cores .venv/bin/python benchmark/sweep.py "$@"
