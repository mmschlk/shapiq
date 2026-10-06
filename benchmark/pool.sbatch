#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --no-requeue
set -euo pipefail
# Admission owns resource/budget/source checks and a byte-identical dependency bootstrap.
# It supplies BOOTSTRAP followed by its arguments; scientific Python is never started here.
bootstrap=$1
shift
exec bash "$bootstrap" "$@"
