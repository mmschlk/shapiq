#!/usr/bin/env bash
# Re-measure every figure against current main, after a clean rebuild of the C extensions.
#
# main now carries both #588 (Quadrature-TreeSHAP) and #590 (InterventionalTreeSHAPIQ in C++,
# vectorized k-SII aggregation), so every earlier result file is superseded: #590 moves the
# interventional curve by about two orders of magnitude, and the k-SII rewrite moves every
# order-2 path-dependent point with it. Run this after
#     rm -rf build && uv run python setup.py build_ext --inplace
# -- a stale in-place build silently measures the previous version (see CLAUDE.md).
set -euo pipefail
cd "$(dirname "$0")"
export SHAPIQ_BENCH_DATA="${SHAPIQ_BENCH_DATA:-/home/user/bench_data}"

echo "=== interventional (base grid) ==="
uv run python bench_interventional.py --dataset heloc --repeats 3 --stop-after 60

echo "=== path-dependent, real ==="
uv run python bench_depth.py --suite real --repeats 5

echo "=== path-dependent, synthetic ==="
uv run python bench_depth.py --suite synthetic --repeats 5 --ignore-guard \
    --n-samples 20000 --n-features 120

echo "=== accuracy (order 1 and 2) ==="
uv run python bench_accuracy.py --suite synthetic
uv run python bench_accuracy.py --suite real

echo "FINAL SWEEP DONE"
