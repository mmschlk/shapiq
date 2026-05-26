#!/bin/bash
# Build shapiq wheels with cibuildwheel.
#
# On macOS, brew-installed libomp is built against the runner's current macOS
# SDK, which forces MACOSX_DEPLOYMENT_TARGET up to that runner version. To
# produce widely-installable wheels we instead pull a pinned conda-forge
# llvm-openmp package (compiled against an old SDK), wire up the compiler
# flags so the build links against it, and let delocate vendor libomp.dylib
# into the wheel via the rpath baked in by LDFLAGS.
#
# The pin must be llvm-openmp >= 12: our C extensions use
# `#pragma omp for schedule(dynamic, ...)`, and modern clang lowers those into
# a call to __kmpc_dispatch_deinit which was first added to the libomp runtime
# in LLVM 12. Older pins (e.g. 11.1.0) build cleanly but fail at wheel-import
# time with "symbol not found in flat namespace ___kmpc_dispatch_deinit".
#
# This mirrors scikit-learn's approach. See their build script for reference:
# https://github.com/scikit-learn/scikit-learn/blob/main/build_tools/wheels/build_wheels.sh

set -euxo pipefail

if [[ $(uname) == "Darwin" ]]; then
    if [[ "${CIBW_ARCHS_MACOS:-}" == "arm64" ]]; then
        # arm64 wheels: matches SciPy's minimum (kernel-panic workaround).
        # https://github.com/scipy/scipy/issues/14688
        export MACOSX_DEPLOYMENT_TARGET=12.0
        # conda-forge build metadata: __osx >=11.0 — compatible with target 12.0.
        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/19.1.7/download/osx-arm64/llvm-openmp-19.1.7-hdb05f8b_1.conda"
    else
        # x86_64 wheels: matches CPython 3.12's documented minimum.
        export MACOSX_DEPLOYMENT_TARGET=10.13
        # conda-forge build metadata: __osx >=10.13 — compatible with target 10.13.
        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/19.1.7/download/osx-64/llvm-openmp-19.1.7-ha54dae1_1.conda"
    fi

    # Use conda purely as a tarball extractor for the pinned llvm-openmp.
    conda create -y -n build "$OPENMP_URL"

    # setup.py reads LIBOMP_PREFIX to wire up the include / library / rpath
    # flags for the C extensions. The rpath it adds is what lets delocate
    # vendor libomp.dylib into the wheel at repair time.
    export LIBOMP_PREFIX="$CONDA/envs/build"

    # Ensure system clang is used rather than anything conda put on PATH.
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
fi

uvx cibuildwheel==3.4.0 --output-dir wheelhouse
