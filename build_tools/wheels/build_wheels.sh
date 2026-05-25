#!/bin/bash
# Build shapiq wheels with cibuildwheel.
#
# On macOS, brew-installed libomp is built against the runner's current macOS
# SDK, which forces MACOSX_DEPLOYMENT_TARGET up to that runner version. To
# produce widely-installable wheels we instead pull a pinned conda-forge
# llvm-openmp tarball (compiled against an old SDK), wire up the compiler
# flags so the build links against it, and let delocate vendor libomp.dylib
# into the wheel via the rpath baked in by LDFLAGS.
#
# This mirrors scikit-learn's approach. See their build script for reference:
# https://github.com/scikit-learn/scikit-learn/blob/main/build_tools/wheels/build_wheels.sh

set -euxo pipefail

if [[ $(uname) == "Darwin" ]]; then
    if [[ "${CIBW_ARCHS_MACOS:-}" == "arm64" ]]; then
        # arm64 wheels: matches SciPy's minimum (kernel-panic workaround).
        # https://github.com/scipy/scipy/issues/14688
        export MACOSX_DEPLOYMENT_TARGET=12.0
        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-arm64/llvm-openmp-11.1.0-hf3c4609_1.tar.bz2"
    else
        # x86_64 wheels: target an old SDK for broad Intel-Mac compatibility.
        # If cibuildwheel rejects 10.9 for cp312+ (CPython 3.12's documented
        # minimum is 10.13), bump this to 10.13. scikit-learn currently ships
        # cp312 x86_64 wheels with 10.9, so this is expected to work.
        export MACOSX_DEPLOYMENT_TARGET=10.9
        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-64/llvm-openmp-11.1.0-hda6cdc1_1.tar.bz2"
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
