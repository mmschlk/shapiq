#!/bin/bash
# Build shapiq wheels with cibuildwheel.
#
# macOS OpenMP strategy: the interventional extension links libomp *statically
# and hidden* (-Wl,-load_hidden,<prefix>/lib/libomp.a; see setup.py). libomp's
# objects are baked into the extension, giving shapiq a private OpenMP runtime
# that does NOT participate in dyld's process-wide weak-symbol coalescing —
# which otherwise crashes when another libomp image is present (e.g. xgboost's
# Homebrew libomp). No separate libomp.dylib is vendored into the wheel.
#
# Because we static-link, we can use Homebrew's libomp directly and the old
# conda-forge old-SDK pin is no longer needed:
#   * Deployment target: the wheel's per-file minimum macOS is governed by our
#     own -mmacosx-version-min (MACOSX_DEPLOYMENT_TARGET below), not by libomp's.
#     The libomp objects are merged into the extension, so there is no standalone
#     libomp.dylib whose current-SDK `minos` would force the wheel target up.
#   * Symbols: libomp's object code references only ancient, stable libSystem
#     symbols (pthread_*, mach_*, sysctlbyname, __tlv_bootstrap, dlsym), so a
#     current-SDK build still runs on our old deployment targets. The nm guard
#     below fails the build if a future libomp ever uses a newer-only symbol.
#
# Requirement: libomp >= 12. Our C extensions use
# `#pragma omp for schedule(dynamic, ...)`, which modern clang lowers into a call
# to __kmpc_dispatch_deinit, first added to the libomp runtime in LLVM 12.
# Homebrew's libomp is far newer, so this is always satisfied.

set -euxo pipefail

if [[ $(uname) == "Darwin" ]]; then
    if [[ "${CIBW_ARCHS_MACOS:-}" == "arm64" ]]; then
        # arm64 wheels: matches SciPy's minimum (kernel-panic workaround).
        # https://github.com/scipy/scipy/issues/14688
        export MACOSX_DEPLOYMENT_TARGET=12.0
    else
        # x86_64 wheels: matches CPython 3.12's documented minimum.
        export MACOSX_DEPLOYMENT_TARGET=10.13
    fi

    # Static-link Homebrew's libomp into the extension (setup.py reads
    # LIBOMP_PREFIX and links <prefix>/lib/libomp.a). `brew install` is a no-op
    # if libomp is already present on the runner image.
    brew install libomp
    LIBOMP_PREFIX="$(brew --prefix libomp)"
    export LIBOMP_PREFIX

    # Guard: our deployment target is older than the SDK Homebrew built libomp
    # against. That is safe only because libomp references no newer-only macOS
    # APIs. Fail loudly at build time if that ever stops being true, instead of
    # discovering it via crash reports from users on older macOS.
    if nm -u "$LIBOMP_PREFIX/lib/libomp.a" \
        | grep -qiE 'os_unfair_lock|os_log|os_workgroup|pthread_jit|mach_msg2'; then
        echo "ERROR: libomp.a references a newer-only macOS API; revisit the" \
             "deployment target or pin an older libomp." >&2
        exit 1
    fi

    # Ensure system clang is used rather than anything on PATH.
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
fi

uvx cibuildwheel==3.4.0 --output-dir wheelhouse
