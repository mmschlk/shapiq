"""Setup script for shapiq package with C extensions."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext


# Extend the default build_ext class to bootstrap numpy installation
# that are needed to build C extensions.
# see https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
class BuildExt(_build_ext):
    """Custom build_ext command to include numpy headers."""

    def finalize_options(self) -> None:
        """Finalize options and set numpy setup flag."""
        _build_ext.finalize_options(self)
        if isinstance(__builtins__, dict):
            __builtins__["__NUMPY_SETUP__"] = False
        else:
            __builtins__.__NUMPY_SETUP__ = False
        import numpy as np

        self.include_dirs.append(np.get_include())


def get_base_flags() -> dict[str, list[str]]:
    """Get compiler/linker flags for extensions that do NOT use OpenMP."""
    if sys.platform == "win32":  # Windows (MSVC)
        return {
            "extra_compile_args": ["/std:c++17", "/O2"],
            "extra_link_args": [],
            "include_dirs": [],
            "library_dirs": [],
        }
    # macOS and Linux
    return {
        "extra_compile_args": ["-std=c++17", "-O3", "-ffast-math"],
        "extra_link_args": [],
        "include_dirs": [],
        "library_dirs": [],
    }


def get_openmp_flags() -> dict[str, list[str]]:
    """Get OpenMP compiler and linker flags based on platform.

    Used only for the ``interventional`` extension, the only one that uses
    OpenMP. On macOS, libomp is linked STATICALLY with its symbols hidden so the
    extension carries a private OpenMP runtime that does not participate in
    dyld's process-wide weak-symbol coalescing. This prevents the cross-image
    crash where another library's OpenMP barrier (e.g. xgboost via Homebrew
    libomp) lands in shapiq's vendored libomp with a mismatched struct layout.
    Linux and Windows do not have this dyld coalescing problem and keep dynamic OpenMP.
    """
    if sys.platform == "win32":  # Windows (MSVC)
        return {
            "extra_compile_args": ["/std:c++17", "/openmp", "/O2"],
            "extra_link_args": [],
            "include_dirs": [],
            "library_dirs": [],
        }
    if sys.platform == "darwin":  # macOS
        # LIBOMP_PREFIX wins over the brew dirs — the wheel build
        # (build_tools/wheels/build_wheels.sh) sets it to a pinned conda-forge
        # libomp so wheels can target an older MACOSX_DEPLOYMENT_TARGET than
        # brew's current-SDK libomp would allow.
        candidates: list[Path] = []
        if env_prefix := os.environ.get("LIBOMP_PREFIX"):
            candidates.append(Path(env_prefix))
        candidates += [Path("/opt/homebrew/opt/libomp"), Path("/usr/local/opt/libomp")]
        for prefix in candidates:
            include_dir = prefix / "include"
            library_dir = prefix / "lib"
            libomp_archive = library_dir / "libomp.a"
            if include_dir.exists() and libomp_archive.exists():
                return {
                    "extra_compile_args": [
                        "-std=c++17",
                        "-Xpreprocessor",
                        "-fopenmp",
                        "-O3",
                        "-ffast-math",
                        # Hide our own (algorithms::*) symbols; only _PyInit_cext
                        # is exported via the linker allowlist below.
                        "-fvisibility=hidden",
                    ],
                    "extra_link_args": [
                        # Static-link the libomp archive AND hide every symbol it
                        # contributes (-load_hidden). Hidden symbols never enter
                        # the export table, so dyld cannot coalesce them with any
                        # other libomp image in the process. No LC_LOAD_DYLIB for
                        # libomp is recorded, so delocate vendors nothing.
                        f"-Wl,-load_hidden,{libomp_archive}",
                        # Drop libomp objects not reachable from the __kmpc_*
                        # entry points we actually call, bounding the size added
                        # to this single extension.
                        "-Wl,-dead_strip",
                        # Belt-and-suspenders allowlist: export only the module
                        # init symbol. Closes the coalescing surface even if a
                        # weak libomp symbol slipped past -load_hidden.
                        "-Wl,-exported_symbol,_PyInit_cext",
                    ],
                    "include_dirs": [str(include_dir)],
                }
        msg = (
            "OpenMP support on macOS requires a static libomp (libomp.a). Either "
            "install it via Homebrew (`brew install libomp`) or set LIBOMP_PREFIX "
            "to a directory containing include/omp.h and lib/libomp.a."
        )
        raise RuntimeError(msg)
    # Linux and others
    return {
        "extra_compile_args": ["-std=c++17", "-fopenmp", "-O3", "-ffast-math"],
        "extra_link_args": ["-fopenmp"],
        "include_dirs": [],
        "library_dirs": [],
    }


ext_modules = [
    Extension(
        "shapiq.tree.conversion.cext",
        sources=[
            "src/shapiq/tree/conversion/cext/cext.cc",
            "src/shapiq/tree/conversion/cext/xgboost_ubjson.cc",
            "src/shapiq/tree/conversion/cext/lightgbm_text.cc",
            "src/shapiq/tree/conversion/cext/catboost_json.cc",
        ],
        language="c++",
        # No OpenMP: this extension contains no #pragma omp / omp_* usage.
        **get_base_flags(),
    ),
    Extension(
        "shapiq.tree.interventional.cext",
        sources=[
            "src/shapiq/tree/interventional/cext/cext.cc",
        ],
        language="c++",
        # The only extension that uses OpenMP (static+hidden libomp on macOS).
        **get_openmp_flags(),
    ),
    Extension(
        "shapiq.tree.linear.cext",
        sources=[
            "src/shapiq/tree/linear/cext/cext.cc",
        ],
        language="c++",
        # No OpenMP: this extension contains no #pragma omp / omp_* usage.
        **get_base_flags(),
    ),
    Extension(
        "shapiq.graph.cext",
        sources=[
            "src/shapiq/graph/cext/cext.cc",
        ],
        language="c++",
        **get_openmp_flags(),
    ),
]

setup(
    name="shapiq",
    ext_modules=ext_modules,
    setup_requires=["numpy"],
    cmdclass={"build_ext": BuildExt},
)
