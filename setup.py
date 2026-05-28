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


def get_openmp_flags() -> dict[str, list[str]]:
    """Get OpenMP compiler and linker flags based on platform."""
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
            libomp_dylib = library_dir / "libomp.dylib"
            if include_dir.exists() and libomp_dylib.exists():
                return {
                    "extra_compile_args": [
                        "-std=c++17",
                        "-Xpreprocessor",
                        "-fopenmp",
                        "-O3",
                        "-ffast-math",
                    ],
                    # Link libomp by ABSOLUTE PATH (not -lomp). With
                    # setuptools' default -undefined dynamic_lookup, a
                    # plain -lomp gets silently dropped from the
                    # LC_LOAD_DYLIB entries and the resulting wheel fails
                    # to load libomp at runtime ("symbol not found in
                    # flat namespace"). Passing the dylib path positionally
                    # forces the load command to be recorded. The -rpath
                    # then lets delocate find and vendor libomp into the
                    # wheel during repair.
                    "extra_link_args": [
                        str(libomp_dylib),
                        f"-Wl,-rpath,{library_dir}",
                    ],
                    "include_dirs": [str(include_dir)],
                }
        msg = (
            "OpenMP support on macOS requires libomp. Either install it via "
            "Homebrew (`brew install libomp`) or set LIBOMP_PREFIX to a "
            "directory containing include/omp.h and lib/libomp.dylib."
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
        **get_openmp_flags(),
    ),
    Extension(
        "shapiq.tree.interventional.cext",
        sources=[
            "src/shapiq/tree/interventional/cext/cext.cc",
        ],
        language="c++",
        **get_openmp_flags(),
    ),
    Extension(
        "shapiq.tree.linear.cext",
        sources=[
            "src/shapiq/tree/linear/cext/cext.cc",
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
