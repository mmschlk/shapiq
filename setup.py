"""Setup script for shapiq package with C extensions."""

from __future__ import annotations

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
        # Prefer standard Homebrew libomp locations to avoid subprocess calls in setup.
        for brew_prefix in (Path("/opt/homebrew/opt/libomp"), Path("/usr/local/opt/libomp")):
            include_dir = brew_prefix / "include"
            library_dir = brew_prefix / "lib"
            if include_dir.exists() and library_dir.exists():
                return {
                    "extra_compile_args": [
                        "-std=c++17",
                        "-Xpreprocessor",
                        "-fopenmp",
                        "-O3",
                        "-ffast-math",
                    ],
                    "extra_link_args": ["-lomp"],
                    "include_dirs": [str(include_dir)],
                    "library_dirs": [str(library_dir)],
                }
        msg = (
            "OpenMP support on macOS requires libomp. Please install it via Homebrew: "
            "brew install libomp"
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
