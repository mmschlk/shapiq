"""Setup script for shapiq package with C extensions."""

from __future__ import annotations

import sys
import subprocess
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


def get_openmp_flags():
    """Get OpenMP compiler and linker flags based on platform."""
    if sys.platform == "darwin":  # macOS
        # Try to find libomp installation from Homebrew
        try:
            brew_prefix = subprocess.check_output(["brew", "--prefix", "libomp"], text=True).strip()
            return {
                "extra_compile_args": [
                    "-Xpreprocessor",
                    "-fopenmp",
                    "-O3",
                    "-march=native",
                    "-ffast-math",
                ],
                "extra_link_args": ["-lomp"],
                "include_dirs": [f"{brew_prefix}/include"],
                "library_dirs": [f"{brew_prefix}/lib"],
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: libomp not found. Install with: brew install libomp")
            print("Building without OpenMP support.")
            return {
                "extra_compile_args": ["-O3"],
                "extra_link_args": [],
                "include_dirs": [],
                "library_dirs": [],
            }
    else:  # Linux and others
        return {
            "extra_compile_args": ["-fopenmp", "-O3", "-march=native", "-ffast-math"],
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
]

setup(
    name="shapiq",
    ext_modules=ext_modules,
    setup_requires=["numpy", "scipy"],
    cmdclass={"build_ext": BuildExt},
)
