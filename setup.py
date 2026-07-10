"""Build script for the optional shapiq C extensions."""

from __future__ import annotations

import sys

from setuptools import Extension, setup


def _compile_args() -> list[str]:
    """Return C++17 optimization flags for the current platform."""
    if sys.platform == "win32":
        return ["/std:c++17", "/O2"]
    return ["-std=c++17", "-O3"]


setup(
    ext_modules=[
        Extension(
            name="shapiq.trees._interventional_cext",
            sources=["src/shapiq/trees/cext/interventional.cc"],
            extra_compile_args=_compile_args(),
            optional=True,  # pure-python installs stay functional without a compiler
        ),
    ],
)
