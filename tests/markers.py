"""This module contains all markers for the tests."""

import pytest

try:
    import tabpfn  # noqa

    importorskip_tabpfn = pytest.mark.skipif(False, reason="tabpfn is installed")
except ImportError:
    importorskip_tabpfn = pytest.mark.skip(reason="tabpfn is not installed")
