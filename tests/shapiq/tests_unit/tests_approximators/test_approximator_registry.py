"""Tests for the per-index approximator registry lists in ``shapiq.approximator``."""

from __future__ import annotations

import pytest

from shapiq import approximator as approximator_module
from shapiq.approximator import InconsistentKernelSHAPIQ, KernelSHAPIQ


@pytest.mark.parametrize(
    ("registry_name", "index"),
    [
        ("SV_APPROXIMATORS", "SV"),
        ("SII_APPROXIMATORS", "SII"),
        ("STII_APPROXIMATORS", "STII"),
        ("FSII_APPROXIMATORS", "FSII"),
        ("FBII_APPROXIMATORS", "FBII"),
    ],
)
def test_registry_members_support_their_index(registry_name: str, index: str):
    """Every approximator listed for an index must declare that index in its valid_indices."""
    registry = getattr(approximator_module, registry_name)
    for approximator_class in registry:
        assert index in approximator_class.valid_indices, (
            f"{approximator_class.__name__} is listed in {registry_name} but does not "
            f"support the index '{index}' (valid indices: {approximator_class.valid_indices})."
        )


def test_kernelshapiq_not_listed_for_unsupported_indices():
    """KernelSHAPIQ variants only support SV/SII/k-SII and must not be listed for STII/FSII."""
    for registry_name in ("STII_APPROXIMATORS", "FSII_APPROXIMATORS"):
        registry = getattr(approximator_module, registry_name)
        assert KernelSHAPIQ not in registry
        assert InconsistentKernelSHAPIQ not in registry
