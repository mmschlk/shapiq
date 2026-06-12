"""Framework-agnostic guarantees for the vision package.

The vision modules are designed so that importing them does not eagerly import
heavy ML frameworks.  In particular, ``torch`` must only be imported lazily
(inside the methods that need it) and must never appear in any vision module's
top-level namespace.
"""

from __future__ import annotations

import sys

import pytest

import shapiq.vision.explainer  # noqa: F401 — ensure module is in sys.modules for the check below


class TestNoModuleLevelTorch:
    """No vision module imports torch at module level."""

    @pytest.mark.parametrize(
        "module_name",
        [
            "shapiq.vision.architecture",
            "shapiq.vision.explainer",
            "shapiq.vision.imputer",
            "shapiq.vision.masking",
            "shapiq.vision.players",
            "shapiq.vision.utils",
        ],
    )
    def test_torch_not_in_module_namespace(self, module_name: str) -> None:
        mod = sys.modules[module_name]
        assert "torch" not in vars(mod), (
            f"torch found at module level in {module_name}. "
            "Use a local 'import torch' inside the methods that need it."
        )
