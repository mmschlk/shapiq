"""Tests for optional-dependency error handling in the vision package.

When ``torch`` or ``scikit-image`` is not installed the vision modules raise an
:exc:`ImportError` with a helpful install hint.  These tests simulate missing
packages by patching ``sys.modules`` and reloading the affected modules, then
verify that the correct error is surfaced.
"""

from __future__ import annotations

import importlib
import re
import sys
from unittest.mock import patch

import numpy as np
import pytest

_INSTALL_HINT = re.escape("pip install shapiq[vision]")


class TestImportError:
    @pytest.mark.parametrize(
        "module_name",
        [
            "shapiq.vision.utils",
            "shapiq.vision.masking",
            "shapiq.vision.architecture",
            "shapiq.vision.imputer",
        ],
    )
    def test_raises_import_error_when_torch_missing(self, module_name):
        """Modules that require torch raise ImportError with install hint."""
        original = sys.modules.get(module_name)
        with (
            patch.dict(sys.modules, {"torch": None}),
            pytest.raises(ImportError, match=_INSTALL_HINT),
        ):
            importlib.reload(importlib.import_module(module_name))
        if original is not None:
            sys.modules[module_name] = original
        importlib.reload(importlib.import_module(module_name))

    def test_superpixel_get_masks_raises(self):
        """SuperpixelStrategy raises ImportError with install hint when skimage is absent."""
        dummy = np.zeros((16, 16, 3), dtype=np.uint8)
        with patch.dict(sys.modules, {"skimage": None, "skimage.segmentation": None}):
            import shapiq.vision.players as players_mod

            importlib.reload(players_mod)
            strategy = players_mod.SuperpixelStrategy(n_segments=4)
            with pytest.raises(ImportError, match=_INSTALL_HINT):
                strategy.get_masks(dummy)
