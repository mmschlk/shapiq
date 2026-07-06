"""Tests for optional-dependency error handling in the vision package.

When ``torch`` or ``scikit-image`` is not installed the vision modules raise an
:exc:`ImportError` with a helpful install hint.  These tests simulate missing
packages by patching ``sys.modules`` and reloading the affected modules, then
verify that the correct error is surfaced.
"""

from __future__ import annotations

import importlib
import re
import subprocess
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


class TestLazyImport:
    """The vision subpackage is imported lazily (PEP 562) so torch loads only on use."""

    def test_import_shapiq_does_not_load_vision_submodules(self):
        """``import shapiq`` must not pull the torch-heavy vision submodules."""
        code = (
            "import sys, shapiq;"
            "eager = sorted(m for m in sys.modules if m.startswith('shapiq.vision.'));"
            "assert not eager, eager"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr

    def test_attribute_access_resolves_real_classes(self):
        import shapiq
        from shapiq import vision
        from shapiq.vision.players import PatchStrategy

        assert vision.PatchStrategy is PatchStrategy
        assert shapiq.ImageExplainer.__name__ == "ImageExplainer"

    def test_dir_lists_public_names(self):
        from shapiq import vision

        listed = dir(vision)
        assert "ImageExplainer" in listed
        assert "PatchStrategy" in listed

    def test_unknown_attribute_raises(self):
        import shapiq
        from shapiq import vision

        with pytest.raises(AttributeError):
            _ = vision.DefinitelyNotAThing
        with pytest.raises(AttributeError):
            _ = shapiq.DefinitelyNotAThing

    def test_image_explainer_placeholder_when_torch_missing(self):
        """With torch absent, ``ImageExplainer`` resolves to a placeholder that raises."""
        from shapiq import vision

        submodules = [
            "shapiq.vision.explainer",
            "shapiq.vision.imputer",
            "shapiq.vision.architecture",
            "shapiq.vision.masking",
            "shapiq.vision.utils",
        ]
        saved = {name: sys.modules.pop(name, None) for name in submodules}
        try:
            with patch.dict(sys.modules, {"torch": None}):
                placeholder = vision.__getattr__("ImageExplainer")
                with pytest.raises(ImportError, match=_INSTALL_HINT):
                    placeholder()
                with pytest.raises(ImportError, match=_INSTALL_HINT):
                    vision.__getattr__("CNNArchitecture")
        finally:
            for name, module in saved.items():
                if module is not None:
                    sys.modules[name] = module
            for name in submodules:
                importlib.import_module(name)
