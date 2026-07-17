"""Tests for optional-dependency error handling in the vision package.

When ``torch`` or ``scikit-image`` is not installed the vision modules raise an
:exc:`ImportError` with a helpful install hint. These tests simulate a missing
package and check that the friendly error is surfaced.

Each such test runs in a **subprocess** for two reasons. First, faking a missing
dependency in-process would mutate the interpreter's module graph (reloading a
module rebinds its classes to new objects, desynchronising the ``from x import
Y`` references other modules hold) and silently break unrelated tests that run
afterwards. Second, the missing package is simulated with a ``meta_path`` finder
that raises :exc:`ModuleNotFoundError`, *not* by assigning ``sys.modules[name] =
None`` -- the latter leaves a ``None`` sentinel that third-party code probing
``sys.modules`` (e.g. scipy's array-API torch detection) dereferences into an
``AttributeError``. Blocking the import cleanly reproduces "not installed"
without that landmine.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

_INSTALL_HINT = "pip install shapiq[vision]"


def _run_isolated(body: str, *, without: tuple[str, ...]) -> None:
    """Run ``body`` in a fresh interpreter with ``without`` packages unimportable.

    A ``meta_path`` finder raises ModuleNotFoundError for the blocked top-level
    packages, so ``import x`` behaves as if they were not installed while leaving
    ``sys.modules`` untouched. The snippet must ``sys.exit(0)`` on success and
    ``sys.exit(<message>)`` on failure, so an unmet expectation surfaces as an
    assertion carrying that message.
    """
    blocker = textwrap.dedent(
        """
        import sys
        import importlib.abc


        class _Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".", 1)[0] in __BLOCKED__:
                    raise ModuleNotFoundError("No module named " + repr(name), name=name)
                return None


        sys.meta_path.insert(0, _Blocker())
        """
    ).replace("__BLOCKED__", repr(without))
    result = subprocess.run(
        [sys.executable, "-c", blocker + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


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
        _run_isolated(
            f"""
            import importlib
            try:
                importlib.import_module("{module_name}")
            except ImportError as err:
                assert {_INSTALL_HINT!r} in str(err), str(err)
                sys.exit(0)
            sys.exit("no ImportError raised for {module_name}")
            """,
            without=("torch",),
        )

    def test_superpixel_get_masks_raises(self):
        """SuperpixelStrategy raises ImportError with install hint when skimage is absent."""
        _run_isolated(
            f"""
            import numpy as np
            from shapiq.vision.players import SuperpixelStrategy

            strategy = SuperpixelStrategy(n_segments=4)
            try:
                strategy.get_masks(np.zeros((16, 16, 3), dtype=np.uint8))
            except ImportError as err:
                assert {_INSTALL_HINT!r} in str(err), str(err)
                sys.exit(0)
            sys.exit("SuperpixelStrategy.get_masks did not raise ImportError")
            """,
            without=("skimage",),
        )


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

    def test_every_exported_name_resolves(self):
        """``__all__`` and ``_LAZY_MODULES`` must agree, or the export is dead on arrival.

        A name in ``__all__`` without a ``_LAZY_MODULES`` entry raises AttributeError
        on access, which is exactly how ``GridStrategy`` and ``CustomPlayerStrategy``
        were unreachable despite being public.
        """
        from shapiq import vision

        for name in vision.__all__:
            assert getattr(vision, name) is not None, name

    def test_lazy_module_map_matches_all(self):
        from shapiq import vision

        assert set(vision._LAZY_MODULES) == set(vision.__all__)

    def test_exported_names_are_the_real_objects(self):
        """Lazy resolution must hand back the module's object, not a copy or placeholder."""
        from shapiq import vision
        from shapiq.vision.players import CustomPlayerStrategy, GridStrategy, labels_to_masks

        assert vision.GridStrategy is GridStrategy
        assert vision.CustomPlayerStrategy is CustomPlayerStrategy
        assert vision.labels_to_masks is labels_to_masks

    def test_unknown_attribute_raises(self):
        import shapiq
        from shapiq import vision

        with pytest.raises(AttributeError):
            _ = vision.DefinitelyNotAThing
        with pytest.raises(AttributeError):
            _ = shapiq.DefinitelyNotAThing

    def test_image_explainer_placeholder_when_torch_missing(self):
        """With torch absent, ``ImageExplainer`` resolves to a placeholder that raises."""
        _run_isolated(
            f"""
            from shapiq import vision

            placeholder = vision.__getattr__("ImageExplainer")
            try:
                placeholder()
            except ImportError as err:
                assert {_INSTALL_HINT!r} in str(err), str(err)
            else:
                sys.exit("ImageExplainer placeholder did not raise")

            try:
                vision.__getattr__("ClassificationArchitecture")
            except ImportError as err:
                assert {_INSTALL_HINT!r} in str(err), str(err)
                sys.exit(0)
            sys.exit("ClassificationArchitecture did not raise")
            """,
            without=("torch",),
        )
