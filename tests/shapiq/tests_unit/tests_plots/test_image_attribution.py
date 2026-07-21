"""Tests for the image attribution heatmap plot.

These exercise :func:`shapiq.plot.vision.image_attribution_plot` and the
``InteractionValues.plot_image_attributions`` convenience method without pulling
in the optional vision (torch) stack: the interaction values and player masks are
constructed directly.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues
from shapiq.plot import image_attribution_plot


@pytest.fixture
def attribution_setup() -> tuple[InteractionValues, np.ndarray, np.ndarray]:
    """First-order interaction values with four players over a quadrant-tiled image."""
    rng = np.random.default_rng(0)
    image = rng.random((8, 8, 3))
    masks = np.zeros((4, 8, 8), dtype=bool)
    masks[0, :4, :4] = True
    masks[1, :4, 4:] = True
    masks[2, 4:, :4] = True
    masks[3, 4:, 4:] = True
    iv = InteractionValues(
        n_players=4,
        values=np.array([0.5, -0.3, 0.1, -0.2]),
        index="SV",
        min_order=1,
        max_order=1,
        estimated=False,
        baseline_value=0.0,
    )
    return iv, image, masks


def test_heatmap_only_returns_single_axis(attribution_setup) -> None:
    iv, image, masks = attribution_setup
    fig, ax = image_attribution_plot(iv, image, masks, show=False, heatmap_only=True)
    assert fig is not None
    assert ax is not None
    plt.close(fig)


def test_with_bar_returns_two_axes(attribution_setup) -> None:
    iv, image, masks = attribution_setup
    fig, (ax_heatmap, ax_bar) = image_attribution_plot(
        iv, image, masks, show=False, heatmap_only=False, region_label="Patch"
    )
    assert ax_heatmap is not None
    assert ax_bar is not None
    plt.close(fig)


def test_custom_cmap_string(attribution_setup) -> None:
    iv, image, masks = attribution_setup
    fig, _ = image_attribution_plot(iv, image, masks, show=False, cmap="viridis")
    plt.close(fig)


def test_show_returns_none(attribution_setup, monkeypatch) -> None:
    iv, image, masks = attribution_setup
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)
    assert image_attribution_plot(iv, image, masks, show=True) is None


def test_skimage_absent_falls_back(attribution_setup) -> None:
    iv, image, masks = attribution_setup
    with patch.dict(sys.modules, {"skimage": None, "skimage.segmentation": None}):
        fig, _ = image_attribution_plot(iv, image, masks, show=False)
    assert fig is not None
    plt.close(fig)


def test_interaction_values_method_delegates(attribution_setup) -> None:
    iv, image, masks = attribution_setup
    fig, ax = iv.plot_image_attributions(image, masks, show=False)
    assert fig is not None
    assert ax is not None
    plt.close(fig)
