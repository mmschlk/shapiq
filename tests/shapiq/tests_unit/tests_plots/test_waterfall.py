"""This module contains all tests for the waterfall plots."""

from __future__ import annotations

from io import BytesIO
from itertools import pairwise

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_rgba
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from shapiq import ExactComputer, InteractionValues, waterfall_plot


def test_waterfall_cooking_game(cooking_game):
    """Test the waterfall plot function with concrete values from the cooking game."""
    exact_computer = ExactComputer(game=cooking_game, n_players=cooking_game.n_players)
    interaction_values = exact_computer(index="k-SII", order=2)
    waterfall_plot(interaction_values, show=False)
    plt.close("all")

    # visual inspection:
    # - E[f(X)] = 10
    # - f(x) = 15
    # - 0, 1, and 2 should individually have negative contributions (go left)
    # - all interactions should have a positive +7 contribution (go right)

    waterfall_plot(interaction_values, show=False, max_display=2)
    # visual inspection:
    # - E[f(X)] = 10
    # - f(x) = 15
    # - 0x1 should have +7 contribution (go right)
    # - remaining 5 interactions should have -2 negative contributions (go left)
    # - Nowhere should there be any cutoffs
    plt.close("all")


def test_waterfall_plot(interaction_values_list: list[InteractionValues]):
    """Test the waterfall plot function."""
    iv = interaction_values_list[0]
    n_players = iv.n_players
    feature_names = [f"feature-{i}" for i in range(n_players)]
    feature_names = np.array(feature_names)

    wp = waterfall_plot(iv, show=False)
    assert wp is not None
    assert isinstance(wp, plt.Axes)
    plt.close()

    wp = waterfall_plot(iv, show=False, feature_names=feature_names)
    assert isinstance(wp, plt.Axes)
    plt.close()

    iv = iv.get_n_order(1)
    wp = waterfall_plot(iv, show=False)
    assert isinstance(wp, plt.Axes)
    plt.close()

    wp = iv.plot_waterfall(show=False)
    assert isinstance(wp, plt.Axes)
    plt.close()


@pytest.fixture
def waterfall_with_feature_values():
    """Render main effects and an interaction with explicit feature values."""
    iv = InteractionValues(
        values={(0,): 0.25, (1,): -0.15, (0, 1): 0.35},
        index="k-SII",
        max_order=2,
        min_order=1,
        n_players=2,
        baseline_value=0.5,
    )
    fig = plt.figure(dpi=100)
    try:
        iv.plot_waterfall(
            feature_names=np.array(["Age=25", "Km=10"]),
            abbreviate=False,
            show=False,
        )
        fig.canvas.draw()
        yield fig, fig.axes[0]
    finally:
        plt.close(fig)


def test_waterfall_feature_value_colors(waterfall_with_feature_values):
    """Color both values black in main-effect and interaction labels."""
    _, ax = waterfall_with_feature_values
    texts = ax.findobj(match=Text)
    for label, color, count in [
        ("Age=", "#999999", 2),
        ("25", "black", 2),
        (" x ", "#999999", 1),
        ("Km=", "#999999", 2),
        ("10", "black", 2),
    ]:
        matching = [text for text in texts if text.get_text() == label]
        assert len(matching) == count
        assert all(to_rgba(text.get_color()) == to_rgba(color) for text in matching)


@pytest.mark.parametrize("rendering", ["resize", "dpi", "tight", "svg", "pdf", "layout"])
def test_waterfall_label_alignment(waterfall_with_feature_values, rendering):
    """Keep interaction labels aligned and their segments adjacent on each redraw."""
    fig, ax = waterfall_with_feature_values
    texts = ax.findobj(match=Text)
    # The interaction has the largest contribution and occupies the top row.
    segments = [
        max(
            (text for text in texts if text.get_text() == label),
            key=lambda text: text.get_window_extent().y1,
        )
        for label in ["Age=", "25", " x ", "Km=", "10"]
    ]
    measurements = []

    def record_alignment(event):
        bounds = [text.get_window_extent(renderer=event.renderer) for text in segments]
        label_bounds = Bbox.union(bounds)
        row_y = ax.transData.transform((0, 2))[1]
        right_edge = ax.transAxes.transform((0, 0))[0] - event.renderer.points_to_pixels(5)
        measurements.append(
            [
                (label_bounds.y0 + label_bounds.y1) / 2 - row_y,
                label_bounds.x1 - right_edge,
                *[right.x0 - left.x1 for left, right in pairwise(bounds)],
            ],
        )

    callback = fig.canvas.mpl_connect("draw_event", record_alignment)
    try:
        if rendering == "resize":
            fig.set_size_inches(fig.get_figwidth() * 0.8, fig.get_figheight() * 1.5)
            fig.canvas.draw()
        elif rendering == "layout":
            fig.tight_layout()
            fig.canvas.draw()
        else:
            with BytesIO() as output:
                if rendering == "dpi":
                    fig.savefig(output, format="png", dpi=200)
                elif rendering == "tight":
                    fig.savefig(output, format="png", bbox_inches="tight")
                else:
                    fig.savefig(output, format=rendering)
    finally:
        fig.canvas.mpl_disconnect(callback)

    assert measurements
    # Font hinting can make glyph bounds slightly smaller than the layout box.
    np.testing.assert_allclose(measurements, 0, atol=0.5, rtol=0)
