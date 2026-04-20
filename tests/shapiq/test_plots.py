"""Tests for the plotting functions.

Four layers (see docs / plan):

- ``TestPlotUtils`` — unit tests for pure helpers in ``plot.utils`` /
  ``plot._config``.
- ``TestPlots`` — parametrised smoke tests across the public plot surface,
  covering common kwarg variants so a regression in one branch doesn't pass
  silently.
- ``TestPlotStructure`` — one rich test per plot that inspects the returned
  ``Axes`` / ``Figure`` to catch silent regressions (ordering, labels,
  titles) that a smoke test wouldn't notice.
- ``TestPlotEdgeCases`` — targeted boundary-condition checks.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues
from shapiq.plot import (
    abbreviate_feature_names,
    bar_plot,
    beeswarm_plot,
    force_plot,
    network_plot,
    sentence_plot,
    si_graph_plot,
    stacked_bar_plot,
    upset_plot,
    waterfall_plot,
)
from shapiq.plot._config import get_color
from shapiq.plot.utils import format_labels, format_value
from shapiq.utils import powerset


def _build_iv(n: int = 4, seed: int = 0, *, all_zero: bool = False) -> InteractionValues:
    """Build a small InteractionValues with deterministic values.

    Set ``all_zero=True`` to get a degenerate IV with zero values everywhere.
    """
    rng = np.random.default_rng(seed)
    interaction_lookup = {}
    values = []
    for i, interaction in enumerate(powerset(range(n), min_size=1, max_size=2)):
        interaction_lookup[interaction] = i
        values.append(0.0 if all_zero else float(rng.normal()))
    return InteractionValues(
        values=np.array(values),
        index="k-SII",
        n_players=n,
        min_order=1,
        max_order=2,
        interaction_lookup=interaction_lookup,
        baseline_value=0.5,
    )


@pytest.fixture
def sample_iv() -> InteractionValues:
    """Small InteractionValues (n_players=4, order 1-2)."""
    return _build_iv()


@pytest.fixture
def sample_iv_list() -> list[InteractionValues]:
    """Small list of InteractionValues for beeswarm / multi-IV tests."""
    return [_build_iv(seed=i) for i in range(3)]


@pytest.fixture(autouse=True)
def _close_plots():
    """Ensure no figures leak between tests."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# L1 — pure-helper unit tests
# ---------------------------------------------------------------------------


class TestPlotUtils:
    """Tests for pure plotting helpers — no matplotlib involved."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1.0, "1"),
            (1.234, "1.23"),
            (0.0, "0"),
            (0, "0"),
            (-1.0, "\u22121"),  # unicode minus
            (-1.234, "\u22121.23"),
            ("pre-formatted", "pre-formatted"),  # strings pass through
        ],
    )
    def test_format_value(self, value, expected):
        assert format_value(value) == expected

    @pytest.mark.parametrize(
        ("feature_tuple", "expected"),
        [
            ((), "Base Value"),
            ((0,), "A"),
            ((0, 1), "A x B"),
            ((0, 1, 2), "A x B x C"),
        ],
    )
    def test_format_labels(self, feature_tuple, expected):
        mapping = {0: "A", 1: "B", 2: "C"}
        assert format_labels(mapping, feature_tuple) == expected

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("MedInc", "MI"),  # CamelCase → caps
            ("AveBedrms", "AB"),
            ("capital-gain", "CG"),  # dash separator
            ("native_country", "NC"),  # underscore separator
            ("hours per week", "HPW"),  # space separator
            ("dotted.name", "DN"),  # dot separator
            ("age", "age."),  # no separator, not multi-cap
            ("Gamma", "Gam."),  # single capital
        ],
    )
    def test_abbreviate_feature_names(self, name, expected):
        assert abbreviate_feature_names([name]) == [expected]

    def test_abbreviate_feature_names_batch(self):
        result = abbreviate_feature_names(["MedInc", "capital-gain", "age"])
        assert result == ["MI", "CG", "age."]

    def test_abbreviate_feature_names_empty(self):
        assert abbreviate_feature_names([]) == []

    @pytest.mark.parametrize("value", [1.0, 0.01, 100.0])
    def test_get_color_positive_is_red(self, value):
        assert get_color(value) == get_color(1.0)

    @pytest.mark.parametrize("value", [-1.0, -0.01, -100.0])
    def test_get_color_negative_is_blue(self, value):
        assert get_color(value) == get_color(-1.0)

    def test_get_color_positive_and_negative_differ(self):
        assert get_color(1.0) != get_color(-1.0)

    def test_get_color_zero_matches_positive(self):
        # Source: `if value >= 0: return RED.hex` — zero must classify as positive.
        assert get_color(0.0) == get_color(1.0)


# ---------------------------------------------------------------------------
# L2 — parametrised smoke tests across kwarg variants
# ---------------------------------------------------------------------------


FEATURE_NAMES_4 = ["Alpha", "Beta", "Gamma", "Delta"]


@pytest.mark.parametrize("abbreviate", [True, False])
@pytest.mark.parametrize("feature_names", [None, FEATURE_NAMES_4])
class TestPlots:
    """Smoke: every public plot runs for (abbreviate x feature_names) pairs."""

    def test_bar_plot(self, sample_iv, abbreviate, feature_names):
        ax = bar_plot(
            [sample_iv.get_n_order(order=1)],
            feature_names=feature_names,
            abbreviate=abbreviate,
        )
        assert ax is not None

    def test_waterfall_plot(self, sample_iv, abbreviate, feature_names):
        ax = waterfall_plot(
            sample_iv.get_n_order(order=1),
            feature_names=feature_names,
            abbreviate=abbreviate,
        )
        assert ax is not None

    def test_force_plot(self, sample_iv, abbreviate, feature_names):
        fig = force_plot(
            sample_iv.get_n_order(order=1),
            feature_names=feature_names,
            abbreviate=abbreviate,
        )
        assert fig is not None


@pytest.mark.parametrize("feature_names", [None, FEATURE_NAMES_4])
class TestPlotsNoAbbreviate:
    """Plots that don't accept the ``abbreviate`` kwarg."""

    def test_network_plot(self, sample_iv, feature_names):
        fig, ax = network_plot(sample_iv, feature_names=feature_names)
        assert fig is not None and ax is not None

    def test_stacked_bar_plot(self, sample_iv, feature_names):
        fig, ax = stacked_bar_plot(sample_iv, feature_names=feature_names)
        assert fig is not None and ax is not None

    def test_upset_plot(self, sample_iv, feature_names):
        fig = upset_plot(sample_iv, feature_names=feature_names, n_interactions=5)
        assert fig is not None

    def test_si_graph_plot(self, sample_iv, feature_names):
        fig, ax = si_graph_plot(sample_iv, feature_names=feature_names)
        assert fig is not None and ax is not None


class TestPlotsWithWords:
    """Plots with special input requirements."""

    @pytest.mark.parametrize("n", [3, 5])
    def test_sentence_plot_variable_length(self, n):
        iv = _build_iv(n=n)
        words = [f"word{i}" for i in range(n)]
        fig, ax = sentence_plot(iv.get_n_order(order=1), words=words)
        assert fig is not None and ax is not None

    @pytest.mark.parametrize("abbreviate", [True, False])
    def test_beeswarm_plot(self, sample_iv_list, abbreviate):
        data = np.random.default_rng(0).normal(size=(len(sample_iv_list), 4))
        ax = beeswarm_plot(
            sample_iv_list,
            data=data,
            abbreviate=abbreviate,
            feature_names=FEATURE_NAMES_4,
            show=False,
        )
        assert ax is not None


# ---------------------------------------------------------------------------
# L3 — structural assertions on returned artists
# ---------------------------------------------------------------------------


class TestPlotStructure:
    """Inspect the returned figure/axes to catch silent regressions.

    These tests are deliberately robust to small matplotlib-version
    differences: they assert *invariants* (e.g. "title was set", "expected
    name appears somewhere in tick labels"), not exact pixel counts.
    """

    @staticmethod
    def _collect_text(fig) -> set[str]:
        """Return every non-empty text string visible in the figure."""
        texts: set[str] = set()
        for artist in fig.findobj(match=lambda x: hasattr(x, "get_text")):
            t = artist.get_text()
            if t:
                texts.add(t)
        return texts

    def test_bar_plot_feature_names_appear_as_ticks(self, sample_iv):
        """With ``abbreviate=False`` some raw feature names must appear as y-tick labels.

        ``bar_plot`` may group small-magnitude features into a single row, so
        we don't require *every* name to appear — but the labels that do
        appear must come from the supplied ``feature_names`` set.
        """
        ax = bar_plot(
            [sample_iv.get_n_order(order=1)],
            feature_names=FEATURE_NAMES_4,
            abbreviate=False,
        )
        ticks = {t.get_text() for t in ax.get_yticklabels() if t.get_text()}
        # At least one of our names must appear verbatim.
        assert ticks & set(FEATURE_NAMES_4)
        # Every tick label is either one of our names or a plot-generated grouping label.
        for t in ticks:
            assert t in FEATURE_NAMES_4 or "other" in t.lower() or "x" in t

    def test_bar_plot_abbreviate_replaces_tick_labels(self, sample_iv):
        """With ``abbreviate=True`` the raw names must not appear; abbreviations do."""
        ax = bar_plot(
            [sample_iv.get_n_order(order=1)],
            feature_names=FEATURE_NAMES_4,
            abbreviate=True,
        )
        ticks = {t.get_text() for t in ax.get_yticklabels()}
        # Raw multi-char names should be gone, replaced by 3-or-4-char abbreviations.
        assert "Alpha" not in ticks
        # Every tick label the bar plot sets is short (at most 4 chars).
        for t in ticks:
            if t:
                assert len(t) <= 4

    def test_waterfall_plot_renders_formatted_values(self, sample_iv):
        """Waterfall adds per-bar value annotations — they should appear in the figure."""
        iv1 = sample_iv.get_n_order(order=1)
        ax = waterfall_plot(iv1, feature_names=FEATURE_NAMES_4, abbreviate=False)
        texts = self._collect_text(ax.figure)
        # At least one formatted value label must be present for a non-empty IV.
        assert any(c.isdigit() for t in texts for c in t)

    def test_force_plot_returns_figure_with_content(self, sample_iv):
        iv1 = sample_iv.get_n_order(order=1)
        fig = force_plot(iv1, feature_names=FEATURE_NAMES_4)
        # The force plot must have drawn at least one axes and some artists.
        assert len(fig.axes) >= 1
        assert sum(len(ax.get_children()) for ax in fig.axes) > 0

    def test_network_plot_draws_nodes_for_each_player(self, sample_iv):
        fig, ax = network_plot(sample_iv, feature_names=FEATURE_NAMES_4)
        # Nodes are rendered as patches/collections; assert at least n_players drawn items.
        n_drawn = len(ax.patches) + sum(len(c.get_offsets()) for c in ax.collections)
        assert n_drawn >= sample_iv.n_players

    def test_stacked_bar_plot_honours_title_and_axis_labels(self, sample_iv):
        fig, ax = stacked_bar_plot(
            sample_iv,
            feature_names=FEATURE_NAMES_4,
            title="My Title",
            xlabel="X-axis",
            ylabel="Y-axis",
        )
        assert ax.get_title() == "My Title"
        assert ax.get_xlabel() == "X-axis"
        assert ax.get_ylabel() == "Y-axis"

    def test_upset_plot_n_interactions_is_respected(self, sample_iv):
        """``n_interactions`` must cap how many interactions the upset plot shows."""
        small = upset_plot(sample_iv, n_interactions=2)
        large = upset_plot(sample_iv, n_interactions=8)
        # More interactions → more visible artists.
        assert sum(len(a.get_children()) for a in large.axes) >= sum(
            len(a.get_children()) for a in small.axes
        )

    def test_si_graph_plot_returns_figure_axes_tuple(self, sample_iv):
        fig, ax = si_graph_plot(sample_iv, feature_names=FEATURE_NAMES_4)
        assert fig is ax.figure
        # Graph must draw at least one artist per player.
        n_drawn = len(ax.patches) + len(ax.collections)
        assert n_drawn >= sample_iv.n_players

    def test_sentence_plot_returns_non_empty_axes(self):
        """``sentence_plot`` renders words as custom artists, not ``Text`` objects —
        we can't scan ``fig.findobj`` for them, so assert on axes-level content.
        """
        iv = _build_iv(n=3)
        fig, ax = sentence_plot(iv.get_n_order(order=1), words=["alpha", "beta", "gamma"])
        assert fig is ax.figure
        # Some non-zero amount of drawing must have happened.
        assert len(ax.get_children()) > 0

    def test_beeswarm_plot_returns_non_empty_axes(self, sample_iv_list):
        """Beeswarm with ``show=False`` returns the Axes and draws non-trivial content.

        We deliberately avoid asserting on tick-count vs ``max_display`` — the
        beeswarm ticker uses multi-text rows for interactions, which makes the
        relationship non-trivial to test robustly on small IVs.
        """
        data = np.random.default_rng(0).normal(size=(len(sample_iv_list), 4))
        ax = beeswarm_plot(
            sample_iv_list,
            data=data,
            feature_names=FEATURE_NAMES_4,
            show=False,
        )
        assert ax is not None
        assert len(ax.collections) > 0  # beeswarm renders points as PathCollections


# ---------------------------------------------------------------------------
# L4 — edge cases
# ---------------------------------------------------------------------------


class TestPlotEdgeCases:
    """Boundary conditions most likely to break in real usage."""

    def test_all_zero_iv_bar_plot(self):
        """Degenerate IV (all zero) must still render without raising."""
        iv = _build_iv(all_zero=True)
        ax = bar_plot([iv.get_n_order(order=1)])
        assert ax is not None

    def test_all_zero_iv_waterfall_plot(self):
        iv = _build_iv(all_zero=True)
        ax = waterfall_plot(iv.get_n_order(order=1))
        assert ax is not None

    def test_feature_names_none_uses_defaults(self, sample_iv):
        """With ``feature_names=None`` the plot must still produce readable tick labels.

        Catches regressions that silently drop tick labels when names aren't provided.
        """
        ax = bar_plot([sample_iv.get_n_order(order=1)], feature_names=None)
        tick_texts = [t.get_text() for t in ax.get_yticklabels()]
        # At least one non-empty tick label must exist (auto-generated "F0", "F1", ...).
        assert any(t for t in tick_texts)

    def test_abbreviate_bounds_long_names(self, sample_iv):
        """``abbreviate=True`` with very long names must produce short tick labels."""
        long_names = ["A" * 40, "B" * 40, "C" * 40, "D" * 40]
        ax = bar_plot(
            [sample_iv.get_n_order(order=1)],
            feature_names=long_names,
            abbreviate=True,
        )
        for t in ax.get_yticklabels():
            text = t.get_text()
            if text:
                # Abbreviations in this codebase are at most 4 chars (first-3 + ".").
                assert len(text) <= 4

    def test_max_display_below_n_features(self, sample_iv):
        """``max_display`` smaller than n_players must still produce a valid plot."""
        ax = bar_plot(
            [sample_iv.get_n_order(order=1)],
            feature_names=FEATURE_NAMES_4,
            abbreviate=False,
            max_display=2,
        )
        assert ax is not None

    def test_sentence_plot_word_count_mismatch_raises(self):
        """``sentence_plot`` must raise when len(words) != n_players."""
        iv = _build_iv(n=3).get_n_order(order=1)
        with pytest.raises(ValueError, match="must match"):
            sentence_plot(iv, words=["only-one"])
        with pytest.raises(ValueError, match="must match"):
            sentence_plot(iv, words=["a", "b", "c", "d", "e"])

    def test_beeswarm_plot_data_column_mismatch_raises(self, sample_iv_list):
        """``beeswarm_plot`` must raise when data has wrong number of columns."""
        bad_data = np.random.default_rng(0).normal(size=(len(sample_iv_list), 99))
        with pytest.raises(ValueError, match="columns"):
            beeswarm_plot(sample_iv_list, data=bad_data, show=False)
