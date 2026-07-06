"""Tests for :mod:`leaderboard.scoring.cd_scorer`.

Covers:
    * The studentized-range-based Nemenyi q_alpha (replaces the old fixed
      Demsar Table 5 lookup) -- correctness against the published table
      *and* generalization beyond k = 10 classifiers.
    * The Nemenyi CD formula built on top of q_alpha.
    * Alpha validation (now any value in (0, 1), not just {0.05, 0.10}).
    * Fractional-rank assignment with ties.
    * The Friedman test wrapper (Wilcoxon fallback for k=2, chi-square
      otherwise).
    * Clique detection.
    * The end-to-end ``score()`` pipeline, including with > 10 approximators.
    * The CD diagram plotting logic.
"""

from __future__ import annotations

import itertools
import math

import matplotlib.figure
import numpy as np
import pytest

from leaderboard.scoring.cd_scorer import (
    CriticalDifferenceResult,
    CriticalDifferenceScorer,
    _nemenyi_q_alpha,
)

# ---------------------------------------------------------------------------
# Reference values, Demsar (2006) Table 5: two-tailed Nemenyi q_alpha for
# k = 2..10 classifiers (including control).
# https://jmlr.org/papers/volume7/demsar06a/demsar06a.pdf
# ---------------------------------------------------------------------------
_DEMSAR_TABLE_5_Q005 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
}
_DEMSAR_TABLE_5_Q010 = {
    2: 1.645,
    3: 2.052,
    4: 2.291,
    5: 2.459,
    6: 2.589,
    7: 2.693,
    8: 2.780,
    9: 2.855,
    10: 2.920,
}


def _make_records(
    n_approx: int,
    n_groups: int,
    *,
    metric: str = "mse",
    consistent_ranking: bool = True,
) -> list[dict[str, object]]:
    """Build synthetic benchmark records with a known, deterministic ranking.

    Args:
        n_approx: Number of approximators to generate (``approx_0`` is
            always the best performer under ``mse`` -- lower is better).
        n_groups: Number of distinct (game, budget) comparable groups.
        metric: Metric column name to populate.
        consistent_ranking: If True, every group ranks approximators in
            exactly the same order (approx_0 best, approx_{n-1} worst) --
            this is the "clearly significant differences" case. If False,
            each group applies a cyclic rotation of the ranking, which
            balances out to a "no significant difference" case.

    Returns:
        List of flat record dicts consumable by :meth:`CriticalDifferenceScorer.score`.
    """
    approx_names = [f"approx_{i}" for i in range(n_approx)]
    records: list[dict[str, object]] = []
    for g in range(n_groups):
        shift = 0 if consistent_ranking else g % n_approx
        for i, approx in enumerate(approx_names):
            # Rotate which approximator gets which value when not consistent,
            # so that averaged across all groups every approximator sees
            # every rank exactly once (a balanced Latin square).
            effective_rank_index = (i + shift) % n_approx
            value = float(effective_rank_index)
            records.extend(
                {
                    "approximator_name": approx,
                    "game_name": f"game_{g}",
                    "index": "SV",
                    "max_order": 1,
                    "budget": 100,
                    "ground_truth_method": "exact",
                    "seed": seed,
                    metric: value,
                }
                for seed in range(2)
            )
    return records


# ---------------------------------------------------------------------------
# Nemenyi q_alpha
# ---------------------------------------------------------------------------


class TestNemenyiQAlpha:
    @pytest.mark.parametrize("k, expected", sorted(_DEMSAR_TABLE_5_Q005.items()))
    def test_matches_demsar_table_5_alpha_005(self, k: int, expected: float) -> None:
        q = _nemenyi_q_alpha(0.05, k)
        assert q == pytest.approx(expected, abs=1e-2)

    @pytest.mark.parametrize("k, expected", sorted(_DEMSAR_TABLE_5_Q010.items()))
    def test_matches_demsar_table_5_alpha_010(self, k: int, expected: float) -> None:
        q = _nemenyi_q_alpha(0.10, k)
        assert q == pytest.approx(expected, abs=1e-2)

    def test_generalizes_beyond_table_5(self) -> None:
        """k > 10 should still produce a finite, sensible q_alpha."""
        q10 = _nemenyi_q_alpha(0.05, 10)
        q15 = _nemenyi_q_alpha(0.05, 15)
        q25 = _nemenyi_q_alpha(0.05, 25)
        assert math.isfinite(q15)
        assert math.isfinite(q25)
        # q_alpha grows monotonically with the number of classifiers.
        assert q10 < q15 < q25

    def test_monotonic_in_k(self) -> None:
        values = [_nemenyi_q_alpha(0.05, k) for k in range(2, 21)]
        assert all(a < b for a, b in itertools.pairwise(values))

    def test_smaller_alpha_gives_larger_q(self) -> None:
        """Stricter significance threshold -> larger critical value."""
        for k in (2, 5, 10, 20):
            assert _nemenyi_q_alpha(0.01, k) > _nemenyi_q_alpha(0.05, k) > _nemenyi_q_alpha(0.10, k)

    def test_k_less_than_2_returns_zero(self) -> None:
        assert _nemenyi_q_alpha(0.05, 1) == 0.0
        assert _nemenyi_q_alpha(0.05, 0) == 0.0


# ---------------------------------------------------------------------------
# Alpha validation (generalized from the old {0.05, 0.10}-only set)
# ---------------------------------------------------------------------------


class TestAlphaValidation:
    @pytest.mark.parametrize("alpha", [0.01, 0.05, 0.10, 0.20, 0.001, 0.4999])
    def test_accepts_any_alpha_in_open_unit_interval(self, alpha: float) -> None:
        scorer = CriticalDifferenceScorer(alpha=alpha)
        assert scorer.alpha == alpha

    @pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, 2.0])
    def test_rejects_alpha_outside_open_unit_interval(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            CriticalDifferenceScorer(alpha=alpha)


# ---------------------------------------------------------------------------
# Nemenyi CD formula
# ---------------------------------------------------------------------------


class TestNemenyiCD:
    def test_formula(self) -> None:
        scorer = CriticalDifferenceScorer(alpha=0.05)
        n_approx, n_groups = 8, 20
        cd = scorer._nemenyi_cd(n_approx=n_approx, n_groups=n_groups)
        expected = _nemenyi_q_alpha(0.05, n_approx) * math.sqrt(
            n_approx * (n_approx + 1) / (6.0 * n_groups)
        )
        assert cd == pytest.approx(expected)

    def test_more_than_ten_approximators_supported(self) -> None:
        """This is the main capability unlocked by the studentized-range change."""
        scorer = CriticalDifferenceScorer(alpha=0.05)
        cd = scorer._nemenyi_cd(n_approx=17, n_groups=30)
        assert cd > 0
        assert math.isfinite(cd)

    def test_cd_decreases_with_more_groups(self) -> None:
        scorer = CriticalDifferenceScorer(alpha=0.05)
        cd_small_n = scorer._nemenyi_cd(n_approx=6, n_groups=5)
        cd_large_n = scorer._nemenyi_cd(n_approx=6, n_groups=50)
        assert cd_large_n < cd_small_n

    def test_cd_increases_with_more_approximators(self) -> None:
        scorer = CriticalDifferenceScorer(alpha=0.05)
        cd_few = scorer._nemenyi_cd(n_approx=3, n_groups=20)
        cd_many = scorer._nemenyi_cd(n_approx=12, n_groups=20)
        assert cd_many > cd_few

    @pytest.mark.parametrize(("n_approx", "n_groups"), [(1, 10), (0, 10), (5, 0)])
    def test_degenerate_inputs_return_zero(self, n_approx: int, n_groups: int) -> None:
        scorer = CriticalDifferenceScorer(alpha=0.05)
        assert scorer._nemenyi_cd(n_approx=n_approx, n_groups=n_groups) == 0.0


# ---------------------------------------------------------------------------
# Fractional ranking (with ties)
# ---------------------------------------------------------------------------


class TestRanking:
    def test_no_ties_lower_is_better(self) -> None:
        scorer = CriticalDifferenceScorer(alpha=0.05)
        records = [
            {"approximator_name": "a", "mse": 0.3},
            {"approximator_name": "b", "mse": 0.1},
            {"approximator_name": "c", "mse": 0.2},
        ]
        ranks = scorer._rank_records(records, metric_name="mse", higher_is_better=False)
        assert ranks == {"b": 1.0, "c": 2.0, "a": 3.0}

    def test_no_ties_higher_is_better(self) -> None:
        scorer = CriticalDifferenceScorer(alpha=0.05)
        records = [
            {"approximator_name": "a", "r2": 0.9},
            {"approximator_name": "b", "r2": 0.5},
            {"approximator_name": "c", "r2": 0.7},
        ]
        ranks = scorer._rank_records(records, metric_name="r2", higher_is_better=True)
        assert ranks == {"a": 1.0, "c": 2.0, "b": 3.0}

    def test_ties_receive_averaged_rank(self) -> None:
        scorer = CriticalDifferenceScorer(alpha=0.05)
        records = [
            {"approximator_name": "a", "mse": 0.1},
            {"approximator_name": "b", "mse": 0.2},
            {"approximator_name": "c", "mse": 0.2},
            {"approximator_name": "d", "mse": 0.4},
        ]
        ranks = scorer._rank_records(records, metric_name="mse", higher_is_better=False)
        # b and c tie for positions 2 and 3 -> both get 2.5.
        assert ranks == {"a": 1.0, "b": 2.5, "c": 2.5, "d": 4.0}

    def test_three_way_tie(self) -> None:
        scorer = CriticalDifferenceScorer(alpha=0.05)
        records = [
            {"approximator_name": "a", "mse": 0.5},
            {"approximator_name": "b", "mse": 0.5},
            {"approximator_name": "c", "mse": 0.5},
        ]
        ranks = scorer._rank_records(records, metric_name="mse", higher_is_better=False)
        assert ranks == {"a": 2.0, "b": 2.0, "c": 2.0}


# ---------------------------------------------------------------------------
# Friedman test wrapper
# ---------------------------------------------------------------------------


class TestFriedmanTest:
    def test_too_few_groups_or_approximators_returns_neutral_result(self) -> None:
        stat, p = CriticalDifferenceScorer._friedman_test(np.empty((0, 3)))
        assert (stat, p) == (0.0, 1.0)

        stat, p = CriticalDifferenceScorer._friedman_test(np.ones((5, 1)))
        assert (stat, p) == (0.0, 1.0)

    def test_identical_rankings_across_blocks_is_significant(self) -> None:
        """Same ranking in every block -> maximal Friedman statistic, tiny p."""
        n_groups, n_approx = 10, 5
        row = np.arange(1, n_approx + 1, dtype=float)
        rank_matrix = np.tile(row, (n_groups, 1))
        stat, p = CriticalDifferenceScorer._friedman_test(rank_matrix)
        assert stat == pytest.approx(n_groups * (n_approx - 1), rel=1e-6)
        assert p < 0.001

    def test_balanced_latin_square_is_not_significant(self) -> None:
        """Every approximator sees every rank equally often -> no difference."""
        n_approx = 5

        rows = [
            [((i + shift) % n_approx) + 1 for i in range(n_approx)] for shift in range(n_approx)
        ]
        rank_matrix = np.array(rows, dtype=float)
        stat, p = CriticalDifferenceScorer._friedman_test(rank_matrix)
        assert stat == pytest.approx(0.0, abs=1e-8)
        assert p == pytest.approx(1.0, abs=1e-6)

    def test_two_approximators_uses_wilcoxon(self) -> None:
        rank_matrix = np.array([[1.0, 2.0], [1.0, 2.0], [2.0, 1.0], [1.0, 2.0]])
        stat, p = CriticalDifferenceScorer._friedman_test(rank_matrix)
        assert stat >= 0.0
        assert 0.0 <= p <= 1.0

    def test_two_approximators_all_tied_returns_neutral(self) -> None:
        rank_matrix = np.array([[1.5, 1.5], [1.5, 1.5]])
        stat, p = CriticalDifferenceScorer._friedman_test(rank_matrix)
        assert (stat, p) == (0.0, 1.0)


# ---------------------------------------------------------------------------
# Clique detection
# ---------------------------------------------------------------------------


class TestCliqueDetection:
    def test_finds_expected_maximal_cliques(self) -> None:
        mean_ranks = {"a": 1.0, "b": 1.2, "c": 1.5, "d": 3.0, "e": 3.1}
        cliques = CriticalDifferenceScorer._find_cliques(list(mean_ranks), mean_ranks, cd=0.6)
        clique_sets = {frozenset(c) for c in cliques}
        assert clique_sets == {frozenset({"a", "b", "c"}), frozenset({"d", "e"})}

    def test_no_cliques_when_cd_is_zero(self) -> None:
        mean_ranks = {"a": 1.0, "b": 1.2, "c": 1.5}
        cliques = CriticalDifferenceScorer._find_cliques(list(mean_ranks), mean_ranks, cd=0.0)
        assert cliques == []

    def test_single_clique_when_cd_is_huge(self) -> None:
        mean_ranks = {"a": 1.0, "b": 5.0, "c": 10.0}
        cliques = CriticalDifferenceScorer._find_cliques(list(mean_ranks), mean_ranks, cd=100.0)
        assert len(cliques) == 1
        assert set(cliques[0]) == {"a", "b", "c"}

    def test_no_subset_cliques_returned(self) -> None:
        mean_ranks = {"a": 1.0, "b": 1.1, "c": 1.2}
        cliques = CriticalDifferenceScorer._find_cliques(list(mean_ranks), mean_ranks, cd=0.5)
        # {a,b}, {b,c}, and {a,b,c} would all qualify pairwise; only the
        # maximal {a,b,c} should survive.
        assert len(cliques) == 1
        assert set(cliques[0]) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


class TestScorePipeline:
    def test_basic_pipeline_orders_rows_by_mean_rank(self) -> None:
        records = _make_records(n_approx=4, n_groups=8, consistent_ranking=True)
        scorer = CriticalDifferenceScorer(alpha=0.05, metric_names=["mse"])
        result = scorer.score(records)

        cd_result = result.metadata["cd_result"]
        assert cd_result.n_approximators == 4
        assert cd_result.n_groups == 8
        assert cd_result.friedman_significant is True

        approx_order = [row.approximator_name for row in result.rows]
        assert approx_order == ["approx_0", "approx_1", "approx_2", "approx_3"]

        # Ranks increase, scores (negated mean rank) decrease.
        ranks = [row.rank for row in result.rows]
        assert ranks == [1, 2, 3, 4]
        scores = [row.score for row in result.rows]
        assert scores == sorted(scores, reverse=True)
        for row in result.rows:
            assert row.score == pytest.approx(-row.metadata["mean_rank"])
            assert row.higher_is_better is False

    def test_balanced_design_yields_no_significant_difference(self) -> None:
        records = _make_records(n_approx=5, n_groups=5, consistent_ranking=False)
        scorer = CriticalDifferenceScorer(alpha=0.05, metric_names=["mse"])
        result = scorer.score(records)
        cd_result = result.metadata["cd_result"]
        assert cd_result.friedman_significant is False
        # All approximators should end up with (near) identical mean rank.
        ranks = list(cd_result.mean_ranks.values())
        assert max(ranks) - min(ranks) == pytest.approx(0.0, abs=1e-8)

    def test_supports_more_than_ten_approximators(self) -> None:
        """The original table only covered k in [2, 10]; this must now work."""
        n_approx = 17
        records = _make_records(n_approx=n_approx, n_groups=10, consistent_ranking=True)
        scorer = CriticalDifferenceScorer(alpha=0.05, metric_names=["mse"])
        result = scorer.score(records)
        cd_result = result.metadata["cd_result"]

        assert cd_result.n_approximators == n_approx
        assert cd_result.critical_difference > 0
        assert math.isfinite(cd_result.critical_difference)
        assert len(result.rows) == n_approx
        # Best-to-worst ordering should follow the synthetic construction.
        approx_order = [row.approximator_name for row in result.rows]
        assert approx_order == [f"approx_{i}" for i in range(n_approx)]

    def test_empty_records_returns_empty_result(self) -> None:
        scorer = CriticalDifferenceScorer(alpha=0.05, metric_names=["mse"])
        result = scorer.score([])
        cd_result = result.metadata["cd_result"]
        assert cd_result.n_approximators == 0
        assert cd_result.n_groups == 0
        assert result.rows == []

    def test_metadata_matches_cd_result_fields(self) -> None:
        records = _make_records(n_approx=6, n_groups=6, consistent_ranking=True)
        scorer = CriticalDifferenceScorer(alpha=0.10, metric_names=["mse"])
        result = scorer.score(records)
        cd_result = result.metadata["cd_result"]
        assert result.metadata["alpha"] == 0.10
        assert result.metadata["n_approximators"] == cd_result.n_approximators
        assert result.metadata["n_groups"] == cd_result.n_groups
        assert result.metadata["critical_difference"] == cd_result.critical_difference
        assert result.metadata["friedman_p_value"] == cd_result.friedman_p_value
        assert result.metadata["friedman_significant"] == cd_result.friedman_significant


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


class TestPlotting:
    @pytest.fixture
    def cd_result(self) -> CriticalDifferenceResult:
        mean_ranks = {"a": 1.2, "b": 1.8, "c": 3.0, "d": 4.0, "e": 4.5}
        return CriticalDifferenceResult(
            mean_ranks=mean_ranks,
            critical_difference=1.0,
            friedman_statistic=12.3,
            friedman_p_value=0.001,
            friedman_significant=True,
            n_groups=20,
            n_approximators=5,
            alpha=0.05,
            cliques=[["a", "b"], ["c", "d", "e"]],
        )

    def test_plot_cd_diagram_returns_figure(self, cd_result: CriticalDifferenceResult) -> None:
        scorer = CriticalDifferenceScorer(alpha=0.05)
        fig = scorer.plot_cd_diagram(cd_result, title="My CD diagram")
        assert isinstance(fig, matplotlib.figure.Figure)
        assert fig.axes[0].get_title() == "My CD diagram"

    def test_plot_cd_diagram_uses_provided_axes(self, cd_result: CriticalDifferenceResult) -> None:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        returned_fig = CriticalDifferenceScorer(alpha=0.05).plot_cd_diagram(cd_result, ax=ax)
        assert returned_fig is fig

    def test_plot_cd_diagram_requires_at_least_two_approximators(self) -> None:
        cd_result = CriticalDifferenceResult(
            mean_ranks={"only_one": 1.0},
            critical_difference=0.0,
            friedman_statistic=0.0,
            friedman_p_value=1.0,
            friedman_significant=False,
            n_groups=5,
            n_approximators=1,
            alpha=0.05,
            cliques=[],
        )
        scorer = CriticalDifferenceScorer(alpha=0.05)
        with pytest.raises(ValueError, match="at least 2"):
            scorer.plot_cd_diagram(cd_result)

    def test_plot_handles_many_approximators(self) -> None:
        """Diagram should still render for > 10 approximators without error."""
        n = 20
        mean_ranks = {f"approx_{i}": float(i + 1) for i in range(n)}
        cd_result = CriticalDifferenceResult(
            mean_ranks=mean_ranks,
            critical_difference=2.5,
            friedman_statistic=99.0,
            friedman_p_value=1e-10,
            friedman_significant=True,
            n_groups=30,
            n_approximators=n,
            alpha=0.05,
            cliques=[[f"approx_{i}" for i in range(5)]],
        )
        scorer = CriticalDifferenceScorer(alpha=0.05)
        fig = scorer.plot_cd_diagram(cd_result)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_convenience_method_reads_metadata(self) -> None:
        records = _make_records(n_approx=4, n_groups=8, consistent_ranking=True)
        scorer = CriticalDifferenceScorer(alpha=0.05, metric_names=["mse"])
        result = scorer.score(records)
        fig = scorer.plot(result, title="From pipeline")
        assert isinstance(fig, matplotlib.figure.Figure)
        assert fig.axes[0].get_title() == "From pipeline"

    def test_plot_missing_cd_result_raises_key_error(self) -> None:
        from leaderboard.scoring import ScoringResult
        from leaderboard.scoring.result import ScoringContext

        scorer = CriticalDifferenceScorer(alpha=0.05)
        empty_result = ScoringResult(
            scorer_name="other",
            context=ScoringContext(
                game_names=[], indices=[], budgets=[], metric_names=[], group_keys=[]
            ),
            rows=[],
            group_results=[],
            metadata={},
        )
        with pytest.raises(KeyError):
            scorer.plot(empty_result)
