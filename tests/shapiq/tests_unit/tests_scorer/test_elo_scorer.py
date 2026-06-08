from __future__ import annotations

import pytest

from leaderboard.scoring.elo_scorer import EloScorer, PairwiseMatch
from leaderboard.scoring.scorer_utils import filter_valid_records, group_records
from tests.shapiq.tests_unit.tests_scorer.test_group_rank_scorer import _make_test_records


def test_elo_scorer_filters_matches_by_metric_and_budget():
    scorer = EloScorer(metric_names=["spearman"], budgets=[500])
    records = _make_test_records()

    valid_records = filter_valid_records(records)
    selected_records = scorer._filter_records_by_context(valid_records)
    groups = group_records(selected_records, scorer.group_keys)
    matches = scorer._build_pairwise_matches(groups)

    assert matches
    assert all(match.metric_name == "spearman" for match in matches)
    assert all(match.group_key["budget"] == 500 for match in matches)


def test_expected_score_equal_ratings():
    """Test that equal Elo ratings produce an expected score of 0.5."""
    scorer = EloScorer()

    expected_score = scorer._expected_score(
        rating_a=1000.0,
        rating_b=1000.0,
    )

    assert expected_score == pytest.approx(0.5)


def test_expected_score_lower_rated_against_higher_rated():
    """Test expected score for a 1000-rated player against a 1200-rated player."""
    scorer = EloScorer()

    expected_score = scorer._expected_score(
        rating_a=1000.0,
        rating_b=1200.0,
    )

    assert expected_score == pytest.approx(0.240253, rel=1e-5)


def test_update_ratings_when_equal_rated_a_wins():
    """Test that a winning approximator gains Elo and the loser loses Elo."""
    scorer = EloScorer(k_factor=16.0)

    new_rating_a, new_rating_b = scorer._update_ratings(
        rating_a=1000.0,
        rating_b=1000.0,
        score_a=1.0,
        score_b=0.0,
    )

    assert new_rating_a > 1000.0
    assert new_rating_b < 1000.0

    assert new_rating_a == pytest.approx(1008.0)
    assert new_rating_b == pytest.approx(992.0)


def _make_record(
    *,
    run_id: str,
    game_id: str = "game-1",
    budget: int = 100,
    approximator_name: str,
    approx_seed: int = 0,
    mse: float,
    spearman: float = 1.0,
) -> dict[str, object]:
    """Create one minimal benchmark record for Elo scorer tests."""
    return {
        "run_id": run_id,
        "game_id": game_id,
        "game_name": "CaliforniaHousing",
        "index": "SV",
        "max_order": 1,
        "budget": budget,
        "ground_truth_method": "ExactComputer",
        "approximator_name": approximator_name,
        "approx_seed": approx_seed,
        "run_failed": False,
        "metrics": {
            "mse": mse,
            "spearman": spearman,
        },
    }


def _make_two_budget_records() -> list[dict[str, object]]:
    """Create records where A wins at budget 100 and B wins at budget 500."""
    return [
        # Budget 100: A wins by lower MSE.
        _make_record(
            run_id="a-budget-100-seed-0",
            budget=100,
            approximator_name="ApproximatorA",
            approx_seed=0,
            mse=0.01,
            spearman=0.90,
        ),
        _make_record(
            run_id="a-budget-100-seed-1",
            budget=100,
            approximator_name="ApproximatorA",
            approx_seed=1,
            mse=0.03,
            spearman=0.80,
        ),
        _make_record(
            run_id="b-budget-100-seed-0",
            budget=100,
            approximator_name="ApproximatorB",
            approx_seed=0,
            mse=0.05,
            spearman=0.60,
        ),
        _make_record(
            run_id="b-budget-100-seed-1",
            budget=100,
            approximator_name="ApproximatorB",
            approx_seed=1,
            mse=0.07,
            spearman=0.50,
        ),
        # Budget 500: B wins by lower MSE.
        _make_record(
            run_id="a-budget-500-seed-0",
            budget=500,
            approximator_name="ApproximatorA",
            approx_seed=0,
            mse=0.08,
            spearman=0.40,
        ),
        _make_record(
            run_id="a-budget-500-seed-1",
            budget=500,
            approximator_name="ApproximatorA",
            approx_seed=1,
            mse=0.10,
            spearman=0.30,
        ),
        _make_record(
            run_id="b-budget-500-seed-0",
            budget=500,
            approximator_name="ApproximatorB",
            approx_seed=0,
            mse=0.02,
            spearman=0.95,
        ),
        _make_record(
            run_id="b-budget-500-seed-1",
            budget=500,
            approximator_name="ApproximatorB",
            approx_seed=1,
            mse=0.04,
            spearman=0.85,
        ),
    ]


def test_expected_score_equal_ratings():
    """Test that equal Elo ratings produce an expected score of 0.5."""
    scorer = EloScorer()

    expected_score = scorer._expected_score(
        rating_a=1000.0,
        rating_b=1000.0,
    )

    assert expected_score == pytest.approx(0.5)


def test_expected_score_lower_rated_against_higher_rated():
    """Test expected score for a 1000-rated player against a 1200-rated player."""
    scorer = EloScorer()

    expected_score = scorer._expected_score(
        rating_a=1000.0,
        rating_b=1200.0,
    )

    assert expected_score == pytest.approx(0.240253, rel=1e-5)


def test_update_ratings_when_equal_rated_a_wins():
    """Test that a winning approximator gains Elo and the loser loses Elo."""
    scorer = EloScorer(k_factor=16.0)

    new_rating_a, new_rating_b = scorer._update_ratings(
        rating_a=1000.0,
        rating_b=1000.0,
        score_a=1.0,
        score_b=0.0,
    )

    assert new_rating_a == pytest.approx(1008.0)
    assert new_rating_b == pytest.approx(992.0)


def test_compute_elo_updates_ratings_and_match_stats():
    """Test Elo computation over two wins for the same approximator."""
    scorer = EloScorer(k_factor=16.0)

    matches = [
        PairwiseMatch(
            approximator_a="ApproximatorA",
            approximator_b="ApproximatorB",
            metric_name="mse",
            metric_value_a=0.01,
            metric_value_b=0.05,
            score_a=1.0,
            score_b=0.0,
            group_key={"budget": 100},
        ),
        PairwiseMatch(
            approximator_a="ApproximatorA",
            approximator_b="ApproximatorB",
            metric_name="mse",
            metric_value_a=0.02,
            metric_value_b=0.06,
            score_a=1.0,
            score_b=0.0,
            group_key={"budget": 500},
        ),
    ]

    ratings, stats = scorer._compute_elo(matches)

    assert ratings["ApproximatorA"] > 1000.0
    assert ratings["ApproximatorB"] < 1000.0

    assert stats["ApproximatorA"] == {
        "n_matches": 2,
        "wins": 2,
        "losses": 0,
        "ties": 0,
    }
    assert stats["ApproximatorB"] == {
        "n_matches": 2,
        "wins": 0,
        "losses": 2,
        "ties": 0,
    }


def test_elo_scorer_filters_by_metric_and_budget():
    """Test that the Elo scorer can build a context-specific leaderboard."""
    scorer = EloScorer(
        metric_names=["mse"],
        budgets=[100],
        k_factor=16.0,
    )

    result = scorer.score(_make_two_budget_records())

    rows_by_approximator = {row.approximator_name: row for row in result.rows}

    assert result.context.metric_names == ["mse"]
    assert result.context.budgets == [100]
    assert result.metadata["n_selected_records"] == 4
    assert result.metadata["n_matches"] == 1

    assert rows_by_approximator["ApproximatorA"].rank == 1
    assert rows_by_approximator["ApproximatorA"].score > 1000.0
    assert rows_by_approximator["ApproximatorA"].metadata["wins"] == 1

    assert rows_by_approximator["ApproximatorB"].rank == 2
    assert rows_by_approximator["ApproximatorB"].score < 1000.0
    assert rows_by_approximator["ApproximatorB"].metadata["losses"] == 1


def test_elo_scorer_can_rank_different_budget_contexts_differently():
    """Test that different budget filters can produce different winners."""
    records = _make_two_budget_records()

    budget_100_result = EloScorer(
        metric_names=["mse"],
        budgets=[100],
    ).score(records)

    budget_500_result = EloScorer(
        metric_names=["mse"],
        budgets=[500],
    ).score(records)

    assert budget_100_result.rows[0].approximator_name == "ApproximatorA"
    assert budget_500_result.rows[0].approximator_name == "ApproximatorB"
