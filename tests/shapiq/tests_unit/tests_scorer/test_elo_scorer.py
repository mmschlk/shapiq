"""Tests for the Elo scorer."""

from __future__ import annotations

import pytest

from leaderboard.scoring.elo_scorer import EloScorer, PairwiseMatch
from leaderboard.scoring.scorer_utils import filter_valid_records, group_records


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
        # Budget 100: A wins by lower MSE and higher Spearman.
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
        # Budget 500: B wins by lower MSE and higher Spearman.
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


def test_elo_scorer_filters_matches_by_metric_and_budget():
    """Test that pairwise matches are restricted by metric and budget filters."""
    scorer = EloScorer(metric_names=["spearman"], budgets=[500])
    records = _make_two_budget_records()

    valid_records = filter_valid_records(records)
    selected_records = scorer._filter_records_by_context(valid_records)
    groups = group_records(selected_records, scorer.group_keys)
    comparable_groups = scorer._build_comparable_groups(groups)
    matches = scorer._build_pairwise_matches(comparable_groups)

    assert matches
    assert len(matches) == 1
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
    assert rows_by_approximator["ApproximatorA"].metadata["score_std"] == pytest.approx(0.0)
    assert rows_by_approximator["ApproximatorA"].metadata["n_rating_samples"] == 1

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


def test_compute_elo_returns_expected_rating_after_three_matches():
    """Test concrete Elo ratings after three sequential wins."""
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
        PairwiseMatch(
            approximator_a="ApproximatorA",
            approximator_b="ApproximatorB",
            metric_name="mse",
            metric_value_a=0.03,
            metric_value_b=0.07,
            score_a=1.0,
            score_b=0.0,
            group_key={"budget": 1000},
        ),
    ]

    ratings, stats = scorer._compute_elo(matches)

    assert ratings["ApproximatorA"] == pytest.approx(1022.9139101641008)
    assert ratings["ApproximatorB"] == pytest.approx(977.0860898358992)

    assert stats["ApproximatorA"] == {
        "n_matches": 3,
        "wins": 3,
        "losses": 0,
        "ties": 0,
    }
    assert stats["ApproximatorB"] == {
        "n_matches": 3,
        "wins": 0,
        "losses": 3,
        "ties": 0,
    }

def _make_cyclic_matches() -> list[PairwiseMatch]:
    """Create matches where order can affect final Elo ratings."""
    return [
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
            approximator_a="ApproximatorB",
            approximator_b="ApproximatorC",
            metric_name="mse",
            metric_value_a=0.01,
            metric_value_b=0.05,
            score_a=1.0,
            score_b=0.0,
            group_key={"budget": 500},
        ),
        PairwiseMatch(
            approximator_a="ApproximatorC",
            approximator_b="ApproximatorA",
            metric_name="mse",
            metric_value_a=0.01,
            metric_value_b=0.05,
            score_a=1.0,
            score_b=0.0,
            group_key={"budget": 1000},
        ),
    ]

def test_generate_match_orderings_with_one_permutation_returns_one_ordering():
    """Test that n_permutations=1 returns exactly one deterministic ordering."""
    scorer = EloScorer(n_permutations=1)
    matches = _make_cyclic_matches()

    orderings = scorer._generate_match_orderings(matches)

    assert len(orderings) == 1
    assert orderings[0] == matches
    assert orderings[0] is not matches

def test_generate_match_orderings_with_multiple_permutations():
    """Test that multiple match orderings are generated."""
    scorer = EloScorer(
        n_permutations=5,
        permutations_random_state=0,
    )
    matches = _make_cyclic_matches()

    orderings = scorer._generate_match_orderings(matches)

    assert len(orderings) == 5
    assert all(len(ordering) == len(matches) for ordering in orderings)
    assert all(sorted(ordering, key=repr) == sorted(matches, key=repr) for ordering in orderings)
    assert any(ordering != matches for ordering in orderings)

def test_generate_match_orderings_is_reproducible_with_same_random_state():
    """Test that the same random state produces the same match orderings."""
    matches = _make_cyclic_matches()

    scorer_a = EloScorer(
        n_permutations=10,
        permutations_random_state=42,
    )
    scorer_b = EloScorer(
        n_permutations=10,
        permutations_random_state=42,
    )

    orderings_a = scorer_a._generate_match_orderings(matches)
    orderings_b = scorer_b._generate_match_orderings(matches)

    assert orderings_a == orderings_b

def test_compute_elo_ratings_per_sample_collects_one_rating_per_permutation():
    """Test that each approximator receives one Elo rating per permutation."""
    scorer = EloScorer(
        n_permutations=7,
        permutations_random_state=0,
        k_factor=16.0,
    )
    matches = _make_cyclic_matches()

    approximator_ratings_map = scorer._compute_elo_ratings_per_sample(matches)

    assert set(approximator_ratings_map) == {
        "ApproximatorA",
        "ApproximatorB",
        "ApproximatorC",
    }
    assert all(len(ratings) == 7 for ratings in approximator_ratings_map.values())

def test_build_leaderboard_rows_from_rating_samples_adds_sample_metadata():
    """Test that leaderboard rows include rating sample metadata."""
    scorer = EloScorer()

    approximator_ratings_map = {
        "ApproximatorA": [1000.0, 1010.0, 1020.0],
        "ApproximatorB": [990.0, 995.0, 1000.0],
    }
    match_stats = {
        "ApproximatorA": {
            "n_matches": 3,
            "wins": 2,
            "losses": 1,
            "ties": 0,
        },
        "ApproximatorB": {
            "n_matches": 3,
            "wins": 1,
            "losses": 2,
            "ties": 0,
        },
    }

    rows = scorer._build_leaderboard_rows_from_rating_samples(
        approximator_ratings_map=approximator_ratings_map,
        match_stats=match_stats,
    )

    rows_by_approximator = {row.approximator_name: row for row in rows}

    assert rows_by_approximator["ApproximatorA"].score == pytest.approx(1010.0)
    assert rows_by_approximator["ApproximatorA"].metadata["score_std"] == pytest.approx(10.0)
    assert rows_by_approximator["ApproximatorA"].metadata["n_rating_samples"] == 3
    assert rows_by_approximator["ApproximatorA"].metadata["wins"] == 2

    assert rows_by_approximator["ApproximatorA"].rank == 1
    assert rows_by_approximator["ApproximatorB"].rank == 2


def test_elo_scorer_rejects_non_positive_permutation_count():
    """Test that invalid permutation counts are rejected."""
    with pytest.raises(ValueError, match="n_permutations must be at least 1"):
        EloScorer(n_permutations=0)

def test_elo_scorer_rejects_negative_bootstrap_sample_count():
    """Test that negative bootstrap sample counts are rejected."""
    with pytest.raises(ValueError, match="n_bootstrap_samples must be at least 0"):
        EloScorer(n_bootstrap_samples=-1)


def _make_comparable_groups(scorer: EloScorer, records: list[dict[str, object]]):
    groups = group_records(records, scorer.group_keys)
    return scorer._build_comparable_groups(groups)

@pytest.mark.parametrize("confidence_level", [0.0, 1.0, -0.1, 1.1])
def test_elo_scorer_rejects_invalid_confidence_level(confidence_level: float):
    """Test that confidence levels must be strictly between zero and one."""
    with pytest.raises(
        ValueError,
        match="confidence_level must be between 0.0 and 1.0",
    ):
        EloScorer(confidence_level=confidence_level)


def test_build_comparable_groups_preserves_group_count():
    """Test that grouped records are converted into comparable group objects."""
    scorer = EloScorer(metric_names=["mse"])
    records = _make_two_budget_records()

    comparable_groups = _make_comparable_groups(scorer, records)

    assert len(comparable_groups) == 2
    assert all(comparable_group.records for comparable_group in comparable_groups)


def test_generate_bootstrap_group_samples_disabled_returns_original_sample():
    """Test that disabled bootstrapping returns one copy of the original groups."""
    scorer = EloScorer(metric_names=["mse"], n_bootstrap_samples=0)
    comparable_groups = _make_comparable_groups(scorer, _make_two_budget_records())

    bootstrap_samples = scorer._generate_bootstrap_group_samples(comparable_groups)

    assert len(bootstrap_samples) == 1
    assert bootstrap_samples[0] == comparable_groups
    assert bootstrap_samples[0] is not comparable_groups


def test_generate_bootstrap_group_samples_returns_requested_number_of_samples():
    """Test that bootstrapping creates the requested number of group samples."""
    scorer = EloScorer(
        metric_names=["mse"],
        n_bootstrap_samples=7,
        bootstrap_random_state=0,
    )
    comparable_groups = _make_comparable_groups(scorer, _make_two_budget_records())

    bootstrap_samples = scorer._generate_bootstrap_group_samples(comparable_groups)

    assert len(bootstrap_samples) == 7
    assert all(len(sample) == len(comparable_groups) for sample in bootstrap_samples)


def test_generate_bootstrap_group_samples_is_reproducible():
    """Test that the same bootstrap random state creates the same samples."""
    records = _make_two_budget_records()

    scorer_a = EloScorer(
        metric_names=["mse"],
        n_bootstrap_samples=10,
        bootstrap_random_state=42,
    )
    scorer_b = EloScorer(
        metric_names=["mse"],
        n_bootstrap_samples=10,
        bootstrap_random_state=42,
    )

    comparable_groups_a = _make_comparable_groups(scorer_a, records)
    comparable_groups_b = _make_comparable_groups(scorer_b, records)

    samples_a = scorer_a._generate_bootstrap_group_samples(comparable_groups_a)
    samples_b = scorer_b._generate_bootstrap_group_samples(comparable_groups_b)

    assert samples_a == samples_b


def test_generate_bootstrap_group_samples_handles_empty_groups():
    """Test that empty comparable-group input is handled explicitly."""
    scorer = EloScorer(
        metric_names=["mse"],
        n_bootstrap_samples=5,
        bootstrap_random_state=0,
    )

    bootstrap_samples = scorer._generate_bootstrap_group_samples([])

    assert bootstrap_samples == [[]]


def test_compute_bootstrap_elo_ratings_collects_one_rating_per_bootstrap_sample():
    """Test that each bootstrap sample contributes one rating per approximator."""
    scorer = EloScorer(
        metric_names=["mse"],
        n_bootstrap_samples=5,
        bootstrap_random_state=0,
    )
    comparable_groups = _make_comparable_groups(scorer, _make_two_budget_records())

    approximator_ratings_map = scorer._compute_bootstrap_elo_ratings(comparable_groups)

    assert set(approximator_ratings_map) == {"ApproximatorA", "ApproximatorB"}
    assert len(approximator_ratings_map["ApproximatorA"]) == 5
    assert len(approximator_ratings_map["ApproximatorB"]) == 5


def test_compute_bootstrap_elo_ratings_combines_bootstrap_and_permutations():
    """Test that bootstrap samples and match-order permutations both add ratings."""
    scorer = EloScorer(
        metric_names=["mse"],
        n_bootstrap_samples=5,
        bootstrap_random_state=0,
        n_permutations=3,
        permutations_random_state=0,
    )
    comparable_groups = _make_comparable_groups(scorer, _make_two_budget_records())

    approximator_ratings_map = scorer._compute_bootstrap_elo_ratings(comparable_groups)

    assert set(approximator_ratings_map) == {"ApproximatorA", "ApproximatorB"}
    assert len(approximator_ratings_map["ApproximatorA"]) == 15
    assert len(approximator_ratings_map["ApproximatorB"]) == 15


def test_confidence_interval_uses_configured_confidence_level():
    """Test that confidence intervals use the configured quantile bounds."""
    scorer = EloScorer(confidence_level=0.5)
    values = [0.0, 10.0, 20.0, 30.0]

    ci_lower, ci_upper = scorer._confidence_interval(values)

    assert ci_lower == pytest.approx(7.5)
    assert ci_upper == pytest.approx(22.5)


def test_confidence_interval_rejects_empty_values():
    """Test that confidence intervals cannot be computed from empty samples."""
    scorer = EloScorer()

    with pytest.raises(
        ValueError,
        match="Cannot compute confidence interval of empty values",
    ):
        scorer._confidence_interval([])


def test_elo_scorer_with_bootstrap_adds_confidence_metadata():
    """Test that score results contain bootstrap confidence interval metadata."""
    scorer = EloScorer(
        metric_names=["mse"],
        n_bootstrap_samples=10,
        bootstrap_random_state=0,
        confidence_level=0.95,
    )

    result = scorer.score(_make_two_budget_records())

    assert result.metadata["rating_sample_method"] == "group_bootstrap"
    assert result.metadata["n_bootstrap_samples"] == 10
    assert result.metadata["bootstrap_random_state"] == 0
    assert result.metadata["confidence_level"] == 0.95
    assert result.metadata["confidence_interval_method"] == "bootstrap_quantile"
    assert result.metadata["n_total_rating_samples"] == 10

    assert len(result.rows) == 2

    for row in result.rows:
        assert row.metadata["n_rating_samples"] == 10
        assert "ci_lower" in row.metadata
        assert "ci_upper" in row.metadata
        assert "ci_minus" in row.metadata
        assert "ci_plus" in row.metadata
        assert row.metadata["ci_lower"] <= row.metadata["ci_upper"]


def test_elo_scorer_with_bootstrap_and_permutations_adds_combined_metadata():
    """Test metadata when group bootstrap and match-order permutations are combined."""
    scorer = EloScorer(
        metric_names=["mse"],
        n_bootstrap_samples=10,
        bootstrap_random_state=0,
        n_permutations=3,
        permutations_random_state=0,
    )

    result = scorer.score(_make_two_budget_records())

    assert (
        result.metadata["rating_sample_method"]
        == "group_bootstrap_with_match_order_permutation"
    )
    assert result.metadata["n_total_rating_samples"] == 30

    for row in result.rows:
        assert row.metadata["n_rating_samples"] == 30