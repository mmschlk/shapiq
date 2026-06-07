import pytest

from leaderboard.scoring.group_rank_scorer import GroupRankScorer


def _make_test_records() -> list[dict[str, object]]:
    """Create two comparable groups with multiple seeds per approximator."""
    return [
        # Group 1: CaliforniaHousing / SV / budget 100
        {
            "run_id": "g1-strat-seed-0",
            "game_id": "california-1",
            "game_name": "CaliforniaHousing",
            "index": "SV",
            "max_order": 1,
            "budget": 100,
            "ground_truth_method": "ExactComputer",
            "approximator_name": "StratifiedSamplingSV",
            "approx_seed": 0,
            "run_failed": False,
            "metrics": {
                "mse": 0.010,
                "spearman": 0.90,
            },
        },
        {
            "run_id": "g1-strat-seed-1",
            "game_id": "california-1",
            "game_name": "CaliforniaHousing",
            "index": "SV",
            "max_order": 1,
            "budget": 100,
            "ground_truth_method": "ExactComputer",
            "approximator_name": "StratifiedSamplingSV",
            "approx_seed": 1,
            "run_failed": False,
            "metrics": {
                "mse": 0.020,
                "spearman": 0.80,
            },
        },
        {
            "run_id": "g1-perm-seed-0",
            "game_id": "california-1",
            "game_name": "CaliforniaHousing",
            "index": "SV",
            "max_order": 1,
            "budget": 100,
            "ground_truth_method": "ExactComputer",
            "approximator_name": "PermutationSamplingSV",
            "approx_seed": 0,
            "run_failed": False,
            "metrics": {
                "mse": 0.030,
                "spearman": 0.70,
            },
        },
        {
            "run_id": "g1-perm-seed-1",
            "game_id": "california-1",
            "game_name": "CaliforniaHousing",
            "index": "SV",
            "max_order": 1,
            "budget": 100,
            "ground_truth_method": "ExactComputer",
            "approximator_name": "PermutationSamplingSV",
            "approx_seed": 1,
            "run_failed": False,
            "metrics": {
                "mse": 0.040,
                "spearman": 0.60,
            },
        },
        {
            "run_id": "g1-proxy-seed-0",
            "game_id": "california-1",
            "game_name": "CaliforniaHousing",
            "index": "SV",
            "max_order": 1,
            "budget": 100,
            "ground_truth_method": "ExactComputer",
            "approximator_name": "ProxySHAP",
            "approx_seed": 0,
            "run_failed": False,
            "metrics": {
                "mse": 0.050,
                "spearman": 0.50,
            },
        },

        # Group 2: CaliforniaHousing / SV / budget 500
        {
            "run_id": "g2-strat-seed-0",
            "game_id": "california-1",
            "game_name": "CaliforniaHousing",
            "index": "SV",
            "max_order": 1,
            "budget": 500,
            "ground_truth_method": "ExactComputer",
            "approximator_name": "StratifiedSamplingSV",
            "approx_seed": 0,
            "run_failed": False,
            "metrics": {
                "mse": 0.040,
                "spearman": 0.60,
            },
        },
        {
            "run_id": "g2-strat-seed-1",
            "game_id": "california-1",
            "game_name": "CaliforniaHousing",
            "index": "SV",
            "max_order": 1,
            "budget": 500,
            "ground_truth_method": "ExactComputer",
            "approximator_name": "StratifiedSamplingSV",
            "approx_seed": 1,
            "run_failed": False,
            "metrics": {
                "mse": 0.060,
                "spearman": 0.40,
            },
        },
        {
            "run_id": "g2-perm-seed-0",
            "game_id": "california-1",
            "game_name": "CaliforniaHousing",
            "index": "SV",
            "max_order": 1,
            "budget": 500,
            "ground_truth_method": "ExactComputer",
            "approximator_name": "PermutationSamplingSV",
            "approx_seed": 0,
            "run_failed": False,
            "metrics": {
                "mse": 0.010,
                "spearman": 0.95,
            },
        },
        {
            "run_id": "g2-perm-seed-1",
            "game_id": "california-1",
            "game_name": "CaliforniaHousing",
            "index": "SV",
            "max_order": 1,
            "budget": 500,
            "ground_truth_method": "ExactComputer",
            "approximator_name": "PermutationSamplingSV",
            "approx_seed": 1,
            "run_failed": False,
            "metrics": {
                "mse": 0.030,
                "spearman": 0.85,
            },
        },
        {
            "run_id": "g2-proxy-seed-0",
            "game_id": "california-1",
            "game_name": "CaliforniaHousing",
            "index": "SV",
            "max_order": 1,
            "budget": 500,
            "ground_truth_method": "ExactComputer",
            "approximator_name": "ProxySHAP",
            "approx_seed": 0,
            "run_failed": False,
            "metrics": {
                "mse": 0.070,
                "spearman": 0.30,
            },
        },
    ]


def _full_record(
        *,
        run_id: str,
        game_id: str,
        budget: int,
        approximator_name: str,
        approx_seed: int,
        metrics: dict[str, float],
) -> dict[str, object]:
    """Create a realistic raw run record with all common benchmark fields."""
    return {
        "run_id": run_id,
        "game_name": "CaliforniaHousing",
        "game_id": game_id,
        "game_params": {
            "x": 0,
            "model_name": "decision_tree",
            "imputer": "marginal",
            "normalize": True,
            "verbose": False,
            "random_state": 42,
        },
        "n_players": 8,
        "approximator_name": approximator_name,
        "approximator_params": {},
        "shapiq_version": "1.4.2.dev156+g2889126ea",
        "index": "SV",
        "max_order": 1,
        "budget": budget,
        "approx_seed": approx_seed,
        "ground_truth_method": "ExactComputer",
        "run_failed": False,
        "error_message": None,
        "metrics": metrics,
        "runtime_seconds": 0.073,
        "timestamp": "2026-06-07T15:02:08.165990+00:00",
        "hardware": {
            "cpu": "x86_64",
            "ram_gb": None,
            "python_version": "3.12.3",
        },
        "notes": "",
        "_id": run_id,
    }


def _make_full_test_records() -> list[dict[str, object]]:
    """Create two comparable groups using realistic full raw run records."""
    return [
        # Group 1: budget 100
        _full_record(
            run_id="g1-strat-0",
            game_id="CaliforniaHousing_LocalExplanation_Game_1",
            budget=100,
            approximator_name="StratifiedSamplingSV",
            approx_seed=0,
            metrics={
                "mse": 0.010,
                "mae": 0.10,
                "mse_normalized": 0.010,
                "spearman": 0.90,
                "kendall_tau": 0.90,
                "precision_at_k": 1.00,
            },
        ),
        _full_record(
            run_id="g1-strat-1",
            game_id="CaliforniaHousing_LocalExplanation_Game_1",
            budget=100,
            approximator_name="StratifiedSamplingSV",
            approx_seed=1,
            metrics={
                "mse": 0.020,
                "mae": 0.20,
                "mse_normalized": 0.020,
                "spearman": 0.80,
                "kendall_tau": 0.80,
                "precision_at_k": 0.90,
            },
        ),
        _full_record(
            run_id="g1-perm-0",
            game_id="CaliforniaHousing_LocalExplanation_Game_1",
            budget=100,
            approximator_name="PermutationSamplingSV",
            approx_seed=0,
            metrics={
                "mse": 0.030,
                "mae": 0.30,
                "mse_normalized": 0.030,
                "spearman": 0.70,
                "kendall_tau": 0.70,
                "precision_at_k": 0.80,
            },
        ),
        _full_record(
            run_id="g1-perm-1",
            game_id="CaliforniaHousing_LocalExplanation_Game_1",
            budget=100,
            approximator_name="PermutationSamplingSV",
            approx_seed=1,
            metrics={
                "mse": 0.040,
                "mae": 0.40,
                "mse_normalized": 0.040,
                "spearman": 0.60,
                "kendall_tau": 0.60,
                "precision_at_k": 0.70,
            },
        ),
        _full_record(
            run_id="g1-proxy-0",
            game_id="CaliforniaHousing_LocalExplanation_Game_1",
            budget=100,
            approximator_name="ProxySHAP",
            approx_seed=0,
            metrics={
                "mse": 0.050,
                "mae": 0.50,
                "mse_normalized": 0.050,
                "spearman": 0.50,
                "kendall_tau": 0.50,
                "precision_at_k": 0.60,
            },
        ),

        # Group 2: budget 500
        _full_record(
            run_id="g2-strat-0",
            game_id="CaliforniaHousing_LocalExplanation_Game_1",
            budget=500,
            approximator_name="StratifiedSamplingSV",
            approx_seed=0,
            metrics={
                "mse": 0.040,
                "mae": 0.40,
                "mse_normalized": 0.040,
                "spearman": 0.60,
                "kendall_tau": 0.60,
                "precision_at_k": 0.70,
            },
        ),
        _full_record(
            run_id="g2-strat-1",
            game_id="CaliforniaHousing_LocalExplanation_Game_1",
            budget=500,
            approximator_name="StratifiedSamplingSV",
            approx_seed=1,
            metrics={
                "mse": 0.060,
                "mae": 0.60,
                "mse_normalized": 0.060,
                "spearman": 0.40,
                "kendall_tau": 0.40,
                "precision_at_k": 0.50,
            },
        ),
        _full_record(
            run_id="g2-perm-0",
            game_id="CaliforniaHousing_LocalExplanation_Game_1",
            budget=500,
            approximator_name="PermutationSamplingSV",
            approx_seed=0,
            metrics={
                "mse": 0.010,
                "mae": 0.10,
                "mse_normalized": 0.010,
                "spearman": 0.95,
                "kendall_tau": 0.95,
                "precision_at_k": 1.00,
            },
        ),
        _full_record(
            run_id="g2-perm-1",
            game_id="CaliforniaHousing_LocalExplanation_Game_1",
            budget=500,
            approximator_name="PermutationSamplingSV",
            approx_seed=1,
            metrics={
                "mse": 0.030,
                "mae": 0.30,
                "mse_normalized": 0.030,
                "spearman": 0.85,
                "kendall_tau": 0.85,
                "precision_at_k": 0.90,
            },
        ),
        _full_record(
            run_id="g2-proxy-0",
            game_id="CaliforniaHousing_LocalExplanation_Game_1",
            budget=500,
            approximator_name="ProxySHAP",
            approx_seed=0,
            metrics={
                "mse": 0.070,
                "mae": 0.70,
                "mse_normalized": 0.070,
                "spearman": 0.30,
                "kendall_tau": 0.30,
                "precision_at_k": 0.40,
            },
        ),
    ]

def _minimal_record(
    *,
    approximator_name: str,
    mse: float | None = None,
    spearman: float | None = None,
    budget: int = 100,
    approx_seed: int = 0,
    run_failed: bool = False,
    flattened: bool = False,
) -> dict[str, object]:
    """Create a minimal comparable benchmark record for scorer tests."""
    record: dict[str, object] = {
        "run_id": f"{approximator_name}-{budget}-{approx_seed}",
        "game_id": "game-1",
        "game_name": "CaliforniaHousing",
        "index": "SV",
        "max_order": 1,
        "budget": budget,
        "ground_truth_method": "ExactComputer",
        "approximator_name": approximator_name,
        "approx_seed": approx_seed,
        "run_failed": run_failed,
    }

    metric_values: dict[str, float] = {}
    if mse is not None:
        metric_values["mse"] = mse
    if spearman is not None:
        metric_values["spearman"] = spearman

    if flattened:
        record.update(metric_values)
    else:
        record["metrics"] = metric_values

    return record

def test_group_rank_scorer_aggregates_seeds_and_ranks_groups():
    """Test group-wise ranking with seed aggregation."""
    scorer = GroupRankScorer()
    records = _make_test_records()

    result = scorer.score(records)

    assert result.scorer_name == "group_rank"
    assert result.metadata["n_input_records"] == 10
    assert result.metadata["n_valid_records"] == 10
    assert result.metadata["n_groups"] == 2

    mse_group_results = [
        group_result
        for group_result in result.group_results
        if group_result.metric_name == "mse"
    ]

    assert len(mse_group_results) == 2

    group_100 = next(
        group_result
        for group_result in mse_group_results
        if group_result.group_key["budget"] == 100
    )
    group_500 = next(
        group_result
        for group_result in mse_group_results
        if group_result.group_key["budget"] == 500
    )

    group_100_rows = {
        row.approximator_name: row
        for row in group_100.rows
    }

    group_500_rows = {
        row.approximator_name: row
        for row in group_500.rows
    }

    # Group 1 / budget 100:
    # Stratified mean MSE = mean(0.010, 0.020) = 0.015 -> rank 1
    # Permutation mean MSE = mean(0.030, 0.040) = 0.035 -> rank 2
    # Proxy mean MSE = 0.050 -> rank 3
    assert group_100_rows["StratifiedSamplingSV"].metric_value == 0.015
    assert group_100_rows["StratifiedSamplingSV"].rank == 1
    assert group_100_rows["PermutationSamplingSV"].metric_value == 0.035
    assert group_100_rows["PermutationSamplingSV"].rank == 2
    assert group_100_rows["ProxySHAP"].metric_value == 0.050
    assert group_100_rows["ProxySHAP"].rank == 3

    # Group 2 / budget 500:
    # Permutation mean MSE = mean(0.010, 0.030) = 0.020 -> rank 1
    # Stratified mean MSE = mean(0.040, 0.060) = 0.050 -> rank 2
    # Proxy mean MSE = 0.070 -> rank 3
    assert group_500_rows["PermutationSamplingSV"].metric_value == 0.020
    assert group_500_rows["PermutationSamplingSV"].rank == 1
    assert group_500_rows["StratifiedSamplingSV"].metric_value == 0.050
    assert group_500_rows["StratifiedSamplingSV"].rank == 2
    assert group_500_rows["ProxySHAP"].metric_value == 0.070
    assert group_500_rows["ProxySHAP"].rank == 3


def test_group_rank_scorer_builds_average_rank_leaderboard():
    """Test that group ranks are aggregated into average leaderboard ranks."""
    scorer = GroupRankScorer()
    records = _make_test_records()

    result = scorer.score(records)

    rows_by_approximator = {
        row.approximator_name: row
        for row in result.rows
    }

    # Expected MSE ranks:
    # Group 100: Stratified=1, Permutation=2, Proxy=3
    # Group 500: Stratified=2, Permutation=1, Proxy=3
    #
    # Expected Spearman ranks:
    # Group 100: Stratified=1, Permutation=2, Proxy=3
    # Group 500: Permutation=1, Stratified=2, Proxy=3
    #
    # Therefore average ranks across MSE + Spearman group rankings:
    # Stratified = mean(1, 2, 1, 2) = 1.5
    # Permutation = mean(2, 1, 2, 1) = 1.5
    # Proxy = mean(3, 3, 3, 3) = 3.0
    assert rows_by_approximator["StratifiedSamplingSV"].score == 1.5
    assert rows_by_approximator["PermutationSamplingSV"].score == 1.5
    assert rows_by_approximator["ProxySHAP"].score == 3.0

    assert rows_by_approximator["StratifiedSamplingSV"].metadata["n_rankings"] == 4
    assert rows_by_approximator["PermutationSamplingSV"].metadata["n_rankings"] == 4
    assert rows_by_approximator["ProxySHAP"].metadata["n_rankings"] == 4


def test_group_rank_scorer_accepts_full_raw_run_records():
    """Test that the scorer works with realistic full raw run records."""
    scorer = GroupRankScorer()
    records = _make_full_test_records()

    result = scorer.score(records)

    assert result.scorer_name == "group_rank"
    assert result.metadata["n_input_records"] == 10
    assert result.metadata["n_valid_records"] == 10
    assert result.metadata["n_groups"] == 2

    # 2 groups * 6 available metrics = 12 group scoring results
    assert result.metadata["n_group_results"] == 12
    assert len(result.group_results) == 12

    assert result.context.game_names == ["CaliforniaHousing"]
    assert result.context.indices == ["SV"]
    assert result.context.budgets == [100, 500]

def test_group_rank_scorer_aggregates_seeds_from_full_records():
    """Test that full records are aggregated over seeds before ranking."""
    scorer = GroupRankScorer()
    records = _make_full_test_records()

    result = scorer.score(records)

    mse_group_results = [
        group_result
        for group_result in result.group_results
        if group_result.metric_name == "mse"
    ]

    group_100 = next(
        group_result
        for group_result in mse_group_results
        if group_result.group_key["budget"] == 100
    )

    rows_by_approximator = {
        row.approximator_name: row
        for row in group_100.rows
    }

    assert rows_by_approximator["StratifiedSamplingSV"].metric_value == pytest.approx(0.015)
    assert rows_by_approximator["StratifiedSamplingSV"].rank == 1
    assert rows_by_approximator["StratifiedSamplingSV"].metadata["n_records"] == 2

    assert rows_by_approximator["PermutationSamplingSV"].metric_value == pytest.approx(0.035)
    assert rows_by_approximator["PermutationSamplingSV"].rank == 2
    assert rows_by_approximator["PermutationSamplingSV"].metadata["n_records"] == 2

    assert rows_by_approximator["ProxySHAP"].metric_value == pytest.approx(0.050)
    assert rows_by_approximator["ProxySHAP"].rank == 3
    assert rows_by_approximator["ProxySHAP"].metadata["n_records"] == 1

def test_group_rank_scorer_builds_average_rank_leaderboard_from_full_records():
    """Test average-rank leaderboard creation from realistic full records."""
    scorer = GroupRankScorer()
    records = _make_full_test_records()

    result = scorer.score(records)

    rows_by_approximator = {
        row.approximator_name: row
        for row in result.rows
    }

    assert rows_by_approximator["StratifiedSamplingSV"].score == pytest.approx(1.5)
    assert rows_by_approximator["PermutationSamplingSV"].score == pytest.approx(1.5)
    assert rows_by_approximator["ProxySHAP"].score == pytest.approx(3.0)

    # 2 groups * 6 metrics = 12 rankings per approximator
    assert rows_by_approximator["StratifiedSamplingSV"].metadata["n_rankings"] == 12
    assert rows_by_approximator["PermutationSamplingSV"].metadata["n_rankings"] == 12
    assert rows_by_approximator["ProxySHAP"].metadata["n_rankings"] == 12

def test_group_rank_scorer_ignores_failed_runs():
    """Test that failed runs are ignored before grouping and ranking."""
    scorer = GroupRankScorer()
    records = [
        _minimal_record(
            approximator_name="StratifiedSamplingSV",
            mse=0.10,
            run_failed=False,
        ),
        _minimal_record(
            approximator_name="PermutationSamplingSV",
            mse=0.01,
            run_failed=True,
        ),
        _minimal_record(
            approximator_name="ProxySHAP",
            mse=0.20,
            run_failed=False,
        ),
    ]

    result = scorer.score(records)

    assert result.metadata["n_input_records"] == 3
    assert result.metadata["n_valid_records"] == 2

    mse_group_results = [
        group_result
        for group_result in result.group_results
        if group_result.metric_name == "mse"
    ]

    assert len(mse_group_results) == 1

    rows_by_approximator = {
        row.approximator_name: row
        for row in mse_group_results[0].rows
    }

    assert "PermutationSamplingSV" not in rows_by_approximator
    assert rows_by_approximator["StratifiedSamplingSV"].rank == 1
    assert rows_by_approximator["ProxySHAP"].rank == 2

def test_group_rank_scorer_skips_records_without_metric_value():
    """Test that records missing a metric are skipped for that metric ranking."""
    scorer = GroupRankScorer()
    records = [
        _minimal_record(
            approximator_name="StratifiedSamplingSV",
            mse=0.10,
        ),
        _minimal_record(
            approximator_name="PermutationSamplingSV",
            spearman=0.90,
        ),
        _minimal_record(
            approximator_name="ProxySHAP",
            mse=0.20,
        ),
    ]

    result = scorer.score(records)

    mse_group_results = [
        group_result
        for group_result in result.group_results
        if group_result.metric_name == "mse"
    ]

    assert len(mse_group_results) == 1

    rows_by_approximator = {
        row.approximator_name: row
        for row in mse_group_results[0].rows
    }

    assert "PermutationSamplingSV" not in rows_by_approximator
    assert rows_by_approximator["StratifiedSamplingSV"].rank == 1
    assert rows_by_approximator["ProxySHAP"].rank == 2

def test_group_rank_scorer_accepts_flattened_metric_records():
    """Test that the scorer accepts records with flattened metric fields."""
    scorer = GroupRankScorer()
    records = [
        _minimal_record(
            approximator_name="StratifiedSamplingSV",
            mse=0.10,
            flattened=True,
        ),
        _minimal_record(
            approximator_name="PermutationSamplingSV",
            mse=0.20,
            flattened=True,
        ),
    ]

    result = scorer.score(records)

    mse_group_results = [
        group_result
        for group_result in result.group_results
        if group_result.metric_name == "mse"
    ]

    assert len(mse_group_results) == 1

    rows_by_approximator = {
        row.approximator_name: row
        for row in mse_group_results[0].rows
    }

    assert rows_by_approximator["StratifiedSamplingSV"].metric_value == pytest.approx(0.10)
    assert rows_by_approximator["StratifiedSamplingSV"].rank == 1
    assert rows_by_approximator["PermutationSamplingSV"].metric_value == pytest.approx(0.20)
    assert rows_by_approximator["PermutationSamplingSV"].rank == 2

def test_group_rank_scorer_handles_equal_metric_values():
    """Test current behavior for equal metric values. This should not hinder the ranking."""
    scorer = GroupRankScorer()
    records = [
        _minimal_record(
            approximator_name="StratifiedSamplingSV",
            mse=0.10,
        ),
        _minimal_record(
            approximator_name="PermutationSamplingSV",
            mse=0.10,
        ),
    ]

    result = scorer.score(records)

    mse_group_results = [
        group_result
        for group_result in result.group_results
        if group_result.metric_name == "mse"
    ]

    assert len(mse_group_results) == 1
    assert len(mse_group_results[0].rows) == 2

    rows_by_approximator = {
        row.approximator_name: row
        for row in mse_group_results[0].rows
    }

    assert rows_by_approximator["StratifiedSamplingSV"].metric_value == pytest.approx(0.10)
    assert rows_by_approximator["PermutationSamplingSV"].metric_value == pytest.approx(0.10)

    assigned_ranks = {
        rows_by_approximator["StratifiedSamplingSV"].rank,
        rows_by_approximator["PermutationSamplingSV"].rank,
    }
    assert assigned_ranks == {1, 2}
