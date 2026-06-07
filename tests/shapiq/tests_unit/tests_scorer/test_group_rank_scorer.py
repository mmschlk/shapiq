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