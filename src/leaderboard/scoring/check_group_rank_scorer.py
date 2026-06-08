"""Manual check script for the group-rank scorer."""

from __future__ import annotations

from leaderboard.scoring.display import print_scoring_result
from leaderboard.scoring.group_rank_scorer import GroupRankScorer


def make_record(
    *,
    run_id: str,
    game_id: str,
    budget: int,
    approximator_name: str,
    approx_seed: int,
    mse: float,
    spearman: float,
) -> dict[str, object]:
    """Create one minimal raw benchmark record."""
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


def make_records() -> list[dict[str, object]]:
    """Create sample records for two comparable groups."""
    return [
        # Group 1: budget 100
        make_record(
            run_id="g1-strat-0",
            game_id="california-1",
            budget=100,
            approximator_name="StratifiedSamplingSV",
            approx_seed=0,
            mse=0.010,
            spearman=0.90,
        ),
        make_record(
            run_id="g1-strat-1",
            game_id="california-1",
            budget=100,
            approximator_name="StratifiedSamplingSV",
            approx_seed=1,
            mse=0.020,
            spearman=0.80,
        ),
        make_record(
            run_id="g1-perm-0",
            game_id="california-1",
            budget=100,
            approximator_name="PermutationSamplingSV",
            approx_seed=0,
            mse=0.030,
            spearman=0.70,
        ),
        make_record(
            run_id="g1-perm-1",
            game_id="california-1",
            budget=100,
            approximator_name="PermutationSamplingSV",
            approx_seed=1,
            mse=0.040,
            spearman=0.60,
        ),
        make_record(
            run_id="g1-proxy-0",
            game_id="california-1",
            budget=100,
            approximator_name="ProxySHAP",
            approx_seed=0,
            mse=0.050,
            spearman=0.50,
        ),
        # Group 2: budget 500
        make_record(
            run_id="g2-strat-0",
            game_id="california-1",
            budget=500,
            approximator_name="StratifiedSamplingSV",
            approx_seed=0,
            mse=0.040,
            spearman=0.60,
        ),
        make_record(
            run_id="g2-strat-1",
            game_id="california-1",
            budget=500,
            approximator_name="StratifiedSamplingSV",
            approx_seed=1,
            mse=0.060,
            spearman=0.40,
        ),
        make_record(
            run_id="g2-perm-0",
            game_id="california-1",
            budget=500,
            approximator_name="PermutationSamplingSV",
            approx_seed=0,
            mse=0.010,
            spearman=0.95,
        ),
        make_record(
            run_id="g2-perm-1",
            game_id="california-1",
            budget=500,
            approximator_name="PermutationSamplingSV",
            approx_seed=1,
            mse=0.030,
            spearman=0.85,
        ),
        make_record(
            run_id="g2-proxy-0",
            game_id="california-1",
            budget=500,
            approximator_name="ProxySHAP",
            approx_seed=0,
            mse=0.070,
            spearman=0.30,
        ),
    ]


def main() -> None:
    """Run the group-rank scorer on sample records and print the result."""
    records = make_records()

    scorer = GroupRankScorer()
    result = scorer.score(records)

    print_scoring_result(result)


if __name__ == "__main__":
    main()
