"""Manual check script for the Elo scorer."""

from __future__ import annotations

from leaderboard.scoring.display import print_scoring_result
from leaderboard.scoring.elo_scorer import EloScorer


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
        # Stratified wins both MSE and Spearman in this group.
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
        # Permutation wins both MSE and Spearman in this group.
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


def print_section(title: str) -> None:
    """Print a readable section header."""
    print()
    print("=" * len(title))
    print(title)
    print("=" * len(title))


def main() -> None:
    """Run the Elo scorer on sample records and print several scoring views."""
    records = make_records()

    print_section("Global Elo scoring over all metrics and budgets")
    global_scorer = EloScorer()
    global_result = global_scorer.score(records)
    print_scoring_result(global_result)

    print_section("Elo scoring for MSE only")
    mse_scorer = EloScorer(metric_names=["mse"])
    mse_result = mse_scorer.score(records)
    print_scoring_result(mse_result)

    print_section("Elo scoring for Spearman only")
    spearman_scorer = EloScorer(metric_names=["spearman"])
    spearman_result = spearman_scorer.score(records)
    print_scoring_result(spearman_result)

    print_section("Elo scoring for MSE at budget 100")
    budget_100_scorer = EloScorer(
        metric_names=["mse"],
        budgets=[100],
    )
    budget_100_result = budget_100_scorer.score(records)
    print_scoring_result(budget_100_result)

    print_section("Elo scoring for MSE at budget 500")
    budget_500_scorer = EloScorer(
        metric_names=["mse"],
        budgets=[500],
    )
    budget_500_result = budget_500_scorer.score(records)
    print_scoring_result(budget_500_result)


if __name__ == "__main__":
    main()