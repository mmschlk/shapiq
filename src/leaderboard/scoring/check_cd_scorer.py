"""Manual check script for the Critical Difference scorer."""

from __future__ import annotations

import logging
from pathlib import Path

from leaderboard.scoring.cd_scorer import CriticalDifferenceScorer

logging.basicConfig(level=logging.INFO, format="%(message)s")
LOGGER = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

# Approximators ordered from (roughly) best to worst, purely so the dummy
# data below has a known, inspectable structure.
APPROXIMATORS = [
    "StratifiedSamplingSV",
    "PermutationSamplingSV",
    "OwenSamplingSV",
    "KernelSHAP",
    "ProxySHAP",
    "RandomBaseline",
]


def make_record(
    *,
    run_id: str,
    game_id: str,
    game_name: str,
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
        "game_name": game_name,
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


def pull_records_from_jsonl(jsonl_path: Path) -> list[dict[str, object]]:
    """Pull records from a JSONL file."""
    import json

    records: list[dict[str, object]] = []
    with Path.open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            records.append(record)
    return records


def print_section(title: str) -> None:
    """Log a readable section header."""
    LOGGER.info("")
    LOGGER.info("=" * len(title))
    LOGGER.info(title)
    LOGGER.info("=" * len(title))


def main() -> None:
    """Run the CD scorer on historical records, print the ranking, and save the CD diagram."""
    records = pull_records_from_jsonl(
        Path(__file__).resolve().parent.parent / "data" / "results_23Jun_1658.jsonl"
    )

    print_section("Critical Difference scoring for MSE")
    cd_scorer = CriticalDifferenceScorer(alpha=0.05, metric_names=["mse"])
    cd_scoring_result = cd_scorer.score(records)

    cd_result = cd_scoring_result.metadata["cd_result"]
    LOGGER.info("")
    LOGGER.info("n_approximators: %d", cd_result.n_approximators)
    LOGGER.info("n_groups (comparable group x metric): %d", cd_result.n_groups)
    LOGGER.info(
        "Friedman: statistic=%.4f  p_value=%.6g  significant=%s",
        cd_result.friedman_statistic,
        cd_result.friedman_p_value,
        cd_result.friedman_significant,
    )
    LOGGER.info("Critical difference (CD): %.4f", cd_result.critical_difference)
    LOGGER.info("Cliques (not significantly different): %s", cd_result.cliques)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "cd_diagram_mse.png"
    fig = cd_scorer.plot(cd_scoring_result, title="Critical Difference diagram (MSE)")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    LOGGER.info("Saved CD diagram to %s", output_path)


if __name__ == "__main__":
    main()
