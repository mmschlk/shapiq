"""Module for database utilites.

Includes:
- functions for processing raw run documents from the database;
- function for aggregating run records by game, approximator, and budget, computing mean and std for each metric.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from leaderboard.metrics import METRICS
from leaderboard.storage.data_classes.run_config import RunConfig


def _process(raw_runs: list[dict]) -> pd.DataFrame:
    """Process raw run documents from the database, filter failures, flatten metrics, and aggregate.

    Args:
        raw_runs: List of raw run documents as dictionaries.

    Returns:
        A DataFrame ready for the leaderboard UI with columns:
            game_name, approximator_name, budget,
            mse_mean, mse_std, mae_mean, mae_std,
            ground_truth_method,
            runtime_mean, runtime_min, runtime_max,
            n_seeds.
    """
    if not raw_runs:
        return _empty_dataframe()

    runs_df = pd.DataFrame(raw_runs)
    runs_df = runs_df[~runs_df["run_failed"]]

    metrics_df = pd.json_normalize(runs_df["metrics"])
    runs_df = pd.concat([runs_df.drop(columns=["metrics"]), metrics_df], axis=1)

    # Normalise seed column name
    if "seed" in runs_df.columns and "approx_seed" not in runs_df.columns:
        runs_df = runs_df.rename(columns={"seed": "approx_seed"})
    if "seed" in runs_df.columns and "approx_seed" in runs_df.columns:
        runs_df["approx_seed"] = runs_df["approx_seed"].combine_first(runs_df["seed"])
        runs_df = runs_df.drop(columns=["seed"])

    return _aggregate(runs_df)


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate runs by game, approximator, and budget, computing mean and std for each metric."""
    avail_metrics = [m for m in METRICS if m in df.columns]
    metric_aggs = {}
    for m in avail_metrics:
        metric_aggs[f"{m}_mean"] = (m, "mean")
        metric_aggs[f"{m}_std"] = (m, "std")

    return (
        df.groupby(["game_name", "approximator_name", "budget"])
        .agg(
            **metric_aggs,
            ground_truth_method=("ground_truth_method", "first"),
            runtime_mean=("runtime_seconds", "mean"),
            runtime_min=("runtime_seconds", "min"),
            runtime_max=("runtime_seconds", "max"),
            n_seeds=("approx_seed", "count"),
        )
        .reset_index()
    )


def _empty_dataframe() -> pd.DataFrame:
    """Return an empty DataFrame with the minimal expected columns for the leaderboard UI."""
    return pd.DataFrame(
        [
            {
                "game_name": "N/A",
                "approximator_name": "N/A",
                "budget": 0,
                "mse": 0,
                "mae": 0,
                "ground_truth_method": "N/A",
                "runtime_seconds": 0,
                "approx_seed": 0,
            }
        ]
    )


def _get_seed(document: dict[str, Any]) -> int | None:
    """Extract the 'seed' field from *document*, if present and an integer."""
    seed = document.get("seed")

    if not seed:
        seed = document.get("approx_seed")

    return seed


def _matches_config(document: dict[str, Any], config: RunConfig) -> bool:
    """Return True if *document* contains all key/value pairs in *config*."""
    return (
        document.get("game_name") == config.game_name
        and document.get("n_players") == config.n_players
        and document.get("approximator_name") == config.approximator_name
        and document.get("index") == config.index
        and document.get("max_order") == config.max_order
        and document.get("budget") == config.budget
        and document.get("ground_truth_method") == config.ground_truth_method
    )


def _matches_config_with_seed(document: dict[str, Any], other_document: dict[str, Any]) -> bool:
    """Return True if *document* matches *config* on all fields including 'seed'."""
    document_config = RunConfig.from_dict(document)

    matches_config = _matches_config(other_document, document_config)
    return matches_config and _get_seed(document) == _get_seed(other_document)
