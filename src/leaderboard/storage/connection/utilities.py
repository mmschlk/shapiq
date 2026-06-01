"""
Module for database utilites. 

Includes:
- functions for processing raw run documents from the database;
- function for aggregating run records by game, approximator, and budget, computing mean and std for each metric.
"""

import pandas as pd
from leaderboard.metrics import METRICS


def _process(raw_runs: list[dict]) -> pd.DataFrame:
    """
    Process raw run documents from the database, filter failures, flatten metrics, and aggregate.
    
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

    df = pd.DataFrame(raw_runs)
    df = df[~df["run_failed"]]

    metrics_df = pd.json_normalize(df["metrics"])
    df = pd.concat([df.drop(columns=["metrics"]), metrics_df], axis=1)

    # Normalise seed column name
    if "seed" in df.columns and "approx_seed" not in df.columns:
        df = df.rename(columns={"seed": "approx_seed"})
    if "seed" in df.columns and "approx_seed" in df.columns:
        df["approx_seed"] = df["approx_seed"].combine_first(df["seed"])
        df = df.drop(columns=["seed"])

    return _aggregate(df)


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

    return pd.DataFrame([{
        "game_name": "N/A", "approximator_name": "N/A",
        "budget": 0, "mse": 0, "mae": 0,
        "ground_truth_method": "N/A",
        "runtime_seconds": 0, "approx_seed": 0,
    }])