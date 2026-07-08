"""Build a tiny pairwise decision-learning dataset from raw leaderboard runs."""

from __future__ import annotations

import itertools
import json
import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd

from leaderboard.metrics.registry import METRIC_SPECS

LOGGER = logging.getLogger(__name__)

DEFAULT_METRICS = ("mse", "spearman")

RUN_COLUMNS = [
    "game_name",
    "game_id",
    "game_params",
    "n_players",
    "approximator_name",
    "approximator_params",
    "index",
    "max_order",
    "budget",
    "approx_seed",
    "ground_truth_method",
    "runtime_seconds",
    "shapiq_version",
    "hardware",
    "mse",
    "spearman",
]

SCORE_GROUP_COLUMNS = [
    "game_name",
    "game_params_json",
    "n_players",
    "index",
    "max_order",
    "ground_truth_method",
    "budget",
    "approximator_name",
]

PAIR_CONTEXT_COLUMNS = [
    "game_name",
    "game_params_json",
    "n_players",
    "index",
    "max_order",
    "ground_truth_method",
    "budget",
    "metric",
]


def flatten_runs(raw_runs: list[dict]) -> pd.DataFrame:
    """Return non-failed raw runs with nested metric values flattened into columns."""
    records = [dict(run) for run in raw_runs if run.get("run_failed") is not True]
    if not records:
        return _empty_flattened_frame()

    runs_df = pd.DataFrame(records)
    if "seed" in runs_df.columns and "approx_seed" not in runs_df.columns:
        runs_df = runs_df.rename(columns={"seed": "approx_seed"})

    metrics = runs_df["metrics"] if "metrics" in runs_df.columns else pd.Series([{}] * len(runs_df))
    metrics_df = pd.json_normalize(
        metrics.map(lambda value: value if isinstance(value, dict) else {})
    )
    runs_df = pd.concat([runs_df.drop(columns=["metrics"], errors="ignore"), metrics_df], axis=1)

    for column in RUN_COLUMNS:
        if column not in runs_df.columns:
            runs_df[column] = pd.NA

    runs_df["game_params_json"] = runs_df["game_params"].map(_stable_json)
    runs_df["approximator_params_json"] = runs_df["approximator_params"].map(_stable_json)

    return runs_df


def aggregate_metric_scores(
    df: pd.DataFrame,
    metric_names: tuple[str, ...] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Aggregate requested metric scores across approximation seeds."""
    available_metrics = [metric for metric in metric_names if metric in df.columns]
    rows = []

    for metric in available_metrics:
        metric_df = df.copy()
        metric_df["score"] = pd.to_numeric(metric_df[metric], errors="coerce")
        grouped = (
            metric_df.groupby(SCORE_GROUP_COLUMNS, dropna=False)
            .agg(score_mean=("score", "mean"), n_seeds=("score", "count"))
            .reset_index()
        )
        grouped["metric"] = metric
        rows.append(grouped)

    columns = [*SCORE_GROUP_COLUMNS, "metric", "score_mean", "n_seeds"]
    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.concat(rows, ignore_index=True)[columns]


def build_pairwise_dataset(
    scores_df: pd.DataFrame,
    metric_names: tuple[str, ...] = DEFAULT_METRICS,
) -> pd.DataFrame:
    """Compare approximators pairwise within each benchmark context."""
    _validate_metrics(scores_df, metric_names)
    rows: list[dict[str, Any]] = []

    for context, group in scores_df.groupby(PAIR_CONTEXT_COLUMNS, dropna=False, sort=True):
        context_values = dict(zip(PAIR_CONTEXT_COLUMNS, context, strict=True))
        metric = str(context_values["metric"])
        higher_is_better = METRIC_SPECS[metric].higher_is_better
        sorted_group = group.sort_values("approximator_name")

        for left, right in itertools.combinations(sorted_group.to_dict("records"), 2):
            score_a = _finite_float(left.get("score_mean"))
            score_b = _finite_float(right.get("score_mean"))
            if score_a is None or score_b is None or score_a == score_b:
                continue

            label = int(score_a > score_b) if higher_is_better else int(score_a < score_b)
            budget = _finite_float(context_values["budget"])
            row = {
                **context_values,
                "log_budget": math.log(budget) if budget and budget > 0 else math.nan,
                "approximator_A": left["approximator_name"],
                "approximator_B": right["approximator_name"],
                "score_A": score_a,
                "score_B": score_b,
                "score_diff": score_a - score_b,
                "label": label,
            }
            rows.append(row)

    columns = [
        "game_name",
        "game_params_json",
        "n_players",
        "index",
        "max_order",
        "ground_truth_method",
        "budget",
        "log_budget",
        "metric",
        "approximator_A",
        "approximator_B",
        "score_A",
        "score_B",
        "score_diff",
        "label",
    ]
    return pd.DataFrame(rows, columns=columns)


def _stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _empty_flattened_frame() -> pd.DataFrame:
    columns = [*RUN_COLUMNS, "game_params_json", "approximator_params_json"]
    return pd.DataFrame(columns=columns)


def _finite_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, int | float):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _validate_metrics(scores_df: pd.DataFrame, metric_names: tuple[str, ...]) -> None:
    requested = set(metric_names)
    unexpected = sorted(set(scores_df.get("metric", pd.Series(dtype=object)).dropna()) - requested)
    if unexpected:
        msg = f"Unexpected metric(s) in scores_df: {unexpected}. Requested metrics: {sorted(requested)}."
        raise ValueError(msg)

    missing_specs = sorted(requested - set(METRIC_SPECS))
    if missing_specs:
        msg = f"Metric direction is not registered for: {missing_specs}."
        raise ValueError(msg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    input_path = Path("leaderboard/data/results_23Jun_1658.jsonl")
    output_path = Path("pairwise_dataset.csv")

    LOGGER.info("Lese Rohdaten aus %s...", input_path)

    # 1. Daten einlesen (JSON-Zeilenweise)
    try:
        raw_runs = [
            json.loads(line)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except FileNotFoundError:
        msg = f"Fehler: Datei '{input_path}' nicht gefunden."
        raise SystemExit(msg) from None

    # 2. Pipeline Schritt für Schritt ausführen
    LOGGER.info("Schritt 1: Verarbeite und glätte rohe JSON-Daten...")
    runs_df = flatten_runs(raw_runs)

    LOGGER.info("Schritt 2: Berechne Durchschnittswerte über die Seeds...")
    scores_df = aggregate_metric_scores(runs_df)

    LOGGER.info("Schritt 3: Baue mathematisch exakte paarweise Vergleiche...")
    pairwise_df = build_pairwise_dataset(scores_df)

    # 3. Als CSV abspeichern
    if not pairwise_df.empty:
        pairwise_df.to_csv(output_path, index=False)
        LOGGER.info(
            "Erfolg! %s paarweise Vergleiche wurden in '%s' gespeichert.",
            len(pairwise_df),
            output_path,
        )
    else:
        LOGGER.info("Keine auswertbaren Paare gefunden.")
