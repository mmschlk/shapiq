"""Train a diagnostic pairwise classifier from local leaderboard run records."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from leaderboard.pdl.pairwise_dataset import (
    aggregate_metric_scores,
    build_pairwise_dataset,
    flatten_runs,
)
from leaderboard.storage.connection import DatabaseClientFactory

if TYPE_CHECKING:
    import pandas as pd

LOGGER = logging.getLogger(__name__)

DEFAULT_PATH = Path("src/leaderboard/data/results_23Jun_1658.jsonl")
CATEGORICAL_FEATURES = [
    "game_name",
    "game_params_json",
    "index",
    "ground_truth_method",
    "metric",
    "approximator_A",
    "approximator_B",
]
NUMERIC_FEATURES = ["n_players", "max_order", "budget", "log_budget"]
CONTEXT_COLUMNS = [
    "game_name",
    "game_params_json",
    "n_players",
    "index",
    "max_order",
    "ground_truth_method",
    "budget",
    "metric",
]


def main() -> None:
    """Load local runs, build pairwise examples, and print diagnostic model metrics."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    raw_runs = _load_raw_runs(args.path)
    flattened = flatten_runs(raw_runs)
    scores = aggregate_metric_scores(flattened)
    pairwise = build_pairwise_dataset(scores)

    _print_profile(raw_runs, flattened, scores, pairwise)
    if len(pairwise) < 2 or pairwise["label"].nunique() < 2:
        msg = "Need at least two pairwise examples with both classes to train."
        raise SystemExit(msg)

    train_df, test_df = _split_pairwise(pairwise)
    pipeline = _build_pipeline()
    pipeline.fit(train_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES], train_df["label"])

    predictions = pipeline.predict(test_df[CATEGORICAL_FEATURES + NUMERIC_FEATURES])
    LOGGER.info("train size: %s", len(train_df))
    LOGGER.info("test size: %s", len(test_df))
    LOGGER.info("accuracy: %.4f", accuracy_score(test_df["label"], predictions))
    LOGGER.info("balanced accuracy: %.4f", balanced_accuracy_score(test_df["label"], predictions))
    LOGGER.info(
        "classification report:\n%s",
        classification_report(test_df["label"], predictions, zero_division=0),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny PDL pairwise classifier prototype.")
    parser.add_argument(
        "--path", type=Path, default=DEFAULT_PATH, help="Path to a local JSONL run file."
    )
    return parser.parse_args()


def _load_raw_runs(path: Path) -> list[dict]:
    client = DatabaseClientFactory.create_client("local", {"LOCAL_DB_PATH": str(path)})
    return client.get_all()


def _print_profile(
    raw_runs: list[dict],
    flattened: pd.DataFrame,
    scores: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    metric_columns = [metric for metric in ("mse", "spearman") if metric in flattened.columns]
    LOGGER.info("data profile:")
    LOGGER.info("raw run count: %s", len(raw_runs))
    LOGGER.info("flattened row count: %s", len(flattened))
    LOGGER.info("unique games: %s", _nunique(flattened, "game_name"))
    LOGGER.info("unique budgets: %s", _nunique(flattened, "budget"))
    LOGGER.info("unique approximators: %s", _nunique(flattened, "approximator_name"))
    LOGGER.info("available metric columns: %s", metric_columns)
    LOGGER.info("non-null mse count: %s", _non_null_count(flattened, "mse"))
    LOGGER.info("non-null spearman count: %s", _non_null_count(flattened, "spearman"))
    LOGGER.info("aggregated score row count: %s", len(scores))
    LOGGER.info("pairwise example count: %s", len(pairwise))
    LOGGER.info("class balance: %s", pairwise["label"].value_counts().sort_index().to_dict())


def _split_pairwise(pairwise: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = pairwise[CONTEXT_COLUMNS].astype(str).agg("|".join, axis=1)
    if groups.nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(pairwise, pairwise["label"], groups=groups))
        return pairwise.iloc[train_idx], pairwise.iloc[test_idx]

    LOGGER.info("warning: only one comparison context; evaluation is diagnostic only.")
    stratify = pairwise["label"] if pairwise["label"].value_counts().min() >= 2 else None
    train_df, test_df = train_test_split(
        pairwise,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )
    return train_df, test_df


def _build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=42, n_estimators=100)),
        ]
    )


def _nunique(df: pd.DataFrame, column: str) -> int:
    return int(df[column].nunique(dropna=True)) if column in df.columns else 0


def _non_null_count(df: pd.DataFrame, column: str) -> int:
    return int(df[column].notna().sum()) if column in df.columns else 0


if __name__ == "__main__":
    main()
