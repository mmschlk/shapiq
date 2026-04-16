"""Helpers to load datasets and models from string identifiers."""

from __future__ import annotations

from collections.abc import Callable

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shapiq_games.benchmark.setup import GameBenchmarkSetup

from .bench_types import BenchmarkDataset

ModelBuilder = Callable[[int | None, int], object]


_MODEL_BUILDERS: dict[tuple[str, str], ModelBuilder] = {
    ("decision_tree", "classification"): lambda random_state, _n_estimators: (
        DecisionTreeClassifier(random_state=random_state)
    ),
    ("decision_tree", "regression"): lambda random_state, _n_estimators: (
        DecisionTreeRegressor(random_state=random_state)
    ),
    ("random_forest", "classification"): lambda random_state, n_estimators: (
        RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    ),
    ("random_forest", "regression"): lambda random_state, n_estimators: (
        RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    ),
}


def register_model_builder(
    model_name: str, data_type: str, builder: ModelBuilder
) -> None:
    """Register a new model builder for future extensions."""
    key = (model_name.lower(), data_type.lower())
    if key in _MODEL_BUILDERS:
        msg = f"Model builder already registered for {key[0]} ({key[1]})."
        raise ValueError(msg)
    _MODEL_BUILDERS[key] = builder


def load_data_from_str(
    data_str: str,
    *,
    random_state: int | None = 42,
    test_size: float = 0.2,
) -> BenchmarkDataset:
    """Load a dataset from a string identifier.

    Args:
            data_str: Dataset identifier (e.g. "adult_census").
            random_state: Random state used for dataset shuffling and split.
            test_size: Fraction of data used for testing.

    Returns:
            A BenchmarkDataset containing train/test splits and metadata.
    """
    setup = GameBenchmarkSetup(
        dataset_name=data_str,
        model_name=None,
        verbose=False,
        random_state=random_state,
        test_size=test_size,
    )

    if setup.x_test.shape[0] > 0:
        x_explain = setup.x_test[0]
    else:
        x_explain = setup.x_train[0]

    return BenchmarkDataset(
        x_train=setup.x_train,
        y_train=setup.y_train,
        x_test=setup.x_test,
        y_test=setup.y_test,
        data_type=setup.dataset_type,
        x_explain=x_explain,
    )


def load_model_from_str(
    model_str: str,
    dataset: BenchmarkDataset,
    *,
    random_state: int | None = 42,
    n_estimators: int = 10,
) -> object:
    """Create an unfitted model from a string identifier.

    Args:
            model_str: Model identifier (e.g. "decision_tree").
            dataset: Dataset metadata used to choose classifier vs regressor.
            random_state: Random state for the model.
            n_estimators: Number of estimators for random forest models.

    Returns:
            An unfitted model instance.
    """
    key = (model_str.lower(), dataset.data_type.lower())
    if key not in _MODEL_BUILDERS:
        supported = sorted({name for name, _dtype in _MODEL_BUILDERS})
        msg = (
            f"Unsupported model '{model_str}' for data type '{dataset.data_type}'. "
            f"Supported models: {', '.join(supported)}"
        )
        raise ValueError(msg)

    return _MODEL_BUILDERS[key](random_state, n_estimators)
