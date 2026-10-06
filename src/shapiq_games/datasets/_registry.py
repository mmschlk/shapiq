"""The dataset registry: names, task types, loaders, and the :class:`Dataset` container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "Dataset",
    "DatasetSpec",
    "DatasetSplit",
    "Task",
    "get_dataset_spec",
    "list_datasets",
    "load_dataset",
    "register_dataset",
]

type Task = Literal["classification", "regression"]
type Loader = Callable[..., tuple[pd.DataFrame, pd.Series]]

_MIN_TEST_SAMPLES = 30


@dataclass(frozen=True)
class DatasetSpec:
    """Registry entry describing how to obtain a dataset.

    Attributes:
        name: The registry name, e.g. ``"california_housing"``.
        task: ``"classification"`` or ``"regression"``, declared explicitly.
        loader: A function returning the features as a DataFrame and the target as a Series.
        source: A human-readable description of where the data comes from.
        kind: ``"tabular"`` for real-world data, ``"synthetic"`` for seeded generators, and
            ``"tabarena"`` for the TabArena-v0.1 collection (requires ``openml``).
    """

    name: str
    task: Task
    loader: Loader
    source: str
    kind: Literal["tabular", "synthetic", "tabarena"] = "tabular"


@dataclass(frozen=True)
class Dataset:
    """A loaded dataset with numeric features.

    Attributes:
        name: The registry name of the dataset.
        task: ``"classification"`` or ``"regression"``.
        x: The features as a float matrix of shape ``(n_samples, n_features)``.
        y: The target of shape ``(n_samples,)``. For classification, the labels are encoded as
            integers ``0, ..., n_classes - 1`` in the order of :attr:`class_names`.
        feature_names: The names of the features.
        class_names: The original class labels for classification, empty for regression.
        params: The keyword arguments the dataset was loaded with (e.g. for synthetic data).
    """

    name: str
    task: Task
    x: np.ndarray
    y: np.ndarray
    feature_names: tuple[str, ...]
    class_names: tuple[str, ...] = ()
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """The number of samples."""
        return int(self.x.shape[0])

    @property
    def n_features(self) -> int:
        """The number of features."""
        return int(self.x.shape[1])

    @property
    def n_classes(self) -> int | None:
        """The number of classes for classification datasets, ``None`` for regression."""
        return len(self.class_names) if self.task == "classification" else None

    def split(self, *, test_size: float = 0.2, random_state: int = 42) -> DatasetSplit:
        """Split the dataset into a training and a test set, deterministically.

        Classification datasets are split stratified by class when every class has at least two
        samples. The test set holds at least 30 samples.

        Args:
            test_size: The fraction of samples used for testing. Defaults to ``0.2``.
            random_state: The seed of the split. Defaults to ``42``.

        Returns:
            The split.

        Raises:
            ValueError: If the dataset is too small for a test set of 30 samples.
        """
        n_test = max(round(test_size * self.n_samples), _MIN_TEST_SAMPLES)
        if n_test >= self.n_samples:
            msg = (
                f"Dataset '{self.name}' has only {self.n_samples} samples, which is too few for "
                f"a test set of {n_test} samples."
            )
            raise ValueError(msg)
        stratify = None
        if self.task == "classification":
            _, counts = np.unique(self.y, return_counts=True)
            if counts.min() >= 2 and n_test >= len(counts):
                stratify = self.y
        x_train, x_test, y_train, y_test = train_test_split(
            self.x,
            self.y,
            test_size=n_test,
            random_state=random_state,
            stratify=stratify,
        )
        return DatasetSplit(
            dataset=self,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            random_state=random_state,
        )


@dataclass(frozen=True)
class DatasetSplit:
    """A deterministic train/test split of a :class:`Dataset`."""

    dataset: Dataset
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    random_state: int

    @property
    def task(self) -> Task:
        """The task of the underlying dataset."""
        return self.dataset.task


_REGISTRY: dict[str, DatasetSpec] = {}


def register_dataset(spec: DatasetSpec) -> DatasetSpec:
    """Add a dataset to the registry.

    Args:
        spec: The dataset specification.

    Returns:
        The specification.

    Raises:
        ValueError: If a dataset with the same name is already registered.
    """
    if spec.name in _REGISTRY:
        msg = f"A dataset named '{spec.name}' is already registered."
        raise ValueError(msg)
    _REGISTRY[spec.name] = spec
    return spec


def get_dataset_spec(name: str) -> DatasetSpec:
    """Return the registry entry of a dataset.

    Raises:
        ValueError: If the dataset is unknown.
    """
    try:
        return _REGISTRY[name]
    except KeyError:
        msg = f"Unknown dataset '{name}'. Available datasets: {', '.join(list_datasets())}."
        raise ValueError(msg) from None


def list_datasets(
    *,
    task: Task | None = None,
    kind: Literal["tabular", "synthetic", "tabarena"] | None = None,
) -> list[str]:
    """List the names of the registered datasets, optionally filtered.

    Args:
        task: Only list datasets of this task.
        kind: Only list datasets of this kind.

    Returns:
        The sorted dataset names.
    """
    return sorted(
        name
        for name, spec in _REGISTRY.items()
        if (task is None or spec.task == task) and (kind is None or spec.kind == kind)
    )


def _feature_names(columns: pd.Index) -> tuple[str, ...]:
    if all(isinstance(column, str) for column in columns):
        return tuple(columns)
    return tuple(f"feature_{i}" for i in range(len(columns)))


def load_dataset(name: str, **params: Any) -> Dataset:
    """Load a registered dataset.

    Data files are downloaded on first use and cached locally (see
    :func:`~shapiq_games.datasets.get_data_dir`). Synthetic datasets are generated from a seed.

    Args:
        name: The registry name of the dataset (see :func:`list_datasets`).
        **params: Parameters of synthetic generators, e.g. ``n_samples`` or ``random_state``.

    Returns:
        The dataset with numeric features.

    Raises:
        ValueError: If the dataset is unknown, parameters are given for a non-synthetic dataset,
            or the features are not numeric.
    """
    spec = get_dataset_spec(name)
    if params and spec.kind != "synthetic":
        msg = f"Dataset '{name}' does not take parameters, got {sorted(params)}."
        raise ValueError(msg)
    x_frame, y_series = spec.loader(**params)

    try:
        x = x_frame.to_numpy(dtype=float)
    except (TypeError, ValueError) as error:
        msg = f"Dataset '{name}' has non-numeric features after preprocessing."
        raise ValueError(msg) from error

    y_raw = np.asarray(y_series)
    class_names: tuple[str, ...] = ()
    if spec.task == "classification":
        classes, y = np.unique(y_raw, return_inverse=True)
        class_names = tuple(str(label) for label in classes)
    else:
        y = y_raw.astype(float)

    return Dataset(
        name=name,
        task=spec.task,
        x=x,
        y=y.reshape(-1),
        feature_names=_feature_names(x_frame.columns),
        class_names=class_names,
        params=dict(params),
    )
