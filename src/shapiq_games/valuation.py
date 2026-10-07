"""Data valuation and dataset valuation games.

Both games share one implementation: the players are disjoint groups of training rows, and the
value of a coalition is the test metric of a model trained on the union of its groups. Data
valuation (Ghorbani and Zou, 2019) uses one training point per player; dataset valuation uses
groups such as data sources or owners. Mathematically, data valuation is the singleton-group
special case of dataset valuation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np

from shapiq.game import Game
from shapiq_games._base import ConfigMixin, as_bool_coalitions
from shapiq_games._setup import configure
from shapiq_games._training import (
    Metric,
    MetricName,
    fit_and_score,
    resolve_metric,
    resolve_task,
)
from shapiq_games.models import build_model

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["DataValuation", "DatasetValuation"]


class _GroupValuation(ConfigMixin, Game):
    """Shared implementation: players are groups of training rows, a model is trained per coalition.

    Attributes:
        task: ``"classification"`` or ``"regression"``.
        model: The unfitted model that is cloned and trained per coalition.
        groups: The training-row indices of every player.
        empty_value: The value of the empty coalition (a model trained on no data).
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        groups: Sequence[np.ndarray],
        *,
        task: str | None = None,
        metric: MetricName | Metric | None,
        empty_value: float,
        normalize: bool,
        verbose: bool,
    ) -> None:
        self.model = model
        self.task = resolve_task(task, model)
        self._metric = resolve_metric(metric, self.task)
        self._x_train, self._y_train = np.asarray(x_train), np.asarray(y_train)
        self._x_test, self._y_test = np.asarray(x_test), np.asarray(y_test)
        self.groups = [np.asarray(group, dtype=int) for group in groups]
        self.empty_value = float(empty_value)
        super().__init__(
            len(self.groups),
            normalize=normalize,
            normalization_value=self.empty_value,
            verbose=verbose,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the test metric of the model trained on the union of the coalition's groups."""
        coalitions = as_bool_coalitions(coalitions)
        values = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            players = np.flatnonzero(coalition)
            if players.size == 0:
                values[i] = self.empty_value
                continue
            rows = np.concatenate([self.groups[player] for player in players])
            values[i] = fit_and_score(
                self.model,
                self._x_train[rows],
                self._y_train[rows],
                self._x_test,
                self._y_test,
                task=self.task,
                metric=self._metric,
            )
        return values


class DataValuation(_GroupValuation):
    """The data valuation game (Data Shapley): every training point is a player.

    The value of a coalition of training points is the test metric of a fresh copy of the model
    trained on them. A model trained on no data has no natural score, so the value of the empty
    coalition is the explicit parameter ``empty_value`` (default ``0``).

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, n_features=5, random_state=0)
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> model = DecisionTreeClassifier(random_state=0)  # unfitted: refit for every coalition
        >>> game = DataValuation(model, X[:8], y[:8], X[100:], y[100:])
        >>> game.n_players  # one player per training point
        8
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        *,
        task: str | None = None,
        metric: MetricName | Metric | None = None,
        empty_value: float = 0.0,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the data valuation game.

        Args:
            model: An unfitted scikit-learn compatible estimator (cloned per evaluation).
            x_train: The training points (the players).
            y_train: The training labels.
            x_test: The test features.
            y_test: The test labels.
            task: ``"classification"`` or ``"regression"``, or ``None`` (default) to infer it
                from the model.
            metric: ``"accuracy"``, ``"r2"``, ``"neg_mse"``, ``"neg_mae"``, a callable, or ``None``
                for accuracy (classification) or R² (regression).
            empty_value: The value of the empty coalition. Defaults to ``0``.
            normalize: Whether to center the game by ``empty_value``. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.
        """
        groups = [np.array([row]) for row in range(np.asarray(x_train).shape[0])]
        super().__init__(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            groups,
            task=task,
            metric=metric,
            empty_value=empty_value,
            normalize=normalize,
            verbose=verbose,
        )

    @classmethod
    def from_config(
        cls,
        *,
        dataset: str,
        model: str = "decision_tree",
        n_players: int = 10,
        metric: MetricName | None = None,
        empty_value: float = 0.0,
        random_state: int = 42,
        test_size: float = 0.2,
        model_params: dict[str, Any] | None = None,
        dataset_params: dict[str, Any] | None = None,
        normalize: bool = True,
    ) -> Self:
        """Build the game with ``n_players`` seeded training points of a registered dataset.

        The points are drawn stratified by class where possible; the test split is the test set.

        Args:
            dataset: The dataset name.
            model: The model name. Defaults to ``"decision_tree"``.
            n_players: The number of training points, i.e. players. Defaults to ``10``.
            metric: The metric name, or ``None`` for the default of the task.
            empty_value: The value of the empty coalition. Defaults to ``0``.
            random_state: The seed of the split, the sample of points, and the model.
            test_size: The fraction of the data used as test set. Defaults to ``0.2``.
            model_params: Hyperparameters of the model.
            dataset_params: Parameters of synthetic datasets.
            normalize: Whether to center the game.

        Returns:
            The configured game.
        """
        setup = configure(
            dataset=dataset,
            model=None,
            random_state=random_state,
            test_size=test_size,
            dataset_params=dataset_params,
        )
        split = setup.split
        rows = _sample_rows(split.y_train, n_players, split.task, random_state)
        model_params = dict(model_params or {})
        estimator = build_model(model, split.task, random_state=random_state, **model_params)
        game = cls(
            estimator,
            split.x_train[rows],
            split.y_train[rows],
            split.x_test,
            split.y_test,
            task=split.task,
            metric=metric,
            empty_value=empty_value,
            normalize=normalize,
        )
        config = {**setup.config, "model": model, "model_params": model_params}
        return game._set_config(
            **config,
            n_players=n_players,
            metric=metric,
            empty_value=empty_value,
            normalize=normalize,
        )


class DatasetValuation(_GroupValuation):
    """The dataset valuation game: every player is a group of training rows (e.g. a data source).

    The value of a coalition of datasets is the test metric of a fresh copy of the model trained
    on their union. The groups are either given explicitly or obtained by splitting the training
    data. The value of the empty coalition is the explicit parameter ``empty_value``.

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, n_features=5, random_state=0)
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> sources = [np.arange(0, 20), np.arange(20, 60), np.arange(60, 100)]
        >>> game = DatasetValuation(
        ...     DecisionTreeClassifier(random_state=0), X[:100], y[:100], X[100:], y[100:], groups=sources
        ... )
        >>> game.n_players  # one player per data source
        3
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        *,
        task: str | None = None,
        groups: Sequence[np.ndarray] | None = None,
        n_players: int = 10,
        player_sizes: Literal["uniform", "increasing", "random"] | Sequence[float] = "uniform",
        metric: MetricName | Metric | None = None,
        empty_value: float = 0.0,
        random_state: int = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the dataset valuation game.

        Args:
            model: An unfitted scikit-learn compatible estimator (cloned per evaluation).
            x_train: The training features of all groups.
            y_train: The training labels of all groups.
            x_test: The test features.
            y_test: The test labels.
            task: ``"classification"`` or ``"regression"``, or ``None`` (default) to infer it
                from the model.
            groups: The training-row indices of every player. If ``None``, the shuffled training
                rows are split into ``n_players`` groups according to ``player_sizes``.
            n_players: The number of groups when ``groups`` is ``None``. Defaults to ``10``.
            player_sizes: ``"uniform"`` (equal sizes), ``"increasing"`` (sizes proportional to
                ``1, ..., n``), ``"random"``, or relative sizes. Defaults to ``"uniform"``.
            metric: ``"accuracy"``, ``"r2"``, ``"neg_mse"``, ``"neg_mae"``, a callable, or ``None``
                for accuracy (classification) or R² (regression).
            empty_value: The value of the empty coalition. Defaults to ``0``.
            random_state: The seed of the split into groups. Defaults to ``42``.
            normalize: Whether to center the game by ``empty_value``. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.
        """
        if groups is None:
            groups = _split_into_groups(
                np.asarray(x_train).shape[0], n_players, player_sizes, random_state
            )
        super().__init__(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            groups,
            task=task,
            metric=metric,
            empty_value=empty_value,
            normalize=normalize,
            verbose=verbose,
        )

    @classmethod
    def from_config(
        cls,
        *,
        dataset: str,
        model: str = "decision_tree",
        n_players: int = 10,
        player_sizes: Literal["uniform", "increasing", "random"] = "uniform",
        n_train: int | None = None,
        metric: MetricName | None = None,
        empty_value: float = 0.0,
        random_state: int = 42,
        test_size: float = 0.2,
        model_params: dict[str, Any] | None = None,
        dataset_params: dict[str, Any] | None = None,
        normalize: bool = True,
    ) -> Self:
        """Build the game by splitting the training data of a registered dataset into groups.

        Args:
            dataset: The dataset name.
            model: The model name. Defaults to ``"decision_tree"``.
            n_players: The number of groups. Defaults to ``10``.
            player_sizes: ``"uniform"``, ``"increasing"``, or ``"random"``.
            n_train: Use a seeded subset of this many training rows (``None`` for all).
            metric: The metric name, or ``None`` for the default of the task.
            empty_value: The value of the empty coalition. Defaults to ``0``.
            random_state: The seed of the split, the groups, and the model.
            test_size: The fraction of the data used as test set. Defaults to ``0.2``.
            model_params: Hyperparameters of the model.
            dataset_params: Parameters of synthetic datasets.
            normalize: Whether to center the game.

        Returns:
            The configured game.
        """
        setup = configure(
            dataset=dataset,
            model=None,
            random_state=random_state,
            test_size=test_size,
            dataset_params=dataset_params,
        )
        split = setup.split
        x_train, y_train = split.x_train, split.y_train
        if n_train is not None and n_train < x_train.shape[0]:
            rows = _sample_rows(y_train, n_train, split.task, random_state)
            x_train, y_train = x_train[rows], y_train[rows]
        model_params = dict(model_params or {})
        estimator = build_model(model, split.task, random_state=random_state, **model_params)
        game = cls(
            estimator,
            x_train,
            y_train,
            split.x_test,
            split.y_test,
            task=split.task,
            n_players=n_players,
            player_sizes=player_sizes,
            metric=metric,
            empty_value=empty_value,
            random_state=random_state,
            normalize=normalize,
        )
        config = {**setup.config, "model": model, "model_params": model_params}
        return game._set_config(
            **config,
            n_players=n_players,
            player_sizes=player_sizes,
            n_train=n_train,
            metric=metric,
            empty_value=empty_value,
            normalize=normalize,
        )


def _sample_rows(y: np.ndarray, n: int, task: str, random_state: int) -> np.ndarray:
    """Draw ``n`` row indices, stratified by class for classification where possible."""
    from sklearn.model_selection import train_test_split

    indices = np.arange(y.shape[0])
    if n >= indices.shape[0]:
        return indices
    stratify = y if task == "classification" else None
    try:
        rows, _ = train_test_split(
            indices, train_size=n, random_state=random_state, stratify=stratify
        )
    except ValueError:  # too few points per class to stratify
        rows, _ = train_test_split(indices, train_size=n, random_state=random_state)
    return np.sort(rows)


def _split_into_groups(
    n_rows: int,
    n_players: int,
    player_sizes: Literal["uniform", "increasing", "random"] | Sequence[float],
    random_state: int,
) -> list[np.ndarray]:
    """Shuffle the rows and split them into groups with the given relative sizes."""
    rng = np.random.default_rng(random_state)
    if isinstance(player_sizes, str):
        if player_sizes == "uniform":
            sizes = np.ones(n_players)
        elif player_sizes == "increasing":
            sizes = np.arange(1, n_players + 1, dtype=float)
        elif player_sizes == "random":
            sizes = rng.random(n_players)
        else:
            msg = "player_sizes must be 'uniform', 'increasing', 'random', or a sequence of sizes."
            raise ValueError(msg)
    else:
        sizes = np.asarray(player_sizes, dtype=float)
    shares = sizes / sizes.sum()
    boundaries = np.round(np.cumsum(shares) * n_rows).astype(int)
    boundaries[-1] = n_rows
    permutation = rng.permutation(n_rows)
    groups = np.split(permutation, boundaries[:-1])
    if any(group.size == 0 for group in groups):
        msg = f"Cannot split {n_rows} rows into {len(groups)} non-empty groups."
        raise ValueError(msg)
    return [np.sort(group) for group in groups]
