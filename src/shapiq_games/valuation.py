"""Data valuation and dataset valuation games.

Both games share one implementation: the players are disjoint groups of training rows, and the
value of a coalition is the test metric of a model trained on the union of its groups. Data
valuation (Ghorbani and Zou, 2019) uses one training point per player; dataset valuation uses
groups such as data sources or owners. Mathematically, data valuation is the singleton-group
special case of dataset valuation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions
from shapiq_games._training import (
    Metric,
    MetricName,
    fit_and_score,
    resolve_metric,
    resolve_task,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["DataValuation", "DatasetValuation"]


class _GroupValuation(Game):
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
        random_state: int | None,
        normalize: bool,
        verbose: bool,
    ) -> None:
        self.model = model
        self.random_state = random_state
        self.task = resolve_task(task, model)
        self._metric = resolve_metric(metric, self.task)
        self._x_train, self._y_train = np.asarray(x_train), np.asarray(y_train)
        self._x_test, self._y_test = np.asarray(x_test), np.asarray(y_test)
        self.groups = [_group_rows(group) for group in groups]
        rows = np.concatenate(self.groups) if self.groups else np.zeros(0, dtype=int)
        if np.unique(rows).shape[0] != rows.shape[0]:
            msg = "The groups must be disjoint: a training row belongs to at most one player."
            raise ValueError(msg)
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
                random_state=self.random_state,
            )
        return values


def _group_rows(group: np.ndarray | Sequence[int]) -> np.ndarray:
    """Return the row indices of a group given as indices or as a boolean mask."""
    group = np.asarray(group)
    if group.dtype == bool:
        return np.flatnonzero(group)
    return group.astype(int).reshape(-1)


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
        random_state: int = 42,
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
            random_state: The seed of the model's clones if the model leaves its ``random_state``
                unset, so that every coalition has one value. Defaults to ``42``.
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
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
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
            groups: The training rows of every player, as indices or boolean masks; the groups
                must be disjoint. If ``None``, the shuffled training rows are split into
                ``n_players`` groups according to ``player_sizes``.
            n_players: The number of groups when ``groups`` is ``None``. Defaults to ``10``.
            player_sizes: ``"uniform"`` (equal sizes), ``"increasing"`` (sizes proportional to
                ``1, ..., n``), ``"random"``, or relative sizes. Defaults to ``"uniform"``.
            metric: ``"accuracy"``, ``"r2"``, ``"neg_mse"``, ``"neg_mae"``, a callable, or ``None``
                for accuracy (classification) or R² (regression).
            empty_value: The value of the empty coalition. Defaults to ``0``.
            random_state: The seed of the split into groups, and of the model's clones if the
                model leaves its ``random_state`` unset. Defaults to ``42``.
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
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )


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
