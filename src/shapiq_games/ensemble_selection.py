"""Ensemble selection games: the test performance of a sub-ensemble of fitted models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np
from scipy.stats import mode

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions
from shapiq_games._training import resolve_metric, resolve_task

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shapiq.typing import CoalitionMatrix, GameValues
    from shapiq_games.typing import Metric, MetricName, Task

__all__ = ["EnsembleSelection", "RandomForestEnsembleSelection"]


class EnsembleSelection(Game):
    """The ensemble selection game: the test metric of a sub-ensemble of fitted members.

    The players are the fitted ensemble members. A coalition predicts by averaging its members'
    predictions (regression) or by majority vote (classification; ties go to the smallest class
    index). The value is the test metric of that prediction. The predictions are computed once at
    construction, so evaluating the game is cheap. An empty ensemble cannot predict, so the value of
    the empty coalition is the explicit parameter ``empty_value``.

    Attributes:
        members: The fitted ensemble members.
        member_names: A name per member.
        task: ``"classification"`` or ``"regression"``.
        predictions: The test predictions of every member, of shape ``(n_members, n_test)``.

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, n_features=5, random_state=0)
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> members = [
        ...     DecisionTreeClassifier(max_depth=depth, random_state=0).fit(X[:150], y[:150])
        ...     for depth in (1, 2, 3, 4)
        ... ]
        >>> game = EnsembleSelection(members, X[150:], y[150:])
        >>> game.n_players
        4
    """

    def __init__(
        self,
        members: Sequence[Any],
        x_test: np.ndarray,
        y_test: np.ndarray,
        *,
        task: Task | None = None,
        metric: MetricName | Metric | None = None,
        empty_value: float = 0.0,
        member_names: Sequence[str] | None = None,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the ensemble selection game.

        Args:
            members: The fitted ensemble members (scikit-learn compatible).
            x_test: The test features.
            y_test: The test labels.
            task: ``"classification"`` or ``"regression"``, or ``None`` (default) to infer it
                from the model.
            metric: ``"accuracy"``, ``"r2"``, ``"neg_mse"``, ``"neg_mae"``, a callable, or ``None``
                for accuracy (classification) or R² (regression).
            empty_value: The value of the empty coalition. Defaults to ``0``.
            member_names: A name per member, used as player names.
            normalize: Whether to center the game by ``empty_value``. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.
        """
        self.members = list(members)
        if not self.members:
            msg = "An ensemble needs at least one member."
            raise ValueError(msg)
        self.task = resolve_task(task, self.members[0])
        self.member_names = (
            [str(name) for name in member_names]
            if member_names is not None
            else [f"{i}_{type(member).__name__}" for i, member in enumerate(self.members)]
        )
        self._metric = resolve_metric(metric, self.task)
        self._y_test = np.asarray(y_test)
        self.predictions = np.stack(
            [np.asarray(member.predict(x_test)).reshape(-1) for member in self.members]
        )
        if self.task == "classification":  # vote on codes, so that any label type works
            self._classes, codes = np.unique(
                np.concatenate([self.predictions.reshape(-1), self._y_test]), return_inverse=True
            )
            self._codes = codes[: self.predictions.size].reshape(self.predictions.shape)
        else:
            self.predictions = self.predictions.astype(float)
        self.empty_value = float(empty_value)
        super().__init__(
            len(self.members),
            normalize=normalize,
            normalization_value=self.empty_value,
            verbose=verbose,
            player_names=self.member_names,
        )

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the test metric of each coalition's combined prediction."""
        coalitions = as_bool_coalitions(coalitions)
        values = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            if not coalition.any():
                values[i] = self.empty_value
                continue
            if self.task == "regression":
                prediction = self.predictions[coalition].mean(axis=0)
            else:
                votes = mode(self._codes[coalition], axis=0, keepdims=False).mode
                prediction = self._classes[votes]
            values[i] = self._metric(self._y_test, prediction)
        return values


class RandomForestEnsembleSelection(EnsembleSelection):
    """Ensemble selection over the trees of a fitted random forest.

    The players are the trees of the forest. Because a random forest classifier averages class
    probabilities while this game takes majority votes, the full ensemble can differ slightly from
    the forest's own prediction for classification.

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, n_features=5, random_state=0)
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> forest = RandomForestClassifier(n_estimators=6, random_state=0).fit(X[:150], y[:150])
        >>> game = RandomForestEnsembleSelection.from_forest(forest, X[150:], y[150:])
        >>> game.n_players  # one player per tree
        6
    """

    @classmethod
    def from_forest(
        cls,
        forest: Any,  # noqa: ANN401
        x_test: np.ndarray,
        y_test: np.ndarray,
        *,
        task: Task | None = None,
        metric: MetricName | Metric | None = None,
        empty_value: float = 0.0,
        normalize: bool = True,
    ) -> Self:
        """Build the game from a fitted scikit-learn random forest.

        Args:
            forest: A fitted ``RandomForestClassifier`` or ``RandomForestRegressor``.
            x_test: The test features.
            y_test: The test labels.
            task: ``"classification"`` or ``"regression"``, or ``None`` (default) to infer it
                from the model.
            metric: The metric, or ``None`` for the default of the task.
            empty_value: The value of the empty coalition. Defaults to ``0``.
            normalize: Whether to center the game by ``empty_value``.

        Returns:
            The game.
        """
        trees = list(getattr(forest, "estimators_", []))
        if not trees:
            msg = "Expected a fitted scikit-learn random forest with `estimators_`."
            raise TypeError(msg)
        classes = getattr(forest, "classes_", None)
        if classes is not None:  # the trees predict class indices into the forest's classes_
            y_test = np.asarray(y_test)
            if not np.isin(y_test, classes).all():
                msg = f"y_test has labels that the forest does not know: {sorted(set(y_test) - set(classes))}."
                raise ValueError(msg)
            y_test = np.searchsorted(classes, y_test)
        return cls(
            trees,
            x_test,
            y_test,
            task=task,
            metric=metric,
            empty_value=empty_value,
            member_names=[f"tree_{i}" for i in range(len(trees))],
            normalize=normalize,
        )
