"""Ensemble selection games: the test performance of a sub-ensemble of fitted models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq.game import Game
from shapiq.utils.modules import safe_isinstance
from shapiq_games._base import as_bool_coalitions
from shapiq_games._training import resolve_row_metric, resolve_task

if TYPE_CHECKING:
    from collections.abc import Sequence

    from shapiq.typing import CoalitionMatrix, GameValues
    from shapiq_games.typing import Metric, MetricName, Task

__all__ = ["EnsembleSelection", "RandomForestEnsembleSelection"]

_MAX_ELEMENTS = 2**22  # the most predictions or votes held at once
_FORESTS = [
    f"sklearn.ensemble.{kind}{task}"
    for kind in ("RandomForest", "ExtraTrees")
    for task in ("Classifier", "Regressor")
]


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
            y_test: The test labels of one target, of shape ``(n_test,)`` or ``(n_test, 1)``.
            task: ``"classification"`` or ``"regression"``, or ``None`` (default) to infer it
                from the model.
            metric: ``"accuracy"``, ``"r2"``, ``"neg_mse"``, ``"neg_mae"``, a callable, or ``None``
                for accuracy (classification) or R² (regression).
            empty_value: The value of the empty coalition. Defaults to ``0``.
            member_names: A name per member, used as player names.
            normalize: Whether to center the game by ``empty_value``. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.

        Raises:
            ValueError: If there are no members, or ``y_test`` holds more than one target.
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
        self._row_metric = resolve_row_metric(metric, self.task)
        self._y_test = np.asarray(y_test)
        if self._y_test.ndim == 2 and self._y_test.shape[1] == 1:  # a column vector
            self._y_test = self._y_test[:, 0]
        if self._y_test.ndim != 1:
            msg = f"Expected one target, got y_test of shape {self._y_test.shape}."
            raise ValueError(msg)
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
        values = np.full(coalitions.shape[0], self.empty_value)
        n_classes = self._classes.shape[0] if self.task == "classification" else 1
        step = max(1, _MAX_ELEMENTS // (self.predictions.shape[1] * n_classes))
        for start in range(0, coalitions.shape[0], step):
            chunk = coalitions[start : start + step]
            present = chunk.any(axis=1)  # the empty ensemble has the value empty_value
            if present.any():
                predictions = self._combined_predictions(chunk[present])
                values[start : start + step][present] = self._row_metric(self._y_test, predictions)
        return values

    def _combined_predictions(self, coalitions: CoalitionMatrix) -> np.ndarray:
        """Return every (non-empty) coalition's prediction, of shape ``(n_coalitions, n_test)``.

        Regression averages the members' predictions, added member after member (the order of
        ``predictions[coalition].mean(axis=0)``); classification counts the votes for every class
        and takes the most frequent one, the smallest on ties (as ``scipy.stats.mode``).
        """
        n_test = self.predictions.shape[1]
        if self.task == "regression":
            sums = np.zeros((coalitions.shape[0], n_test))
            for member, prediction in enumerate(self.predictions):
                sums[coalitions[:, member]] += prediction
            return sums / np.sum(coalitions, axis=1, keepdims=True)
        votes = np.zeros((coalitions.shape[0], n_test, self._classes.shape[0]), dtype=np.int32)
        tests = np.arange(n_test)
        for member, codes in enumerate(self._codes):
            votes[:, tests, codes] += coalitions[:, member, None]
        return self._classes[np.argmax(votes, axis=2)]


class RandomForestEnsembleSelection(EnsembleSelection):
    """Ensemble selection over the trees of a fitted random forest.

    The players are the trees of the forest. Because a random forest classifier averages class
    probabilities while this game takes majority votes, the full ensemble can differ slightly from
    the forest's own prediction for classification.

    Attributes:
        forest: The explained random forest.

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, n_features=5, random_state=0)
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> forest = RandomForestClassifier(n_estimators=6, random_state=0).fit(X[:150], y[:150])
        >>> game = RandomForestEnsembleSelection(forest, X[150:], y[150:])
        >>> game.n_players  # one player per tree
        6
    """

    def __init__(
        self,
        forest: Any,  # noqa: ANN401
        x_test: np.ndarray,
        y_test: np.ndarray,
        *,
        task: Task | None = None,
        metric: MetricName | Metric | None = None,
        empty_value: float = 0.0,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the game from a fitted scikit-learn random forest.

        Args:
            forest: A fitted ``RandomForestClassifier`` or ``RandomForestRegressor``.
            x_test: The test features.
            y_test: The test labels.
            task: ``"classification"`` or ``"regression"``, or ``None`` (default) to infer it
                from the model.
            metric: The metric, or ``None`` for the default of the task.
            empty_value: The value of the empty coalition. Defaults to ``0``.
            normalize: Whether to center the game by ``empty_value``.
            verbose: Whether to show a progress bar when evaluating the game.

        Raises:
            TypeError: If ``forest`` is not a fitted scikit-learn random forest or extra-trees
                ensemble.
            ValueError: If ``y_test`` has labels the forest does not know.
        """
        trees = list(getattr(forest, "estimators_", []))
        if not (safe_isinstance(forest, _FORESTS) and trees):
            # other ensembles' members can predict labels instead of indices into classes_
            msg = (
                "Expected a fitted scikit-learn random forest (or extra trees) with `estimators_`."
            )
            raise TypeError(msg)
        classes = getattr(forest, "classes_", None)
        if classes is not None:  # the trees predict class indices into the forest's classes_
            y_test = np.asarray(y_test)
            if not np.isin(y_test, classes).all():
                unknown = sorted(set(y_test) - set(classes))
                msg = f"y_test has labels that the forest does not know: {unknown}."
                raise ValueError(msg)
            y_test = np.searchsorted(classes, y_test)
        self.forest = forest
        super().__init__(
            trees,
            x_test,
            y_test,
            task=task,
            metric=metric,
            empty_value=empty_value,
            member_names=[f"tree_{i}" for i in range(len(trees))],
            normalize=normalize,
            verbose=verbose,
        )
