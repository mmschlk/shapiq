"""Feature selection games: the test performance of a model retrained on a coalition of features."""

from __future__ import annotations

from typing import Any

import numpy as np

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions
from shapiq_games._training import (
    Metric,
    MetricName,
    empty_model_score,
    fit_and_score,
    resolve_metric,
    resolve_task,
)

__all__ = ["FeatureSelection"]


class FeatureSelection(Game):
    """The feature selection game: the test metric of a model retrained on a feature subset.

    The players are the features. The value of a coalition is the test metric of a fresh copy of
    the model trained only on the features in the coalition. The empty coalition is a model
    without features (majority class or training mean). Every evaluation retrains the model, so
    the game is expensive; consider subsampling the training data.

    Attributes:
        task: ``"classification"`` or ``"regression"``.
        model: The unfitted model that is cloned and trained per coalition.

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, n_features=5, random_state=0)
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> model = DecisionTreeClassifier(random_state=0)  # unfitted: refit for every coalition
        >>> game = FeatureSelection(model, X[:150], y[:150], X[150:], y[150:])
        >>> game.n_players, game.task
        (5, 'classification')
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
        random_state: int = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the feature selection game.

        Args:
            model: An unfitted scikit-learn compatible estimator (cloned per evaluation).
            x_train: The training features.
            y_train: The training labels.
            x_test: The test features.
            y_test: The test labels.
            task: ``"classification"`` or ``"regression"``, or ``None`` (default) to infer it
                from the model.
            metric: ``"accuracy"``, ``"r2"``, ``"neg_mse"``, ``"neg_mae"``, a callable
                ``metric(y_true, y_pred)``, or ``None`` for accuracy (classification) or R²
                (regression). Higher is better.
            random_state: The seed of the model's clones if the model leaves its ``random_state``
                unset, so that every coalition has one value. Defaults to ``42``.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.
        """
        self.model = model
        self.random_state = random_state
        self.task = resolve_task(task, model)
        self._metric = resolve_metric(metric, self.task)
        self._x_train, self._y_train = np.asarray(x_train), np.asarray(y_train)
        self._x_test, self._y_test = np.asarray(x_test), np.asarray(y_test)
        self.empty_value = empty_model_score(
            self._y_train, self._x_test, self._y_test, task=self.task, metric=self._metric
        )
        super().__init__(
            self._x_train.shape[1],
            normalize=normalize,
            normalization_value=self.empty_value,
            verbose=verbose,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the test metric of the model retrained on each coalition of features."""
        coalitions = as_bool_coalitions(coalitions)
        values = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            if not coalition.any():
                values[i] = self.empty_value
                continue
            values[i] = fit_and_score(
                self.model,
                self._x_train[:, coalition],
                self._y_train,
                self._x_test[:, coalition],
                self._y_test,
                task=self.task,
                metric=self._metric,
                random_state=self.random_state,
            )
        return values
