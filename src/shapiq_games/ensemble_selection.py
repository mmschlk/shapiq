"""Ensemble selection games: the test performance of a sub-ensemble of fitted models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self

import numpy as np
from scipy.stats import mode

from shapiq.game import Game
from shapiq_games._base import ConfigMixin, as_bool_coalitions
from shapiq_games._setup import configure
from shapiq_games._training import Metric, MetricName, resolve_metric
from shapiq_games.models import build_model

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["DEFAULT_MEMBER_POOL", "EnsembleSelection", "RandomForestEnsembleSelection"]

DEFAULT_MEMBER_POOL: tuple[str, ...] = (
    "linear",
    "decision_tree",
    "random_forest",
    "svm",
    "knn",
    "xgboost",
)
"""The model names ensemble members are drawn from by default."""


class EnsembleSelection(ConfigMixin, Game):
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
    """

    def __init__(
        self,
        members: Sequence[Any],
        x_test: np.ndarray,
        y_test: np.ndarray,
        *,
        task: str,
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
            task: ``"classification"`` or ``"regression"``.
            metric: ``"accuracy"``, ``"r2"``, ``"neg_mse"``, ``"neg_mae"``, a callable, or ``None``
                for accuracy (classification) or R² (regression).
            empty_value: The value of the empty coalition. Defaults to ``0``.
            member_names: A name per member, used as player names.
            normalize: Whether to center the game by ``empty_value``. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.
        """
        self.members = list(members)
        self.task = task
        self.member_names = (
            [str(name) for name in member_names]
            if member_names is not None
            else [f"{i}_{type(member).__name__}" for i, member in enumerate(self.members)]
        )
        self._metric = resolve_metric(metric, task)
        self._y_test = np.asarray(y_test)
        self.predictions = np.stack(
            [np.asarray(member.predict(x_test), dtype=float).reshape(-1) for member in self.members]
        )
        self.empty_value = float(empty_value)
        super().__init__(
            len(self.members),
            normalize=normalize,
            normalization_value=self.empty_value,
            verbose=verbose,
            player_names=self.member_names,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
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
                prediction = mode(self.predictions[coalition], axis=0, keepdims=False).mode
            values[i] = self._metric(self._y_test, prediction)
        return values

    @classmethod
    def from_config(
        cls,
        *,
        dataset: str,
        members: Sequence[str] | None = None,
        n_members: int = 10,
        metric: MetricName | None = None,
        empty_value: float = 0.0,
        random_state: int = 42,
        test_size: float = 0.2,
        dataset_params: dict[str, Any] | None = None,
        normalize: bool = True,
    ) -> Self:
        """Build the game by fitting ensemble members on a registered dataset.

        Args:
            dataset: The dataset name.
            members: The model name of every member. If ``None``, ``n_members`` names are drawn
                (seeded) from :data:`DEFAULT_MEMBER_POOL`.
            n_members: The number of members when ``members`` is ``None``. Defaults to ``10``.
            metric: The metric name, or ``None`` for the default of the task.
            empty_value: The value of the empty coalition. Defaults to ``0``.
            random_state: The seed of the split and the members (member ``i`` gets
                ``random_state + i``).
            test_size: The fraction of the data used as test set. Defaults to ``0.2``.
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
        if members is None:
            rng = np.random.default_rng(random_state)
            members = [str(name) for name in rng.choice(DEFAULT_MEMBER_POOL, size=n_members)]
        fitted = []
        for i, name in enumerate(members):
            params = {"n_neighbors": 3} if name == "knn" else {}
            model = build_model(name, split.task, random_state=random_state + i, **params)
            fitted.append(model.fit(split.x_train, split.y_train))
        game = cls(
            fitted,
            split.x_test,
            split.y_test,
            task=split.task,
            metric=metric,
            empty_value=empty_value,
            member_names=[f"{i}_{name}" for i, name in enumerate(members)],
            normalize=normalize,
        )
        return game._set_config(
            **setup.config,
            members=list(members),
            metric=metric,
            empty_value=empty_value,
            normalize=normalize,
        )


class RandomForestEnsembleSelection(EnsembleSelection):
    """Ensemble selection over the trees of a fitted random forest.

    The players are the trees of the forest. Because a random forest classifier averages class
    probabilities while this game takes majority votes, the full ensemble can differ slightly from
    the forest's own prediction for classification.
    """

    @classmethod
    def from_forest(
        cls,
        forest: Any,  # noqa: ANN401
        x_test: np.ndarray,
        y_test: np.ndarray,
        *,
        task: str,
        metric: MetricName | Metric | None = None,
        empty_value: float = 0.0,
        normalize: bool = True,
    ) -> Self:
        """Build the game from a fitted scikit-learn random forest.

        Args:
            forest: A fitted ``RandomForestClassifier`` or ``RandomForestRegressor``.
            x_test: The test features.
            y_test: The test labels, encoded as the forest's class indices for classification.
            task: ``"classification"`` or ``"regression"``.
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

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        *,
        dataset: str,
        n_members: int = 10,
        metric: MetricName | None = None,
        empty_value: float = 0.0,
        random_state: int = 42,
        test_size: float = 0.2,
        model_params: dict[str, Any] | None = None,
        dataset_params: dict[str, Any] | None = None,
        normalize: bool = True,
    ) -> Self:
        """Build the game from a random forest with ``n_members`` trees on a registered dataset.

        Args:
            dataset: The dataset name.
            n_members: The number of trees, i.e. players. Defaults to ``10``.
            metric: The metric name, or ``None`` for the default of the task.
            empty_value: The value of the empty coalition. Defaults to ``0``.
            random_state: The seed of the split and the forest.
            test_size: The fraction of the data used as test set. Defaults to ``0.2``.
            model_params: Further hyperparameters of the forest.
            dataset_params: Parameters of synthetic datasets.
            normalize: Whether to center the game.

        Returns:
            The configured game.
        """
        model_params = {**(model_params or {}), "n_estimators": n_members}
        setup = configure(
            dataset=dataset,
            model="random_forest",
            random_state=random_state,
            test_size=test_size,
            model_params=model_params,
            dataset_params=dataset_params,
        )
        game = cls.from_forest(
            setup.model,
            setup.split.x_test,
            setup.split.y_test,
            task=setup.split.task,
            metric=metric,
            empty_value=empty_value,
            normalize=normalize,
        )
        return game._set_config(  # noqa: SLF001 - the game was just built by this class
            **setup.config, metric=metric, empty_value=empty_value, normalize=normalize
        )
