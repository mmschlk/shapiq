"""The interventional game of a tree model (the game explained by interventional TreeSHAP)."""

from __future__ import annotations

from typing import Any, Self

import numpy as np

from shapiq.game import Game
from shapiq.utils.modules import safe_isinstance
from shapiq_games._base import ConfigMixin, as_bool_coalitions, resolve_class_index, resolve_x
from shapiq_games._setup import configure

__all__ = ["InterventionalTreeGame"]


class InterventionalTreeGame(ConfigMixin, Game):
    r"""The interventional (marginal) game of a model over a reference dataset.

    The value of a coalition :math:`S` is the mean model output over the reference rows
    :math:`z`, where the features in :math:`S` are replaced by the explained point :math:`x`:

    .. math::
        v(S) = \frac{1}{|Z|} \sum_{z \in Z} f(x_S, z_{\bar S})

    For tree models, this is the game that interventional TreeSHAP-IQ explains, so its exact values
    are available through :class:`~shapiq.tree.interventional.InterventionalTreeSHAPIQ`. The output
    space matches that explainer: margins (log-odds) for XGBoost and LightGBM classifiers, class
    probabilities for other classifiers, and predictions for regressors. The game is not normalized
    by default.

    Attributes:
        model: The model.
        reference_data: The reference (background) rows.
        x: The explained point.
        class_index: The explained class for classifiers, ``None`` for regressors.
    """

    def __init__(
        self,
        model: Any,  # noqa: ANN401
        reference_data: np.ndarray,
        x: np.ndarray,
        *,
        class_index: int | None = None,
        normalize: bool = False,
    ) -> None:
        """Initialize the interventional game.

        Args:
            model: The fitted model.
            reference_data: The reference rows of shape ``(n_reference, n_features)``.
            x: The explained point of shape ``(n_features,)``.
            class_index: The explained class for classifiers. Defaults to ``None``, which means
                class ``1`` for classifiers (the convention of the shapiq explainers).
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``False``.
        """
        self.model = model
        self.reference_data = np.asarray(reference_data)
        self.x = np.asarray(x).reshape(-1)
        self.class_index = resolve_class_index(model, class_index)
        n_players = self.x.shape[0]
        empty_value = float(self._evaluate(np.zeros((1, n_players), dtype=bool))[0])
        super().__init__(n_players, normalize=normalize, normalization_value=empty_value)

    def _model_output(self, data: np.ndarray) -> np.ndarray:
        """Return the model output for the explained class (or the regression output)."""
        if self.class_index is None:
            return np.asarray(self.model.predict(data), dtype=float)
        if safe_isinstance(self.model, "xgboost.sklearn.XGBClassifier"):
            import xgboost as xgb

            margins = self.model.get_booster().predict(xgb.DMatrix(data), output_margin=True)
        elif safe_isinstance(self.model, "lightgbm.LGBMClassifier"):
            margins = self.model.predict(data, raw_score=True)
        else:
            return np.asarray(self.model.predict_proba(data), dtype=float)[:, self.class_index]
        margins = np.asarray(margins, dtype=float)
        if margins.ndim == 1:  # binary classification: one margin for the positive class
            return margins if self.class_index == 1 else -margins
        return margins[:, self.class_index]

    def _evaluate(self, coalitions: np.ndarray) -> np.ndarray:
        values = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            data = np.where(coalition, self.x, self.reference_data)
            values[i] = float(np.mean(self._model_output(data)))
        return values

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the mean model output over the reference data for the coalitions."""
        return self._evaluate(as_bool_coalitions(coalitions))

    @classmethod
    def from_config(
        cls,
        *,
        dataset: str,
        model: str = "decision_tree",
        x: int = 0,
        n_reference: int = 100,
        class_index: int | None = None,
        random_state: int = 42,
        test_size: float = 0.2,
        preset: str | None = None,
        model_params: dict[str, Any] | None = None,
        normalize: bool = False,
    ) -> Self:
        """Build the game for a registered dataset and a model from the model registry.

        The reference data is a seeded random subset of the training split; the explained point
        is taken from the test split.

        Args:
            dataset: The dataset name.
            model: A tree model name: ``"decision_tree"``, ``"random_forest"``, ``"xgboost"``,
                ``"lightgbm"``, or ``"catboost"``. Defaults to ``"decision_tree"``.
            x: The index of the explained point in the test split. Defaults to ``0``.
            n_reference: The number of reference rows. Defaults to ``100``.
            class_index: The explained class for classifiers (``None`` means class ``1``).
            random_state: The seed of the split, the model, and the reference rows.
            test_size: The fraction of the data used as test set. Defaults to ``0.2``.
            preset: The hyperparameter preset of the model (``"tuned"`` or ``None``).
            model_params: Hyperparameters of the model.
            normalize: Whether to center the game.

        Returns:
            The configured game.
        """
        setup = configure(
            dataset=dataset,
            model=model,
            random_state=random_state,
            test_size=test_size,
            preset=preset,
            model_params=model_params,
        )
        x_train = setup.split.x_train
        rng = np.random.default_rng(random_state)
        rows = rng.choice(x_train.shape[0], size=min(n_reference, x_train.shape[0]), replace=False)
        game = cls(
            setup.model,
            x_train[np.sort(rows)],
            resolve_x(x, setup.split.x_test),
            class_index=class_index,
            normalize=normalize,
        )
        return game._set_config(
            **setup.config,
            x=x,
            n_reference=n_reference,
            class_index=class_index,
            normalize=normalize,
        )
