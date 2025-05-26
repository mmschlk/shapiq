"""This module contains the base class for the uncertainty explanation game."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import entropy

from shapiq.games.base import Game
from shapiq.games.benchmark.setup import get_x_explain
from shapiq.games.imputer import ConditionalImputer, MarginalImputer

if TYPE_CHECKING:
    from shapiq.utils.custom_types import Model


class UncertaintyExplanation(Game):
    """The UncertaintyExplanation class for uncertainty explanation.

    The UncertaintyExplanation benchmark class is used to explain the uncertainty of a model's
    predictions with attribution methods. This was first proposed by Watson et al. (2023) [1]_.

    Attributes:
        empty_prediction_value: The model's prediction on an empty data point (all features missing).
        x: The explanation point to use the imputer to.

    References:
        .. [1] Watson, D., O'Hara, J., Tax, N., Mudd, R., & Guy, I. (2023). Explaining Predictive Uncertainty with Information Theoretic Shapley Values. In *Advances in Neural Information Processing Systems (NeurIPS 2023)*. https://proceedings.neurips.cc/paper_files/paper/2023/file/16e4be78e61a3897665fa01504e9f452-Paper-Conference.pdf

    """

    def __init__(
        self,
        *,
        data: np.ndarray,
        model: Model,
        x: np.ndarray | int = None,
        imputer: str = "marginal",
        normalize: bool = True,
        random_state: int | None = 42,
        verbose: bool = False,
        uncertainty_to_explain: str = "total",
    ) -> None:
        """Initialize the UncertaintyExplanation game.

        Args:
            data: The background data to use for the explainer as a two-dimensional array
                with shape ``(n_samples, n_features)``.

            model: The model to explain as a callable function expecting data points as input and
                returning the model's predictions.

            x: The explanation point to use the imputer to. If ``None``, then the first data point
                is used. If an integer, then the data point at the given index is used. If a numpy
                array, then the data point is used as is. Defaults to ``None``.

            uncertainty_to_explain: The type of uncertainty to explain. Can be either ``'total'``,
                ``'aleatoric'`` or ``'epistemic'``. Defaults to ``'total'``.

            imputer: The imputer to use for the game. Can be either ``'marginal'`` or
                ``'conditional'``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            random_state: The random state to use for sampling. Defaults to ``None``.

            verbose: A flag to enable verbose output. Defaults to ``False``.
        """
        from sklearn.ensemble import RandomForestClassifier

        # validate the inputs
        if uncertainty_to_explain not in ["total", "aleatoric", "epistemic"]:
            msg = (
                f"Invalid class label provided. Should be 'total', 'aleatoric' or 'epistemic' "
                f"but got {uncertainty_to_explain}."
            )
            raise ValueError(msg)

        # get x_explain
        self.x = get_x_explain(x, data)
        self._model = model
        self._uncertainty_to_explain = uncertainty_to_explain

        if isinstance(model, RandomForestClassifier):
            self._predict = self._predict_rf
        else:
            msg = f"Invalid model provided. Should be RandomForestClassifier but got {model}."
            raise TypeError(msg)

        if imputer == "marginal":
            self._imputer = MarginalImputer(
                model=self._predict,
                sample_size=20,
                data=data,
                x=self.x,
                random_state=random_state,
                normalize=False,
            )
        elif imputer == "conditional":
            self._imputer = ConditionalImputer(
                model=self._predict,
                data=data[:1000],
                x=self.x,
                random_state=random_state,
                normalize=False,
            )
        else:
            msg = (
                f"Invalid imputer provided. Should be 'marginal' or 'conditional' but got "
                f"{imputer}."
            )
            raise ValueError(msg)

        self.empty_prediction_value: float = self._imputer.empty_prediction
        # init the base game
        super().__init__(
            data.shape[1],
            normalize=normalize,
            normalization_value=self.empty_prediction_value,
            verbose=verbose,
        )

    def _uncertainty(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate the uncertainty of the predictions.

        Args:
            predictions: The predictions of the model.

        Returns:
            The uncertainty of the predictions.
        """
        predictions_mean = predictions.mean(axis=0)
        if self._uncertainty_to_explain == "total":
            uncertainty = entropy(predictions_mean, axis=1, base=2)
        elif self._uncertainty_to_explain == "aleatoric":
            uncertainty = entropy(predictions, axis=2, base=2).mean(axis=0)
        elif self._uncertainty_to_explain == "epistemic":
            uncertainty = entropy(predictions_mean, axis=1, base=2) - entropy(
                predictions,
                axis=2,
                base=2,
            ).mean(axis=0)
        else:
            msg = (
                f"Invalid class label provided. Should be 'total', 'aleatoric' or 'epistemic' "
                f"but got {self._uncertainty_to_explain}."
            )
            raise ValueError(msg)
        return uncertainty

    def _predict_rf(self, x: np.ndarray) -> np.ndarray:
        predictions = np.array(
            [estimator.predict_proba(x) for estimator in self._model.estimators_],
        )
        return self._uncertainty(predictions)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Calculate the value function of the game.

        Args:
            coalitions: The coalitions of the game.

        Returns:
            The value function of the game.
        """
        return self._imputer(coalitions)
