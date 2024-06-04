"""This module contains the base class for the uncertainty explanation game."""

from typing import Optional, Union

import numpy as np
from scipy.stats import entropy

from shapiq.games.base import Game
from shapiq.games.imputer import ConditionalImputer, MarginalImputer
from shapiq.utils.types import Model

from .._config import get_x_explain


class UncertaintyExplanation(Game):

    def __init__(
        self,
        *,
        data: np.ndarray,
        model: Model,
        x: Union[np.ndarray, int] = None,
        imputer: str = "marginal",
        normalize: bool = True,
        random_state: Optional[int] = 42,
        verbose: bool = False,
        uncertainty_to_explain: str = "total",
    ) -> None:
        from sklearn.ensemble import RandomForestClassifier

        # validate the inputs
        if uncertainty_to_explain not in ["total", "aleatoric", "epistemic"]:
            raise ValueError(
                f"Invalid class label provided. Should be 'total', 'aleatoric' or 'epistemic' "
                f"but got {uncertainty_to_explain}."
            )

        # get x_explain
        self.x = get_x_explain(x, data)
        self._model = model
        self._uncertainty_to_explain = uncertainty_to_explain

        if isinstance(model, RandomForestClassifier):
            self._predict = self._predict_rf
        else:
            raise ValueError(
                f"Invalid model provided. Should be RandomForestClassifier but got {model}."
            )

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
            raise ValueError(
                f"Invalid imputer provided. Should be 'marginal' or 'conditional' but got "
                f"{imputer}."
            )

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
                predictions, axis=2, base=2
            ).mean(axis=0)
        else:
            raise ValueError(
                f"Invalid class label provided. Should be 'total', 'aleatoric' or 'epistemic' "
                f"but got {self._uncertainty_to_explain}."
            )
        return uncertainty

    def _predict_rf(self, x: np.ndarray) -> np.ndarray:
        predictions = np.array(
            [estimator.predict_proba(x) for estimator in self._model.estimators_]
        )
        return self._uncertainty(predictions)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Calculate the value function of the game.

        Args:
            coalitions: The coalitions of the game.

        Returns:
            The value function of the game.
        """
        uncertainty = self._imputer(coalitions)
        return uncertainty
