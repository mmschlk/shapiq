"""This module contains the local explanation game."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from shapiq.game import Game
from shapiq.imputer import (
    BaselineImputer,
    GenerativeConditionalImputer,
    MarginalImputer,
    TabPFNImputer,
)

from .utils import get_x_explain

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from shapiq.typing import CoalitionMatrix, GameValues, NumericArray


def get_imputer(
    imputer: Literal["baseline", "marginal", "conditional"],
    model: Callable[[NumericArray], NumericArray],
    data: NumericArray,
    x: NumericArray,
    *,
    random_state: int | None = 42,
) -> BaselineImputer | MarginalImputer | GenerativeConditionalImputer:
    """Get the appropriate imputer based on the input."""
    match imputer:
        case "baseline":
            return BaselineImputer(
                model=model,
                data=data,
                x=x,
                random_state=random_state,
                normalize=False,
            )
        case "marginal":
            return MarginalImputer(
                model=model,
                data=data,
                x=x,
                random_state=random_state,
                normalize=False,
            )
        case "conditional":
            # use only a random subset of the data for the GenerativeConditionalImputer
            return GenerativeConditionalImputer(
                model=model,
                # give only first 2_000 samples to the GenerativeConditionalImputer
                data=data,
                x=x,
                random_state=random_state,
                normalize=False,
            )
        case _:
            msg = (
                f"Imputer {imputer} not available. Choose from "
                f"{'baseline', 'marginal', 'conditional', 'tabpfn'}."
            )
            raise ValueError(msg)


class TabularLocalExplanation(Game):
    """The TabularLocalExplanation game class.

    The `TabularLocalExplanation` game is a game that performs local explanation of a model at a
    specific data point as a coalition game. The game evaluates the model's prediction on feature
    subsets around a specific data point.

    """

    imputer: BaselineImputer | MarginalImputer | GenerativeConditionalImputer | TabPFNImputer
    """The Tabular Imputer used to turn a machine learning model into a cooperative game."""

    def __init__(
        self,
        data: np.ndarray,
        model: Callable[[np.ndarray], np.ndarray],
        *,
        x: np.ndarray | int | None = None,
        imputer: BaselineImputer
        | MarginalImputer
        | GenerativeConditionalImputer
        | TabPFNImputer
        | Literal["baseline", "marginal", "conditional", "tabpfn"] = "marginal",
        normalize: bool = True,
        random_state: int | None = 42,
        verbose: bool = False,
    ) -> None:
        """Initialize the LocalExplanation game.

        Args:
            data: The background data used to fit the imputer. Should be a 2d matrix of shape
                ``(n_samples, n_features)``.

            model: The model to explain as a callable function expecting data points as input and
                returning the model's predictions. The input should be a 2d matrix of shape
                ``(n_samples, n_features)`` and the output a 1d matrix of shape ``(n_samples,)``.

            imputer: The imputer to use. Defaults to ``'marginal'``.

            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select a random data point
                from the background data.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        # get x_explain
        x = get_x_explain(x, data)

        # init the imputer which serves as the workhorse of this Game
        if isinstance(imputer, str):
            if imputer == "tabpfn":
                msg = "TabPFN imputer is not yet implemented in shapiq."
                raise NotImplementedError(msg)

            self.imputer = get_imputer(
                imputer,
                model=model,
                data=data,
                x=x,
                random_state=random_state,
            )
        else:
            self.imputer = imputer

        self._empty_prediction_value: float = self.imputer.empty_prediction

        # init the base game
        super().__init__(
            data.shape[1],
            normalize=normalize,
            normalization_value=self._empty_prediction_value,
            verbose=verbose,
        )

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Calls the model and returns the prediction.

        Args:
            coalitions: The coalitions as a one-hot matrix for which the game is to be evaluated.

        Returns:
            The output of the model on feature subsets.

        """
        return self.imputer(coalitions)

    @property
    def empty_prediction_value(self) -> float:
        """The model prediction of the empty set of features."""
        return self._empty_prediction_value
