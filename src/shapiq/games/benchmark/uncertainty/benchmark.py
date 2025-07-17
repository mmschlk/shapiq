"""This module contains tabular benchmark games for uncertainty explanation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapiq.games.benchmark.setup import GameBenchmarkSetup, get_x_explain
from shapiq.games.benchmark.uncertainty.base import UncertaintyExplanation

if TYPE_CHECKING:
    import numpy as np


class AdultCensus(UncertaintyExplanation):
    """The Adult Census dataset as an UncertaintyExplanation benchmark game."""

    def __init__(
        self,
        *,
        uncertainty_to_explain: str = "total",
        imputer: str = "marginal",
        x: np.ndarray | int | None = None,
        model_name: str = "random_forest",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Adult Census UncertaintyExplanation benchmark game.

        Args:
            model_name: The model to use for the game. Can only be ``'random_forest'``.
                Defaults to ``'random_forest'``.

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

        self.setup = GameBenchmarkSetup(
            dataset_name="adult_census",
            model_name=None,
            verbose=verbose,
            random_state=random_state,
        )

        # train a model with limited depth such that we get non-degenerate distributions
        if model_name == "random_forest":
            model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=random_state)
            model.fit(self.setup.x_train, self.setup.y_train)
        else:
            msg = f"Invalid model name provided. Should be 'random_forest' but got {model_name}."
            raise ValueError(msg)

        # get x_explain
        x = get_x_explain(x, self.setup.x_test)

        # call the super constructor
        super().__init__(
            x=x,
            data=self.setup.x_train,
            imputer=imputer,
            model=model,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
            uncertainty_to_explain=uncertainty_to_explain,
        )
