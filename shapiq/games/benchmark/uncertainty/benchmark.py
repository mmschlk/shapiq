"""This module contains tabular benchmark games for uncertainty explanation."""

from typing import Optional, Union

import numpy as np

from .._config import get_x_explain
from .._setup import GameBenchmarkSetup
from .base import UncertaintyExplanation


class AdultCensus(UncertaintyExplanation):

    def __init__(
        self,
        *,
        uncertainty_to_explain: str = "total",
        imputer: str = "marginal",
        x: Optional[Union[np.ndarray, int]] = None,
        model_name: str = "random_forest",
        normalize: bool = True,
        verbose: bool = False,
        random_state: Optional[int] = 42,
    ) -> None:
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
            raise ValueError(
                f"Invalid model name provided. Should be 'random_forest' but got {model_name}."
            )

        print(f"Trained model {model_name} for the adult_census dataset.")
        print(f"Score on training data: {model.score(self.setup.x_test, self.setup.y_test)}")

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
