"""This module contains tabular benchmark games for local explanation."""

from typing import Optional, Union

import numpy as np

from ..setup import BenchmarkSetup
from .base import LocalExplanation, _get_x_explain


class AdultCensus(LocalExplanation):
    """The AdultCensus dataset as a local explanation game.

    Args:
        class_to_explain: The class label to explain. Defaults to 1.
        x: The data point to explain. Can be an index of the background data or a 1d matrix
            of shape (n_features).
        model_name: The model to explain as a string. Defaults to 'decision_tree'. Available models
            are 'decision_tree', 'random_forest', and 'gradient_boosting'.
        random_state: The random state to use for the imputer. Defaults to `None`.
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`.
        verbose: A flag to print the validation score of the model if trained. Defaults to `True`.
    """

    def __init__(
        self,
        *,
        class_to_explain: int = 1,
        x: Optional[Union[np.ndarray, int]] = None,
        model_name: str = "decision_tree",
        random_state: Optional[int] = None,
        normalize: bool = True,
        verbose: bool = True,
    ) -> None:
        # validate the inputs
        if class_to_explain not in [0, 1]:
            raise ValueError(
                f"Invalid class label provided. Should be 0 or 1 but got {class_to_explain}."
            )

        setup = BenchmarkSetup(
            dataset_name="adult_census",
            model_name=model_name,
            verbose=verbose,
        )

        # get x_explain
        x = _get_x_explain(x, setup.x_test)

        def predict_function(x):
            return setup.predict_function(x)[:, class_to_explain]

        # call the super constructor
        super().__init__(
            x=x,
            data=setup.x_train,
            model=predict_function,
            random_state=random_state,
            normalize=normalize,
        )


class BikeSharing(LocalExplanation):
    """The BikeSharing dataset as a Local Explanation game.

    Args:
        x: The data point to explain. Can be an index of the background data or a 1d matrix
            of shape (n_features).
        model_name: The model to explain as a string. Defaults to 'decision_tree'. Available models
            are 'decision_tree', 'random_forest', and 'gradient_boosting'.
        random_state: The random state to use for the imputer. Defaults to `None`.
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`.
        verbose: A flag to print the validation score of the model if trained. Defaults to `True`.
    """

    def __init__(
        self,
        *,
        x: Optional[Union[np.ndarray, int]] = None,
        model_name: str = "decision_tree",
        random_state: Optional[int] = None,
        normalize: bool = True,
        verbose: bool = True,
    ) -> None:

        setup = BenchmarkSetup(
            dataset_name="bike_sharing",
            model_name=model_name,
            verbose=verbose,
        )

        # get x_explain
        x = _get_x_explain(x, setup.x_test)

        predict_function = setup.predict_function

        # call the super constructor
        super().__init__(
            x=x,
            data=setup.x_test,
            model=predict_function,
            random_state=random_state,
            normalize=normalize,
        )


class CaliforniaHousing(LocalExplanation):
    """The CaliforniaHousing dataset as a LocalExplanation game.

    Args:
        x: The data point to explain. Can be an index of the background data or a 1d matrix
            of shape (n_features).
        model_name: The model to explain as a string. Defaults to 'decision_tree'. Available models
            are 'decision_tree', 'random_forest', 'gradient_boosting', and 'neural_network'.
        random_state: The random state to use for the imputer. Defaults to `None`.
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`.
        verbose: A flag to print the validation score of the model if trained. Defaults to `True`.
    """

    def __init__(
        self,
        *,
        x: Optional[Union[np.ndarray, int]] = None,
        model_name: str = "decision_tree",
        random_state: Optional[int] = None,
        normalize: bool = True,
        verbose: bool = True,
    ) -> None:

        setup = BenchmarkSetup(
            dataset_name="california_housing",
            model_name=model_name,
            verbose=verbose,
        )

        # get x_explain
        x = _get_x_explain(x, setup.x_test)

        # call the super constructor
        super().__init__(
            x=x,
            data=setup.x_test,
            model=setup.predict_function,
            random_state=random_state,
            normalize=normalize,
        )
