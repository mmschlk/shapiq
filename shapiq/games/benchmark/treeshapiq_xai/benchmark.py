"""This module contains the TreeSHAPIQ explanation benchmark games."""

from typing import Optional, Union

import numpy as np

from .._config import GameBenchmarkSetup, get_x_explain
from .base import TreeSHAPIQXAI


class AdultCensus(TreeSHAPIQXAI):
    """The Adult Census dataset as a TreeSHAP-IQ explanation game.

    Args:
        x: The feature vector to be explained.
        model_name: The model to explain as a string. Defaults to 'decision_tree'. Available models
            are 'decision_tree' and 'random_forest'.
        index: The type of interaction index to be computed. The default value is "k-SII".
        class_label: The class label to be explained. The default value is None.
        max_order: The maximum order of interactions to be computed. The default value is 2.
        min_order: The minimum order of interactions to be computed. The default value is 1.
        normalize: A boolean flag to normalize/center the game values. The default value is True.
        verbose: A flag to print the validation score of the model if trained. Defaults to `True`.
        random_state: The random state to use for the imputer. Defaults to 42.
    """

    def __init__(
        self,
        x: Optional[Union[np.ndarray, int]] = None,
        model_name: str = "decision_tree",
        index: str = "k-SII",
        class_label: Optional[int] = None,
        max_order: int = 2,
        min_order: int = 1,
        normalize: bool = True,
        verbose: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:

        # TODO: add xgb to TreeSHAQ-IQ, yet
        assert model_name in [
            "decision_tree",
            "random_forest",
        ], "Model name must be either decision_tree' or 'random_forest'."

        setup = GameBenchmarkSetup(
            dataset_name="adult_census",
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )

        # get x_explain
        x = get_x_explain(x, setup.x_test)

        # call the super constructor
        super().__init__(
            x=x,
            tree_model=setup.model,
            normalize=normalize,
            index=index,
            class_label=class_label,
            max_order=max_order,
            min_order=min_order,
            verbose=verbose,
        )


class BikeSharing(TreeSHAPIQXAI):
    """The Bike Sharing dataset as a TreeSHAP-IQ explanation game.

    Args:
        x: The feature vector to be explained.
        model_name: The model to explain as a string. Defaults to 'decision_tree'. Available models
            are 'decision_tree' and 'random_forest'.
        index: The type of interaction index to be computed. The default value is "k-SII".
        max_order: The maximum order of interactions to be computed. The default value is 2.
        min_order: The minimum order of interactions to be computed. The default value is 1.
        normalize: A boolean flag to normalize/center the game values. The default value is True.
        verbose: A flag to print the validation score of the model if trained. Defaults to `True`.
        random_state: The random state to use for the imputer. Defaults to 42.
    """

    def __init__(
        self,
        x: Optional[Union[np.ndarray, int]] = None,
        model_name: str = "decision_tree",
        index: str = "k-SII",
        max_order: int = 2,
        min_order: int = 1,
        normalize: bool = True,
        verbose: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:

        # TODO: add xgb to TreeSHAQ-IQ, yet
        assert model_name in [
            "decision_tree",
            "random_forest",
        ], "Model name must be either decision_tree' or 'random_forest'."

        setup = GameBenchmarkSetup(
            dataset_name="bike_sharing",
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )

        # get x_explain
        x = get_x_explain(x, setup.x_test)

        # call the super constructor
        super().__init__(
            x=x,
            tree_model=setup.model,
            normalize=normalize,
            index=index,
            max_order=max_order,
            min_order=min_order,
            verbose=verbose,
        )


class CaliforniaHousing(TreeSHAPIQXAI):
    """The California Housing dataset as a TreeSHAP-IQ explanation game.

    Args:
        x: The feature vector to be explained.
        model_name: The model to explain as a string. Defaults to 'decision_tree'. Available models
            are 'decision_tree' and 'random_forest'.
        index: The type of interaction index to be computed. The default value is "k-SII".
        max_order: The maximum order of interactions to be computed. The default value is 2.
        min_order: The minimum order of interactions to be computed. The default value is 1.
        normalize: A boolean flag to normalize/center the game values. The default value is True.
        verbose: A flag to print the validation score of the model if trained. Defaults to `True`.
        random_state: The random state to use for the imputer. Defaults to 42.
    """

    def __init__(
        self,
        x: Optional[Union[np.ndarray, int]] = None,
        model_name: str = "decision_tree",
        index: str = "k-SII",
        max_order: int = 2,
        min_order: int = 1,
        normalize: bool = True,
        verbose: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:

        # TODO: add xgb to TreeSHAQ-IQ, yet
        assert model_name in [
            "decision_tree",
            "random_forest",
        ], "Model name must be either decision_tree' or 'random_forest'."

        setup = GameBenchmarkSetup(
            dataset_name="california_housing",
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )

        # get x_explain
        x = get_x_explain(x, setup.x_test)

        # call the super constructor
        super().__init__(
            x=x,
            tree_model=setup.model,
            normalize=normalize,
            index=index,
            max_order=max_order,
            min_order=min_order,
            verbose=verbose,
        )


class SynthData(TreeSHAPIQXAI):

    def __init__(
        self,
        x: int = 0,
        n_features: int = 20,
        classification: bool = True,
        model_name: str = "decision_tree",
        index: str = "k-SII",
        max_order: int = 2,
        min_order: int = 1,
        normalize: bool = True,
        verbose: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        from sklearn.datasets import make_classification, make_regression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        n_samples = 10_000
        n_informative = int(n_features * 2 / 3)
        n_redundant = 0
        if classification:
            x_data, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_clusters_per_class=3,
                random_state=random_state,
            )
        else:
            x_data, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_targets=1,
                random_state=random_state,
            )

        if model_name == "decision_tree":
            if classification:
                model = DecisionTreeClassifier(random_state=random_state, max_depth=15)
            else:
                model = DecisionTreeRegressor(random_state=random_state, max_depth=15)
        else:
            if classification:
                model = RandomForestClassifier(
                    random_state=random_state, n_estimators=10, max_depth=15
                )
            else:
                model = RandomForestRegressor(
                    random_state=random_state, n_estimators=10, max_depth=15
                )

        # get x_explain
        x = get_x_explain(x, x_data)

        # call the super constructor
        super().__init__(
            x=x,
            tree_model=model,
            normalize=normalize,
            index=index,
            max_order=max_order,
            min_order=min_order,
            verbose=verbose,
        )
