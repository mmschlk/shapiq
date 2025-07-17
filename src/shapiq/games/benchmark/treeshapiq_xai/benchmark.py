"""This module contains the TreeSHAPIQ explanation benchmark games."""

from __future__ import annotations

from typing import TYPE_CHECKING

from shapiq.games.benchmark.setup import GameBenchmarkSetup, get_x_explain
from shapiq.games.benchmark.treeshapiq_xai.base import TreeSHAPIQXAI

if TYPE_CHECKING:
    import numpy as np


class AdultCensus(TreeSHAPIQXAI):
    """The Adult Census dataset as a TreeSHAP-IQ explanation game."""

    def __init__(
        self,
        *,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        class_label: int | None = None,
        normalize: bool = True,
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Adult Census TreeSHAP-IQ explanation game.

        Args:
            x: The feature vector to be explained. If ``None``, then the first data point is used.
                If an integer, then the data point at the given index is used. If a numpy array,
                then the data point is used as is. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            class_label: The class label to use for the model. If ``None``, then the default
                behaviour of :class:`~shapiq.explainer.tree.TreeExplainer` is used.

            normalize: A boolean flag to normalize/center the game values. The default value is
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        if model_name not in ["decision_tree", "random_forest"]:
            msg = "Model name must be either decision_tree' or 'random_forest'."
            raise ValueError(msg)

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
            class_label=class_label,
            verbose=verbose,
        )


class BikeSharing(TreeSHAPIQXAI):
    """The Bike Sharing dataset as a TreeSHAP-IQ explanation game."""

    def __init__(
        self,
        *,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        normalize: bool = True,
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Bike Sharing TreeSHAP-IQ explanation game.

        Args:
            x: The feature vector to be explained. If ``None``, then the first data point is used.
                If an integer, then the data point at the given index is used. If a numpy array,
                then the data point is used as is. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            normalize: A boolean flag to normalize/center the game values. The default value is
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        if model_name not in ["decision_tree", "random_forest"]:
            msg = "Model name must be either decision_tree' or 'random_forest'."
            raise ValueError(msg)

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
            verbose=verbose,
        )


class CaliforniaHousing(TreeSHAPIQXAI):
    """The California Housing dataset as a TreeSHAP-IQ explanation game."""

    def __init__(
        self,
        *,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        normalize: bool = True,
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the California Housing TreeSHAP-IQ explanation game.

        Args:
            x: The feature vector to be explained. If ``None``, then the first data point is used.
                If an integer, then the data point at the given index is used. If a numpy array,
                then the data point is used as is. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            normalize: A boolean flag to normalize/center the game values. The default value is
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        if model_name not in ["decision_tree", "random_forest"]:
            msg = "Model name must be either decision_tree' or 'random_forest'."
            raise ValueError(msg)

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
            verbose=verbose,
        )


class SynthData(TreeSHAPIQXAI):
    """A synthetic dataset as a TreeSHAP-IQ explanation game."""

    def __init__(
        self,
        *,
        x: int = 0,
        n_features: int = 30,
        classification: bool = True,
        model_name: str = "decision_tree",
        normalize: bool = True,
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the synthetic dataset TreeSHAP-IQ explanation game.

        Args:
            x: The index of the data point to be explained. Defaults to ``0``.

            n_features: The number of features in the synthetic dataset. Defaults to ``30``.

            classification: A flag to indicate if the dataset is a classification or regression
                task. Defaults to ``True`` (classification).

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            normalize: A boolean flag to normalize/center the game values. The default value is
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        from sklearn.datasets import make_classification, make_regression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        n_samples = 10_000
        n_informative = int(n_features * 2 / 3)
        n_redundant = 0
        if classification:
            x_data, y_data = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_clusters_per_class=3,
                random_state=random_state,
            )
            class_label = 0
        else:
            x_data, y_data = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_targets=1,
                random_state=random_state,
            )
            class_label = None

        if model_name == "decision_tree":
            if classification:
                model = DecisionTreeClassifier(random_state=random_state, max_depth=15)
            else:
                model = DecisionTreeRegressor(random_state=random_state, max_depth=15)
        elif classification:
            model = RandomForestClassifier(
                random_state=random_state,
                n_estimators=10,
                max_depth=15,
            )
        else:
            model = RandomForestRegressor(
                random_state=random_state,
                n_estimators=10,
                max_depth=15,
            )
        # fit the model
        model.fit(x_data, y_data)

        # get x_explain
        x = get_x_explain(x, x_data)

        # call the super constructor
        super().__init__(
            x=x,
            class_label=class_label,
            tree_model=model,
            normalize=normalize,
            verbose=verbose,
        )
