"""This module contains tabular benchmark games for local explanation."""

from typing import Optional

from .._setup import GameBenchmarkSetup
from .base import GlobalExplanation


class AdultCensus(GlobalExplanation):
    """The AdultCensus dataset as a global explanation game.

    Args:
        model_name: The model to explain as a string. Defaults to 'decision_tree'. Available models
            are 'decision_tree', 'random_forest', and 'gradient_boosting'.
        loss_function: The loss function to use for the model. Defaults to 'accuracy_score'.
            Available loss functions are described in the `BenchmarkSetup` class.
        n_samples_eval: The number of samples to use for the evaluation of the value function.
            Defaults to 10.
        n_samples_empty: The number of samples to use for the empty subset estimation. Defaults to
            200.
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`.
        verbose: A flag to print the validation score of the model if trained. Defaults to `True`.
        random_state: The random state to use for the imputer. Defaults to 42.
    """

    def __init__(
        self,
        *,
        model_name: str = "decision_tree",
        loss_function: str = "accuracy_score",
        n_samples_eval: int = 10,
        n_samples_empty: int = 200,
        normalize: bool = True,
        verbose: bool = False,
        random_state: Optional[int] = 42,
    ) -> None:

        setup = GameBenchmarkSetup(
            dataset_name="adult_census",
            model_name=model_name,
            loss_function=loss_function,
            verbose=verbose,
            random_state=random_state,
        )

        # call the super constructor
        super().__init__(
            data=setup.x_train,
            model=setup.predict_function,
            loss_function=setup.loss_function,
            n_samples_eval=n_samples_eval,
            n_samples_empty=n_samples_empty,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )


class BikeSharing(GlobalExplanation):
    """The Bike Sharing regression dataset as a global explanation game.

    Args:
        model_name: The model to explain as a string. Defaults to 'decision_tree'. Available models
            are 'decision_tree', 'random_forest', and 'gradient_boosting'.
        loss_function: The loss function to use for the model. Defaults to 'mean_absolute_error'.
            Available loss functions are described in the `BenchmarkSetup` class.
        n_samples_eval: The number of samples to use for the evaluation of the value function.
            Defaults to 10.
        n_samples_empty: The number of samples to use for the empty subset estimation. Defaults to
            200.
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`.
        verbose: A flag to print the validation score of the model if trained. Defaults to `True`.
        random_state: The random state to use for the imputer. Defaults to 42.
    """

    def __init__(
        self,
        *,
        model_name: str = "decision_tree",
        loss_function: str = "mean_absolute_error",
        n_samples_eval: int = 10,
        n_samples_empty: int = 200,
        normalize: bool = True,
        verbose: bool = False,
        random_state: Optional[int] = 42,
    ) -> None:

        setup = GameBenchmarkSetup(
            dataset_name="bike_sharing",
            model_name=model_name,
            loss_function=loss_function,
            verbose=verbose,
            random_state=random_state,
        )

        # call the super constructor
        super().__init__(
            data=setup.x_train,
            model=setup.predict_function,
            loss_function=setup.loss_function,
            n_samples_eval=n_samples_eval,
            n_samples_empty=n_samples_empty,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )


class CaliforniaHousing(GlobalExplanation):
    """The California Housing regression dataset as a global explanation game.

    Args:
        model_name: The model to explain as a string. Defaults to 'decision_tree'. Available models
            are 'decision_tree', 'random_forest', and 'gradient_boosting'.
        loss_function: The loss function to use for the model. Defaults to 'mean_absolute_error'.
            Available loss functions are described in the `BenchmarkSetup` class.
        n_samples_eval: The number of samples to use for the evaluation of the value function.
            Defaults to 10.
        n_samples_empty: The number of samples to use for the empty subset estimation. Defaults to
            200.
        random_state: The random state to use for the imputer. Defaults to `None`.
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`.
        verbose: A flag to print the validation score of the model if trained. Defaults to `True`.
    """

    def __init__(
        self,
        *,
        model_name: str = "decision_tree",
        loss_function: str = "mean_absolute_error",
        n_samples_eval: int = 10,
        n_samples_empty: int = 200,
        normalize: bool = True,
        verbose: bool = False,
        random_state: Optional[int] = 42,
    ) -> None:

        setup = GameBenchmarkSetup(
            dataset_name="california_housing",
            model_name=model_name,
            loss_function=loss_function,
            verbose=verbose,
            random_state=random_state,
        )

        # call the super constructor
        super().__init__(
            data=setup.x_train,
            model=setup.predict_function,
            n_samples_eval=n_samples_eval,
            n_samples_empty=n_samples_empty,
            loss_function=setup.loss_function,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )
