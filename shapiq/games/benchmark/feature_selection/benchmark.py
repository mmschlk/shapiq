"""This module contains all benchmark games for the feature selection setting."""

from shapiq.games.benchmark._setup import GameBenchmarkSetup

from .base import FeatureSelection


class AdultCensus(FeatureSelection):
    """The Adult Census dataset as a Feature Selection benchmark game."""

    def __init__(self, model_name: str, normalize: bool = True, verbose: bool = True) -> None:

        setup = GameBenchmarkSetup(
            dataset_name="adult_census",
            model_name=model_name,
            verbose=verbose,
        )

        super().__init__(
            x_train=setup.x_train,
            x_test=setup.x_test,
            y_train=setup.y_train,
            y_test=setup.y_test,
            fit_function=setup.fit_function,
            score_function=setup.score_function,
            normalize=normalize,
        )


class BikeSharing(FeatureSelection):
    """The Bike Sharing dataset as a Feature Selection benchmark game."""

    def __init__(self, model_name: str, normalize: bool = True, verbose: bool = True) -> None:
        setup = GameBenchmarkSetup(
            dataset_name="bike_sharing",
            model_name=model_name,
            verbose=verbose,
        )

        super().__init__(
            x_train=setup.x_train,
            x_test=setup.x_test,
            y_train=setup.y_train,
            y_test=setup.y_test,
            fit_function=setup.fit_function,
            score_function=setup.score_function,
            normalize=normalize,
        )


class CaliforniaHousing(FeatureSelection):
    """The California Housing dataset as a Feature Selection benchmark game."""

    def __init__(self, model_name: str, normalize: bool = True, verbose: bool = True) -> None:
        setup = GameBenchmarkSetup(
            dataset_name="california_housing",
            model_name=model_name,
            verbose=verbose,
        )

        super().__init__(
            x_train=setup.x_train,
            x_test=setup.x_test,
            y_train=setup.y_train,
            y_test=setup.y_test,
            fit_function=setup.fit_function,
            score_function=setup.score_function,
            normalize=normalize,
        )


# Path: shapiq/games/benchmark/feature_selection/benchmark.py
