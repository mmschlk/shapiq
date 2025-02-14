"""This module contains the benchmark DatasetValuation games."""

from shapiq.games.benchmark.data_valuation.base import DataValuation
from shapiq.games.benchmark.setup import GameBenchmarkSetup


class CaliforniaHousing(DataValuation):
    """The California Housing dataset as a DataValuation game."""

    def __init__(
        self,
        *,
        n_data_points: int = 14,
        model_name: str = "decision_tree",
        random_state: int | None = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:

        setup = GameBenchmarkSetup(
            dataset_name="california_housing",
            model_name=model_name,
            random_state=random_state,
            verbose=verbose,
        )

        super().__init__(
            n_data_points=n_data_points,
            x_data=setup.x_data[:5000],
            y_data=setup.y_data[:5000],
            fit_function=setup.fit_function,
            predict_function=setup.predict_function,
            loss_function=setup.loss_function,
            random_state=random_state,
            empty_data_value=0.0,
            normalize=normalize,
            verbose=verbose,
        )


class BikeSharing(DataValuation):
    """The Bike Sharing dataset as a DataValuation game."""

    def __init__(
        self,
        *,
        n_data_points: int = 14,
        model_name: str = "decision_tree",
        random_state: int | None = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:

        setup = GameBenchmarkSetup(
            dataset_name="bike_sharing",
            model_name=model_name,
            random_state=random_state,
            verbose=verbose,
        )

        super().__init__(
            n_data_points=n_data_points,
            x_data=setup.x_data[:5000],
            y_data=setup.y_data[:5000],
            fit_function=setup.fit_function,
            predict_function=setup.predict_function,
            loss_function=setup.loss_function,
            random_state=random_state,
            empty_data_value=0.0,
            normalize=normalize,
            verbose=verbose,
        )


class AdultCensus(DataValuation):
    """The Adult Census dataset as a DataValuation game."""

    def __init__(
        self,
        *,
        n_data_points: int = 14,
        model_name: str = "decision_tree",
        random_state: int | None = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:

        setup = GameBenchmarkSetup(
            dataset_name="adult_census",
            model_name=model_name,
            random_state=random_state,
            verbose=verbose,
        )

        super().__init__(
            n_data_points=n_data_points,
            x_data=setup.x_data[:5000],
            y_data=setup.y_data[:5000],
            fit_function=setup.fit_function,
            predict_function=setup.predict_function,
            loss_function=setup.loss_function,
            random_state=random_state,
            empty_data_value=0.0,
            normalize=normalize,
            verbose=verbose,
        )
