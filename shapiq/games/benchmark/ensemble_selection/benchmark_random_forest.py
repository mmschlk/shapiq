"""This benchmark module contains the benchmark games for ensemble selection."""

from shapiq.games.benchmark.ensemble_selection.base import RandomForestEnsembleSelection
from shapiq.games.benchmark.setup import GameBenchmarkSetup


class AdultCensus(RandomForestEnsembleSelection):
    """The AdultCensus dataset as a random forest ensemble selection game.

    Args:
        loss_function: The loss function to use for the ensemble selection game. Defaults to
            'accuracy_score'. See `shapiq.games.benchmark.setup.BenchmarkSetup` for available
            loss functions.
        n_members: The number of ensemble members to use. Defaults to 10, which is the same random
            forest model used in the other benchmark games with this model type for this dataset.
        random_state: The random state to use for the ensemble selection game. Defaults to 42, which
            is the same random state used in the other benchmark games with this model type for this
            dataset.
    """

    def __init__(
        self,
        *,
        loss_function: str = "accuracy_score",
        n_members: int = 10,
        random_state: int | None = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        setup = GameBenchmarkSetup(
            dataset_name="adult_census",
            loss_function=loss_function,
            model_name="random_forest",
            verbose=verbose,
            random_forest_n_estimators=n_members,
            random_state=random_state,
        )
        super().__init__(
            random_forest=setup.model,
            x_train=setup.x_train,
            y_train=setup.y_train,
            x_test=setup.x_test,
            y_test=setup.y_test,
            loss_function=setup.loss_function,
            dataset_type=setup.dataset_type,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )


class BikeSharing(RandomForestEnsembleSelection):
    """The BikeSharing dataset as a random forest ensemble selection game.

    Args:
        loss_function: The loss function to use for the ensemble selection game. Defaults to
            'r2_score'. See `shapiq.games.benchmark.setup.BenchmarkSetup` for available
            loss functions.
        n_members: The number of ensemble members to use. Defaults to 10, which is the same random
            forest model used in the other benchmark games with this model type for this dataset.
        random_state: The random state to use for the ensemble selection game. Defaults to 42, which
            is the same random state used in the other benchmark games with this model type for this
            dataset.
    """

    def __init__(
        self,
        *,
        loss_function: str = "r2_score",
        n_members: int = 10,
        random_state: int | None = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        setup = GameBenchmarkSetup(
            dataset_name="bike_sharing",
            loss_function=loss_function,
            model_name="random_forest",
            verbose=verbose,
            random_forest_n_estimators=n_members,
            random_state=random_state,
        )
        super().__init__(
            random_forest=setup.model,
            x_train=setup.x_train,
            y_train=setup.y_train,
            x_test=setup.x_test,
            y_test=setup.y_test,
            loss_function=setup.loss_function,
            dataset_type=setup.dataset_type,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )


class CaliforniaHousing(RandomForestEnsembleSelection):
    """The CaliforniaHousing dataset as a random forest ensemble selection game.

    Args:
        loss_function: The loss function to use for the ensemble selection game. Defaults to
            'r2_score'. See `shapiq.games.benchmark.setup.BenchmarkSetup` for available
            loss functions.
        n_members: The number of ensemble members to use. Defaults to 10, which is the same random
            forest model used in the other benchmark games with this model type for this dataset.
        random_state: The random state to use for the ensemble selection game. Defaults to 42, which
            is the same random state used in the other benchmark games with this model type for this
            dataset.
    """

    def __init__(
        self,
        *,
        loss_function: str = "r2_score",
        n_members: int = 10,
        random_state: int | None = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        setup = GameBenchmarkSetup(
            dataset_name="california_housing",
            loss_function=loss_function,
            model_name="random_forest",
            verbose=verbose,
            random_forest_n_estimators=n_members,
            random_state=random_state,
        )
        super().__init__(
            random_forest=setup.model,
            x_train=setup.x_train,
            y_train=setup.y_train,
            x_test=setup.x_test,
            y_test=setup.y_test,
            loss_function=setup.loss_function,
            dataset_type=setup.dataset_type,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )
