"""This benchmark module contains the benchmark games for ensemble selection."""

from __future__ import annotations

from shapiq.games.benchmark.ensemble_selection.base import EnsembleSelection
from shapiq.games.benchmark.setup import GameBenchmarkSetup


class AdultCensus(EnsembleSelection):
    """The AdultCensus dataset as an ensemble selection game."""

    def __init__(
        self,
        *,
        loss_function: str = "accuracy_score",
        ensemble_members: list[str] | None = None,
        n_members: int = 10,
        random_state: int | None = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initializes the AdultCensus EnsembleSelection game.

        Args:
            loss_function: The loss function to use for the ensemble selection game. Defaults to
                ``'accuracy_score'``. See :class:`~shapiq.games.benchmark.setup.GameBenchmarkSetup`
                for available loss functions.

            ensemble_members: A optional list of ensemble members to use. Defaults to ``None``. If
                ``None``, then the ensemble members are determined by the game. Available ensemble
                members are:
                - ``'regression'`` (will use linear regression for regression datasets and logistic
                    regression for classification datasets)
                - ``'decision_tree'``
                - ``'random_forest'``
                - ``'gradient_boosting'``
                - ``'knn'``
                - ``'svm'``

            n_members: The number of ensemble members to use. Defaults to ``10``. Ignored if
                ``ensemble_members`` is not ``None``.

            verbose: A flag to enable verbose output. Defaults to ``False``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            random_state: The random state to use for the ensemble selection game. Defaults to
                ``42``, which is the same random state used in the other benchmark games with this
                model type for this dataset.
        """
        setup = GameBenchmarkSetup(
            dataset_name="adult_census",
            loss_function=loss_function,
            model_name=None,
            verbose=verbose,
        )
        super().__init__(
            x_train=setup.x_train,
            y_train=setup.y_train,
            x_test=setup.x_test,
            y_test=setup.y_test,
            loss_function=setup.loss_function,
            dataset_type=setup.dataset_type,
            ensemble_members=ensemble_members,
            n_members=n_members,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )


class BikeSharing(EnsembleSelection):
    """The BikeSharing dataset as an ensemble selection game."""

    def __init__(
        self,
        *,
        loss_function: str = "r2_score",
        ensemble_members: list[str] | None = None,
        n_members: int = 10,
        random_state: int | None = 42,
        verbose: bool = False,
        normalize: bool = True,
    ) -> None:
        """Initializes the BikeSharing EnsembleSelection game.

        Args:
            loss_function: The loss function to use for the ensemble selection game. Defaults to
                ``'r2_score'``. See :class:`~shapiq.games.benchmark.setup.GameBenchmarkSetup` for
                available loss functions.

            ensemble_members: A optional list of ensemble members to use. Defaults to ``None``. If
                ``None``, then the ensemble members are determined by the game. Available ensemble
                members are:
                - ``'regression'`` (will use linear regression for regression datasets and logistic
                    regression for classification datasets)
                - ``'decision_tree'``
                - ``'random_forest'``
                - ``'gradient_boosting'``
                - ``'knn'``
                - ``'svm'``

            n_members: The number of ensemble members to use. Defaults to ``10``. Ignored if
                ``ensemble_members`` is not ``None``.

            verbose: A flag to enable verbose output. Defaults to ``False``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            random_state: The random state to use for the ensemble selection game. Defaults to
                ``42``, which is the same random state used in the other benchmark games with this
                model type for this dataset.
        """
        setup = GameBenchmarkSetup(
            dataset_name="bike_sharing",
            loss_function=loss_function,
            model_name=None,
            verbose=verbose,
        )
        super().__init__(
            x_train=setup.x_train,
            y_train=setup.y_train,
            x_test=setup.x_test,
            y_test=setup.y_test,
            loss_function=setup.loss_function,
            dataset_type=setup.dataset_type,
            ensemble_members=ensemble_members,
            n_members=n_members,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )


class CaliforniaHousing(EnsembleSelection):
    """The CaliforniaHousing dataset as an ensemble selection game."""

    def __init__(
        self,
        *,
        loss_function: str = "r2_score",
        ensemble_members: list[str] | None = None,
        n_members: int = 10,
        random_state: int | None = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initializes the BikeSharing EnsembleSelection game.

        Args:
            loss_function: The loss function to use for the ensemble selection game. Defaults to
                ``'r2_score'``. See :class:`~shapiq.games.benchmark.setup.GameBenchmarkSetup` for
                available loss functions.

            ensemble_members: A optional list of ensemble members to use. Defaults to ``None``. If
                ``None``, then the ensemble members are determined by the game. Available ensemble
                members are:
                - ``'regression'`` (will use linear regression for regression datasets and logistic
                    regression for classification datasets)
                - ``'decision_tree'``
                - ``'random_forest'``
                - ``'gradient_boosting'``
                - ``'knn'``
                - ``'svm'``

            n_members: The number of ensemble members to use. Defaults to ``10``. Ignored if
                ``ensemble_members`` is not ``None``.

            verbose: A flag to enable verbose output. Defaults to ``False``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            random_state: The random state to use for the ensemble selection game. Defaults to
                ``42``, which is the same random state used in the other benchmark games with this
                model type for this dataset.
        """
        setup = GameBenchmarkSetup(
            dataset_name="california_housing",
            loss_function=loss_function,
            model_name=None,
            verbose=verbose,
        )
        super().__init__(
            x_train=setup.x_train,
            y_train=setup.y_train,
            x_test=setup.x_test,
            y_test=setup.y_test,
            loss_function=setup.loss_function,
            dataset_type=setup.dataset_type,
            ensemble_members=ensemble_members,
            n_members=n_members,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )
