"""This module contains all base game classes for the unserpervised benchmark games."""

from __future__ import annotations

from typing import Any

from shapiq.games.benchmark.setup import GameBenchmarkSetup

from .base import UnsupervisedData


class AdultCensus(UnsupervisedData):
    """The Adult Census dataset as a benchmark unsupervised data game."""

    def __init__(
        self,
        *,
        verbose: bool = False,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initializes the Adult Census UnsupervisedData game.

        Args:
            verbose: A flag to enable verbose imputation, which will print a progress bar for model
                predictions. Defaults to ``False``.
            *args: Additional positional arguments (not used).
            **kwargs: Additional keyword arguments (not used).
        """
        setup = GameBenchmarkSetup("adult_census", verbose=verbose)
        data = setup.x_data

        super().__init__(data=data, verbose=verbose)


class BikeSharing(UnsupervisedData):
    """The Bike Sharing dataset as a benchmark unsupervised data game."""

    def __init__(
        self,
        *,
        verbose: bool = False,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initializes the Bike Sharing UnsupervisedData game.

        Args:
            verbose: A flag to enable verbose imputation, which will print a progress bar for model
                predictions. Defaults to ``False``.
            **kwargs: Additional keyword arguments (not used).
        """
        setup = GameBenchmarkSetup("bike_sharing", verbose=verbose)
        data = setup.x_data

        super().__init__(data=data, verbose=verbose)


class CaliforniaHousing(UnsupervisedData):
    """The California Housing dataset as a benchmark unsupervised data game."""

    def __init__(
        self,
        *,
        verbose: bool = False,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initializes the California Housing UnsupervisedData game.

        Args:
            verbose: A flag to enable verbose imputation, which will print a progress bar for model
                predictions. Defaults to ``False``.
            **kwargs: Additional keyword arguments (not used).
        """
        setup = GameBenchmarkSetup("california_housing", verbose=verbose)
        data = setup.x_data

        super().__init__(data=data, verbose=verbose)
