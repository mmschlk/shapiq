"""This module contains all base game classes for the unserpervised benchmark games."""

from .._config import GameBenchmarkSetup
from .base import UnsupervisedData


class AdultCensus(UnsupervisedData):
    """The Adult Census game as an unsupervised data game."""

    def __init__(self, verbose: bool = False, *args, **kwargs) -> None:

        setup = GameBenchmarkSetup("adult_census", verbose=verbose)
        data = setup.x_data

        super().__init__(data=data, verbose=verbose)


class BikeSharing(UnsupervisedData):
    """The Bike Sharing game as an unsupervised data game."""

    def __init__(self, verbose: bool = False, *args, **kwargs) -> None:

        setup = GameBenchmarkSetup("bike_sharing", verbose=verbose)
        data = setup.x_data

        super().__init__(data=data, verbose=verbose)


class CaliforniaHousing(UnsupervisedData):
    """The California Housing game as an unsupervised data game."""

    def __init__(self, verbose: bool = False, *args, **kwargs) -> None:

        setup = GameBenchmarkSetup("california_housing", verbose=verbose)
        data = setup.x_data

        super().__init__(data=data, verbose=verbose)
