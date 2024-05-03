"""This module contains all base game classes for the unserpervised benchmark games."""

from .._config import GameBenchmarkSetup
from .base import UnsupervisedData


class AdultCensus(UnsupervisedData):
    """The Adult Census game as an unsupervised data game."""

    def __init__(self) -> None:

        setup = GameBenchmarkSetup("adult_census")
        data = setup.x_data

        super().__init__(data=data)


class BikeSharing(UnsupervisedData):
    """The Bike Sharing game as an unsupervised data game."""

    def __init__(self) -> None:

        setup = GameBenchmarkSetup("bike_sharing")
        data = setup.x_data

        super().__init__(data=data)


class CaliforniaHousing(UnsupervisedData):
    """The California Housing game as an unsupervised data game."""

    def __init__(self) -> None:

        setup = GameBenchmarkSetup("california_housing")
        data = setup.x_data

        super().__init__(data=data)
