"""This module contains all base game classes for the unserpervised benchmark games."""

from ..setup import BenchmarkSetup
from .base import UnsupervisedData


class AdultCensus(UnsupervisedData):
    """The Adult Census game as an unsupervised data game."""

    def __init__(self, normalize: bool = True) -> None:

        setup = BenchmarkSetup("adult_census")
        data = setup.x_data

        super().__init__(
            data=data,
            normalize=normalize,
        )


class BikeSharing(UnsupervisedData):
    """The Bike Sharing game as an unsupervised data game."""

    def __init__(self, normalize: bool = True) -> None:

        setup = BenchmarkSetup("bike_sharing")
        data = setup.x_data

        super().__init__(
            data=data,
            normalize=normalize,
        )


class CaliforniaHousing(UnsupervisedData):
    """The California Housing game as an unsupervised data game."""

    def __init__(self, normalize: bool = True) -> None:

        setup = BenchmarkSetup("california_housing")
        data = setup.x_data

        super().__init__(
            data=data,
            normalize=normalize,
        )
