"""This module contains the cluster explanation benchmark games the shapiq package."""

from typing import Optional

from .._config import GameBenchmarkSetup
from .base import ClusterExplanation


class AdultCensus(ClusterExplanation):
    """The Adult Census game as a clustering explanation game."""

    def __init__(
        self,
        cluster_method: str = "dbscan",
        score_method: str = "calinski_harabasz_score",
        random_state: Optional[int] = None,
        normalize: bool = True,
    ) -> None:

        setup = GameBenchmarkSetup("adult_census")
        data = setup.x_data

        cluster_params = None
        if cluster_method == "kmeans":
            cluster_params = {"n_clusters": 3}
        if cluster_method == "agglomerative":
            cluster_params = {"n_clusters": 3}

        # standardize the data
        data = (data - data.mean()) / data.std()

        super().__init__(
            data=data,
            cluster_method=cluster_method,
            score_method=score_method,
            cluster_params=cluster_params,
            random_state=random_state,
            normalize=normalize,
        )


class BikeSharing(ClusterExplanation):
    """The Bike Sharing game as a clustering explanation game."""

    def __init__(
        self,
        cluster_method: str = "kmeans",
        score_method: str = "calinski_harabasz_score",
        random_state: Optional[int] = None,
        normalize: bool = True,
    ) -> None:

        setup = GameBenchmarkSetup("bike_sharing")
        data = setup.x_data

        cluster_params = None
        if cluster_method == "kmeans":
            cluster_params = {"n_clusters": 3}
        if cluster_method == "agglomerative":
            cluster_params = {"n_clusters": 3}

        # standardize the data
        data = (data - data.mean()) / data.std()

        super().__init__(
            data=data,
            cluster_method=cluster_method,
            score_method=score_method,
            cluster_params=cluster_params,
            random_state=random_state,
            normalize=normalize,
        )


class CaliforniaHousing(ClusterExplanation):
    """The California Housing game as a clustering explanation game."""

    def __init__(
        self,
        cluster_method: str = "kmeans",
        score_method: str = "calinski_harabasz_score",
        random_state: Optional[int] = None,
        normalize: bool = True,
    ) -> None:

        setup = GameBenchmarkSetup("california_housing")
        data = setup.x_data

        cluster_params = None
        if cluster_method == "kmeans":
            cluster_params = {"n_clusters": 3}
        if cluster_method == "agglomerative":
            cluster_params = {"n_clusters": 3}

        # standardize the data
        data = (data - data.mean()) / data.std()

        super().__init__(
            data=data,
            cluster_method=cluster_method,
            score_method=score_method,
            cluster_params=cluster_params,
            random_state=random_state,
            normalize=normalize,
        )
