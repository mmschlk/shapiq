"""This module contains the cluster explanation benchmark games the shapiq package."""

from __future__ import annotations

from shapiq.games.benchmark.setup import GameBenchmarkSetup
from shapiq.games.benchmark.unsupervised_cluster.base import ClusterExplanation


class AdultCensus(ClusterExplanation):
    """The Adult Census game as a clustering explanation game.

    See :class:`ClusterExplanation` for more information.
    """

    def __init__(
        self,
        *,
        cluster_method: str = "kmeans",
        score_method: str = "calinski_harabasz_score",
        normalize: bool = True,
        random_state: int | None = 42,
        verbose: bool = False,
    ) -> None:
        """Initializes the Adult Census ClusterExplanation game.

        Args:
            cluster_method: The clustering algorithm to use as a string. Defaults to ``'kmeans'``.
                Alternative available clustering algorithms are ``'dbscan'`` and ``'agglomerative'``
                with 3 clusters.

            score_method: The score method to use for the clustering algorithm. Available score
                methods are ``'calinski_harabasz_score'`` and ``'silhouette_score'``. Defaults to
                ``'calinski_harabasz_score'``.

            normalize: Whether to normalize the data before clustering. Defaults to ``True``.

            random_state: The random state to use for the clustering algorithm. Defaults to ``42``.

            verbose: Whether to print additional information. Defaults to ``False``.
        """
        setup = GameBenchmarkSetup("adult_census", verbose=False)
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
            normalize=normalize,
            random_state=random_state,
            verbose=verbose,
        )


class BikeSharing(ClusterExplanation):
    """The Bike Sharing game as a clustering explanation game.

    See :class:`ClusterExplanation` for more information.
    """

    def __init__(
        self,
        *,
        cluster_method: str = "kmeans",
        score_method: str = "calinski_harabasz_score",
        normalize: bool = True,
        random_state: int | None = 42,
        verbose: bool = False,
    ) -> None:
        """Initializes the Bike Sharing ClusterExplanation game.

        Args:
            cluster_method: The clustering algorithm to use as a string. Defaults to ``'kmeans'``.
                Alternative available clustering algorithms are ``'dbscan'`` and ``'agglomerative'``
                with 3 clusters.

            score_method: The score method to use for the clustering algorithm. Available score
                methods are ``'calinski_harabasz_score'`` and ``'silhouette_score'``. Defaults to
                ``'calinski_harabasz_score'``.

            normalize: Whether to normalize the data before clustering. Defaults to ``True``.

            random_state: The random state to use for the clustering algorithm. Defaults to ``42``.

            verbose: Whether to print additional information. Defaults to ``False``.
        """
        setup = GameBenchmarkSetup("bike_sharing", verbose=False)
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
            normalize=normalize,
            random_state=random_state,
            verbose=verbose,
        )


class CaliforniaHousing(ClusterExplanation):
    """The California Housing game as a clustering explanation game.

    See :class:`ClusterExplanation` for more information.
    """

    def __init__(
        self,
        *,
        cluster_method: str = "kmeans",
        score_method: str = "calinski_harabasz_score",
        normalize: bool = True,
        random_state: int | None = 42,
        verbose: bool = False,
    ) -> None:
        """Initializes the California Housing ClusterExplanation game.

        Args:
            cluster_method: The clustering algorithm to use as a string. Defaults to ``'kmeans'``.
                Alternative available clustering algorithms are ``'dbscan'`` and ``'agglomerative'``
                with 3 clusters.

            score_method: The score method to use for the clustering algorithm. Available score
                methods are ``'calinski_harabasz_score'`` and ``'silhouette_score'``. Defaults to
                ``'calinski_harabasz_score'``.

            normalize: Whether to normalize the data before clustering. Defaults to ``True``.

            random_state: The random state to use for the clustering algorithm. Defaults to ``42``.

            verbose: Whether to print additional information. Defaults to ``False``.
        """
        setup = GameBenchmarkSetup("california_housing", verbose=False)
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
            normalize=normalize,
            random_state=random_state,
            verbose=verbose,
        )
