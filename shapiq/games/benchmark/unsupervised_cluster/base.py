"""This module contains all base game classes for the unserpervised benchmark games."""

from typing import Optional, Union

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from ...base import Game


class ClusterExplanation(Game):
    """The Cluster Explanation game.

    The cluster explanation game models clustering problems as cooperative games. The players are
    features of the data and the value of a coalition. Given a

    Args:
        data: The data to cluster as a numpy array of shape (n_samples, n_features).
        cluster_method: The clustering algorithm to use as a string. Available clustering algorithms
            are 'kmeans', or 'agglomerative'. Defaults to 'kmeans'.
        score_method: The score method to use for the clustering algorithm. Available score methods
            are 'calinski_harabasz_score' and 'silhouette_score'. Defaults to
            'calinski_harabasz_score'.
        random_state: The random state to use for the clustering algorithm. Defaults to 42.
        normalize: Whether to normalize the data before clustering. Defaults to True.
        empty_cluster_value: The worth of an empty cluster. Defaults to 0.0 (which automatically
             in a normalized game).
    """

    def __init__(
        self,
        data: np.ndarray,
        cluster_method: str = "kmeans",
        score_method: str = "calinski_harabasz_score",
        cluster_params: Optional[dict] = None,
        normalize: bool = True,
        empty_cluster_value: float = 0.0,
        random_state: Optional[int] = 42,
        verbose: bool = False,
    ) -> None:

        if cluster_params is None:
            cluster_params = {}

        # get a clustering algorithm
        self.cluster: Optional[Union[KMeans, AgglomerativeClustering]] = None
        if cluster_method == "kmeans":
            cluster_params["random_state"] = random_state
            self.cluster = KMeans(**cluster_params)
        elif cluster_method == "agglomerative":
            self.cluster = AgglomerativeClustering(**cluster_params)
        else:
            raise ValueError(
                f"Invalid clustering method provided. Got {cluster_method} but expected one of "
                f"['kmeans', 'agglomerative']."
            )

        # get a score function for the clustering
        self.score: Optional[Union[calinski_harabasz_score, silhouette_score]] = None
        if score_method == "calinski_harabasz_score":
            self.score = calinski_harabasz_score
        elif score_method == "silhouette_score":
            self.score = silhouette_score
        else:
            raise ValueError(
                f"Invalid score method provided. Got {score_method} but expected one of "
                f"['calinski_harabasz_score', 'silhouette_score']."
            )

        self.data = data

        self.random_state = random_state
        self.empty_cluster_value = empty_cluster_value

        super().__init__(
            data.shape[1],
            normalize=normalize,
            normalization_value=self.empty_cluster_value,
            verbose=verbose,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Fits the clustering algorithm and returns the score of the clustering.

        Args:
            coalitions: The coalitions as a one-hot matrix for which the game is to be evaluated.

        Returns:
            The score of the clustering algorithm for the given coalitions.
        """
        n_coalitions = coalitions.shape[0]
        worth = np.zeros(n_coalitions, dtype=float)
        for i, coalition in enumerate(coalitions):
            if sum(coalition) == 0:
                worth[i] = self.empty_cluster_value
                continue
            data_selection = self.data[:, coalition]
            self.cluster.fit(data_selection)
            labels = self.cluster.labels_
            worth[i] = self.score(data_selection, labels)
        return worth
