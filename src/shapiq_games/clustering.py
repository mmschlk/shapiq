"""Cluster explanation games: how well a clustering on a coalition of features separates the data."""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions

__all__ = ["ClusterExplanation"]

_SCORES = {"calinski_harabasz": calinski_harabasz_score, "silhouette": silhouette_score}


class ClusterExplanation(Game):
    """The cluster explanation game: the quality of a clustering using only some features.

    The players are the features. The value of a coalition is the clustering score
    (Calinski-Harabasz or silhouette) of a fresh clustering (k-means or agglomerative) fitted on
    the features in the coalition. Clustering without features is undefined, so the empty
    coalition has the explicit value ``empty_value``. If a clustering finds a single cluster, its
    score is undefined and ``empty_value`` is used as well.

    Attributes:
        data: The clustered data.
        method: The clustering method.
        n_clusters: The number of clusters.
        score: The clustering score.

    Examples:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(n_samples=200, n_features=5, random_state=0)
        >>> game = ClusterExplanation(X, method="kmeans", n_clusters=3)
        >>> game.n_players
        5
    """

    def __init__(
        self,
        data: np.ndarray,
        *,
        method: Literal["kmeans", "agglomerative"] = "kmeans",
        n_clusters: int = 3,
        score: Literal["calinski_harabasz", "silhouette"] = "calinski_harabasz",
        empty_value: float = 0.0,
        random_state: int = 42,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the cluster explanation game.

        Args:
            data: The data to cluster, of shape ``(n_samples, n_features)``.
            method: ``"kmeans"`` or ``"agglomerative"``. Defaults to ``"kmeans"``.
            n_clusters: The number of clusters. Defaults to ``3``.
            score: ``"calinski_harabasz"`` or ``"silhouette"``. Defaults to
                ``"calinski_harabasz"``.
            empty_value: The value of the empty coalition. Defaults to ``0``.
            random_state: The seed of k-means. Defaults to ``42``.
            normalize: Whether to center the game by ``empty_value``. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.
        """
        if method not in ("kmeans", "agglomerative"):
            msg = f"method must be 'kmeans' or 'agglomerative', got {method!r}."
            raise ValueError(msg)
        if score not in _SCORES:
            msg = f"score must be 'calinski_harabasz' or 'silhouette', got {score!r}."
            raise ValueError(msg)
        self.data = np.asarray(data, dtype=float)
        self.method = method
        self.n_clusters = n_clusters
        self.score = score
        self.random_state = random_state
        self.empty_value = float(empty_value)
        super().__init__(
            self.data.shape[1],
            normalize=normalize,
            normalization_value=self.empty_value,
            verbose=verbose,
        )

    def _cluster_labels(self, data: np.ndarray) -> np.ndarray:
        if self.method == "kmeans":
            model = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=self.random_state)
        else:
            model = AgglomerativeClustering(n_clusters=self.n_clusters)
        return model.fit_predict(data)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the clustering score on the features of each coalition."""
        coalitions = as_bool_coalitions(coalitions)
        values = np.full(coalitions.shape[0], self.empty_value)
        for i, coalition in enumerate(coalitions):
            if not coalition.any():
                continue
            data = self.data[:, coalition]
            labels = self._cluster_labels(data)
            if np.unique(labels).shape[0] > 1:
                values[i] = float(_SCORES[self.score](data, labels))
        return values
