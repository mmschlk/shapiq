"""This module contains the base game for the unsupervised data analysis setting."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats

from shapiq.games.base import Game


class UnsupervisedData(Game):
    """The Unsupervised Data game.

    The unsupervised data game models unsupervised data analysis problems as cooperative games. The
    players are features of the data. The value of a coalition is the total correlation explained by
    the features in the coalition. For more information, refer to the paper by Balestra et al.
    (2022) [1]_.

    References:
        .. [1] Balestra, C., Huber, F., Mayr, A., Müller, E. (2022). Unsupervised Features Ranking via Coalitional Game Theory for Categorical Data. Cooperative game. https://arxiv.org/abs/2205.09060
    """

    def __init__(
        self,
        *,
        data: np.ndarray,
        verbose: bool = False,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the Unsupervised Data game.

        Args:
            data: The data to analyze as a numpy array of shape ``(n_samples, n_features)``.
            verbose: Whether to print additional information. Defaults to ``False``.
            **kwargs: Additional keyword arguments (not used).
        """
        self.data = data
        self._n_features = data.shape[1]

        # discretize the data
        from sklearn.preprocessing import KBinsDiscretizer

        discretizer = KBinsDiscretizer(
            n_bins=20,
            encode="ordinal",
            strategy="uniform",
            subsample=200000,
        )
        self.data_discrete = np.zeros_like(data)
        for i in range(self._n_features):
            self.data_discrete[:, i] = discretizer.fit_transform(data[:, i].reshape(-1, 1)).ravel()
        self.data_discrete = self.data_discrete.astype(int).astype(str)

        super().__init__(
            n_players=self._n_features,
            normalize=True,
            normalization_value=0.0,
            verbose=verbose,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Calculate the value of the coalitions.

        Args:
            coalitions: The coalitions to calculate the value of as a numpy array of shape
                ``(n_coalitions, n_players)``.

        Returns:
            The value of the coalitions as a numpy array of shape (n_coalitions,).
        """
        values = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            if sum(coalition) == 0:
                values[i] = 0.0  # total correlation of the empty set is always 0 (not defined)
                continue
            data_subset = self.data_discrete[:, np.where(coalition)[0]]
            values[i] = total_correlation(data_subset)  # compute total correlation of the subset
        return values


def total_correlation(data: np.ndarray) -> float:
    """Compute the total correlation of a data subset.

    The total correlation is the sum of the entropies of the marginal distributions minus the joint
    entropy of the joint distribution.

    Args:
        data: The data subset as a numpy array of shape ``(n_samples, n_features)``.

    Returns:
        The total correlation of the data subset.
    """
    n_samples, n_features = data.shape

    # entropy of the marginal distributions
    entropy = np.zeros(n_features)
    for i in range(n_features):
        frequencies = np.unique(data[:, i], return_counts=True)[1]
        entropy[i] = stats.entropy(frequencies)

    # joint entropy of the joint distribution
    joint_entropy = entropy[0]
    if n_features > 1:
        joint_data = np.apply_along_axis(lambda x: " ".join(x), 1, data)
        joint_frequencies = np.unique(joint_data, return_counts=True)[1]
        joint_entropy = stats.entropy(joint_frequencies)

    return np.sum(entropy) - joint_entropy
