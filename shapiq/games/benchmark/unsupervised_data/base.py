"""This module contains the base game for the unsupervised data analysis setting."""

import numpy as np

from ...base import Game


class UnsupervisedData(Game):
    """The Unsupervised Data game.

    The unsupervised data game models unsupervised data analysis problems as cooperative games. The
    players are features of the data. The value of a coalition is the total correlation explained by
    the features in the coalition.

    For more information we refer to the following paper: https://arxiv.org/pdf/2205.09060.pdf

    Note:
        This game requires the pyitlib and sklearn package to be installed. You can install it via
        pip:
        ```
        pip install pyitlib scikit-learn
        ```

    Args:
        data: The data to analyze as a numpy array of shape (n_samples, n_features).
        normalize: Whether to normalize the data before analysis. Defaults to False.
        empty_coalition_value: The value of an empty coalition. Defaults to 0.0.
    """

    def __init__(
        self,
        data: np.ndarray,
        normalize: bool = False,
        empty_coalition_value: float = 0.0,
    ) -> None:
        self.data = data
        self._n_features = data.shape[1]
        self.empty_coalition_value = empty_coalition_value

        # discretize the data
        from sklearn.preprocessing import KBinsDiscretizer

        discretizer = KBinsDiscretizer(n_bins=20, encode="ordinal", strategy="uniform")
        self.data_discrete = np.zeros_like(data)
        for i in range(self._n_features):
            self.data_discrete[:, i] = discretizer.fit_transform(data[:, i].reshape(-1, 1)).ravel()
        self.data_discrete = self.data_discrete.astype(int)

        super().__init__(
            n_players=self._n_features,
            normalization_value=empty_coalition_value,
            normalize=normalize,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Calculate the value of the coalitions.

        Args:
            coalitions: The coalitions to calculate the value of as a numpy array of shape
                (n_coalitions, n_players).

        Returns:
            The value of the coalitions as a numpy array of shape (n_coalitions,).
        """
        values = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            if sum(coalition) == 0:
                values[i] = self.empty_coalition_value
                continue
            data_subset = self.data_discrete[:, np.where(coalition)[0]]
            values[i] = total_correlation(data_subset)  # compute total correlation of the subset
        return values


def total_correlation(data) -> float:
    """Compute the total correlation of a data subset.

    The computation computes the total correlation C of a set of random variables X_1,...,X_n such
    that C(X_1,...,X_n) = H(X_1) + ... + H(X_n) - H(X_1,...,X_n). For more information see:
    https://arxiv.org/pdf/2205.09060.pdf

    Args:
        data: The data subset as a numpy array of shape (n_samples, n_features).

    Returns:
        The total correlation of the data subset.

    Note:
        This function requires the pyitlib package to be installed. You can install it via pip:
        ```
        pip install pyitlib
        ```
    """
    from pyitlib import discrete_random_variable as drv

    return drv.information_multi(data)


def entropy(data):
    """Compute the Shannon entropy H of a set of random variables X_1,...,X_n.

    Args:
        data: The data subset as a numpy array of shape (n_samples, n_features).

    Returns:
        The Shannon entropy of the data subset.

    Note:
        This function requires the pyitlib package to be installed. You can install it via pip:
        ```
        pip install pyitlib
        ```
    """
    from pyitlib import discrete_random_variable as drv

    return drv.entropy_joint(data)
