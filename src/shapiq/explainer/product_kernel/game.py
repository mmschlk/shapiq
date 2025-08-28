"""Product Kernel Game.

This module implements the product kernel game defined in https://arxiv.org/abs/2505.16516.
It is based on machine learning models using (product) kernels as decision functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

from shapiq.game import Game

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, GameValues

    from .base import ProductKernelModel


class ProductKernelGame(Game):
    r"""Implements the product kernel game.

    For models using the product kernel as the decision function the game can be formulated as
    ..math::
        v(S) = \alpha^T (K(X_S, x_S))

    where K(., .) is the product kernel function, X_S are the samples (support vectors) restricted to the features in S and x_S is the point to explain restricted to the features in S.

    See https://arxiv.org/abs/2505.16516 for more details.

    """

    def __init__(
        self,
        n_players: int,
        explain_point: np.ndarray,
        model: ProductKernelModel,
        *,
        normalize: bool = False,
    ) -> None:
        """Initializes the product kernel game.

        Args:
            n_players (int): The number of players in the game.
            explain_point (np.ndarray): The point to explain.
            model (ProductKernelModel): The product kernel model.
            normalize (bool): Whether to normalize the game values.

        """
        self.model = model
        self.explain_point = explain_point
        self.n, self.d = self.model.X_train.shape
        self._X_train = self.model.X_train
        # The empty value can generally be defined by: \sum_{i=1}^n \alpha_i K(x^i, x) - \beta, where x^i are training points / support vectors.
        normalization_value: float = float(self.model.alpha.sum()) + model.intercept

        super().__init__(n_players, normalization_value=normalization_value, normalize=normalize)

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """The product kernel game value function.

        Args:
            coalitions (CoalitionMatrix): The coalitions to evaluate.

        Raises:
            NotImplementedError: If the kernel type is not supported.

        Returns:
            GameValues: The values of the game for each coalition.
        """
        alpha = self.model.alpha
        n_coalitions, _ = coalitions.shape
        res = []
        if self.model.kernel_type == "rbf":
            for coalition in range(n_coalitions):
                current_coalition = coalitions[coalition, :]

                # The baseline value
                if current_coalition.sum() == 0:
                    res.append(float(self.model.alpha.sum()) + self.model.intercept)
                    continue
                # Extract X_S and x_S
                coalition_features = self.explain_point[current_coalition]
                X_train = self.model.X_train[:, current_coalition]

                # Reshape into twodimensional vectors
                if len(coalition_features.shape) == 1:
                    coalition_features = coalition_features.reshape(1, -1)
                if len(X_train.shape) == 1:
                    X_train = X_train.reshape(1, -1)

                # Compute the RBF kernel
                kernel_values = rbf_kernel(X=X_train, Y=coalition_features, gamma=self.model.gamma)

                # Compute the decision value
                res.append((alpha @ kernel_values + self.model.intercept).squeeze())
        else:
            msg = f"Kernel type '{self.model.kernel_type}' not supported"
            raise NotImplementedError(msg)
        return np.array(res)

    def compute_kernel_vectors(self, X: np.ndarray, x: np.ndarray) -> list[np.ndarray]:
        """Compute kernel vectors for a given dataset X and instance x.

        Args:
            X: The dataset (2D array) used to train the model.
            x: The instance (1D array) for which to compute Shapley values.

        Returns:
            List of kernel vectors corresponding to each feature. Length = number of features.
        """
        # Initialize the kernel matrix
        kernel_vectors = []

        # For each sample and each feature, compute k(x_i^j, x^j)
        for i in range(self.d):
            kernel_vec = rbf_kernel(
                X[:, i].reshape(-1, 1),
                x[..., np.newaxis][i].reshape(1, -1),
                gamma=self.model.gamma,
            )
            kernel_vectors.append(kernel_vec.squeeze())

        return kernel_vectors


class ProductKernelGameAuthor(ProductKernelGame):
    """Implement Kernel Game based on Author Code."""

    def value_function(self, coalitions: CoalitionMatrix) -> np.ndarray:
        """Compute v(S): the inner product of alpha with the elementwise product of kernel_vectors columns in S.

        Args:
            coalitions: CoalitionMatrix representing the coalitions to consider.

        Returns:
            Scalar value: alpha^T (elementwise product of columns in S).
        """
        # Compute the kernel vectors for the coalitions
        kernel_vectors = self.compute_kernel_vectors(self._X_train, self.explain_point)

        # Ensure kernel_vectors is (n, d)
        if isinstance(kernel_vectors, list):
            kernel_vectors = np.array(kernel_vectors).T  # shape (n, d)
        elif kernel_vectors.shape[0] != self.n:
            kernel_vectors = kernel_vectors.T  # shape (n, d)
        res = []
        for coalition in coalitions:
            if len(coalition) == 0:
                prod = np.ones(self.n)
            else:
                prod = np.prod(kernel_vectors[:, list(coalition)], axis=1)
            res.append(np.dot(self.model.alpha, prod))
        return np.array(res)
