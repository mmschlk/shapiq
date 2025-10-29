"""Implementation of ProductKernelSHAPIQ for computing Shapley values."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import numpy as np
from sklearn.metrics.pairwise import rbf_kernel

if TYPE_CHECKING:
    from shapiq.explainer.product_kernel.base import ProductKernelModel

ProductKernelSHAPIQIndices = Literal["SV"]


class ProductKernelComputer:
    """The Product Kernel Computer for product kernel-based models.

    This class computes the Shapley values for product kernel-based models.
    The functions are obtained from the PKeX-Shapley LocalExplainer. https://github.com/Majeed7/RKHS-ExactSHAP/blob/main/explainer/LocalExplainer.py#L3

    References:
        -- [pkex-shapley] Majid Mohammadi and Siu Lun Chau, Krikamol Muandet. (2025). Computing Exact Shapley Values in Polynomial Time for Product-Kernel Methods. https://arxiv.org/abs/2505.16516

    Attributes:
        model: The product kernel model to explain.
        kernel_type: The type of kernel to be used. Defaults to ``"rbf"``.
        max_order: The maximum interaction order to be computed. Defaults to ``1``.
        index: The type of interaction to be computed. Defaults to ``"SV"``.
        d: The number of features in the model.

    """

    def __init__(
        self,
        model: ProductKernelModel,
        *,
        max_order: int = 1,
        index: ProductKernelSHAPIQIndices = "SV",
    ) -> None:
        """Initializes the ProductKernelSHAPIQ explainer.

        Args:
            model: A product kernel-based model to explain.
            max_order: The maximum interaction order to be computed. Defaults to ``1``.
            index: The type of interaction to be computed. Defaults to ``"SV"``.

        Returns:
            None.
        """
        self.model = model
        self.kernel_type = self.model.kernel_type
        self.max_order = max_order
        self.index = index
        self.d = model.d

    def precompute_weights(self, num_features: int) -> np.ndarray:
        """Precompute model weights (mu coefficients) for computing Shapley values.

        Args:
            num_features: Number of features.

        Returns:
            List of precomputed mu coefficients.
        """
        unnormalized_factors = [
            (math.factorial(q) * math.factorial(num_features - q - 1)) for q in range(num_features)
        ]

        return np.array(unnormalized_factors) / math.factorial(num_features)

    def compute_elementary_symmetric_polynomials(self, kernel_vectors: list) -> list:
        """Compute elementary symmetric polynomials using a dynamic programming approach.

        Args:
            kernel_vectors: List of kernel vectors (1D arrays).

        Returns:
            List of elementary symmetric polynomials.
        """
        # Initialize with e_0 = 1
        max_abs_k = max(np.max(np.abs(k)) for k in kernel_vectors) or 1.0
        scaled_kernel = [k / max_abs_k for k in kernel_vectors]

        # Initialize polynomial coefficients: P_0(x) = 1
        e = [np.ones_like(scaled_kernel[0], dtype=np.float64)]

        for k in scaled_kernel:
            # Prepend and append zeros to handle polynomial multiplication (x - k)
            new_e = [np.zeros_like(e[0])] * (len(e) + 1)
            # new_e[0] corresponds to the constant term after multiplying by (x - k)
            new_e[0] = -k * e[0]
            # Compute the rest of the terms
            for i in range(1, len(e)):
                new_e[i] = e[i - 1] - k * e[i]
            # The highest degree term is x^{len(e)}, coefficient is e[-1] (which is 1 initially)
            new_e[len(e)] = e[-1].copy()
            e = new_e

        # Extract elementary symmetric polynomials from the coefficients
        n = len(scaled_kernel)
        elementary = [np.ones_like(e[0])]  # e_0 = 1
        for r in range(1, n + 1):
            sign = (-1) ** r
            elementary_r = sign * e[n - r] * (max_abs_k**r)
            elementary.append(elementary_r)

        return elementary

    def compute_shapley_value(self, kernel_vectors: list, feature_index: int) -> float:
        """Compute the Shapley value for a specific feature of an instance.

        Args:
            kernel_vectors: List of kernel vectors (1D arrays).
            feature_index: Index of the feature.

        Returns:
           Shapley value for the specified feature.
        """
        alpha = self.model.alpha
        cZ_minus_j = [kernel_vectors[i] for i in range(self.model.d) if i != feature_index]
        e_polynomials = self.compute_elementary_symmetric_polynomials(cZ_minus_j)
        mu_coefficients = self.precompute_weights(self.model.d)

        # Compute kernel vector for the chosen feature
        k_j = kernel_vectors[feature_index]
        onevec = np.ones_like(k_j)

        # Compute the Shapley value
        result = np.zeros_like(k_j)
        for q in range(self.model.d):
            if q < len(e_polynomials):
                result += mu_coefficients[q] * e_polynomials[q]

        shapley_value = alpha.dot((k_j - onevec) * result)

        return shapley_value.item()

    def compute_kernel_vectors(self, X: np.ndarray, x: np.ndarray) -> list:
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
