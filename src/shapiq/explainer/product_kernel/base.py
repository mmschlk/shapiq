"""The base class for product kernel model conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class ProductKernelModel:
    """A dataclass for storing the information of a product kernel model.

    Attributes:
         alpha: The alpha parameter of the product kernel model.
         X_train: The training data used to fit the product kernel model.
         n: The number of samples in the training data.
         d: The number of features in the training data.
         gamma: The gamma parameter of the product kernel model.

    """

    def __init__(
        self,
        X_train: np.ndarray,
        alpha: np.ndarray,
        n: int,
        d: int,
        gamma: float | None = None,
        kernel_type: str = "rbf",
    ) -> None:
        """Initializes the ProductKernelModel.

        Args:
            X_train: The training data used to fit the product kernel model.
            alpha: The alpha parameter of the product kernel model.
            n: The number of samples in the training data.
            d: The number of features in the training data.
            gamma: The gamma parameter of the product kernel model. Defaults to 'None'
            kernel_type: The type of the kernel model.
        """
        self.X_train = X_train
        self.alpha = alpha
        self.n = n
        self.d = d
        self.gamma = gamma
        self.kernel_type = kernel_type
