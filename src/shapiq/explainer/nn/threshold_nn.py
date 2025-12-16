"""Implements the Explainer for threshold nearest-neighbor models."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast
from typing_extensions import override

import numpy as np
from scipy.special import comb

if TYPE_CHECKING:
    import numpy.typing as npt
    import sklearn.neighbors

    from shapiq import InteractionValues
    from shapiq.explainer.custom_types import ExplainerIndices


from ._util import (
    assert_valid_index_and_order,
    interaction_values_from_array,
    warn_ignored_parameters,
)
from .base import NNExplainerBase


class ThresholdNNExplainer(NNExplainerBase):
    r"""Explainer for threshold nearest-neighbor models.

    Implements the algorithm for efficiently computing exact Shapley values for threshold nearest neighbor models proposed by Wang et al. (2023) [Wng23]_.
    The algorithm has a runtime complexity of :math:`O(N)` (when explaining a single data point), where :math:`N` is the number of training samples.
    """

    def __init__(
        self,
        model: sklearn.neighbors.RadiusNeighborsClassifier,
        class_index: int | None = None,
        data: np.ndarray | None = None,
        index: ExplainerIndices = "SV",
        max_order: int = 1,
    ) -> None:
        r"""Initializes the class.

        This method extracts the training data and the threshold :math:`\tau` from the provided model and stores it as class members.

        Args:
            model: The model to explain. The model must not use multi-output classification, i.e. the ``y`` value provided to ``model.fit(X, y)`` must be a 1D vector.
            data: This parameter is currently ignored but may be used in future versions.
            class_index: The class index of the model to explain. Defaults to ``1``.
            index: The type of Shapley interaction index to use. Only ``"SV"`` is supported.
            max_order: The maximum interaction order to be computed. Only ``1`` is supported.

        Raises:
            sklearn.exceptions.NotFittedError: The constructor was called with a model that hasn't been fitted.
        """
        assert_valid_index_and_order(index, max_order)
        warn_ignored_parameters(locals(), ["data"], self.__class__.__name__)

        super().__init__(model, class_index=class_index)
        self._model = model
        self.tau = cast("float", model.radius)  # type: ignore[attr-defined]

    @override
    def explain_function(self, x: npt.NDArray[np.floating]) -> InteractionValues:
        # Following Theorem 13 and equation (7) in Wang et al. (2023) DOI: 2308.15709v2
        # Counting queries defined in C.2.2 ibid.
        n_train = self.X_train.shape[0]
        n_classes = self.y_train_indices.shape[0]

        neighbor_indices = self._model.radius_neighbors(x.reshape(1, -1), return_distance=False)
        neighbor_indices = neighbor_indices[0]

        in_neighborhood = np.zeros((n_train,), dtype=bool)
        in_neighborhood[neighbor_indices] = True

        y_train_is_class_index = self.y_train_indices == self.class_index

        # For entire dataset D
        c_D = n_train
        c_x_tau_D = 1 + len(neighbor_indices)
        c_plus_z_tau_D = cast("int", np.sum(in_neighborhood & y_train_is_class_index))

        # For each training point z_i
        c = c_D - 1
        c_x_tau = c_x_tau_D - in_neighborhood.astype(int)
        c_plus_z_tau = c_plus_z_tau_D - cast(
            "npt.NDArray[np.integer]", (in_neighborhood & y_train_is_class_index).astype(int)
        )

        a1 = np.zeros((n_train,), dtype=np.float64)
        mask = in_neighborhood & (c_x_tau >= 2)
        a1[mask] = y_train_is_class_index[mask] / c_x_tau[mask] - c_plus_z_tau[mask] / (
            c_x_tau[mask] * (c_x_tau[mask] - 1)
        )

        a2 = np.zeros((n_train,), dtype=np.float64)
        for i in range(n_train):
            if not in_neighborhood[i] or c_x_tau[i] < 2:
                continue
            for k in range(c + 1):
                binom_term = comb(c - k, c_x_tau[i]) / comb(c + 1, c_x_tau[i])
                a2[i] += 1 / (k + 1) * (1 - binom_term)
            a2[i] -= 1

        first_summand = a1 * a2
        second_summand = np.zeros((n_train,), dtype=np.float64)
        second_summand[in_neighborhood] = (
            y_train_is_class_index[in_neighborhood] - 1 / n_classes
        ) / c_x_tau[in_neighborhood]

        sv = first_summand + second_summand

        return interaction_values_from_array(sv)
