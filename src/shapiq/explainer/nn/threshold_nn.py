"""Implements the Explainer for threshold nearest neighbor models."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, cast
from typing_extensions import override

import numpy as np
from scipy.special import comb

from ._lookup_game import LookupGame

if TYPE_CHECKING:
    import numpy.typing as npt
    import sklearn.neighbors
    from sklearn.neighbors import RadiusNeighborsClassifier

    from shapiq import InteractionValues


from ._util import interaction_values_from_array
from .base import KNNExplainer

MODE_THRESHOLD = "threshold"


class _BruteForceTNNExplainer(KNNExplainer):
    """Brute force approach for explaining TNN Classifiers."""

    def __init__(self, model: RadiusNeighborsClassifier, class_index: int | None = None) -> None:
        super().__init__(model, class_index=class_index)
        # The type of the superclass's `model` attribute is too broad, since it also allows for other KNN explainers
        # To circumvent this, we store the model separately in an attribute with a narrower type
        self.tnn_model = model
        self.tau = cast("float", model.radius)  # type: ignore[attr-defined]

    @property
    @override
    def mode(self) -> str:
        return MODE_THRESHOLD

    @override
    def explain_function(self, x: npt.NDArray[np.floating]) -> InteractionValues:
        n_train = self.X_train.shape[0]
        n_classes = len(self.y_train_indices)

        neighbor_indices = self.tnn_model.radius_neighbors(x.reshape(1, -1), return_distance=False)
        neighbor_indices = neighbor_indices[0]
        in_neighborhood = np.zeros((n_train,), dtype=bool)
        in_neighborhood[neighbor_indices] = True

        y_train_is_class_index = self.y_train_indices == self.class_index

        utilities: dict[tuple[int, ...], float] = {}

        for coalition_generator in product([False, True], repeat=self.X_train.shape[0]):
            coalition = np.array(list(coalition_generator))

            coal_nhood = coalition & in_neighborhood
            coal_nhood_with_class_index = coal_nhood & y_train_is_class_index

            n_coal_nhood = np.sum(coal_nhood)

            # Utility function according to equation (3) in paper by Wang et al. (2023) DOI: 2308.15709v2
            if n_coal_nhood == 0:
                utility = 1 / n_classes
            else:
                utility = np.sum(coal_nhood_with_class_index) / n_coal_nhood

            coal_tuple = tuple(map(int, np.where(coalition)[0]))
            utilities[coal_tuple] = utility

        game = LookupGame(n_players=self.X_train.shape[0], utilities=utilities)
        return game.exact_values("SV", order=1)


class ThresholdNNExplainer(KNNExplainer):
    r"""Explainer for threshold nearest-neighbor models.

    Implements the algorithm for efficiently computing exact Shapley values for threshold nearest neighbor models proposed by Wang et al. (2023) [Wng23]_.
    The algorithm has a runtime complexity of :math:`O(N)` (when explaining a single data point), where :math:`N` is the number of training samples.
    """

    def __init__(
        self, model: sklearn.neighbors.RadiusNeighborsClassifier, class_index: int | None = None
    ) -> None:
        r"""Initializes the class.

        This method extracts the training data and the threshold :math:`\tau` from the provided model and stores it as class members.

        Args:
            model: The model to explain. The model must not use multi-output classification, i.e. the ``y`` value provided to ``model.fit(X, y)`` must be a 1D vector.
            data: This parameter is currently ignored but may be used in future versions.
            labels: This parameter is currently ignored but may be used in future versions.
            class_index: The class index of the model to explain. Defaults to ``1``.

        Raises:
            sklearn.exceptions.NotFittedError: The constructor was called with a model that hasn't been fitted.
        """
        super().__init__(model, class_index=class_index)
        self._model = model

        self.tau = cast("float", model.radius)  # type: ignore[attr-defined]

    @property
    @override
    def mode(self) -> str:
        """This explainer's mode, which is ``"threshold"``."""
        return MODE_THRESHOLD

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
