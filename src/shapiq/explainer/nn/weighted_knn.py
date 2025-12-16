"""Implementation of the explainer for weighted KNN models."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast, overload
from typing_extensions import override

from ._common_knn import _CommonKNNExplainer
from ._util import (
    interaction_values_from_array,
    warn_ignored_parameters,
)

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.neighbors import KNeighborsClassifier

    from shapiq.explainer.custom_types import ExplainerIndices
    from shapiq.interaction_values import InteractionValues

import numpy as np
from scipy.special import comb


class _WeightedKNNExplainerBase(_CommonKNNExplainer):
    """Base class for WKNN explainers that provides a utility function for calculating weights of training data points."""

    @override
    def __init__(
        self,
        model: KNeighborsClassifier,
        class_index: int | None = None,
    ) -> None:
        super().__init__(model, class_index=class_index)

        model_weights = self.knn_model.weights  # type: ignore[attr-defined]
        if model_weights != "distance":
            msg = f"KNeighborsClassifier must use weights='distance', but has weights='{model_weights}'"
            raise ValueError(msg)

    def _get_normalized_weights(
        self, x_val: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.floating]]:
        """Calculate normalized weights of training data points with respect to a validation data point.

        Args:
            x_val: The validation data point.

        Returns:
            A tuple ``(sortperm, weights)``, where both are of type ``numpy.ndarray`` with dimensions ``(n_training_samples,)`` and
                - ``sortperm`` is a permutation that sorts the training data points by decreasing weight
                - ``weights`` contains the weights for each training data point, normalized to the interval [0, 1]
        """
        distances, sortperm = self.knn_model.kneighbors(
            x_val.reshape(1, -1), n_neighbors=self.X_train.shape[0], return_distance=True
        )
        distances = distances[0]
        sortperm = sortperm[0]

        # Replicate sklearn behavior: if any training points are zero distance
        # from the test point, those points get weight 1 and all others 0
        zero_dist = np.isclose(distances, 0)
        if np.any(zero_dist):
            weights = np.zeros_like(distances)
            weights[zero_dist] = 1
        else:
            weights = distances[0] / distances

        return sortperm, weights


class WeightedKNNExplainer(_WeightedKNNExplainerBase):
    r"""Explainer for weighted KNN models.

    Implements the algorithm for efficiently computing exact Shapley values for weighted KNN models proposed by [Wng24]_.
    The algorithm achieves a runtime complexity of :math:`O\bigl(\frac{k^2 N^2 W}{C}\bigr)`, where

    * :math:`k` is the defining hyperparameter of the :math:`k`-nearest neighbors model,
    * :math:`N` is the size of the training dataset,
    * :math:`W = 2^b` (where :math:`b` is the number of discretization bits) is the size of the *discretized weights space*,
    * :math:`C` is the number of classes of the training dataset.

    Since the parameters :math:`k`, :math:`W` and :math:`C` can be considered constants for most purposes, the effective complexity is :math:`O(N^2)`.
    """

    @override
    def __init__(
        self,
        model: KNeighborsClassifier,
        class_index: int | None = None,
        n_bits: int = 3,
        data: np.ndarray | None = None,
        index: ExplainerIndices = "SV",
        max_order: int = 1,
    ) -> None:
        """Initializes the class.

        This method extracts the training data as well as the parameter :math:`k` from the provided KNN model and stores them as class members.

        Args:
            model: The KNN model to explain. Must be an instance of ``sklearn.neighbors.KNeighborsClassifier``.
                The model must not use multi-output classification, i.e. the ``y`` value provided to ``model.fit()`` must be a 1D vector.
            data: This parameter is currently ignored but may be used in future versions.
            class_index: The class index of the model to explain. Defaults to ``1``.
            n_bits: The number of bits to use for discretizing weights. Must be non-negative.
            index: The type of Shapley interaction index to use. Only ``"SV"`` is supported.
            max_order: The maximum interaction order to be computed. Only ``1`` is supported.

        Raises:
            sklearn.exceptions.NotFittedError: The constructor was called with a model that hasn't been fitted.
            shapiq_student.explainer.knn.exceptions.MultiOutputKNNError: The constructor was called with a model that uses multi-output classification.
        """
        # TODO(Zaphoood): Check that index and max_order are valid (only first-order etc.)  # noqa: TD003
        warn_ignored_parameters(locals(), ["data"], self.__class__.__name__)

        super().__init__(model, class_index)

        if self.k <= 1:
            msg = f"Only values of k > 1 are supported, but {self.k=}"
            raise ValueError(msg)

        model_weights = self.knn_model.weights  # type: ignore[attr-defined]
        if model_weights != "distance":
            msg = f"KNeighboursClassifier must use weights='distance', but has weights='{model_weights}'"
            raise ValueError(msg)

        if n_bits < 0:
            msg = f"Number of bits for discretization must be non-negative but was {n_bits}"
            raise ValueError(msg)
        self.n_bits = n_bits

        self.weights_space_size = 2 * self.k * 2**n_bits + 1
        self.weights_space = cast("npt.NDArray[np.integer]", np.arange(self.weights_space_size))
        # Index in the discrete weight space which weight zero is mapped to
        self.weights_space_zero = self.k * cast("int", 2**n_bits)

        self.n_train = self.X_train.shape[0]

    @override
    def explain_function(self, x: npt.NDArray[np.floating]) -> InteractionValues:
        n_players = self.X_train.shape[0]

        n_classes = len(self.y_train_classes)
        if n_classes == 1:
            return interaction_values_from_array(np.full(n_players, 1 / n_players))

        sortperm, weights = self._get_prepared_weights(x)

        sv = np.zeros((n_players,))
        for other_class_index in self._range_without_i(n_classes, self.class_index):
            sv += self._explain_binary(other_class_index, sortperm, weights)
        sv /= n_classes - 1

        return interaction_values_from_array(sv)

    def _explain_binary(
        self,
        y_other: int,
        sortperm: npt.NDArray[np.integer],
        weights: npt.NDArray[np.integer],
    ) -> npt.NDArray[np.floating]:
        y_val = self.class_index
        y_train_sorted = self.y_train_indices[sortperm]
        y_val_mask = y_train_sorted == y_val
        y_other_mask = y_train_sorted == y_other
        subgame_mask = y_val_mask | y_other_mask
        n_subgame = np.sum(subgame_mask)
        # Maps an index from the sorted subgame weights/labels to the sorted multi-class weights/labels
        subgame = np.arange(self.n_train)[subgame_mask]
        weights_subgame = weights[subgame]

        sv = np.zeros(self.n_train)
        for i in range(n_subgame):
            y_i = cast("int", self.y_train_indices[sortperm[subgame[i]]])
            f_i = self._compute_f_i(i, n_subgame, weights_subgame)
            r_i = self._compute_r_i(i, n_subgame, f_i, y_i, y_val, weights_subgame)
            g_i = self._compute_g_i(i, n_subgame, f_i, y_i, y_val, weights_subgame)

            sv[sortperm[subgame[i]]] = self._compute_single_shapley_value(
                i, n_subgame, r_i, g_i, weights_subgame
            )

        return sv

    def _get_prepared_weights(
        self, x_val: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.integer], npt.NDArray[np.integer]]:
        """Returns weights after normalization, discretization and sign-flipping."""
        sortperm, weights = self._get_normalized_weights(x_val)
        # Change sign of weights where class disagrees with class that is to be explained
        weights[(self.y_train_indices != self.class_index)[sortperm]] *= -1
        weights_discrete = self._discretize_weight(weights)

        return sortperm, weights_discrete

    def _range_without_i(self, n: int, i: int) -> npt.NDArray[np.integer]:
        """Returns the range [0, ..., i - 1, i + 1, ..., n - 1]."""
        return np.hstack([np.arange(i), np.arange(i + 1, n)])

    def _compute_f_i(
        self,
        i: int,
        n: int,
        weights: npt.NDArray[np.integer],
    ) -> npt.NDArray[np.floating]:
        f_i = np.zeros((n, self.k - 1, self.weights_space_size))

        indices_without_i = self._range_without_i(n, i)
        f_i[indices_without_i, 0, weights[indices_without_i]] = 1

        for l in range(1, self.k - 1):  # noqa: E741
            for m, weight_m in enumerate(weights[l:n], start=l):
                if m == i:
                    continue

                weight_diff = self._sub_weight(self.weights_space, weight_m)
                weight_diff_in_bounds = (weight_diff >= 0) & (weight_diff < self.weights_space_size)
                f_i[m, l, weight_diff_in_bounds] = np.sum(
                    f_i[:m, l - 1, weight_diff[weight_diff_in_bounds]], axis=0
                )

        return f_i

    def _compute_r_i(
        self,
        i: int,
        n: int,
        f_i: npt.NDArray[np.floating],
        y_i: int,
        y_val: int,
        weights: npt.NDArray[np.integer],
    ) -> npt.NDArray[np.floating]:
        r_i = np.zeros((n,))
        for m in range(max(i + 1, self.k), n):
            if y_i == y_val:
                weight_range_begin = self._flip_weight_sign(weights[i])
                weight_range_end = self._flip_weight_sign(weights[m])
            else:
                weight_range_begin = self._flip_weight_sign(weights[m])
                weight_range_end = self._flip_weight_sign(weights[i])

            if weight_range_begin < weight_range_end:
                r_i[m] = np.sum(f_i[:m, self.k - 2, weight_range_begin:weight_range_end])

        return r_i

    def _compute_g_i(
        self,
        i: int,
        n: int,
        f_i: npt.NDArray[np.floating],
        y_i: int,
        y_val: int,
        weights: npt.NDArray[np.integer],
    ) -> npt.NDArray[np.floating]:
        g_i = np.zeros((self.k,))
        g_i[0] = 0 if self._is_weight_negative(weights[i]) else 1
        for l in range(1, self.k):  # noqa: E741
            if y_i == y_val:
                weight_range_begin = self._flip_weight_sign(weights[i])
                weight_range_end = self.weights_space_zero
            else:
                weight_range_begin = self.weights_space_zero
                weight_range_end = self._flip_weight_sign(weights[i])

            if weight_range_begin < weight_range_end:
                indices_without_i = self._range_without_i(n, i)
                g_i[l] = np.sum(f_i[indices_without_i, l - 1, weight_range_begin:weight_range_end])

        return g_i

    def _compute_single_shapley_value(
        self,
        i: int,
        n: int,
        r_i: npt.NDArray[np.floating],
        g_i: npt.NDArray[np.floating],
        weights: npt.NDArray[np.integer],
    ) -> float:
        modified_weight_sign = self._modified_weight_sign(weights[i])
        first_summand = cast(
            "float",
            sum(g_i[l] / comb(n - 1, l) for l in range(min(self.k, n))) / n,  # noqa: E741
        )
        second_summand = cast(
            "float",
            sum(
                r_i[m - 1] / (m * comb(m - 1, self.k)) for m in range(max(i + 2, self.k + 1), n + 1)
            ),
        )

        return modified_weight_sign * (first_summand + second_summand)

    @overload
    def _discretize_weight(self, weight: float) -> int: ...

    @overload
    def _discretize_weight(self, weight: npt.NDArray[np.floating]) -> npt.NDArray[np.integer]: ...

    def _discretize_weight(
        self, weight: float | npt.NDArray[np.floating]
    ) -> int | npt.NDArray[np.integer]:
        """Turns floating-point weight into an integer index in the discretized weights space.

        Maps a given ``w`` to ``w^disc``, according to the following linear mapping:
        Weight ``-k`` will be mapped to index ``0`` and weight ``k`` to index ``2 * k * 2**n_bits``.
        """
        return self.weights_space_zero + np.round(weight * cast("int", 2**self.n_bits)).astype(int)

    @overload
    def _undiscretize_weight(self, weight_discrete: np.integer) -> float: ...

    @overload
    def _undiscretize_weight(
        self, weight_discrete: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.floating]: ...

    def _undiscretize_weight(
        self, weight_discrete: np.integer | npt.NDArray[np.integer]
    ) -> float | npt.NDArray[np.floating]:
        """Turns discrete weight index into the corresponding floating point weight.

        Returns ``w`` for some ``w^disc``.
        """
        return (weight_discrete - self.weights_space_zero) / cast("int", (2**self.n_bits))

    @overload
    def _sub_weight(
        self, weight_a_discrete: np.integer, weight_b_discrete: np.integer
    ) -> np.integer: ...

    @overload
    def _sub_weight(
        self, weight_a_discrete: npt.NDArray[np.integer], weight_b_discrete: np.integer
    ) -> npt.NDArray[np.integer]: ...

    def _sub_weight(
        self, weight_a_discrete: npt.NDArray[np.integer] | np.integer, weight_b_discrete: np.integer
    ) -> npt.NDArray[np.integer] | np.integer:
        """Computes ``(w_a - w_b)^disc`` for two discrete weight indices ``w_a^disc`` and ``w_b^disc``."""
        return self.weights_space_zero + weight_a_discrete - weight_b_discrete

    @overload
    def _flip_weight_sign(self, weight_discrete: np.integer) -> np.integer: ...

    @overload
    def _flip_weight_sign(
        self, weight_discrete: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.integer]: ...

    def _flip_weight_sign(
        self, weight_discrete: np.integer | npt.NDArray[np.integer]
    ) -> np.integer | npt.NDArray[np.integer]:
        """Given a discretized weight index, returns the discretized index of the corresponding weight with the sign flipped.

        In other words, it returns ``(-w)^disc`` for a given ``w^disc``.
        """
        return 2 * self.weights_space_zero - weight_discrete

    @overload
    def _is_weight_negative(self, weight_discrete: np.integer) -> np.bool: ...

    @overload
    def _is_weight_negative(
        self, weight_discrete: npt.NDArray[np.integer]
    ) -> npt.NDArray[np.bool]: ...

    def _is_weight_negative(
        self, weight_discrete: np.integer | npt.NDArray[np.integer]
    ) -> np.bool | npt.NDArray[np.bool]:
        """Checks whether the weight corresponding to a discretized weight index is negative.

        In other words, it returns ``w < 0`` for some ``w^disc``.
        """
        return weight_discrete < self.weights_space_zero

    def _modified_weight_sign(self, weight_discrete: np.integer) -> int:
        """Implements a modified sign function for discretized weights.

        It acts like a normal sign function but maps 0 to 1: Given some discretized weight index ``w^disc``, returns 1 if ``w >= 0`` and -1 if ``w < 0``.
        """
        if weight_discrete >= self.weights_space_zero:
            return 1

        return -1
