"""Implementation of the ProductKernelComputer for product kernel-based models."""

from __future__ import annotations

import math
import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np

from shapiq.game_theory.indices import get_computation_index

if TYPE_CHECKING:
    from shapiq.explainer.product_kernel.base import ProductKernelModel

ProductKernelSHAPIQIndices = Literal["SV", "BV", "SII", "k-SII", "BII", "Moebius"]

#: Element budget for one quadrature chunk (~32 MiB per float64 temporary).
_MAX_CHUNK_ELEMENTS = 1 << 22

#: Interaction count above which an explanation becomes slow enough to warn about.
_INTERACTION_WARNING_THRESHOLD = 10_000


class ProductKernelInteractionSizeWarning(UserWarning):
    """Warns that the requested interaction order enumerates very many interactions."""


def _gauss_legendre_unit(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on the unit interval ``[0, 1]``."""
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    return 0.5 * (nodes + 1.0), 0.5 * weights


class ProductKernelComputer:
    """The Product Kernel Computer for product kernel-based models.

    This class computes the Shapley and Banzhaf values for product kernel-based models.

    The Shapley values are computed by Gauss-Legendre quadrature of the product-game integral,
    which is exact with ``ceil(d / 2)`` nodes for ``d`` features. See [quadrashap]_ for details.
    The Banzhaf values are computed by a single ``t = 1/2`` evaluation of this product-game polynomial.

    References:
        -- [quadrashap] Majid Mohammadi, Grigory Reznikov, Pavel Sinitcyn, Krikamol Muandet and Siu Lun Chau. (2026). QuadraSHAP: Stable and Scalable Shapley Values for Product Games via Gauss-Legendre Quadrature. https://arxiv.org/abs/2605.05870

    Attributes:
        model: The product kernel model to explain.
        kernel_type: The type of kernel to be used. Defaults to ``"rbf"``.
        max_order: The maximum interaction order to be computed. Defaults to ``1``.
        index: The type of value to be computed, ``"SV"`` or ``"BV"``. Defaults to ``"SV"``.
        d: The number of features in the model.

    """

    def __init__(
        self,
        model: ProductKernelModel,
        *,
        max_order: int = 1,
        min_order: int = 1,
        index: ProductKernelSHAPIQIndices = "SV",
        n_quadrature_points: int | None = None,
    ) -> None:
        """Initializes the product kernel computer.

        Args:
            model: A product kernel-based model to explain.

            max_order: The maximum interaction order to be computed. Defaults to ``1``.

            min_order: The minimum interaction order to be computed. Order ``0`` is treated as
                ``1``, since the empty interaction carries the baseline value. Defaults to ``1``.

            index: The type of value to be computed. ``"SV"``, ``"SII"`` and ``"k-SII"`` are
                computed from the Shapley base, ``"BV"`` and ``"BII"`` from the Banzhaf base,
                and ``"Moebius"`` returns the Moebius coefficients. Defaults to ``"SV"``.

            n_quadrature_points: Number of Gauss-Legendre nodes. Defaults to ``None``, which
                uses the exact bound ``ceil(d / 2)`` for ``d`` features. Smaller values trade
                exactness for speed with a geometrically decaying error and are worthwhile
                only for very high-dimensional models, where a few hundred nodes already
                reach float64 precision. Ignored for ``"BV"``, which is a single evaluation
                point rather than a quadrature rule.

        Raises:
            ValueError: If ``n_quadrature_points`` is not positive.

        """
        if index not in get_args(ProductKernelSHAPIQIndices):
            msg = (
                f"Index '{index}' is not supported by ProductKernelComputer. Supported indices "
                f"are {get_args(ProductKernelSHAPIQIndices)}."
            )
            raise ValueError(msg)
        if n_quadrature_points is not None and n_quadrature_points < 1:
            msg = f"n_quadrature_points must be positive, got {n_quadrature_points}."
            raise ValueError(msg)

        self.model = model
        self.kernel_type = self.model.kernel_type
        self.max_order = max_order
        self.min_order = min_order
        self.index = index
        self.d = model.d

        self._orders = tuple(range(max(min_order, 1), max_order + 1))

        base_index = get_computation_index(index)
        if base_index == "Moebius":
            # at t = 0 every remaining factor collapses to 1, leaving prod_{j in T} (u_j - 1)
            self._nodes, self._weights = np.array([0.0]), np.array([1.0])
        elif base_index == "BII":
            self._nodes, self._weights = np.array([0.5]), np.array([1.0])
        else:
            n_points = n_quadrature_points or math.ceil(self.d / 2)
            self._nodes, self._weights = _gauss_legendre_unit(max(n_points, 1))

        n_interactions = sum(math.comb(self.d, order) for order in self._orders)
        if n_interactions > _INTERACTION_WARNING_THRESHOLD:
            msg = (
                f"Explaining {self.d} features up to order {max_order} enumerates "
                f"{n_interactions:,} interactions per instance. Every interaction of a product "
                f"kernel is generically non-zero, so this cannot be sparsified; consider a lower "
                f"max_order."
            )
            warnings.warn(msg, ProductKernelInteractionSizeWarning, stacklevel=3)

    def compute_kernel_matrix(self, x: np.ndarray) -> np.ndarray:
        """Compute the per-feature kernel factors of the explained instance.

        Args:
            x: The instance (1D array) for which to compute the values.

        Returns:
            The ``(n, d)`` matrix ``u`` with ``u[i, j] = k_j(x_j, X_train[i, j])``.

        Raises:
            NotImplementedError: If the kernel type is not supported.

        """
        if self.kernel_type != "rbf":
            msg = f"Kernel type '{self.kernel_type}' not supported."
            raise NotImplementedError(msg)
        # an unset gamma resolves to 1.0, matching what sklearn's rbf_kernel defaulted to when
        # the previous implementation called it once per single-column feature slice
        gamma = 1.0 if self.model.gamma is None else self.model.gamma
        return np.exp(-gamma * (self.model.X_train - np.asarray(x)[None, :]) ** 2)

    def compute_values(self, x: np.ndarray) -> np.ndarray:
        r"""Compute the first-order values of all features of an instance.

        Use :meth:`compute_interaction_values` for interactions of order two and above.

        Evaluates the quadrature form of the product-game Shapley value

        .. math::
            \phi_\ell = (u_\ell - 1) \sum_{q=1}^{m_q} \omega_q
            \prod_{j \neq \ell} \big((1 - \tau_q) + \tau_q u_j\big),

        where :math:`\{(\tau_q, \omega_q)\}_{q=1}^{m_q}` are the Gauss-Legendre nodes and
        weights on :math:`[0, 1]`, and :math:`u_j` are the per-feature kernel factors of one
        reference point. The values of the model are the :math:`\alpha`-weighted sum of these
        over all reference points.

        Taken literally the formula rebuilds a leave-one-out product for each of the :math:`d`
        features, costing :math:`O(n d^2 m_q)`. Instead the per-feature factors and their full
        product

        .. math::
            T_{q,j} = (1 - \tau_q) + \tau_q u_j, \qquad P_q = \prod_{j=1}^{d} T_{q,j},

        are formed once per node, so that every leave-one-out product is a single division

        .. math::
            \prod_{j \neq \ell} T_{q,j} = \frac{P_q}{T_{q,\ell}},

        which shares the work across features and brings the cost down to :math:`O(n d m_q)`.
        For better accuracy, the division is carried out in log-space,

        .. math::
            \phi_\ell = (u_\ell - 1) \sum_{q=1}^{m_q} \omega_q
            \exp\big(\log P_q - \log T_{q,\ell}\big),

        For ``index="BV"`` the very same expression runs with the single node
        :math:`\tau = 1/2` and weight :math:`\omega = 1`.

        Args:
            x: The instance (1D array) for which to compute the values.

        Returns:
            The Shapley (or Banzhaf) values as a 1D array of length ``d``.

        """
        # u: (ref_samples x features), u[i, j] = k_j(x_j, X_train[i, j])
        u = self.compute_kernel_matrix(x)
        n_samples, n_features = u.shape

        # leading: (ref_samples x features), include alpha_i * (u_{i,j} - 1) for each reference
        # row i and feature j.
        leading = self.model.alpha[:, None] * (u - 1.0)

        # chunk the nodes so the (nodes x ref_samples x features) temporaries stay in budget
        chunk = max(1, _MAX_CHUNK_ELEMENTS // max(n_samples * n_features, 1))
        values = np.zeros(n_features, dtype=np.float64)  # (features,)
        for start in range(0, len(self._nodes), chunk):
            # nodes: (nodes x 1 x 1), tau_q, - the gauss legendre node q, broadcast over
            # reference points and features
            nodes = self._nodes[start : start + chunk][:, None, None]
            # log_factors: (nodes x ref_samples x features)
            # log T_{q,j} for T_{q,j} = (1 - tau_q) + tau_q u_j; for every node q and feature j,
            # computed for every reference point.
            log_factors = np.log((1.0 - nodes) + nodes * u[None, :, :])
            # log_products: (nodes x ref_samples), log P_q, for every node q and reference point,
            # shared by all features.
            log_products = log_factors.sum(axis=-1)
            # leave_one_out: (nodes x ref_samples x features), the leave-one-out product
            # prod_{j != l} T_{q,j} = P_q / T_{q,l}. We broadcast log_products to
            # (nodes x ref_samples x features), subtract log_factors, then exponentiate.
            leave_one_out = np.exp(log_products[:, :, None] - log_factors)
            # weights: (nodes,), the gauss legendre weights
            weights = self._weights[start : start + chunk]
            # An inner product between weights (nodes,) and leave_one_out (nodes x ref_samples x features)
            # along the nodes axis, yielding (ref_samples x features).
            # integrated: (ref_samples x features) include the quadrature sum over the node axis, i.e.
            # the integral term of the Shapley formula for every (reference point, feature)
            integrated = np.tensordot(weights, leave_one_out, axes=(0, 0))
            # scale by the marginal factor and sum the reference points away: -> (features,)
            values += (integrated * leading).sum(axis=0)
        return values

    def compute_interaction_values(self, x: np.ndarray) -> dict[tuple[int, ...], float]:
        r"""Compute the interaction values of all feature subsets of an instance.

        Lifts :meth:`compute_values` from single features to every order in
        ``min_order..max_order``, using the same quadrature rule and the same shared
        log-space product :math:`P_q`; see that method for the polynomial itself.

        The discrete derivative of a product game with respect to a subset :math:`T` is
        :math:`\Delta_T v(S) = \prod_{j \in T}(u_j - 1) \prod_{j \in S} u_j`, so an order-k
        interaction is the first-order expression with two substitutions: the marginal factor
        :math:`(u_\ell - 1)` becomes :math:`\prod_{j \in T}(u_j - 1)`, and the leave-one-out
        product becomes the leave-:math:`T`-out product

        .. math::
            \prod_{j \notin T} T_{q,j} = \frac{P_q}{\prod_{j \in T} T_{q,j}}.

        :math:`P_q` is still formed once per node and shared by every subset, so the extra
        cost per interaction is one length-k sum in log-space.

        Args:
            x: The instance (1D array) for which to compute the values.

        Returns:
            A mapping from each feature subset to its interaction value.

        """
        # u: (ref_samples x features), u[i, j] = k_j(x_j, X_train[i, j])
        u = self.compute_kernel_matrix(x)
        n_samples, n_features = u.shape

        interactions: dict[tuple[int, ...], float] = {}
        for order in self._orders:
            # subsets: (interactions x order), every feature subset T of this order
            subsets = np.array(list(combinations(range(n_features), order)), dtype=np.intp).reshape(
                -1, order
            )
            # leading: (ref_samples x interactions), alpha_i * prod_{j in T} (u_{i,j} - 1)
            leading = self.model.alpha[:, None] * np.prod(u[:, subsets] - 1.0, axis=-1)
            values = np.zeros(len(subsets), dtype=np.float64)  # (interactions,)

            # chunk both axes so the (nodes x ref_samples x ...) temporaries stay in budget
            node_chunk = max(1, _MAX_CHUNK_ELEMENTS // max(n_samples * n_features, 1))
            subset_chunk = max(1, _MAX_CHUNK_ELEMENTS // max(n_samples * node_chunk, 1))
            for node_start in range(0, len(self._nodes), node_chunk):
                # nodes: (nodes x 1 x 1), tau_q, broadcast over reference points and features
                nodes = self._nodes[node_start : node_start + node_chunk][:, None, None]
                # log_factors: (nodes x ref_samples x features), log T_{q,j} for
                # T_{q,j} = (1 - tau_q) + tau_q u_j
                log_factors = np.log((1.0 - nodes) + nodes * u[None, :, :])
                # log_products: (nodes x ref_samples), log P_q, shared by all interactions
                log_products = log_factors.sum(axis=-1)
                weights = self._weights[node_start : node_start + node_chunk]  # (nodes,)

                for subset_start in range(0, len(subsets), subset_chunk):
                    chunk_subsets = subsets[subset_start : subset_start + subset_chunk]
                    # leave_t_out: (nodes x ref_samples x interactions), the leave-T-out product
                    # prod_{j not in T} T_{q,j} = P_q / prod_{j in T} T_{q,j}, in log-space.
                    # log_factors.sum(axis=-1) compute the ``prod_{j in T} T_{q,j}`` expression -
                    # as we are in log-space the prod is replaced by a sum.
                    leave_t_out = np.exp(
                        log_products[:, :, None] - log_factors[:, :, chunk_subsets].sum(axis=-1)
                    )
                    # integrated: (ref_samples x interactions), the quadrature sum over the nodes
                    integrated = np.tensordot(weights, leave_t_out, axes=(0, 0))
                    # scale by the marginal factor and sum the reference points away
                    stop = subset_start + len(chunk_subsets)
                    values[subset_start:stop] += (integrated * leading[:, subset_start:stop]).sum(
                        axis=0
                    )

            interactions.update(
                {
                    tuple(int(j) for j in subset): float(v)
                    for subset, v in zip(subsets, values, strict=False)
                }
            )
        return interactions
