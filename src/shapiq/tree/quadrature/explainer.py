"""Quadrature-based path-dependent TreeSHAP explainer.

Implements the quadrature reformulation of path-dependent TreeSHAP for Shapley values and
any-order Shapley interactions. Shapley (interaction) values are expressed as integrals of
weighted Banzhaf interaction polynomials over the participation probability and evaluated
exactly with Gauss-Legendre quadrature; see `Wettenstein et al. (2026)
<https://arxiv.org/abs/2605.04497>`_ and `TreeGrad <https://arxiv.org/abs/2602.11623>`_ for the
first-order case. Unlike the interpolation-based :class:`~shapiq.tree.treeshapiq.TreeSHAPIQ`
and :class:`~shapiq.tree.linear.explainer.LinearTreeSHAP`, every maintained quantity is a
product of factors ``h*t + c*(1-t)`` in ``(0, 1]``, so the computation is numerically exact in
float64 at any tree depth (issue #545).

The papers (and the XGBoost implementation) parameterize edges by the marginal multiplier
``q = h/c`` and fold the cover products into the leaf's empty prediction; this module instead
keeps the hot indicator ``h`` and the accumulated cover ``c`` separate. The algebra is
identical, but folding the covers in edge by edge is what bounds every intermediate product by
one — the ``q`` form maintains ``prod(1 + (q-1)t)``, which can overflow on deep hot chains
with small covers.
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np

from shapiq.game_theory.indices import get_computation_index
from shapiq.interaction_values import InteractionValues
from shapiq.tree.conversion.edges import create_edge_tree
from shapiq.tree.validation import validate_tree_model
from shapiq.utils.sets import generate_interaction_lookup

if TYPE_CHECKING:
    from shapiq.tree.base import EdgeTree, TreeModel
    from shapiq.typing import Model

QuadratureTreeSHAPIndices = Literal["SV", "SII", "k-SII", "BV", "BII"]
QuadratureImplementation = Literal["auto", "numpy", "cpp"]


def _gauss_legendre_unit(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on the unit interval ``[0, 1]``."""
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    return 0.5 * (nodes + 1.0), 0.5 * weights


class QuadratureTreeSHAP:
    """Quadrature-based path-dependent TreeSHAP for values and any-order interactions.

    Computes the same path-dependent Shapley (interaction) values as
    :class:`~shapiq.tree.treeshapiq.TreeSHAPIQ` but through Gauss-Legendre quadrature of the
    weighted Banzhaf interaction polynomial, which is numerically exact in float64 for any
    tree depth and any number of distinct features per decision path. Banzhaf indices
    (``"BV"``/``"BII"``) are obtained from the same computation by evaluating the polynomial at
    participation probability ``1/2`` instead of integrating.

    Note:
        Like :class:`~shapiq.tree.treeshapiq.TreeSHAPIQ`, this class explains a single tree and
        is used internally by the :class:`~shapiq.tree.explainer.TreeExplainer`.
    """

    def __init__(
        self,
        model: dict | TreeModel | Model,
        *,
        max_order: int = 1,
        min_order: int = 1,
        index: QuadratureTreeSHAPIndices = "SV",
        n_quadrature_points: int | None = None,
        implementation: QuadratureImplementation = "auto",
    ) -> None:
        """Initializes the quadrature TreeSHAP explainer.

        Args:
            model: A single tree model to explain, in any representation accepted by
                :meth:`~shapiq.tree.validation.validate_tree_model`.

            max_order: The maximum interaction order to be computed. An interaction order of
                ``1`` corresponds to the Shapley (or Banzhaf) value. Defaults to ``1``.

            min_order: The minimum interaction order to be computed. Must be ``>= 1``. Defaults
                to ``1``.

            index: The interaction index to compute. ``"SV"``, ``"SII"``, and ``"k-SII"`` are
                computed from the SII base; ``"BV"`` and ``"BII"`` from the Banzhaf base.
                Defaults to ``"SV"``.

            n_quadrature_points: Number of Gauss-Legendre points. Defaults to ``None``, which
                uses the exact bound ``ceil(d / 2)`` where ``d`` is the maximum number of
                distinct features along a root-to-leaf path. A fixed ``8`` (the rule of
                Wettenstein et al.) is only useful on trees with more than ~16 features per
                path — below that the default already uses fewer points — where it measured
                13-35% faster with deviations from the exact result within ``2e-14`` up to
                ``d = 100``. Ignored for Banzhaf indices, which need a single evaluation point.

            implementation: ``"cpp"`` forces the C extension, ``"numpy"`` the pure-Python
                implementation, and ``"auto"`` uses the C extension when available. Defaults to
                ``"auto"``.

        Raises:
            ValueError: If the index or the interaction orders are invalid.

        """
        if index not in get_args(QuadratureTreeSHAPIndices):
            msg = (
                f"Index '{index}' is not supported by QuadratureTreeSHAP. Supported indices are "
                f"{get_args(QuadratureTreeSHAPIndices)}."
            )
            raise ValueError(msg)
        if max_order < min_order or max_order < 1 or min_order < 1:
            msg = (
                "The maximum order must be greater than the minimum order and both must be "
                "greater than 0."
            )
            raise ValueError(msg)
        if implementation not in get_args(QuadratureImplementation):
            msg = f"implementation='{implementation}' must be one of 'auto', 'numpy', or 'cpp'."
            raise ValueError(msg)
        self._max_order: int = max_order
        self._min_order: int = min_order
        self._index: str = index
        self._base_index: str = get_computation_index(self._index)

        # validate and parse the model into the reduced internal feature space; the validated
        # trees are owned by this explainer (validate_tree_model copies TreeModel inputs)
        validated_model = validate_tree_model(model)
        self._tree: TreeModel = validated_model[0]
        self._relevant_features: np.ndarray = np.array(sorted(self._tree.feature_ids), dtype=int)
        self._tree.reduce_feature_complexity()
        self._n_nodes: int = self._tree.n_nodes
        self._n_features_in_tree: int = self._tree.n_features_in_tree
        self._max_feature_id: int = self._tree.max_feature_id
        self._trivial_computation = self._n_features_in_tree <= 1

        # output lookup over the original feature ids, matching TreeSHAPIQ's packaging
        self._interactions_lookup_relevant: dict[tuple, int] = generate_interaction_lookup(
            self._relevant_features,
            self._min_order,
            self._max_order,
        )
        # per-order position lookup over the reduced feature ids, and the start offset of each
        # order's block in the concatenated output array
        self._order_lookups: dict[int, dict[tuple, int]] = {
            order: generate_interaction_lookup(self._n_features_in_tree, order, order)
            for order in range(self._min_order, self._max_order + 1)
        }
        self._output_offsets: dict[int, int] = {}
        offset = 0
        for order in range(self._min_order, self._max_order + 1):
            self._output_offsets[order] = offset
            offset += comb(self._n_features_in_tree, order)
        self._output_size: int = offset

        # the edge representation carries the accumulated per-chain multipliers and ancestors;
        # max_interaction=1 suffices since the quadrature algorithm needs no interaction heights
        self._edge_tree: EdgeTree = create_edge_tree(
            children_left=self._tree.children_left,
            children_right=self._tree.children_right,
            features=self._tree.features,
            node_sample_weight=self._tree.node_sample_weight,
            values=self._tree.values,
            max_interaction=1,
            n_features=self._max_feature_id + 1,
            n_nodes=self._n_nodes,
            subset_updates_pos_store={
                1: {f: np.array([f], dtype=np.int64) for f in range(self._max_feature_id + 1)}
            },
        )

        computed_empty_prediction = float(
            np.sum(self._edge_tree.empty_predictions[self._tree.leaf_mask]),
        )
        tree_empty_prediction = self._tree.empty_prediction
        if tree_empty_prediction is None:
            tree_empty_prediction = computed_empty_prediction
        self.empty_prediction: float = tree_empty_prediction

        # quadrature rule: exact for the degree-(d - order) integrands; Banzhaf indices are a
        # single evaluation of the weighted Banzhaf polynomial at p = 1/2
        self.max_features_per_path: int = int(self._edge_tree.edge_heights.max())
        if self._base_index == "BII":
            self._t = np.array([0.5])
            self._w = np.array([1.0])
        else:
            n_points = n_quadrature_points
            if n_points is None:
                n_points = max((self.max_features_per_path + 1) // 2, 1)
            elif n_points < 1:
                msg = f"n_quadrature_points={n_points} must be a positive integer."
                raise ValueError(msg)
            self._t, self._w = _gauss_legendre_unit(n_points)

        # per-node data in the (h, c) parameterization: c is the accumulated cold factor
        # (product of the cover ratios of the feature's edge chain), h the hot indicator
        with np.errstate(divide="ignore"):
            self._c_acc = np.where(
                self._edge_tree.p_e_values > 0, 1.0 / self._edge_tree.p_e_values, 1.0
            )
        self._edge_feature = np.full(self._n_nodes, -1, dtype=np.int32)
        non_root = self._edge_tree.parents >= 0
        self._edge_feature[non_root] = self._tree.features[self._edge_tree.parents[non_root]]

        self._cpp_available = False
        if implementation in ("auto", "cpp"):
            try:
                from .cext import (
                    quadrature_tree_shap,  # ty: ignore[unresolved-import]  # noqa: F401
                )

                self._cpp_available = True
            except ImportError:
                if implementation == "cpp":
                    msg = "The QuadratureTreeSHAP C extension is not available in this build."
                    raise ImportError(msg) from None
        self._implementation = implementation

        if self._cpp_available and not self._trivial_computation:
            # cache the kernel inputs once so a single explanation call pays no conversion cost
            tree = self._tree
            self._kernel_args = (
                np.ascontiguousarray(tree.thresholds, dtype=np.float64),
                np.ascontiguousarray(tree.features, dtype=np.int32),
                np.ascontiguousarray(tree.children_left, dtype=np.int32),
                np.ascontiguousarray(tree.children_right, dtype=np.int32),
                np.ascontiguousarray(self._edge_tree.parents, dtype=np.int32),
                np.ascontiguousarray(self._edge_tree.ancestors, dtype=np.int32),
                np.ascontiguousarray(self._c_acc, dtype=np.float64),
                np.ascontiguousarray(tree.values, dtype=np.float64),
                int(self._edge_tree.max_depth),
                int(self._n_nodes),
                np.ascontiguousarray(self._t, dtype=np.float64),
                np.ascontiguousarray(self._w, dtype=np.float64),
                int(self._n_features_in_tree),
                int(self._min_order),
                int(self._max_order),
            )
            self._kernel_args_tail = (
                tree.decision_type,
                np.ascontiguousarray(tree.cat_values, dtype=np.int64),
                np.ascontiguousarray(tree.cat_start, dtype=np.int64),
                np.ascontiguousarray(tree.cat_size, dtype=np.int64),
                np.ascontiguousarray(tree.children_left_default, dtype=bool),
            )

    def explain(self, x: np.ndarray) -> InteractionValues:
        """Computes the Shapley interaction values for a given instance ``x``.

        Args:
            x: Instance to be explained as a 1-dimensional array.

        Returns:
            The computed interaction values.

        """
        x_full = np.asarray(x, dtype=float)
        x_relevant = x_full[self._relevant_features]
        n_players = max(x_full.shape[0], self._n_features_in_tree)

        if self._n_features_in_tree == 0:
            interactions = np.zeros(0, dtype=float)
        elif self._trivial_computation:
            interactions = self._compute_trivial_interaction_values(x_full)
        elif self._cpp_available and self._implementation in ("auto", "cpp"):
            interactions = self._explain_cpp(x_relevant)
        else:
            interactions = self._explain_numpy(x_relevant)

        return InteractionValues(
            values=interactions,
            index=self._base_index,
            min_order=self._min_order,
            max_order=self._max_order,
            n_players=n_players,
            estimated=False,
            interaction_lookup=self._interactions_lookup_relevant,
            baseline_value=self.empty_prediction,
            target_index=self._index,
        )

    def _compute_trivial_interaction_values(self, x_full: np.ndarray) -> np.ndarray:
        """The single-feature case: the feature receives the full effect at order one.

        ``predict_one`` maps the reduced tree's internal feature ids back to the original
        feature space, so it routes the full instance, not the reduced one.
        """
        full_prediction = self._tree.predict_one(x_full)
        interactions = np.zeros(self._output_size, dtype=float)
        if self._min_order == 1 and interactions.shape[0] > 0:
            interactions[0] = full_prediction - self.empty_prediction
        return interactions

    def _explain_numpy(self, x_relevant: np.ndarray) -> np.ndarray:
        """Reference numpy implementation of the quadrature traversal for one instance."""
        t, w = self._t, self._w
        one_minus_t = 1.0 - t
        n_quad = t.shape[0]
        tree = self._tree
        children_left = tree.children_left
        children_right = tree.children_right
        ancestors = self._edge_tree.ancestors
        edge_feature = self._edge_feature
        c_acc = self._c_acc
        values = tree.values
        max_depth = int(self._edge_tree.max_depth)

        offsets = self._output_offsets
        out = np.zeros(self._output_size, dtype=float)

        act = np.zeros(self._n_nodes, dtype=bool)
        A = np.ones((max_depth + 2, n_quad))
        E = np.zeros((max_depth + 2, n_quad))
        live_g: dict[int, np.ndarray] = {}
        path_feats: list[int] = []

        def edge_factors(node: int) -> tuple[np.ndarray, np.ndarray]:
            """The homogenized factor ``u`` and the Banzhaf ratio ``g = (h - c)/u`` at a node."""
            h = 1.0 if act[node] else 0.0
            c = c_acc[node]
            u = h * t + c * one_minus_t
            return u, (h - c) / u

        def extract(node: int, depth: int) -> None:
            """Add the edge's contributions to all interactions it completes.

            An interaction receives updates at every edge of every member feature whose other
            members are already on the path; the ancestor term makes those updates telescope,
            so each leaf ends up counted exactly once with each member's deepest factor.
            """
            feature = int(edge_feature[node])
            ancestor = int(ancestors[node])
            if ancestor >= 0 and not act[ancestor]:
                return  # both telescoping terms cancel exactly once the hot chain is broken
            _, g_new = edge_factors(node)
            delta = g_new if ancestor < 0 else g_new - edge_factors(ancestor)[1]
            weighted = w * E[depth] * delta
            if self._min_order == 1:
                out[offsets[1] + self._order_lookups[1][(feature,)]] += weighted.sum()
            if self._max_order > 1:
                others = [j for j in path_feats if j != feature]
                for order in range(max(self._min_order, 2), self._max_order + 1):
                    lookup = self._order_lookups[order]
                    for subset in combinations(others, order - 1):
                        gamma = weighted
                        for j in subset:
                            gamma = gamma * live_g[j]
                        position = lookup[tuple(sorted((*subset, feature)))]
                        out[offsets[order] + position] += gamma.sum()

        def restore(node: int) -> None:
            """Undo this edge's live-state mutation when leaving its subtree."""
            feature = int(edge_feature[node])
            ancestor = int(ancestors[node])
            if ancestor >= 0:
                live_g[feature] = edge_factors(ancestor)[1]
            else:
                del live_g[feature]
                path_feats.remove(feature)

        # iterative depth-first traversal; stages: 0 enter, 1 after left, 2 after right, 3 leave
        stack: list[tuple[int, int, int]] = [(0, 0, 0)]
        while stack:
            node, depth, stage = stack.pop()
            if stage == 0:
                ancestor = int(ancestors[node])
                if node != 0:
                    if ancestor >= 0:
                        act[node] &= act[ancestor]
                    u_new, g_new = edge_factors(node)
                    if ancestor >= 0:
                        A[depth] = A[depth - 1] * (u_new / edge_factors(ancestor)[0])
                    else:
                        A[depth] = A[depth - 1] * u_new
                        path_feats.append(int(edge_feature[node]))
                    live_g[int(edge_feature[node])] = g_new
                left, right = int(children_left[node]), int(children_right[node])
                if left >= 0:
                    go_left = tree.goes_left(node, x_relevant[int(tree.features[node])])
                    act[left], act[right] = go_left, not go_left
                    stack.append((node, depth, 3))
                    stack.append((node, depth, 2))
                    stack.append((right, depth + 1, 0))
                    stack.append((node, depth, 1))
                    stack.append((left, depth + 1, 0))
                else:  # leaf
                    E[depth] = A[depth] * values[node]
                    if node != 0:
                        extract(node, depth)
                        restore(node)
            elif stage == 1:
                E[depth] = E[depth + 1]
            elif stage == 2:
                E[depth] += E[depth + 1]
            elif node != 0:  # stage 3 on a non-root internal node
                extract(node, depth)
                restore(node)
        return out

    def _explain_cpp(self, x_relevant: np.ndarray) -> np.ndarray:
        """C-extension implementation of the quadrature traversal for one instance."""
        from .cext import quadrature_tree_shap  # ty: ignore[unresolved-import]

        out = np.zeros((1, self._output_size), dtype=np.float64)
        quadrature_tree_shap(
            *self._kernel_args,
            np.ascontiguousarray(x_relevant.reshape(1, -1), dtype=np.float64),
            out,
            *self._kernel_args_tail,
        )
        return out[0]
