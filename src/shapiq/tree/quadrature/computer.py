"""Quadrature-based path-dependent TreeSHAP explainer.

Implements the quadrature reformulation of path-dependent TreeSHAP for Shapley values and
any-order Shapley interactions. Shapley (interaction) values are expressed as integrals of
weighted Banzhaf interaction polynomials over the participation probability and evaluated
exactly with Gauss-Legendre quadrature; see Quadrature-TreeSHAP :cite:t:`Wettenstein.2026a`
and, for the first-order case, TreeGrad-Ranker :cite:t:`Li.2026`. Unlike the
interpolation-based :class:`~shapiq.tree.treeshapiq.TreeSHAPIQ`
and :class:`~shapiq.tree.linear.computer.LinearTreeSHAP`, every maintained quantity is a
product of factors ``h*t + c*(1-t)`` in ``(0, 1]``, so the computation is numerically exact in
float64 at any tree depth (issue #545).

The papers (and the XGBoost implementation) parameterize edges by the marginal multiplier
``q = h/c`` and fold the cover products into the leaf's empty prediction; this module instead
keeps the hot indicator ``h`` and the accumulated cover ``c`` separate. The algebra is
identical, but folding the covers in edge by edge is what bounds every intermediate product by
one — the ``q`` form maintains ``prod(1 + (q-1)t)``, which can overflow on deep hot chains
with small covers.

Ensembles are explained in a single C++ kernel invocation: all trees are mapped into one
shared (ensemble-reduced) feature space, their node arrays are concatenated, and the kernel
loops the trees and accumulates the contributions directly into the output buffer — no
per-tree Python objects are involved on the hot path.
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
    from shapiq.tree.base import TreeModel
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
    weighted Banzhaf interaction polynomial :cite:t:`Wettenstein.2026a` :cite:t:`Li.2026`,
    which is numerically exact in float64 for any tree depth and any number of distinct
    features per decision path. Banzhaf indices (``"BV"``/``"BII"``) are obtained from the
    same computation by evaluating the polynomial at participation probability ``1/2``
    instead of integrating.

    The class explains single trees and ensembles alike (ensembles run as one batched kernel
    call) and is the default path-dependent algorithm of the
    :class:`~shapiq.tree.explainer.TreeExplainer`.
    """

    def __init__(
        self,
        model: dict | TreeModel | Model,
        *,
        max_order: int = 1,
        min_order: int = 1,
        index: QuadratureTreeSHAPIndices = "SV",
        class_index: int | None = None,
        n_quadrature_points: int | None = None,
        implementation: QuadratureImplementation = "auto",
    ) -> None:
        """Initializes the quadrature TreeSHAP explainer.

        Args:
            model: A tree model or ensemble to explain, in any representation accepted by
                :meth:`~shapiq.tree.validation.validate_tree_model`.

            max_order: The maximum interaction order to be computed. An interaction order of
                ``1`` corresponds to the Shapley (or Banzhaf) value. Defaults to ``1``.

            min_order: The minimum interaction order to be computed. Must be ``>= 1``. Defaults
                to ``1``.

            index: The interaction index to compute. ``"SV"``, ``"SII"``, and ``"k-SII"`` are
                computed from the SII base; ``"BV"`` and ``"BII"`` from the Banzhaf base.
                Defaults to ``"SV"``.

            class_index: The class index for classification models. Defaults to ``None``.

            n_quadrature_points: Number of Gauss-Legendre points. Defaults to ``None``, which
                uses the exact bound ``ceil(d / 2)`` where ``d`` is the maximum number of
                distinct features along a root-to-leaf path (over all trees). A fixed ``8``
                (the rule of Wettenstein et al.) is only useful on trees with more than ~16
                features per path — below that the default already uses fewer points — where it
                measured 13-35% faster with deviations from the exact result within ``2e-14``
                up to ``d = 100``. Ignored for Banzhaf indices, which need a single evaluation
                point.

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

        # validate and parse the model; the validated trees are owned by this explainer
        # (validate_tree_model copies TreeModel inputs)
        self._trees: list[TreeModel] = validate_tree_model(model, class_label=class_index)
        if len({tree.decision_type for tree in self._trees}) > 1:
            msg = "All trees of an ensemble must share the same decision type."
            raise ValueError(msg)

        # ensemble-shared feature space: the sorted union of the trees' original feature ids
        self._relevant_features: np.ndarray = np.array(
            sorted(set().union(*(tree.feature_ids for tree in self._trees))), dtype=int
        )
        self._n_features_in_tree: int = len(self._relevant_features)
        ensemble_id = {orig: pos for pos, orig in enumerate(self._relevant_features)}
        self._trivial_computation = self._n_features_in_tree == 0

        # output lookup over the original feature ids, matching TreeSHAPIQ's packaging
        self._interactions_lookup_relevant: dict[tuple, int] = generate_interaction_lookup(
            self._relevant_features,
            self._min_order,
            self._max_order,
        )
        # per-order position lookup over the ensemble-space feature ids, and the start offset
        # of each order's block in the concatenated output array
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

        # concatenate all trees into one node-array block; child/parent/ancestor indices are
        # rebased to global positions so the kernels can traverse each tree from its root
        empty_prediction = 0.0
        max_features_per_path = 0
        max_depth = 0
        parts: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "children_left",
                "children_right",
                "parents",
                "ancestors",
                "features",
                "thresholds",
                "values",
                "c_acc",
                "cat_values",
                "cat_start",
                "cat_size",
                "children_left_default",
            )
        }
        roots: list[int] = []
        node_offset = 0
        cat_offset = 0
        for tree in self._trees:
            features_ens = np.array(
                [ensemble_id[f] if f >= 0 else -2 for f in tree.features], dtype=np.int64
            )
            subset_positions = {
                1: {
                    f: np.array([f], dtype=np.int64)
                    for f in range(max(self._n_features_in_tree, 1))
                }
            }
            edge_tree = create_edge_tree(
                children_left=tree.children_left,
                children_right=tree.children_right,
                features=features_ens,
                node_sample_weight=tree.node_sample_weight,
                values=tree.values,
                max_interaction=1,
                n_features=max(self._n_features_in_tree, 1),
                n_nodes=tree.n_nodes,
                subset_updates_pos_store=subset_positions,
            )
            tree_empty = tree.empty_prediction
            if tree_empty is None:
                tree_empty = float(np.sum(edge_tree.empty_predictions[tree.leaf_mask]))
            empty_prediction += float(tree_empty)
            max_features_per_path = max(max_features_per_path, int(edge_tree.edge_heights.max()))
            max_depth = max(max_depth, int(edge_tree.max_depth))
            # zero-cover edges (subtrees no training sample reaches, e.g. in CatBoost's
            # oblivious trees) contribute exactly zero in the limit c -> 0; a tiny positive
            # cover keeps every factor u = h*t + c*(1-t) positive instead of producing 0/0.
            # p_e is inf on the entry edge of a dead subtree and NaN strictly inside it
            # (0/0 cover ratios), so the clamp must test p_e itself before inverting.
            with np.errstate(divide="ignore"):
                c_acc = np.where(
                    np.isfinite(edge_tree.p_e_values) & (edge_tree.p_e_values > 0),
                    1.0 / edge_tree.p_e_values,
                    1e-300,
                )

            def rebase(indices: np.ndarray, offset: int = node_offset) -> np.ndarray:
                rebased = np.asarray(indices, dtype=np.int64).copy()
                rebased[rebased >= 0] += offset
                return rebased

            roots.append(node_offset)
            parts["children_left"].append(rebase(tree.children_left))
            parts["children_right"].append(rebase(tree.children_right))
            parts["parents"].append(rebase(edge_tree.parents))
            parts["ancestors"].append(rebase(edge_tree.ancestors))
            parts["features"].append(features_ens)
            parts["thresholds"].append(np.asarray(tree.thresholds, dtype=np.float64))
            parts["values"].append(np.asarray(tree.values, dtype=np.float64))
            parts["c_acc"].append(c_acc)
            parts["cat_values"].append(np.asarray(tree.cat_values, dtype=np.int64))
            parts["cat_start"].append(np.asarray(tree.cat_start, dtype=np.int64) + cat_offset)
            parts["cat_size"].append(np.asarray(tree.cat_size, dtype=np.int64))
            parts["children_left_default"].append(
                np.asarray(tree.children_left_default, dtype=bool)
            )
            node_offset += tree.n_nodes
            cat_offset += len(tree.cat_values)

        self.empty_prediction: float = empty_prediction
        self.max_features_per_path: int = max_features_per_path
        self._max_depth: int = max_depth
        self._n_nodes_total: int = node_offset
        self._roots = np.array(roots, dtype=np.int32)
        self._arrays = {key: np.concatenate(arrs) for key, arrs in parts.items()}
        self._decision_type: str = self._trees[0].decision_type

        # quadrature rule: exact for the degree-(d - order) integrands; Banzhaf indices are a
        # single evaluation of the weighted Banzhaf polynomial at p = 1/2
        if n_quadrature_points is not None and n_quadrature_points < 1:
            msg = f"n_quadrature_points={n_quadrature_points} must be a positive integer."
            raise ValueError(msg)
        if self._base_index == "BII":
            self._t = np.array([0.5])
            self._w = np.array([1.0])
        else:
            n_points = n_quadrature_points
            if n_points is None:
                n_points = max((self.max_features_per_path + 1) // 2, 1)
            self._t, self._w = _gauss_legendre_unit(n_points)

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
        self._kernel_args: tuple | None = None

    def explain(self, x: np.ndarray) -> InteractionValues:
        """Computes the Shapley interaction values for a given instance ``x``.

        Args:
            x: Instance to be explained as a 1-dimensional array.

        Returns:
            The computed interaction values.

        """
        x_full = self._trees[0].cast_input(np.asarray(x, dtype=np.float64))
        x_relevant = x_full[self._relevant_features]
        n_players = max(x_full.shape[0], self._n_features_in_tree)

        if self._trivial_computation:
            interactions = np.zeros(0, dtype=float)
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

    def explain_function(self, x: np.ndarray) -> InteractionValues:
        """Computes the Shapley interaction values (alias of ``explain``)."""
        return self.explain(x)

    def _explain_numpy(self, x_relevant: np.ndarray) -> np.ndarray:
        """Reference numpy implementation of the quadrature traversal for one instance."""
        t, w = self._t, self._w
        one_minus_t = 1.0 - t
        n_quad = t.shape[0]
        arrays = self._arrays
        children_left = arrays["children_left"]
        children_right = arrays["children_right"]
        parents = arrays["parents"]
        ancestors = arrays["ancestors"]
        features = arrays["features"]
        thresholds = arrays["thresholds"]
        c_acc = arrays["c_acc"]
        values = arrays["values"]
        cat_values = arrays["cat_values"]
        cat_start = arrays["cat_start"]
        cat_size = arrays["cat_size"]
        children_left_default = arrays["children_left_default"]
        strict = self._decision_type == "<"

        offsets = self._output_offsets
        out = np.zeros(self._output_size, dtype=float)

        act = np.zeros(self._n_nodes_total, dtype=bool)
        A = np.ones((self._max_depth + 2, n_quad))
        E = np.zeros((self._max_depth + 2, n_quad))
        live_g: dict[int, np.ndarray] = {}
        path_feats: list[int] = []

        def goes_left(node: int, value: float) -> bool:
            if np.isnan(value):
                return bool(children_left_default[node])
            if cat_size[node] > 0:
                start, end = cat_start[node], cat_start[node] + cat_size[node]
                position = int(np.searchsorted(cat_values[start:end], int(value)))
                return position < end - start and int(cat_values[start + position]) == int(value)
            if strict:
                return value < thresholds[node]
            return value <= thresholds[node]

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
            feature = int(features[parents[node]])
            ancestor = int(ancestors[node])
            if ancestor >= 0 and not act[ancestor]:
                return  # both telescoping terms cancel exactly once the hot chain is broken
            _, g_new = edge_factors(node)
            delta = g_new if ancestor < 0 else g_new - edge_factors(ancestor)[1]
            weighted = w * E[depth] * delta
            if self._min_order == 1:
                out[offsets[1] + feature] += weighted.sum()
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
            feature = int(features[parents[node]])
            ancestor = int(ancestors[node])
            if ancestor >= 0:
                live_g[feature] = edge_factors(ancestor)[1]
            else:
                del live_g[feature]
                path_feats.remove(feature)

        # iterative depth-first traversal per tree; stages: 0 enter, 1 after left,
        # 2 after right, 3 leave
        for tree_root in self._roots:
            root = int(tree_root)
            stack: list[tuple[int, int, int]] = [(root, 0, 0)]
            while stack:
                node, depth, stage = stack.pop()
                if stage == 0:
                    ancestor = int(ancestors[node])
                    if node != root:
                        if ancestor >= 0:
                            act[node] &= act[ancestor]
                        u_new, g_new = edge_factors(node)
                        if ancestor >= 0:
                            A[depth] = A[depth - 1] * (u_new / edge_factors(ancestor)[0])
                        else:
                            A[depth] = A[depth - 1] * u_new
                            path_feats.append(int(features[parents[node]]))
                        live_g[int(features[parents[node]])] = g_new
                    left, right = int(children_left[node]), int(children_right[node])
                    if left >= 0:
                        go_left = goes_left(node, x_relevant[int(features[node])])
                        act[left], act[right] = go_left, not go_left
                        stack.append((node, depth, 3))
                        stack.append((node, depth, 2))
                        stack.append((right, depth + 1, 0))
                        stack.append((node, depth, 1))
                        stack.append((left, depth + 1, 0))
                    else:  # leaf
                        E[depth] = A[depth] * values[node]
                        if node != root:
                            extract(node, depth)
                            restore(node)
                elif stage == 1:
                    E[depth] = E[depth + 1]
                elif stage == 2:
                    E[depth] += E[depth + 1]
                elif node != root:  # stage 3 on a non-root internal node
                    extract(node, depth)
                    restore(node)
        return out

    def _explain_cpp(self, x_relevant: np.ndarray) -> np.ndarray:
        """C-extension implementation: one batched kernel call over all trees."""
        from .cext import quadrature_tree_shap  # ty: ignore[unresolved-import]

        if self._kernel_args is None:
            arrays = self._arrays
            self._kernel_args = (
                np.ascontiguousarray(arrays["thresholds"], dtype=np.float64),
                np.ascontiguousarray(arrays["features"], dtype=np.int32),
                np.ascontiguousarray(arrays["children_left"], dtype=np.int32),
                np.ascontiguousarray(arrays["children_right"], dtype=np.int32),
                np.ascontiguousarray(arrays["parents"], dtype=np.int32),
                np.ascontiguousarray(arrays["ancestors"], dtype=np.int32),
                np.ascontiguousarray(arrays["c_acc"], dtype=np.float64),
                np.ascontiguousarray(arrays["values"], dtype=np.float64),
                int(self._max_depth),
                int(self._n_nodes_total),
                np.ascontiguousarray(self._roots, dtype=np.int32),
                np.ascontiguousarray(self._t, dtype=np.float64),
                np.ascontiguousarray(self._w, dtype=np.float64),
                int(self._n_features_in_tree),
                int(self._min_order),
                int(self._max_order),
                self._decision_type,
                np.ascontiguousarray(arrays["cat_values"], dtype=np.int64),
                np.ascontiguousarray(arrays["cat_start"], dtype=np.int64),
                np.ascontiguousarray(arrays["cat_size"], dtype=np.int64),
                np.ascontiguousarray(arrays["children_left_default"], dtype=bool),
            )
        args = self._kernel_args
        out = np.zeros((1, self._output_size), dtype=np.float64)
        quadrature_tree_shap(
            *args[:16],
            np.ascontiguousarray(x_relevant.reshape(1, -1), dtype=np.float64),
            out,
            *args[16:],
        )
        return out[0]
