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

from math import comb
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np

from shapiq.game_theory.indices import get_computation_index
from shapiq.interaction_values import InteractionValues
from shapiq.tree.conversion.edges import create_edge_tree
from shapiq.tree.validation import validate_tree_model
from shapiq.utils.sets import generate_interaction_lookup

if TYPE_CHECKING:
    from shapiq.tree.base import DecisionType, TreeModel
    from shapiq.typing import Model

QuadratureTreeSHAPIndices = Literal["SV", "SII", "k-SII", "BV", "BII"]


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

        Raises:
            ValueError: If the index or the interaction orders are invalid.
            ImportError: If the C extension is not available in this build.

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
        self._max_order: int = max_order
        self._min_order: int = min_order
        self._index: QuadratureTreeSHAPIndices = index
        self._base_index: str = get_computation_index(self._index)

        self._trees: list[TreeModel] = validate_tree_model(model, class_label=class_index)
        if len({tree.decision_type for tree in self._trees}) > 1:
            msg = "All trees of an ensemble must share the same decision type."
            raise ValueError(msg)
        if len({tree.input_precision for tree in self._trees}) > 1:
            msg = "All trees of an ensemble must share the same input precision."
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
        # per-order lookups over the ensemble-space ids, and each order's output offset
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

        # concatenate all trees into one node-array block with globally rebased indices
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
            # unreachable (zero-cover) subtrees contribute exactly zero in the limit c -> 0; a
            # tiny positive cover realizes that limit without 0/0 factors. p_e is inf on a dead
            # subtree's entry edge and NaN strictly inside it, so test p_e before inverting.
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
        self._decision_type: DecisionType = self._trees[0].decision_type

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

        try:
            from .cext import (
                quadrature_tree_shap,  # ty: ignore[unresolved-import]  # noqa: F401
            )
        except ImportError:
            msg = (
                "The QuadratureTreeSHAP C extension is not available in this build; "
                "reinstall shapiq from a platform wheel."
            )
            raise ImportError(msg) from None
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
        else:
            interactions = self._explain_cpp(x_relevant)

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
