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

from bisect import bisect_left, insort
from itertools import combinations
from typing import TYPE_CHECKING, Literal, get_args

import numpy as np

from shapiq.game_theory.indices import get_computation_index
from shapiq.interaction_values import InteractionValues
from shapiq.tree.conversion.edges import create_edge_tree
from shapiq.tree.validation import validate_tree_model

if TYPE_CHECKING:
    from shapiq.tree.base import DecisionType, TreeModel
    from shapiq.typing import Model

QuadratureTreeSHAPIndices = Literal["SV", "SII", "k-SII", "BV", "BII"]


def _collect_cooccurring_subsets(
    children_left: np.ndarray,
    children_right: np.ndarray,
    features: np.ndarray,
    parents: np.ndarray,
    ancestors: np.ndarray,
    subset_sets: dict[int, set[tuple[int, ...]]],
) -> None:
    """Add every feature subset (order >= 2) co-occurring on some root-to-leaf path.

    Each subset is emitted at the first-occurrence edge of its deepest member (``ancestors``
    marks repeated features), so the work is bounded by the same per-edge enumeration the
    kernel performs for one explanation instead of the much larger per-leaf count.

    Note: Changing the traversal order will break the traverser implemented in the C++ kernel.
    Therefore any changes here must be reflected in the kernel's path bookkeeping (``path_feats``) and
    """
    path: list[int] = []  # sorted distinct features on the current path
    stack: list[tuple[int, int]] = [(0, -1)]  # (node, feature to remove on leave; -1 = enter)
    while stack:
        node, leave_feature = stack.pop()
        if leave_feature >= 0:
            path.remove(leave_feature)
            continue
        new_feature = -1
        if parents[node] >= 0 and ancestors[node] < 0:
            new_feature = int(features[parents[node]])
            for order, subsets in subset_sets.items():
                if len(path) >= order - 1:
                    for chosen in combinations(path, order - 1):
                        position = bisect_left(chosen, new_feature)
                        subsets.add((*chosen[:position], new_feature, *chosen[position:]))
            insort(path, new_feature)
        left = int(children_left[node])
        if left >= 0:
            if new_feature >= 0:
                stack.append((node, new_feature))
            stack.append((int(children_right[node]), -1))
            stack.append((left, -1))
        elif new_feature >= 0:
            path.remove(new_feature)


# Multiplier of the subset index hash; must equal kIndexHashMultiplier in
# cext/quadrature_tree_shap.cc (2^64 / golden ratio).
_INDEX_HASH_MULTIPLIER = np.uint64(0x9E3779B97F4A7C15)


def _build_subset_index(
    n_features: int,
    max_order: int,
    keys: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Builds the lookup from each subset to the output position its value is stored at.

    The C++ extension enumerates subsets along decision paths to have a fast lookup of where subset contribution belong in the output.
    This function builds a array flat hash table for every order >= 2, where each slot holds a subset's key and row id (with -1 marking an empty slot).
    It is built once per explainer.
    Its hashing must match ``merged_position`` in ``cext/quadrature_tree_shap.cc``, which reads it.

    Args:
        n_features: Number of features the ensemble splits on. Subsets use the remapped ids
            ``0 .. n_features - 1``, and each subset's key is read as a number in this base.

        max_order: Highest interaction order. Orders ``2 .. max_order`` each get their own
            block of the index; order 1 needs none, as a feature's position is its id.

        keys: The flat ``int32`` subset tables from ``_subset_table_args``: per order, its
            sorted subsets back to back, ``order`` feature ids each.

        starts: Per order, the position in ``keys`` where its table begins.

        counts: Per order, the number of subsets in its table.

    Returns:
        ``(slot_keys, slot_rows, block_starts, block_shifts)``, the arrays the kernel reads:
        each slot holds a subset's key and row id (``-1`` marks an empty slot), and per order
        ``block_starts`` and ``block_shifts`` give where its slots begin and how many there
        are (``2 ** (64 - shift)``). ``None`` when there is no order ``>= 2`` or the keys would
        overflow an ``int64``; the kernel then searches the subset tables directly.
    """
    if max_order < 2 or n_features**max_order > np.iinfo(np.int64).max:
        return None
    if int(np.max(counts)) > np.iinfo(np.int32).max:  # row ids are stored as int32
        return None
    # Build hash table flat arrays for every order >= 2.
    block_starts = np.zeros(max_order + 1, dtype=np.int64)
    block_shifts = np.full(max_order + 1, 63, dtype=np.int64)
    slot_keys: list[np.ndarray] = []
    slot_rows: list[np.ndarray] = []
    position = 0
    for order in range(2, max_order + 1):
        count = int(counts[order])
        # 2d array of the order's subsets, each row is a sorted tuple of feature ids
        table = keys[starts[order] : starts[order] + count * order].reshape(count, order)
        row_keys = np.zeros(count, dtype=np.int64)
        # Compute a unique integer key for each row by treating the row as a base-n_features
        for column in range(order):
            row_keys = row_keys * n_features + table[:, column]
        # Compute the smallest power-of-two block size that can hold all rows with at least one
        bits = max(1, (2 * count - 1).bit_length())  # 2**bits >= 2 * count
        size = 1 << bits
        # The hash of a row key is the top bits of key * _INDEX_HASH_MULTIPLIER, which is uniformly distributed over the 64-bit space.
        # Standard linear probing is used to resolve collisions, wrapping within the block.
        home = (row_keys.astype(np.uint64) * _INDEX_HASH_MULTIPLIER) >> np.uint64(64 - bits)
        block_keys = np.full(size, -1, dtype=np.int64)
        block_rows = np.full(size, -1, dtype=np.int32)
        slot = home.astype(np.int64)
        pending = np.arange(count)
        # linear probing for all rows at once: every pending row tries its current slot, one
        # row wins each free slot, and the others move on by one
        while pending.size:
            target = slot[pending]
            free = block_keys[target] == -1
            # Find the first free slot for each pending row
            claimed, first = np.unique(target[free], return_index=True)
            winners = pending[free][first]
            block_keys[claimed] = row_keys[winners]
            block_rows[claimed] = winners
            # The remaining rows that didn't find a free slot will try the next slot in the next iteration
            pending = np.setdiff1d(pending, winners, assume_unique=True)
            slot[pending] = (slot[pending] + 1) & (size - 1)
        block_starts[order] = position
        block_shifts[order] = 64 - bits
        slot_keys.append(block_keys)
        slot_rows.append(block_rows)
        position += size
    return np.concatenate(slot_keys), np.concatenate(slot_rows), block_starts, block_shifts


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

    The returned :class:`~shapiq.interaction_values.InteractionValues` enumerates exactly the
    interactions whose features co-occur on at least one decision path — every other
    interaction is structurally zero and therefore not part of the output. This keeps the
    result and the computation sparse at higher orders (the dense ``C(n_features, order)``
    enumeration is never materialized); order 1 remains a dense per-feature block.

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
        subset_sets: dict[int, set[tuple[int, ...]]] = {
            order: set() for order in range(max(self._min_order, 2), self._max_order + 1)
        }
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

            if self._max_order >= 2:
                _collect_cooccurring_subsets(
                    tree.children_left,
                    tree.children_right,
                    features_ens,
                    edge_tree.parents,
                    edge_tree.ancestors,
                    subset_sets,
                )
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

        # sparse interaction support: only subsets whose features co-occur on at least one
        # decision path can be nonzero, so the output enumerates exactly those. Order 1 stays
        # a dense per-feature block (every union feature appears on some path) — the Shapley
        # hot path keeps direct indexing while higher orders avoid the C(F, order) blow-up.
        self._order_lookups: dict[int, dict[tuple, int]] = {}
        self._output_offsets: dict[int, int] = {}
        self._subset_tables: dict[int, np.ndarray] = {}
        offset = 0
        if self._min_order == 1:
            self._order_lookups[1] = {(f,): f for f in range(self._n_features_in_tree)}
            self._output_offsets[1] = 0
            offset = self._n_features_in_tree
        for order in range(max(self._min_order, 2), self._max_order + 1):
            ordered = sorted(subset_sets[order])
            self._order_lookups[order] = {subset: pos for pos, subset in enumerate(ordered)}
            self._output_offsets[order] = offset
            self._subset_tables[order] = np.asarray(ordered, dtype=np.int32).reshape(
                len(ordered), order
            )
            offset += len(ordered)
        self._output_size: int = offset

        # output lookup over the original feature ids for the returned InteractionValues
        lookup: dict[tuple, int] = {}
        original_ids = self._relevant_features
        for order in range(self._min_order, self._max_order + 1):
            base = self._output_offsets[order]
            for subset, position in self._order_lookups[order].items():
                lookup[tuple(int(original_ids[j]) for j in subset)] = base + position
        self._interactions_lookup_relevant: dict[tuple, int] = lookup
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
            subset_tables = self._subset_table_args()
            keys, starts, counts, _ = subset_tables
            # hash index of the subset tables, built once from the very arrays the kernel
            # receives; None leaves the kernel to search the tables by tuple comparison
            subset_index = _build_subset_index(
                int(self._n_features_in_tree), int(self._max_order), keys, starts, counts
            )
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
                *subset_tables,
                self._decision_type,
                np.ascontiguousarray(arrays["cat_values"], dtype=np.int64),
                np.ascontiguousarray(arrays["cat_start"], dtype=np.int64),
                np.ascontiguousarray(arrays["cat_size"], dtype=np.int64),
                np.ascontiguousarray(arrays["children_left_default"], dtype=bool),
                *(() if subset_index is None else subset_index),
            )
        args = self._kernel_args
        out = np.zeros((1, self._output_size), dtype=np.float64)
        quadrature_tree_shap(
            *args[:20],
            np.ascontiguousarray(x_relevant.reshape(1, -1), dtype=np.float64),
            out,
            *args[20:],
        )
        return out[0]

    def _subset_table_args(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The kernel's sparse subset tables: flat keys, per-order starts/counts/offsets."""
        n_orders = self._max_order + 1
        starts = np.zeros(n_orders, dtype=np.int64)
        counts = np.zeros(n_orders, dtype=np.int64)
        offsets = np.zeros(n_orders, dtype=np.int64)
        keys: list[np.ndarray] = []
        position = 0
        for order in range(max(self._min_order, 2), self._max_order + 1):
            table = self._subset_tables[order]
            starts[order] = position
            counts[order] = table.shape[0]
            offsets[order] = self._output_offsets[order]
            keys.append(table.reshape(-1))
            position += table.size
        flat = np.concatenate(keys) if keys else np.zeros(0, dtype=np.int32)
        return np.ascontiguousarray(flat, dtype=np.int32), starts, counts, offsets
