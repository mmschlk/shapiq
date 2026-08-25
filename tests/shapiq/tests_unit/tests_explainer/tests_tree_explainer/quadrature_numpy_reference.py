"""Pure-numpy reference implementation of the quadrature traversal.

Moved out of ``shapiq.tree.quadrature`` — the shipped explainer requires its C extension, and
a missing extension is a packaging error rather than something to silently fall back from.
This module is kept only as the differential test oracle for the C kernel and is slated for
deletion once it is no longer needed.
"""

from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np

from shapiq.interaction_values import InteractionValues

if TYPE_CHECKING:
    from shapiq.tree.quadrature import QuadratureTreeSHAP


def numpy_explain(explainer: QuadratureTreeSHAP, x: np.ndarray) -> InteractionValues:
    """Compute ``explainer.explain(x)`` with the numpy traversal instead of the C kernel."""
    x_full = explainer._trees[0].cast_input(np.asarray(x, dtype=np.float64))
    x_relevant = x_full[explainer._relevant_features]
    n_players = max(x_full.shape[0], explainer._n_features_in_tree)

    if explainer._trivial_computation:
        interactions = np.zeros(0, dtype=float)
    else:
        interactions = _traverse(explainer, x_relevant)

    return InteractionValues(
        values=interactions,
        index=explainer._base_index,
        min_order=explainer._min_order,
        max_order=explainer._max_order,
        n_players=n_players,
        estimated=False,
        interaction_lookup=explainer._interactions_lookup_relevant,
        baseline_value=explainer.empty_prediction,
        target_index=explainer._index,
    )


def _traverse(explainer: QuadratureTreeSHAP, x_relevant: np.ndarray) -> np.ndarray:
    """Reference numpy implementation of the quadrature traversal for one instance."""
    t, w = explainer._t, explainer._w
    one_minus_t = 1.0 - t
    n_quad = t.shape[0]
    arrays = explainer._arrays
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
    strict = explainer._decision_type == "<"

    offsets = explainer._output_offsets
    out = np.zeros(explainer._output_size, dtype=float)

    act = np.zeros(explainer._n_nodes_total, dtype=bool)
    A = np.ones((explainer._max_depth + 2, n_quad))
    E = np.zeros((explainer._max_depth + 2, n_quad))
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
        if explainer._min_order == 1:
            out[offsets[1] + feature] += weighted.sum()
        if explainer._max_order > 1:
            others = [j for j in path_feats if j != feature]
            for order in range(max(explainer._min_order, 2), explainer._max_order + 1):
                lookup = explainer._order_lookups[order]
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
    for tree_root in explainer._roots:
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
