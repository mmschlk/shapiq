"""ProxySPEX adapter helpers for OddSHAP.

Two free-function helpers used inside ``OddSHAP.approximate`` to extract a
sparse Fourier representation of the LightGBM surrogate and select the
top-k odd-cardinality interactions.

Mirrors the interface that the OddSHAP paper code imports:

    from oddshap.proxyspex import lgboost_to_fourier, top_k_interactions

so the OddSHAP class body can call them as

    initial_transform = lgboost_to_fourier(surrogate.booster_.dump_model())
    selected = top_k_interactions(initial_transform, n_interactions, odd=True)

The Fourier conversion uses the per-tree DFS recursion of Gorji et al.
(arXiv:2410.06300): each leaf is a constant function with non-zero weight only
on the empty interaction; each split node combines its children's coefficients
by averaging shared interactions and adding the split feature to a new "odd"
interaction with coefficient ``(left - right) / 2``. Per-tree coefficients are
then summed across the ensemble.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

__all__ = ["lgboost_to_fourier", "top_k_interactions"]


def lgboost_to_fourier(
    model_dict: dict[str, Any],
) -> dict[tuple[int, ...], float]:
    """Convert a fitted LightGBM model to its aggregated Fourier representation.

    Args:
        model_dict: The output of ``model.booster_.dump_model()`` for a fitted
            LightGBM regressor or classifier.

    Returns:
        Dictionary mapping interaction tuples (sorted feature indices) to their
        aggregated Fourier coefficients summed across all trees in the
        ensemble. Zero-valued coefficients are dropped.
    """
    aggregated: dict[tuple[int, ...], float] = defaultdict(float)
    for tree_info in model_dict["tree_info"]:
        for interaction, value in _tree_to_fourier(tree_info).items():
            aggregated[interaction] += value
    return {k: v for k, v in aggregated.items() if v != 0.0}


def top_k_interactions(
    fourier_coeffs: dict[tuple[int, ...], float],
    k: int,
    *,
    odd: bool = True,
) -> dict[tuple[int, ...], float]:
    """Select the top-k interactions by Fourier coefficient magnitude.

    Args:
        fourier_coeffs: Output of :func:`lgboost_to_fourier`.
        k: Maximum number of interactions to retain.
        odd: If True (default), restrict to interactions of odd cardinality —
            matches the OddSHAP paper's restriction since by Theorem 3.2 the
            Shapley value depends only on the odd component of the set
            function.

    Returns:
        Sub-dictionary of ``fourier_coeffs`` containing the top-k entries by
        absolute coefficient magnitude. If fewer than ``k`` qualifying entries
        exist, all of them are returned.
    """
    if odd:
        candidates = {t: v for t, v in fourier_coeffs.items() if len(t) % 2 == 1}
    else:
        candidates = dict(fourier_coeffs)

    sorted_items = sorted(candidates.items(), key=lambda kv: abs(kv[1]), reverse=True)
    return dict(sorted_items[:k])


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _tree_to_fourier(tree_info: dict[str, Any]) -> dict[tuple[int, ...], float]:
    """Compute Fourier coefficients for a single LightGBM tree via DFS.

    Args:
        tree_info: A single ``tree_info`` entry from
            ``model.booster_.dump_model()``.

    Returns:
        Dictionary mapping interaction tuples to per-tree Fourier coefficients.
    """

    def _combine(
        left: dict[tuple[int, ...], float],
        right: dict[tuple[int, ...], float],
        feature_idx: int,
    ) -> dict[tuple[int, ...], float]:
        combined: dict[tuple[int, ...], float] = {}
        for interaction in set(left) | set(right):
            l_val = left.get(interaction, 0.0)
            r_val = right.get(interaction, 0.0)
            combined[interaction] = (l_val + r_val) / 2
            extended = tuple(sorted(set(interaction) | {feature_idx}))
            combined[extended] = (l_val - r_val) / 2
        return combined

    def _dfs(node: dict[str, Any]) -> dict[tuple[int, ...], float]:
        if "leaf_value" in node:
            return {(): node["leaf_value"]}
        left_coeffs = _dfs(node["left_child"])
        right_coeffs = _dfs(node["right_child"])
        return _combine(left_coeffs, right_coeffs, node["split_feature"])

    return _dfs(tree_info["tree_structure"])
