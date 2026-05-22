"""Multi-output interventional tree explainer.

This module is the *multivariate* counterpart of
:class:`shapiq.tree.interventional.explainer.InterventionalTreeExplainer`. It
explains a list of :class:`MultiOutputTreeModel` (or, equivalently, a fitted
``XGBRegressor(multi_strategy="multi_output_tree")``) whose leaves carry a
length-``c`` vector, producing one :class:`~shapiq.interaction_values.InteractionValues`
object *per output dimension*.

The whole forest is preprocessed once via
:func:`preprocess_boolean_trees_multi` and all ``c`` output columns are computed
in a single fused traversal via :func:`compute_interactions_flatten_multi`. The
dense ``(n_outputs, result_size)`` result of that kernel is then mapped back to
sparse ``{tuple: value}`` interaction dicts.

Boolean-tree convention
-----------------------
Like the scalar :class:`ProxySHAP` path, the proxy tree is treated as a
*boolean* tree: coalitions live in ``{0, 1}^n`` and the interventional reference
data is a single all-zeros row ``np.zeros((1, n))``. The fused kernel therefore
runs with ``scaling_factor = 1.0`` (one reference row).

Dense-offset layout
-------------------
The fused kernel returns, for each output, a dense vector laid out as

* order 1: ``result[i]`` for feature ``i`` (``n`` entries),
* order 2: ``result[n + offset]`` where ``offset`` enumerates unordered pairs
  ``(i, j)`` with ``i < j`` in upper-triangle order
  (``offset = i*n - i*(i+1)//2 + (j-i-1)``),
* order 3: ``result[n + C(n,2) + offset]`` with the compact triple offset
  ``offset = i + j*(j-1)//2 + k*(k-1)*(k-2)//6`` for ``i < j < k``.

This is exactly the layout documented by ``_dict_to_dense`` in
``tests/experimental/test_multioutput_kernel.py``;
:func:`build_offset_to_tuple_map` is its inverse.
"""

from __future__ import annotations

from math import comb
from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq.game_theory.indices import get_computation_index
from shapiq.interaction_values import InteractionValues
from shapiq.tree.interventional.cext import (
    compute_interactions_flatten_multi,  # ty: ignore[unresolved-import]
    preprocess_boolean_trees_multi,  # ty: ignore[unresolved-import]
)

from .tree import MultiOutputTreeModel, convert_multioutput_xgboost

if TYPE_CHECKING:
    from numpy.typing import NDArray

# The fused dense kernel mirrors the scalar flatten path: it is only implemented
# for main effects, pairwise and triple interactions.
_SUPPORTED_MAX_ORDERS = (1, 2, 3)


def build_offset_to_tuple_map(n_features: int, max_order: int) -> list[tuple[int, ...]]:
    """Build the inverse of the dense-result offset layout.

    Returns a list ``offset_to_tuple`` of length
    ``sum(C(n_features, k) for k in 1..max_order)`` such that the dense kernel
    result entry ``result[offset]`` is the interaction value of the feature
    tuple ``offset_to_tuple[offset]``.

    The layout matches the fused C kernel (and ``_dict_to_dense`` in the Phase 2
    test): order-1 singletons first, then order-2 pairs in upper-triangle order,
    then order-3 triples in the compact ``index3`` order.

    Args:
        n_features: Number of features (players) ``n``.
        max_order: Maximum interaction order; one of ``{1, 2, 3}``.

    Returns:
        The offset -> interaction-tuple map.
    """
    if max_order not in _SUPPORTED_MAX_ORDERS:
        msg = f"max_order must be one of {_SUPPORTED_MAX_ORDERS}, got {max_order}."
        raise ValueError(msg)

    offset_to_tuple: list[tuple[int, ...]] = []

    # order 1: singleton (i,)
    offset_to_tuple.extend((i,) for i in range(n_features))

    # order 2: unordered pairs (i, j), i < j, upper-triangle order
    if max_order >= 2:
        offset_to_tuple.extend((i, j) for i in range(n_features) for j in range(i + 1, n_features))

    # order 3: unordered triples (i, j, k), i < j < k, in the compact ``index3``
    # order of the fused C kernel. The kernel does NOT lay triples out
    # lexicographically -- it places triple (i, j, k) at the dense offset
    # ``i + j*(j-1)//2 + k*(k-1)*(k-2)//6`` (the same formula used by
    # ``_dict_to_dense`` in the Phase 2 kernel test). The triples must therefore
    # be emitted sorted by that offset, not by plain nested iteration.
    if max_order >= 3:
        triples = [
            (i, j, k)
            for i in range(n_features)
            for j in range(i + 1, n_features)
            for k in range(j + 1, n_features)
        ]
        triples.sort(
            key=lambda t: t[0] + t[1] * (t[1] - 1) // 2 + t[2] * (t[2] - 1) * (t[2] - 2) // 6
        )
        offset_to_tuple.extend(triples)

    expected = sum(comb(n_features, k) for k in range(1, max_order + 1))
    if len(offset_to_tuple) != expected:  # pragma: no cover - defensive
        msg = (
            f"offset map size {len(offset_to_tuple)} does not match expected "
            f"dense result size {expected}."
        )
        raise RuntimeError(msg)
    return offset_to_tuple


class MultiOutputInterventionalTreeExplainer:
    """Any-order interventional Shapley-interaction explainer for multi-output trees.

    This is the multivariate analogue of
    :class:`shapiq.tree.interventional.explainer.InterventionalTreeExplainer`.
    It consumes a list of :class:`MultiOutputTreeModel` (vector-valued leaves) or
    a fitted XGBoost ``multi_strategy="multi_output_tree"`` model and computes,
    for every output dimension ``j = 0 .. c-1``, the requested interaction index
    up to ``max_order``.

    The trees are treated as *boolean* trees (the proxy convention): the
    interventional reference data is a single all-zeros row, so the fused kernel
    runs with ``scaling_factor = 1.0``.

    Attributes:
        trees: The converted multi-output trees.
        n_outputs: The output dimensionality ``c``.
        n_players: Number of features ``n``.
        max_order: Maximum interaction order (one of ``{1, 2, 3}``).
        index: The interaction index (e.g. ``"SV"`` / ``"SII"``).
        baseline_values: Length-``c`` array of per-output interventional
            baselines (mean forest prediction over the all-zeros reference row).
    """

    def __init__(
        self,
        model: list[MultiOutputTreeModel] | Any,  # noqa: ANN401
        *,
        index: str = "SII",
        max_order: int = 2,
        n_players: int | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the multi-output interventional tree explainer.

        Args:
            model: Either a list of :class:`MultiOutputTreeModel` or a fitted
                ``XGBRegressor`` trained with
                ``multi_strategy="multi_output_tree"`` (converted internally).
            index: Interaction index. Supported: ``"SV"`` (with ``max_order=1``)
                and ``"SII"``. Defaults to ``"SII"``.
            max_order: Maximum interaction order; one of ``{1, 2, 3}``. Defaults
                to ``2``.
            n_players: Number of features ``n``. If ``None``, inferred from the
                highest split-feature index, which undercounts when the proxy
                never splits on a trailing feature. Defaults to ``None``.
            verbose: If ``True``, the fused C kernel prints debug information.
                Defaults to ``False``.
        """
        if max_order not in _SUPPORTED_MAX_ORDERS:
            msg = f"max_order must be one of {_SUPPORTED_MAX_ORDERS}, got {max_order}."
            raise ValueError(msg)

        if isinstance(model, list) and all(isinstance(t, MultiOutputTreeModel) for t in model):
            self.trees: list[MultiOutputTreeModel] = model
        else:
            self.trees = convert_multioutput_xgboost(model)

        if not self.trees:
            msg = "The multi-output model contains no trees."
            raise ValueError(msg)

        self.index = index
        self.max_order = max_order
        self.verbose = verbose
        self.n_outputs: int = int(self.trees[0].n_outputs)
        # When n_players is not given explicitly it is inferred from the highest
        # split-feature index. That undercounts whenever the proxy never splits on
        # a (trailing) feature, so callers that know n -- e.g. MultiOutputProxySHAP
        # -- should pass it explicitly.
        if n_players is not None:
            self.n_players: int = int(n_players)
        else:
            self.n_players = int(max(int(t.features.max(initial=-1)) for t in self.trees) + 1)

        # Build the boolean-tree structural arrays once.
        self._preprocess_boolean_trees()

        # Per-output interventional baseline: mean forest prediction over the
        # single all-zeros reference row. With one reference row this is simply
        # the forest prediction (without base score) at the all-zeros input.
        reference_row = np.zeros(self.n_players, dtype=np.float64)
        baseline = np.zeros(self.n_outputs, dtype=np.float64)
        for tree in self.trees:
            baseline += tree.predict_one(reference_row)
        self.baseline_values: NDArray[np.float64] = baseline

        # Inverse offset map, built once.
        self._offset_to_tuple = build_offset_to_tuple_map(self.n_players, self.max_order)

    def _preprocess_boolean_trees(self) -> None:
        """Flatten the forest into the boolean-tree layout the fused kernel needs."""
        values_list = [tree.values.astype(np.float32) for tree in self.trees]
        features_list = [tree.features.astype(np.int64) for tree in self.trees]
        children_left_list = [tree.children_left.astype(np.int64) for tree in self.trees]
        children_right_list = [tree.children_right.astype(np.int64) for tree in self.trees]

        (
            self._e_r_flatten,
            self._e_size_flatten,
            self._r_size_flatten,
            self._feature_in_e,
            self._leaf_id,
            self._leaf_values,
        ) = preprocess_boolean_trees_multi(
            values_list,
            features_list,
            children_left_list,
            children_right_list,
            self.n_players,
        )

    def explain(self) -> list[InteractionValues]:
        """Compute interaction values for every output dimension.

        Runs the fused multi-output kernel once and decodes the dense
        ``(n_outputs, result_size)`` result into one
        :class:`~shapiq.interaction_values.InteractionValues` per output.

        Returns:
            A list of length ``n_outputs``; element ``j`` carries the
            interactions of output column ``j``, with ``interactions[()]`` set
            to that output's interventional baseline.
        """
        computation_index = get_computation_index(self.index)

        # The proxy reference data is a single all-zeros row, so scaling is 1.0.
        dense = compute_interactions_flatten_multi(
            self._leaf_values,
            self._e_r_flatten,
            self._e_size_flatten,
            self._r_size_flatten,
            self._feature_in_e,
            self._leaf_id,
            computation_index,
            len(self._e_r_flatten),  # n_iterations
            self.n_players,
            self.max_order,
            int(self.verbose),
            1.0,  # scaling_factor: single reference row
            None,  # no custom weight table
        )
        dense = np.asarray(dense, dtype=np.float64)

        results: list[InteractionValues] = []
        for j in range(self.n_outputs):
            baseline = float(self.baseline_values[j])
            interactions: dict[tuple[int, ...], float] = {
                self._offset_to_tuple[offset]: float(value) for offset, value in enumerate(dense[j])
            }
            interactions[()] = baseline
            results.append(
                InteractionValues(
                    interactions,
                    max_order=self.max_order,
                    min_order=1,
                    index=computation_index,
                    n_players=self.n_players,
                    baseline_value=baseline,
                    target_index=self.index,
                )
            )
        return results
