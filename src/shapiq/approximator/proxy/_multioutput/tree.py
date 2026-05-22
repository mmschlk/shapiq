"""Experimental multi-output tree container and XGBoost converter.

This module provides an experimental, *self-contained* representation of an XGBoost
``multi_strategy="multi_output_tree"`` model: a single shared-topology tree per
boosting round whose leaf nodes carry a length-``c`` value vector.

Design notes
------------
The field names of :class:`MultiOutputTreeModel` deliberately mirror the relevant
fields of shapiq's :class:`shapiq.tree.base.TreeModel` (``children_left``,
``children_right``, ``features``, ``thresholds``, ``values``, ``leaf_mask``) so the
boolean-tree / interventional traversal kernels can consume either structure with
minimal branching. The only structural difference is that ``values`` here is a 2-D
``(n_nodes, n_outputs)`` array instead of a flat 1-D array.

Sentinels (matching ``TreeModel`` conventions, verified against XGBoost output):

* ``children_left[i] == -1`` and ``children_right[i] == -1`` mark a leaf node. Note
  that XGBoost only guarantees ``left_children == -1`` at leaves; the raw
  ``right_children`` entries at leaf rows are garbage, so the converter sanitises
  them to ``-1``.
* ``features[i] == -2`` marks a leaf node (no decision feature).
* ``thresholds[i] == np.nan`` marks a leaf node (no decision threshold).
* internal-node rows of ``values`` are forced to ``0``.

Extraction route
----------------
Trees are parsed from XGBoost's *lossless* full-model JSON serialization
(``booster.save_raw(raw_format="json")``). That serialization stores each tree's
``base_weights`` as a flat float array of length ``n_nodes * n_outputs`` with **no
truncation** (unlike the human-readable ``get_dump`` output, which truncates long
arrays with a literal ``...`` token). A leaf row of ``base_weights`` holds the raw
length-``c`` leaf vector; split rows are (numerically) zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# XGBoost's multi_output_tree split nodes use a strict "<" comparison to route a
# sample to the left child (value < threshold -> left).
_DECISION_TYPE = "<"


@dataclass
class MultiOutputTreeModel:
    """Container for a single multi-output (vector-valued leaf) decision tree.

    Each attribute is indexed by node id ``0 .. n_nodes - 1``; the root is node ``0``.
    The arrays mirror the layout of :class:`shapiq.tree.base.TreeModel` so downstream
    traversal kernels can treat both structurally the same, except that :attr:`values`
    is 2-D here.

    Attributes:
        children_left: Left-child node id of each node; ``-1`` denotes a leaf.
        children_right: Right-child node id of each node; ``-1`` denotes a leaf.
        children_default: Child node id that missing-value samples are routed to
            (``children_left`` if the node's ``default_left`` flag is set, else
            ``children_right``); ``-1`` at leaves.
        features: Decision-node feature index of each node; ``-2`` denotes a leaf.
        thresholds: Decision-node split threshold of each node; ``np.nan`` at leaves.
        values: ``(n_nodes, n_outputs)`` float64 array. Leaf rows hold the raw
            length-``n_outputs`` leaf vector; internal rows are ``0``.
        leaf_mask: Boolean mask, ``True`` at leaf nodes.
        n_outputs: The output dimensionality ``c``.
        n_nodes: The number of nodes in the tree.
        decision_type: Split comparison used by traversal (always ``"<"`` for the
            XGBoost multi-output trees parsed here).
    """

    children_left: NDArray[np.int_]
    children_right: NDArray[np.int_]
    children_default: NDArray[np.int_]
    features: NDArray[np.int_]
    thresholds: NDArray[np.floating]
    values: NDArray[np.float64]
    leaf_mask: NDArray[np.bool_]
    n_outputs: int
    n_nodes: int
    decision_type: str = _DECISION_TYPE

    def predict_one(self, x: NDArray[np.floating]) -> NDArray[np.float64]:
        """Predict the length-``n_outputs`` raw output of a single instance.

        This is the contribution of *this tree only* (no base score is added).

        XGBoost evaluates split comparisons in single precision internally. Feature
        values are therefore cast to ``float32`` before the threshold comparison so
        that samples sitting close to a split boundary are routed down the same
        branch as ``model.predict`` (a float64 comparison can otherwise flip the
        branch and produce a visibly wrong leaf vector).

        Args:
            x: The instance to predict, as a 1-D array of feature values. ``np.nan``
                entries are routed using each node's default direction.

        Returns:
            The tree's length-``n_outputs`` leaf vector for ``x``.
        """
        x = np.asarray(x, dtype=np.float32)
        node = 0
        while not self.leaf_mask[node]:
            value = x[self.features[node]]
            threshold = np.float32(self.thresholds[node])
            if np.isnan(value):
                node = self.children_default[node]
            elif value < threshold:
                node = self.children_left[node]
            else:
                node = self.children_right[node]
        return self.values[node].astype(np.float64)

    def predict(self, X: NDArray[np.floating]) -> NDArray[np.float64]:
        """Predict the raw output for a batch of instances.

        Args:
            X: A ``(n_samples, n_features)`` array of instances.

        Returns:
            A ``(n_samples, n_outputs)`` array of this tree's leaf vectors.
        """
        X = np.asarray(X, dtype=np.float64)
        out = np.empty((X.shape[0], self.n_outputs), dtype=np.float64)
        for i in range(X.shape[0]):
            out[i] = self.predict_one(X[i])
        return out


def _predict_forest(
    trees: list[MultiOutputTreeModel],
    base_score: NDArray[np.float64],
    X: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Sum the per-tree predictions and add the base score.

    Args:
        trees: The converted multi-output trees.
        base_score: The length-``n_outputs`` prediction offset.
        X: A ``(n_samples, n_features)`` array of instances.

    Returns:
        A ``(n_samples, n_outputs)`` array of forest predictions.
    """
    X = np.asarray(X, dtype=np.float64)
    out = np.tile(base_score.astype(np.float64), (X.shape[0], 1))
    for tree in trees:
        out += tree.predict(X)
    return out


def _parse_base_score(learner_model_param: dict, n_outputs: int) -> NDArray[np.float64]:
    """Parse the (possibly vector-valued) base score from the model JSON.

    For multi-output regression XGBoost stores ``base_score`` as a JSON-list string
    (e.g. ``"[8.25E-2,8.5E-2,...]"``). For a scalar model it is a plain number string.
    Either way the returned array is broadcast to length ``n_outputs``.

    Args:
        learner_model_param: The ``learner -> learner_model_param`` dict.
        n_outputs: The output dimensionality ``c``.

    Returns:
        A length-``n_outputs`` float64 base-score array.
    """
    raw = learner_model_param.get("base_score", "0")
    parsed = json.loads(raw) if isinstance(raw, str) else raw
    arr = np.atleast_1d(np.asarray(parsed, dtype=np.float64))
    if arr.size == 1 and n_outputs > 1:
        arr = np.full(n_outputs, float(arr[0]), dtype=np.float64)
    return arr


def _parse_tree(tree_json: dict) -> MultiOutputTreeModel:
    """Parse a single tree dict from XGBoost's lossless model JSON.

    Args:
        tree_json: One entry of ``learner -> gradient_booster -> model -> trees``.

    Returns:
        The parsed :class:`MultiOutputTreeModel`.
    """
    tree_param = tree_json["tree_param"]
    n_nodes = int(tree_param["num_nodes"])
    # size_leaf_vector is the per-leaf output dimensionality (c) for multi-output trees.
    n_outputs = int(tree_param.get("size_leaf_vector", 1)) or 1

    children_left = np.asarray(tree_json["left_children"], dtype=np.int64)
    children_right = np.asarray(tree_json["right_children"], dtype=np.int64)
    features = np.asarray(tree_json["split_indices"], dtype=np.int64)
    thresholds = np.asarray(tree_json["split_conditions"], dtype=np.float64)
    default_left = np.asarray(tree_json["default_left"], dtype=bool)

    # base_weights is flat of length n_nodes * n_outputs; leaf rows hold the c-vector.
    base_weights = np.asarray(tree_json["base_weights"], dtype=np.float64)
    values = base_weights.reshape(n_nodes, n_outputs)

    # XGBoost only guarantees left_children == -1 at leaves; sanitise the rest.
    leaf_mask = children_left == -1
    children_right = np.where(leaf_mask, -1, children_right)
    features = np.where(leaf_mask, -2, features).astype(np.int64)
    thresholds = np.where(leaf_mask, np.nan, thresholds)

    # internal-node rows of values must be zeroed (they are ~0 already, but exactly 0
    # is what the traversal kernels expect).
    values = values.copy()
    values[~leaf_mask] = 0.0

    children_default = np.where(default_left, children_left, children_right)
    children_default = np.where(leaf_mask, -1, children_default).astype(np.int64)

    return MultiOutputTreeModel(
        children_left=children_left,
        children_right=children_right,
        children_default=children_default,
        features=features,
        thresholds=thresholds,
        values=values,
        leaf_mask=leaf_mask,
        n_outputs=n_outputs,
        n_nodes=n_nodes,
    )


def convert_multioutput_xgboost(model: Any) -> list[MultiOutputTreeModel]:
    """Convert a fitted multi-output XGBoost regressor into multi-output trees.

    The model must be a fitted ``XGBRegressor`` (or ``Booster``) trained with
    ``multi_strategy="multi_output_tree"``. Parsing uses the lossless full-model JSON
    serialization (``booster.save_raw(raw_format="json")``), so the conversion is pure
    Python and touches no shared shapiq C code.

    The returned trees plus the model's base score reproduce ``model.predict`` (up to
    float32 rounding inside XGBoost); use :func:`predict_multioutput` or the
    ``base_score`` attached to the result for the prediction offset.

    Args:
        model: A fitted ``XGBRegressor`` / ``Booster`` with vector-valued leaves.

    Returns:
        The list of :class:`MultiOutputTreeModel` instances, one per boosting round.
        The shared base-score vector is attached to the list's first element via the
        :func:`predict_multioutput` helper; callers needing it directly should use
        :func:`convert_multioutput_xgboost_with_base_score`.
    """
    trees, _ = convert_multioutput_xgboost_with_base_score(model)
    return trees


def convert_multioutput_xgboost_with_base_score(
    model: Any,
) -> tuple[list[MultiOutputTreeModel], NDArray[np.float64]]:
    """Convert a multi-output XGBoost model and also return its base score.

    See :func:`convert_multioutput_xgboost` for details. This variant additionally
    returns the length-``n_outputs`` base-score (prediction offset) vector, which is
    needed to reproduce ``model.predict``.

    Args:
        model: A fitted ``XGBRegressor`` / ``Booster`` with vector-valued leaves.

    Returns:
        A ``(trees, base_score)`` tuple.
    """
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    model_json = json.loads(bytes(booster.save_raw(raw_format="json")).decode("utf-8"))

    learner = model_json["learner"]
    tree_dicts = learner["gradient_booster"]["model"]["trees"]
    trees = [_parse_tree(t) for t in tree_dicts]

    n_outputs = trees[0].n_outputs if trees else 1
    base_score = _parse_base_score(learner["learner_model_param"], n_outputs)
    return trees, base_score


def predict_multioutput(
    model: Any,
    X: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Convert ``model`` and predict ``X``, reproducing ``model.predict``.

    Convenience wrapper around :func:`convert_multioutput_xgboost_with_base_score` and
    :func:`_predict_forest`.

    Args:
        model: A fitted multi-output ``XGBRegressor`` / ``Booster``.
        X: A ``(n_samples, n_features)`` array of instances.

    Returns:
        A ``(n_samples, n_outputs)`` array of predictions.
    """
    trees, base_score = convert_multioutput_xgboost_with_base_score(model)
    return _predict_forest(trees, base_score, X)
