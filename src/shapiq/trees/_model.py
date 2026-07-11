"""The unified tree representation consumed by tree games."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shapiq.games._values import to_host_array

LEAF = -1


@dataclass(frozen=True)
class TreeModel:
    """One decision tree in the unified node-array layout.

    Library-specific converters (see ``to_tree_model``) target this format so
    tree games and explainers understand a single layout. Nodes are indexed
    ``0 .. n_nodes - 1`` with the root at ``0``; a node is a leaf exactly
    when both children are ``-1``. Decision nodes route a sample left when
    ``sample[feature] <= threshold``. Leaf values may be scalars or carry
    trailing value axes (class probabilities), which become the game's
    ``value_shape``; entries at decision nodes are ignored.

    Construction accepts the arrays from any backend (NumPy, JAX, torch,
    nested sequences); they are normalized to host NumPy because routing
    against ``float64`` thresholds is an exact host-side computation.
    """

    children_left: np.ndarray
    children_right: np.ndarray
    features: np.ndarray
    thresholds: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        """Normalize the arrays and validate the node layout."""
        children_left = to_host_array(self.children_left, np.int64)
        children_right = to_host_array(self.children_right, np.int64)
        features = to_host_array(self.features, np.int64)
        thresholds = to_host_array(self.thresholds, np.float64)
        values = to_host_array(self.values, np.float64)
        n_nodes = children_left.shape[0]
        for name, array in (
            ("children_right", children_right),
            ("features", features),
            ("thresholds", thresholds),
        ):
            if array.shape[:1] != (n_nodes,):
                msg = f"{name} must hold one entry per node, expected {n_nodes} entries"
                raise ValueError(msg)
        if values.shape[:1] != (n_nodes,):
            msg = f"values must hold one row per node, expected {n_nodes} rows"
            raise ValueError(msg)
        if n_nodes == 0:
            msg = "a tree needs at least one node"
            raise ValueError(msg)
        leaves = children_left == LEAF
        if not np.array_equal(leaves, children_right == LEAF):
            msg = "a node is a leaf exactly when both children are -1"
            raise ValueError(msg)
        if not leaves.any():
            msg = "a tree needs at least one leaf"
            raise ValueError(msg)
        internal = ~leaves
        if internal.any() and int(features[internal].min()) < 0:
            msg = "decision nodes must carry non-negative feature indices"
            raise ValueError(msg)
        for name, children in (("children_left", children_left), ("children_right", children_right)):
            linked = children[internal]
            if linked.size and (int(linked.min()) < 0 or int(linked.max()) >= n_nodes):
                msg = f"{name} links out of range for {n_nodes} nodes"
                raise ValueError(msg)
        object.__setattr__(self, "children_left", children_left)
        object.__setattr__(self, "children_right", children_right)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "values", values)

    @property
    def n_nodes(self) -> int:
        """Return the number of nodes."""
        return int(self.children_left.shape[0])

    @property
    def leaf_mask(self) -> np.ndarray:
        """Return the boolean leaf mask over nodes."""
        return self.children_left == LEAF

    @property
    def max_feature(self) -> int:
        """Return the largest feature index used by a decision node, or -1."""
        internal = ~self.leaf_mask
        if not internal.any():
            return -1
        return int(self.features[internal].max())

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Return the trailing value axes of the leaf values."""
        return tuple(self.values.shape[1:])


def trusted_tree_model(
    *,
    children_left: np.ndarray,
    children_right: np.ndarray,
    features: np.ndarray,
    thresholds: np.ndarray,
    values: np.ndarray,
) -> TreeModel:
    """Construct a ``TreeModel`` from already-normalized arrays, unvalidated.

    Converter-internal: the validating constructor exists to teach users
    hand-building trees, while converters guarantee the node layout by
    construction and their parity suites pin it — re-validating thousands
    of trees would only cost conversion time. Arrays must already be host
    NumPy in the normalized dtypes (``int64`` structure, ``float64``
    thresholds and values).
    """
    tree = object.__new__(TreeModel)
    object.__setattr__(tree, "children_left", children_left)
    object.__setattr__(tree, "children_right", children_right)
    object.__setattr__(tree, "features", features)
    object.__setattr__(tree, "thresholds", thresholds)
    object.__setattr__(tree, "values", values)
    return tree


def constant_tree(value: np.ndarray | float) -> TreeModel:
    """Return a lone-leaf tree contributing a constant to an ensemble sum.

    Booster converters use this to carry base scores and biases: tree games
    sum their trees, so one leaf holding the constant (a vector for
    multiclass margins) completes the model's prediction.
    """
    return TreeModel(
        children_left=np.asarray([-1]),
        children_right=np.asarray([-1]),
        features=np.asarray([-2]),
        thresholds=np.asarray([np.nan]),
        values=np.asarray([value]),
    )
