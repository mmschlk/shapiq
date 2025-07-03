"""The base class for tree model conversion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .utils import compute_empty_prediction

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class TreeModel:
    """A dataclass for storing the information of a tree model.

    The dataclass stores the information of a tree model in a way that is easy to access and
    manipulate. The dataclass is used to convert tree models from different libraries to a common
    format.

    Attributes:
        children_left: The left children of each node in a tree. Leaf nodes are ``-1``.
        children_right: The right children of each node in a tree. Leaf nodes are ``-1``.
        features: The feature indices of the decision nodes in a tree. Leaf nodes are assumed to be
            ``-2`` but no check is performed.
        thresholds: The thresholds of the decision nodes in a tree. Leaf nodes are set to ``np.NaN``.
        values: The values of the leaf nodes in a tree.
        node_sample_weight: The sample weights of the nodes in a tree.
        empty_prediction: The empty prediction of the tree model. The default value is ``None`.` Then
            the empty prediction is computed from the leaf values and the sample weights.
        leaf_mask: The boolean mask of the leaf nodes in a tree. The default value is ``None``. Then the
            leaf mask is computed from the children left and right arrays.
        n_features_in_tree: The number of features in the tree model. The default value is ``None``.
            Then the number of features in the tree model is computed from the unique feature
            indices in the features array.
        max_feature_id: The maximum feature index in the tree model. The default value is ``None``. Then
            the maximum feature index in the tree model is computed from the features array.
        feature_ids: The feature indices of the decision nodes in the tree model. The default value
            is ``None``. Then the feature indices of the decision nodes in the tree model are computed
            from the unique feature indices in the features array.
        root_node_id: The root node id of the tree model. The default value is ``None``. Then the root
            node id of the tree model is set to ``0``.
        n_nodes: The number of nodes in the tree model. The default value is ``None``. Then the number
            of nodes in the tree model is computed from the children left array.
        nodes: The node ids of the tree model. The default value is ``None``. Then the node ids of the
            tree model are computed from the number of nodes in the tree model.
        feature_map_original_internal: A mapping of feature indices from the original feature
            indices (as in the model) to the internal feature indices (as in the tree model).
        feature_map_internal_original: A mapping of feature indices from the internal feature
            indices (as in the tree model) to the original feature indices (as in the model).
        original_output_type: The original output type of the tree model. The default value is
            ``"raw"``.

    """

    children_left: NDArray[np.int_]
    children_right: NDArray[np.int_]
    features: NDArray[np.int_]
    thresholds: NDArray[np.floating]
    values: NDArray[np.floating]
    node_sample_weight: NDArray[np.floating]
    empty_prediction: float = None  # type: ignore[assignment]
    leaf_mask: NDArray[np.bool_] = None  # type: ignore[assignment]
    n_features_in_tree: int = None  # type: ignore[assignment]
    max_feature_id: int = None  # type: ignore[assignment]
    feature_ids: set = None  # type: ignore[assignment]
    root_node_id: int = None  # type: ignore[assignment]
    n_nodes: int = None  # type: ignore[assignment]
    nodes: NDArray[np.int_] = None  # type: ignore[assignment]
    feature_map_original_internal: dict[int, int] = None  # type: ignore[assignment]
    feature_map_internal_original: dict[int, int] = None  # type: ignore[assignment]
    original_output_type: str = "raw"  # not used at the moment

    def compute_empty_prediction(self) -> None:
        """Compute the empty prediction of the tree model.

        The method computes the empty prediction of the tree model by taking the weighted average of
        the leaf node values. The method modifies the tree model in place.
        """
        self.empty_prediction = compute_empty_prediction(
            self.values[self.leaf_mask],
            self.node_sample_weight[self.leaf_mask],
        )

    def __post_init__(self) -> None:
        """Clean-up after the initialization of the TreeModel dataclass.

        The method sets up the tree model with the information provided in the constructor.
        """
        # setup leaf mask
        if self.leaf_mask is None:
            self.leaf_mask = np.asarray(self.children_left == -1)
        # sanitize features
        self.features = np.where(self.leaf_mask, -2, self.features)
        self.features = self.features.astype(int)  # make features integer type
        # sanitize thresholds
        self.thresholds = np.where(self.leaf_mask, np.nan, self.thresholds)
        # setup empty prediction
        if self.empty_prediction is None:
            self.compute_empty_prediction()
        unique_features = set(np.unique(self.features))
        unique_features.discard(-2)  # remove leaf node "features"
        # setup number of features
        if self.n_features_in_tree is None:
            self.n_features_in_tree = int(len(unique_features))
        # setup max feature id
        if self.max_feature_id is None:
            self.max_feature_id = max(unique_features)
        # setup feature names
        if self.feature_ids is None:
            self.feature_ids = unique_features
        # setup root node id
        if self.root_node_id is None:
            self.root_node_id = 0
        # setup number of nodes
        if self.n_nodes is None:
            self.n_nodes = len(self.children_left)
        # setup nodes
        if self.nodes is None:
            self.nodes = np.arange(self.n_nodes)
        # setup original feature mapping
        if self.feature_map_original_internal is None:
            self.feature_map_original_internal = {i: i for i in unique_features}
        # setup new feature mapping
        if self.feature_map_internal_original is None:
            self.feature_map_internal_original = {i: i for i in unique_features}
        # flatten values if necessary
        if self.values.ndim > 1:
            if self.values.shape[1] != 1:
                msg = "Values array has more than one column."
                raise ValueError(msg)
            self.values = self.values.flatten()
        # set all values of non leaf nodes to zero
        self.values[~self.leaf_mask] = 0

    def reduce_feature_complexity(self) -> None:
        """Reduces the feature complexity of the tree model.

        The method reduces the feature complexity of the tree model by removing unused features and
        reindexing the feature indices of the decision nodes in the tree. The method modifies the
        tree model in place. To see the original feature mappings, use the ``feature_mapping_old_new``
        and ``feature_mapping_new_old`` attributes.

        For example, consider a tree model with the following feature indices:

            [0, 1, 8]

        The method will remove the unused feature indices and reindex the feature indices of the
        decision nodes in the tree to the following:

            [0, 1, 2]

        Feature ``'8'`` is 'renamed' to ``'2'`` such that in the internal representation a one-hot vector
        (and matrices) of length ``3`` suffices to represent the feature indices.
        """
        if self.n_features_in_tree < self.max_feature_id + 1:
            new_feature_ids = set(range(self.n_features_in_tree))
            mapping_old_new = {old_id: new_id for new_id, old_id in enumerate(self.feature_ids)}
            mapping_new_old = dict(enumerate(self.feature_ids))
            new_features = np.zeros_like(self.features)
            for i, old_feature in enumerate(self.features):
                new_value = -2 if old_feature == -2 else mapping_old_new[old_feature]
                new_features[i] = new_value
            self.features = new_features
            self.feature_ids = new_feature_ids
            self.feature_map_original_internal = mapping_old_new
            self.feature_map_internal_original = mapping_new_old
            self.n_features_in_tree = len(new_feature_ids)
            self.max_feature_id = self.n_features_in_tree - 1

    def predict_one(self, x: np.ndarray) -> float:
        """Predicts the output of a single instance.

        Args:
            x: The instance to predict as a 1-dimensional array.

        Returns:
            The prediction of the instance with the tree model.

        """
        node = self.root_node_id
        is_leaf = self.leaf_mask[node]
        while not is_leaf:
            feature_id_internal = self.features[node]
            feature_id_original = self.feature_map_internal_original[feature_id_internal]
            if x[feature_id_original] <= self.thresholds[node]:
                node = self.children_left[node]
            else:
                node = self.children_right[node]
            is_leaf = self.leaf_mask[node]
        return float(self.values[node])


@dataclass
class EdgeTree:
    """A dataclass for storing the information of an edge representation of the tree.

    The dataclass stores the information of an edge representation of the tree in a way that is easy
    to access and manipulate for the TreeSHAP-IQ algorithm.
    """

    parents: np.ndarray
    ancestors: np.ndarray
    ancestor_nodes: dict[int, np.ndarray]
    p_e_values: np.ndarray
    p_e_storages: np.ndarray
    split_weights: np.ndarray
    empty_predictions: np.ndarray
    edge_heights: np.ndarray
    max_depth: int
    last_feature_node_in_path: np.ndarray
    interaction_height_store: dict[int, np.ndarray]
    has_ancestors: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Clean-up after the initialization of the EdgeTree dataclass."""
        # setup has ancestors
        if self.has_ancestors is None:
            self.has_ancestors = self.ancestors > -1
