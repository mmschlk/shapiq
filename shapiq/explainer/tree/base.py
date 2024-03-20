"""This module contains the base class for tree model conversion."""
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .utils import compute_empty_prediction


@dataclass
class TreeModel:
    """A dataclass for storing the information of a tree model.

    The dataclass stores the information of a tree model in a way that is easy to access and
    manipulate. The dataclass is used to convert tree models from different libraries to a common
    format.

    Attributes:
        children_left: The left children of each node in a tree. Leaf nodes are -1.
        children_right: The right children of each node in a tree. Leaf nodes are -1.
        features: The feature indices of the decision nodes in a tree. Leaf nodes are assumed to be
            -2 but no check is performed.
        thresholds: The thresholds of the decision nodes in a tree. Leaf nodes are set to NaN.
        values: The values of the leaf nodes in a tree.
        node_sample_weight: The sample weights of the nodes in a tree.
        empty_prediction: The empty prediction of the tree model. The default value is None. Then
            the empty prediction is computed from the leaf values and the sample weights.
        leaf_mask: The boolean mask of the leaf nodes in a tree. The default value is None. Then the
            leaf mask is computed from the children left and right arrays.
    """

    children_left: np.ndarray[int]
    children_right: np.ndarray[int]
    features: np.ndarray[int]
    thresholds: np.ndarray[float]
    values: np.ndarray[float]
    node_sample_weight: np.ndarray[float]
    empty_prediction: Optional[float] = None
    leaf_mask: Optional[np.ndarray[bool]] = None
    n_features_in_tree: Optional[int] = None
    max_feature_id: Optional[int] = None
    feature_ids: Optional[set] = None
    root_node_id: Optional[int] = None
    n_nodes: Optional[int] = None
    nodes: Optional[np.ndarray[int]] = None
    feature_mapping_old_new: Optional[dict] = None
    feature_mapping_new_old: Optional[dict] = None

    def __getitem__(self, item) -> Any:
        return getattr(self, item)

    def __post_init__(self) -> None:
        # setup leaf mask
        if self.leaf_mask is None:
            self.leaf_mask = np.asarray(self.children_left == -1)
        # sanitize features
        self.features = np.where(self.leaf_mask, -2, self.features)
        # sanitize thresholds
        self.thresholds = np.where(self.leaf_mask, np.nan, self.thresholds)
        # setup empty prediction
        if self.empty_prediction is None:
            self.empty_prediction = compute_empty_prediction(
                self.values[self.leaf_mask], self.node_sample_weight[self.leaf_mask]
            )
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
        if self.feature_mapping_old_new is None:
            self.feature_mapping_old_new = {i: i for i in unique_features}
        # setup new feature mapping
        if self.feature_mapping_new_old is None:
            self.feature_mapping_new_old = {i: i for i in unique_features}

    def reduce_feature_complexity(self) -> None:
        """Reduces the feature complexity of the tree model.

        The method reduces the feature complexity of the tree model by removing unused features and
        reindexing the feature indices of the decision nodes in the tree. The method modifies the
        tree model in place. To see the original feature mappings, use the `feature_mapping_old_new`
        and `feature_mapping_new_old` attributes.

        For example, consider a tree model with the following feature indices:

            [0, 1, 8]

        The method will remove the unused feature indices and reindex the feature indices of the
        decision nodes in the tree to the following:

            [0, 1, 2]

        Feature '8' is 'renamed' to '2' such that in the internal representation a one-hot vector
        (and matrices) of length 3 suffices to represent the feature indices.
        """
        if self.n_features_in_tree < self.max_feature_id + 1:
            new_feature_ids = set(range(self.n_features_in_tree))
            mapping_old_new = {old_id: new_id for new_id, old_id in enumerate(self.feature_ids)}
            mapping_new_old = {new_id: old_id for new_id, old_id in enumerate(self.feature_ids)}
            new_features = np.zeros_like(self.features)
            for i, old_feature in enumerate(self.features):
                new_value = -2 if old_feature == -2 else mapping_old_new[old_feature]
                new_features[i] = new_value
            self.features = new_features
            self.feature_ids = new_feature_ids
            self.feature_mapping_old_new = mapping_old_new
            self.feature_mapping_new_old = mapping_new_old
            self.n_features_in_tree = len(new_feature_ids)
            self.max_feature_id = self.n_features_in_tree - 1


@dataclass
class EdgeTree:
    """A dataclass for storing the information of an edge representation of the tree.

    The dataclass stores the information of an edge representation of the tree in a way that is easy
    to access and manipulate for the TreeSHAP-IQ algorithm.
    """

    parents: np.ndarray[int]
    ancestors: np.ndarray[int]
    ancestor_nodes: dict[int, np.ndarray[int]]
    p_e_values: np.ndarray[float]
    p_e_storages: np.ndarray[float]
    split_weights: np.ndarray[float]
    empty_predictions: np.ndarray[float]
    edge_heights: np.ndarray[int]
    max_depth: int
    last_feature_node_in_path: np.ndarray[int]
    interaction_height_store: dict[int, np.ndarray[int]]
    has_ancestors: Optional[np.ndarray[bool]] = None

    def __getitem__(self, item) -> Any:
        return getattr(self, item)

    def __post_init__(self) -> None:
        # setup has ancestors
        if self.has_ancestors is None:
            self.has_ancestors = self.ancestors > -1
