"""This module contains the base class for tree model conversion."""
from typing import Optional
from dataclasses import dataclass

import numpy as np

from explainer.tree.utils import compute_empty_prediction


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
    n_features: Optional[int] = None
    root_node_id: Optional[int] = None

    def __getitem__(self, item):
        return getattr(self, item)

    def __post_init__(self):
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
        # setup number of features
        # TODO: only features present in the tree should be counted (requires a mask/lookup table). this is a temporary solution
        if self.n_features is None:
            self.n_features = int(np.max(self.features)) + 1
        # setup root node id
        if self.root_node_id is None:
            self.root_node_id = 0


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

    def __getitem__(self, item):
        return getattr(self, item)

    def __post_init__(self):
        # setup has ancestors
        if self.has_ancestors is None:
            self.has_ancestors = self.ancestors > -1
