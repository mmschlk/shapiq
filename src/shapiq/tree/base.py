"""The base class for tree model conversion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .utils import compute_empty_prediction

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TreeModel:
    """Internal representation of a single tree used by the shapiq tree explainers.

    Each library-specific converter (scikit-learn, XGBoost, LightGBM, CatBoost) targets this
    common format so that the downstream algorithms (TreeSHAP-IQ, LinearTreeSHAP,
    InterventionalTreeSHAP) only need to understand one node-array layout.

    Constructor arguments that fall back to a computed default when ``None`` is passed are
    documented on :meth:`__init__`. The attributes below describe what is available on a fully
    initialized instance.

    Attributes:
        children_left: The left children of each node in a tree. Leaf nodes are ``-1``.
        children_right: The right children of each node in a tree. Leaf nodes are ``-1``.
        children_missing: The child each node routes missing-value samples to. Used together with
            ``children_left`` to derive ``children_left_default`` during ``__init__``.
        children_left_default: Boolean mask. ``True`` at index ``i`` if missing-value samples at
            node ``i`` are routed to ``children_left[i]``. Derived from ``children_missing``.
        features: The feature indices of the decision nodes in a tree. Leaf nodes are ``-2``.
        thresholds: The thresholds of the decision nodes in a tree. Leaf nodes are ``np.nan``.
        values: The leaf-node values, flattened to a 1-D array. Non-leaf nodes are set to ``0``.
        node_sample_weight: The sample weights of the nodes in a tree.
        empty_prediction: The empty prediction of the tree model (weighted mean of leaf values).
        leaf_mask: The boolean mask of the leaf nodes in a tree.
        n_features_in_tree: The number of distinct features actually used by decision nodes.
        max_feature_id: The maximum feature index used by any decision node (or ``0`` if none).
        feature_ids: The set of feature indices used by decision nodes.
        root_node_id: The root node id of the tree model. Defaults to ``0``.
        n_nodes: The number of nodes in the tree model.
        decision_type: The split comparison used by :meth:`decision_function`. Either ``"<="``
            (default) or ``"<"``.
        nodes: The node ids of the tree model as ``np.arange(n_nodes)``.
        feature_map_original_internal: Mapping of feature indices from the original feature
            indices (as in the model) to the internal feature indices (as in the tree model).
        feature_map_internal_original: Mapping of feature indices from the internal feature
            indices (as in the tree model) to the original feature indices (as in the model).
        original_output_type: The original output type of the tree model. Defaults to ``"raw"``.
            Currently not used by downstream algorithms.
        intercepts: Per-leaf intercept terms for linear-leaf tree models. Currently unused.
        coeffs: Per-leaf coefficient vectors for linear-leaf tree models. Currently unused.

    """

    children_left: NDArray[np.int_]
    children_right: NDArray[np.int_]
    children_missing: NDArray[np.int_]
    features: NDArray[np.int_]
    thresholds: NDArray[np.floating]
    values: NDArray[np.floating]
    node_sample_weight: NDArray[np.floating]
    children_left_default: NDArray[np.bool_]
    empty_prediction: float
    leaf_mask: NDArray[np.bool_]
    n_features_in_tree: int
    max_feature_id: int
    feature_ids: set[int]
    root_node_id: int
    n_nodes: int
    decision_type: str
    nodes: NDArray[np.int_]
    feature_map_original_internal: dict[int, int]
    feature_map_internal_original: dict[int, int]
    original_output_type: str = "raw"  # not used at the moment
    intercepts: NDArray[np.floating]
    coeffs: NDArray[np.floating]

    def __init__(
        self,
        children_left: NDArray[np.int_],
        children_right: NDArray[np.int_],
        children_missing: NDArray[np.int_],
        features: NDArray[np.int_],
        thresholds: NDArray[np.floating],
        values: NDArray[np.floating],
        node_sample_weight: NDArray[np.floating],
        empty_prediction: float | None = None,
        leaf_mask: NDArray[np.bool_] | None = None,
        n_features_in_tree: int | None = None,
        max_feature_id: int | None = None,
        feature_ids: set[int] | None = None,
        root_node_id: int | None = None,
        n_nodes: int | None = None,
        nodes: NDArray[np.int_] | None = None,
        decision_type: str | None = None,
        feature_map_original_internal: dict[int, int] | None = None,
        feature_map_internal_original: dict[int, int] | None = None,
        original_output_type: str = "raw",  # noqa: ARG002
        intercepts: NDArray[np.floating] | None = None,  # noqa: ARG002
        coeffs: NDArray[np.floating] | None = None,  # noqa: ARG002
    ) -> None:
        """Initialize the :class:`TreeModel`.

        All numpy-array arguments must share a common node ordering. Arguments listed as
        ``None``-able fall back to a value computed from the mandatory arrays.

        Args:
            children_left: Left-child node ids; ``-1`` denotes a leaf.
            children_right: Right-child node ids; ``-1`` denotes a leaf.
            children_missing: Node id to which missing-value samples are routed.
            features: Decision-node feature indices. Leaf positions are sanitized to ``-2``.
            thresholds: Decision-node thresholds. Leaf positions are sanitized to ``np.nan``.
            values: Leaf-node values. Higher-dim arrays are flattened to 1-D; non-leaf positions
                are forced to ``0``.
            node_sample_weight: Per-node sample weights. ``NaN`` at leaves is replaced with ``1``.
            empty_prediction: Pre-computed empty prediction. ``None`` triggers
                :meth:`compute_empty_prediction`.
            leaf_mask: Boolean mask of leaf nodes. ``None`` derives it from ``children_left == -1``.
            n_features_in_tree: Number of distinct features used by decision nodes. ``None``
                derives it from the unique values in ``features`` (excluding ``-2``).
            max_feature_id: Largest feature index used. ``None`` derives it from ``features``.
            feature_ids: Set of feature indices used by decision nodes. ``None`` derives it
                from ``features``.
            root_node_id: Root node id. ``None`` defaults to ``0``.
            n_nodes: Number of nodes. ``None`` derives it from ``len(children_left)``.
            nodes: Node-id array. ``None`` defaults to ``np.arange(n_nodes)``.
            decision_type: Split comparison used by :meth:`decision_function` (``"<="`` or ``"<"``).
                ``None`` defaults to ``"<="``.
            feature_map_original_internal: Mapping from original to internal feature indices.
                ``None`` defaults to the identity mapping on ``feature_ids``.
            feature_map_internal_original: Mapping from internal to original feature indices.
                ``None`` defaults to the identity mapping on ``feature_ids``.
            original_output_type: Currently unused; accepted for forward compatibility.
            intercepts: Currently unused; accepted for forward compatibility with linear-leaf
                trees.
            coeffs: Currently unused; accepted for forward compatibility with linear-leaf trees.
        """
        self.children_left = children_left
        self.children_right = children_right
        self.children_missing = children_missing
        # Set children_missing to 1 if equal to children_left else 0 if equal to children_right
        self.children_left_default = self.children_missing == self.children_left
        self.features = features
        self.thresholds = thresholds
        self.values = values
        self.node_sample_weight = node_sample_weight
        # setup leaf mask
        if leaf_mask is None:
            self.leaf_mask = np.asarray(self.children_left == -1)
        else:
            self.leaf_mask = leaf_mask
        # sanitize features
        self.features = np.where(self.leaf_mask, -2, self.features)
        self.features = self.features.astype(int)  # make features integer type
        # sanitize thresholds
        self.thresholds = np.where(self.leaf_mask, np.nan, self.thresholds)
        #  sanitize node sample weights
        self.node_sample_weight[self.leaf_mask] = np.where(
            np.isnan(self.node_sample_weight[self.leaf_mask]),
            1.0,
            self.node_sample_weight[self.leaf_mask],
        )
        # setup empty prediction
        if empty_prediction is None:
            self.compute_empty_prediction()
        else:
            self.empty_prediction = empty_prediction
        unique_features = set(np.unique(self.features))
        unique_features.discard(-2)  # remove leaf node "features"
        # setup number of features
        if n_features_in_tree is None:
            self.n_features_in_tree = len(unique_features)
        else:
            self.n_features_in_tree = n_features_in_tree
        # setup max feature id
        if max_feature_id is None and len(unique_features) > 0:
            self.max_feature_id = max(unique_features)
        elif max_feature_id is not None:
            self.max_feature_id = max_feature_id
        else:
            self.max_feature_id = 0
        # setup feature names
        if feature_ids is None:
            self.feature_ids = unique_features
        else:
            self.feature_ids = feature_ids
        # setup root node id
        if root_node_id is None:
            self.root_node_id = 0
        else:
            self.root_node_id = root_node_id
        # setup number of nodes
        if n_nodes is None:
            self.n_nodes = len(self.children_left)
        else:
            self.n_nodes = n_nodes
        # setup nodes
        if nodes is None:
            self.nodes = np.arange(self.n_nodes)
        else:
            self.nodes = nodes
        # setup original feature mapping
        if feature_map_original_internal is None:
            self.feature_map_original_internal = {i: i for i in unique_features}
        else:
            self.feature_map_original_internal = feature_map_original_internal
        # setup new feature mapping
        if feature_map_internal_original is None:
            self.feature_map_internal_original = {i: i for i in unique_features}
        else:
            self.feature_map_internal_original = feature_map_internal_original

        # flatten values if necessary
        if self.values.ndim > 1:
            if self.values.shape[1] != 1:
                msg = "Values array has more than one column."
                raise ValueError(msg)
            self.values = self.values.flatten()
        # set all values of non leaf nodes to zero
        self.values[~self.leaf_mask] = 0

        # Set decision function
        self.decision_type = decision_type if decision_type is not None else "<="

    def decision_function(self, value: float, threshold: float, *, left_default: bool) -> bool:
        """Decision function for split nodes.

        The function compares the input value to the threshold using the specified decision type.
        If the value is NaN, the function returns the left_default.

        Args:
            value: The feature value to compare.
            threshold: The threshold to compare the feature value against.
            left_default: The default direction to take if the value is NaN. True for left, False for right.

        Returns:
            A boolean indicating whether to go left (True) or right (False) at the split node.
        """
        if self.decision_type == "<":
            return (value < threshold) if not np.isnan(value) else left_default
        return (value <= threshold) if not np.isnan(value) else left_default

    def compute_empty_prediction(self) -> None:
        """Compute the empty prediction of the tree model.

        The method computes the empty prediction of the tree model by taking the weighted average of
        the leaf node values. The method modifies the tree model in place.
        """
        try:
            self.empty_prediction = compute_empty_prediction(
                self.values[self.leaf_mask],
                self.node_sample_weight[self.leaf_mask],
            )
        except Exception as e:
            msg = f"Could not compute empty prediction: {e}"
            raise ValueError(msg) from e

    def __post_init__(self) -> None:
        """No-op hook retained for forward compatibility.

        Kept so existing callers (e.g. dataclass-aware subclasses) can still invoke it; all
        initialization happens in :meth:`__init__`.
        """

    def reduce_feature_complexity(self) -> None:
        """Reduces the feature complexity of the tree model.

        The method reduces the feature complexity of the tree model by removing unused features and
        reindexing the feature indices of the decision nodes in the tree. The method modifies the
        tree model in place. To see the original feature mappings after the reduction, use the
        ``feature_map_original_internal`` and ``feature_map_internal_original`` attributes.

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

    def predict_one(self, x: NDArray[np.floating]) -> float:
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
            if self.decision_function(
                x[feature_id_original],
                self.thresholds[node],
                left_default=self.children_left_default[node],
            ):
                node = self.children_left[node]
            else:
                node = self.children_right[node]
            is_leaf = self.leaf_mask[node]
        return float(self.values[node])


class EdgeTree:
    """Edge-based representation of a tree used by the TreeSHAP-IQ algorithm.

    Built from a :class:`TreeModel` via :func:`~shapiq.tree.conversion.edges.create_edge_tree`,
    this structure pre-computes the per-edge quantities that TreeSHAP-IQ needs to traverse the
    tree only once per explained instance.

    Attributes:
        parents: Parent node id for each node (root parent is ``-1``).
        ancestors: For each node, the id of the closest ancestor that splits on the same feature
            (``-1`` if no such ancestor exists).
        ancestor_nodes: Mapping ``node_id -> per-feature ancestor id array`` for non-root nodes.
        p_e_values: Per-edge probability factors used by the summary polynomial.
        p_e_storages: Cached storage of ``p_e`` values along each path.
        split_weights: Per-edge split weights (fraction of samples taking each branch).
        empty_predictions: Per-leaf contribution to the empty prediction.
        edge_heights: Per-node edge height used by the Chebyshev interpolation.
        max_depth: Maximum depth of the tree.
        last_feature_node_in_path: Per-node id of the last decision node along the path that
            split on the same feature.
        interaction_height_store: Mapping ``order -> per-node interaction-height array`` used to
            decide which interactions a node contributes to.
        has_ancestors: Boolean mask; ``True`` at node ``i`` if ``ancestors[i] != -1``.
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
    has_ancestors: np.ndarray

    def __init__(
        self,
        parents: np.ndarray,
        ancestors: np.ndarray,
        ancestor_nodes: dict[int, np.ndarray],
        p_e_values: np.ndarray,
        p_e_storages: np.ndarray,
        split_weights: np.ndarray,
        empty_predictions: np.ndarray,
        edge_heights: np.ndarray,
        max_depth: int,
        last_feature_node_in_path: np.ndarray,
        interaction_height_store: dict[int, np.ndarray],
        *,
        has_ancestors: np.ndarray | None = None,
    ) -> None:
        """Initialize an :class:`EdgeTree` from pre-computed per-node / per-edge arrays.

        See the class docstring for the meaning of each attribute. ``has_ancestors`` is derived
        from ``ancestors > -1`` when not supplied.
        """
        self.parents = parents
        self.ancestors = ancestors
        self.ancestor_nodes = ancestor_nodes
        self.p_e_values = p_e_values
        self.p_e_storages = p_e_storages
        self.split_weights = split_weights
        self.empty_predictions = empty_predictions
        self.edge_heights = edge_heights
        self.max_depth = max_depth
        self.last_feature_node_in_path = last_feature_node_in_path
        self.interaction_height_store = interaction_height_store
        if has_ancestors is None:
            self.has_ancestors = self.ancestors > -1
        else:
            self.has_ancestors = has_ancestors
