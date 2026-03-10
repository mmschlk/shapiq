"""Interventional TreeShap Explainer Implementation."""

from __future__ import annotations

from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from scipy.special import binom

from shapiq.interaction_values import InteractionValues
from shapiq.tree.validation import validate_tree_model
from shapiq.utils.modules import safe_isinstance
from shapiq.utils.sets import powerset

from .cext import (
    compute_interactions_batched_sparse,
    compute_interactions_flatten,
)
from .weight import (
    banzhaf_weight_function,
    chaining_weight_function,
    general_weight_fbii,
    interaction_weight_to_moebius_weight,
    interaction_weight_to_moebius_weight_gv,
    shapley_based_weight_function,
)

if TYPE_CHECKING:
    from shapiq.tree.base import TreeModel

INDICES_C_IMPLEMENTATION_CAPABLE = ["SV", "SII", "BII", "BV", "SV", "CHII", "CV", "FBII", "FSII", "STII"]
INDICES_CII_IMPLEMENTATION_CAPABLE = ["WBII", *INDICES_C_IMPLEMENTATION_CAPABLE]

def obtain_E_R_values(tree: TreeModel) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Obtain two arrays E and R indicating for each leaf and each feature whether the leaf is reachable by having the feature equal to 1 (E) or equal to 0 (R)."""
    E = []
    R = []
    leaf_vals = []

    # Iterative DFS — avoids Python recursion overhead and recursion-depth limits.
    # Stack entries: (node_id, e_set, r_set)
    stack = [(0, frozenset(), frozenset())]
    while stack:
        node_id, e_set, r_set = stack.pop()
        if tree.children_left[node_id] == tree.children_right[node_id]:  # leaf
            E.append(np.array(sorted(e_set), dtype=np.int64))
            R.append(np.array(sorted(r_set), dtype=np.int64))
            leaf_vals.append(tree.values[node_id].item())
            continue

        feature = int(tree.features[node_id])

        # Go left: feature must be 0 → add to R (unless already constrained to 1).
        if feature not in e_set:
            stack.append((tree.children_left[node_id], e_set, r_set | {feature}))

        # Go right: feature must be 1 → add to E (unless already constrained to 0).
        if feature not in r_set:
            stack.append((tree.children_right[node_id], e_set | {feature}, r_set))

    return E, R, leaf_vals

def obtain_E_R_values_point(tree:TreeModel, point_to_explain: np.ndarray, reference_point: np.ndarray):
    """Obtain two arrays E and R indicating for each leaf and each feature whether the leaf was reached due to features in point_to_explain (E) or due to features in reference_point (R)."""
    E = []
    R = []
    leaf_vals = []

    # Iterative DFS — avoids Python recursion overhead and recursion-depth limits.
    # Stack entries: (node_id, e_set, r_set)
    stack = [(0, frozenset(), frozenset())]
    while stack:
        node_id, e_set, r_set = stack.pop()
        if tree.children_left[node_id] == tree.children_right[node_id]:  # leaf
            E.append(np.array(sorted(e_set), dtype=np.int64))
            R.append(np.array(sorted(r_set), dtype=np.int64))
            leaf_vals.append(tree.values[node_id].item())
            continue

        feature = int(tree.features[node_id])
        explain_goes_left = tree.decision_function(
            point_to_explain[feature],
            tree.thresholds[node_id],
            tree.children_left_default[node_id],
        )
        child_node_explain = (
            tree.children_left[node_id] if explain_goes_left else tree.children_right[node_id]
        )
        ref_goes_left = tree.decision_function(
            reference_point[feature],
            tree.thresholds[node_id],
            tree.children_left_default[node_id],
        )
        child_node_ref = (
            tree.children_left[node_id] if ref_goes_left else tree.children_right[node_id]  
        )
        if child_node_explain != child_node_ref:
            if feature not in r_set: # Feature is not fixed by the reference point
                stack.append((child_node_explain, e_set | {feature}, r_set))
            if feature not in e_set: # Feature is not fixed by the explain point
                stack.append((child_node_ref, e_set, r_set | {feature}))
        else:
            stack.append((child_node_explain, e_set, r_set))
    return E, R, leaf_vals

class InterventionalTreeExplainer:
    """Interventional Tree Shap explainer for a single decision tree.

    This class implements the interventional TreeSHAP algorithm for a single
    sklearn DecisionTreeClassifier or DecisionTreeRegressor.

    Attributes.
    ----------
    tree : Any
        Validated tree structure returned by validate_tree_model for the fitted
        sklearn tree.
    data : np.ndarray
        Background dataset used to compute the reference point.
    reference_point : np.ndarray
        The baseline/reference feature values (mean over the background data).

    Methods.
    -------
    _compute_interventional_shapley_value(x, reference_points) -> np.ndarray
        Compute interventional Shapley values for one instance x given reference points.
    explain_function(x, **kwargs) -> InteractionValues
        Compute Shapley values for a single instance and return InteractionValues.
    """

    def __init__(
        self,
        model,
        data: np.ndarray,
        class_index: int | None = None,
        debug: bool = False,
        max_order: int = 2,
        index: str = "SII",
        index_func: callable | None = None,
        p: float = 0.5,
        bool_tree:bool = False,
    ) -> None:



        # If Classification model and class_index is None, set to 1
        if class_index is None and hasattr(model, "predict_proba"):
            class_index = 1
        self.tree = validate_tree_model(model, class_label=class_index)
        self.reference_data: np.ndarray = data.astype(np.float32)
        self.debug = debug
        self.max_order = max_order
        self.index = index
        self.n_players = data.shape[1]
        self.index_func = index_func
        self.p = p
        self.bool_tree = bool_tree
        if class_index is not None:
            # If XGBoost model, use DMatrix to obtain logits
            if safe_isinstance(model, "xgboost.sklearn.XGBClassifier"):
                import xgboost as xgb

                dmatrix_data = xgb.DMatrix(self.reference_data)
                logits = model.get_booster().predict(dmatrix_data, output_margin=True)
                if logits.ndim == 1:
                    # Binary classification case
                    if class_index == 1:
                        self.baseline_value = np.mean(logits).astype(np.float64)
                    else:
                        self.baseline_value = np.mean(-logits).astype(np.float64)
                else:
                    self.baseline_value = np.mean(logits[:, class_index]).astype(
                        np.float64
                    )
            else:
                self.baseline_value = np.mean(
                    model.predict_proba(self.reference_data)[:, class_index]
                ).astype(np.float64)
        else:
            # Check if we have the case of a leaf_matrix directly provided
            if isinstance(model, list):
                self.baseline_value = model[0][-1]
            else:
                self.baseline_value = np.mean(
                    model.predict(self.reference_data)
                ).astype(np.float64)
        self.values_list = [
                tree.values.astype(np.float32).flatten() for tree in self.tree
            ]
        self.threshold_list = [
                tree.thresholds.astype(np.float32).flatten() for tree in self.tree
            ]
        self.features_list = [
                tree.features.astype(np.int64).flatten() for tree in self.tree
            ]
        self.children_left_list = [
                tree.children_left.astype(np.int64).flatten() for tree in self.tree
            ]
        self.children_right_list = [
                tree.children_right.astype(np.int64).flatten() for tree in self.tree
            ]
        self.children_left_default_list = [
                tree.children_left_default.astype(np.bool).flatten()
                for tree in self.tree
            ]
        if self.bool_tree:
            self.gather_e_r_statistics_bool_tree()

    def gather_e_r_statistics_bool_tree(self):
        self.n_features = self.reference_data.shape[1]
        E_list: list[np.ndarray] = []
        R_list: list[np.ndarray] = []
        leaf_vals_list: list[float] = []

        for tree in self.tree:
            E, R, leaf_vals = obtain_E_R_values(tree)
            E_list.extend(E)
            R_list.extend(R)
            leaf_vals_list.extend(leaf_vals)

        self.E_list = E_list
        self.R_list = R_list
        self.leaf_vals = np.array(leaf_vals_list, dtype=np.float32)
        n_leafs = len(E_list)

        # Per-leaf sizes — computed once and reused for all flattened arrays.
        e_sizes = np.array([len(e) for e in E_list], dtype=np.int64)
        r_sizes = np.array([len(r) for r in R_list], dtype=np.int64)
        er_sizes = e_sizes + r_sizes

        self.n_features_e = e_sizes
        self.n_features_r = r_sizes

        # Build flatten numpy arrays
        if n_leafs > 0:
            self.E_R_flatten = np.concatenate(
                [np.concatenate([e, r]) for e, r in zip(E_list, R_list)]
            ).astype(np.int64)
            self.feature_in_E = np.concatenate(
                [np.concatenate([np.ones(int(e), dtype=np.int64), np.zeros(int(r), dtype=np.int64)])
                 for e, r in zip(e_sizes, r_sizes)]
            )
        else:
            self.E_R_flatten = np.array([], dtype=np.int64)
            self.feature_in_E = np.array([], dtype=np.int64)

        self.leaf_id = np.repeat(np.arange(n_leafs, dtype=np.int64), er_sizes)
        self.leaf_vals_flatten = np.repeat(self.leaf_vals, er_sizes)
        self.e_size_flatten = np.repeat(e_sizes, er_sizes)
        self.r_size_flatten = np.repeat(r_sizes, er_sizes)
        self.e_length = len(self.E_R_flatten)
        self.n_leafs = n_leafs

    def gather_e_r_statistics(self,explain_point):
        self.n_features = self.reference_data.shape[1]
        E_list: list[np.ndarray] = []
        R_list: list[np.ndarray] = []
        leaf_vals_list: list[float] = []


        for r in self.reference_data:
            for tree in self.tree:
                E, R, leaf_vals = obtain_E_R_values_point(tree, explain_point, r)
                E_list.extend(E)
                R_list.extend(R)
                leaf_vals_list.extend(leaf_vals)

        self.E_list = E_list
        self.R_list = R_list
        self.leaf_vals = np.array(leaf_vals_list, dtype=np.float32)
        n_leafs = len(E_list)

        # Per-leaf sizes — computed once and reused for all flattened arrays.
        e_sizes = np.array([len(e) for e in E_list], dtype=np.int64)
        r_sizes = np.array([len(r) for r in R_list], dtype=np.int64)
        er_sizes = e_sizes + r_sizes

        self.n_features_e = e_sizes
        self.n_features_r = r_sizes

        # Build flatten numpy arrays
        if n_leafs > 0:
            self.E_R_flatten = np.concatenate(
                [np.concatenate([e, r]) for e, r in zip(E_list, R_list)]
            ).astype(np.int64)
            self.feature_in_E = np.concatenate(
                [np.concatenate([np.ones(int(e), dtype=np.int64), np.zeros(int(r), dtype=np.int64)])
                 for e, r in zip(e_sizes, r_sizes)]
            )
        else:
            self.E_R_flatten = np.array([], dtype=np.int64)
            self.feature_in_E = np.array([], dtype=np.int64)

        self.leaf_id = np.repeat(np.arange(n_leafs, dtype=np.int64), er_sizes)
        self.leaf_vals_flatten = np.repeat(self.leaf_vals, er_sizes)
        self.e_size_flatten = np.repeat(e_sizes, er_sizes)
        self.r_size_flatten = np.repeat(r_sizes, er_sizes)
        self.e_length = len(self.E_R_flatten)
        self.n_leafs = n_leafs

    def general_weight_function(self, A, B, N, U, moebius_weight_func):
        """Computes a general weight for given sets A, B, N and U.

        Args:
            A: Set A.
            B: Set B.
            N: Set of all players.
            U: Current coalition.
            möbius_weight_func: Möbius weight function to use.
        Returns:
            The general weight.
        """
        u_0 = len(U.intersection(N.difference(B)))
        a = len(A)
        b = len(B)
        n = len(N)
        u = len(U)
        sign = (-1) ** u_0
        return sign * sum(
            [
                (-1) ** k
                * binom(n - b - u_0, k)
                * moebius_weight_func(coalition_size=k + u_0 + a, interaction_size=u)
                for k in range(n - b - u_0 + 1)
            ]
        )

    def update_values(
        self,
        interaction_to_values,
        const_prediction,
        A,
        B,
        NB,
        max_order,
        weight_func,
    ):
        """Updates the CII based on sets A and NB.

        Args:
            interaction_to_values: Mapping from interactions to their effects.
            const_prediction: Constant prediction value.
            A: Set A.
            B: Set B.
            NB: Set NB.
            max_order: Maximum order of interactions.
            weight_func: Weight function for the update.
        Returns:
            The updated Shapley values.
        """
        # Though A & NB there is already some filtering of which interactions could even be updated. Irrelevant features will not be part of A or NB.
        for U in powerset(A.union(NB), min_size=1, max_size=max_order):
            U = set(U)
            # Compute the weights
            weight = weight_func(A=A, B=B, N=NB.union(B), U=U)
            if self.debug:
                print("Updating interaction:", U)
                print("Weight:", weight)
                print("const_prediction:", const_prediction)
            # Update the values
            # Make U contain not numpy numbers
            U = set(map(int, U))
            U = tuple(sorted(tuple(U)))

            try:
                interaction_to_values[U] += (
                    np.array(weight * const_prediction).astype(np.float64).item()
                )
            except KeyError:
                interaction_to_values[U] = (
                    np.array(weight * const_prediction).astype(np.float64).item()
                )
        return interaction_to_values

    def _compute_interventional_cii_values(
        self,
        x: np.ndarray,
        interactions_dict: dict[tuple[int, ...], float],
        tree: TreeModel,
    ) -> dict[tuple[int, ...], float]:
        """Computes the interventional CII value for a single instance.

        Args:
            x: The instance to explain as a 1-dimensional array.
            interactions_dict: Resulting interactions dictionary.
            tree: The decision tree to explain.
        Returns:
            interventional CII values in a dictionary.
        """
        N = set(range(x.shape[0]))  # number of features
        D = self.reference_data.shape[0]  # number of reference points
        for r in self.reference_data:
            # Implement the interventional CII value computation here
            reference_point = r
            # Initialize Node Stack
            node_stack = [(0, (set(), N))]  # (node_id, (A, B))
            while node_stack:
                node_id, (A, B) = node_stack.pop()

                if self.debug:
                    print("Visiting node:", node_id)
                    print("Current A:", A)
                    print("Current B:", B)
                # Check if inner node
                is_inner_node = (
                    tree.children_left[node_id] != tree.children_right[node_id]
                )
                if is_inner_node:  # Inner Node
                    feature_index = tree.features[node_id]
                    child_node_x = (
                        tree.children_left[node_id]
                        if tree.decision_function(
                            x[feature_index],
                            tree.thresholds[node_id],
                            tree.children_left_default[node_id],
                        )
                        else tree.children_right[node_id]
                    )
                    child_node_ref = (
                        tree.children_left[node_id]
                        if tree.decision_function(
                            reference_point[feature_index],
                            tree.thresholds[node_id],
                            tree.children_left_default[node_id],
                        )
                        else tree.children_right[node_id]
                    )
                    if self.debug:
                        print(
                            f"Feature index: {feature_index}, x value: {x[feature_index]}, ref value: {reference_point[feature_index]}, threshold: {tree.thresholds[node_id]}"
                        )
                        print(
                            f"Child node x: {child_node_x}, Child node ref: {child_node_ref}"
                        )
                    # Update stack based on child nodes
                    if child_node_x == child_node_ref:
                        if self.debug:
                            print("Both go to the same child node.")
                            print("Adding to stack:", child_node_x, (A, B))
                        node_stack.append((child_node_x, (A, B)))
                    else:
                        if feature_index in B:  # Keeping Child of x
                            if self.debug:
                                print("Feature index in B, splitting the path.")
                                print(
                                    "Adding to stack:",
                                    child_node_x,
                                    (A.union({feature_index}), B),
                                )
                            node_stack.append(
                                (child_node_x, (A.union({feature_index}), B))
                            )
                        if feature_index not in A:
                            if self.debug:
                                print("Feature index not in A, splitting the path.")
                                print(
                                    "Adding to stack:",
                                    child_node_ref,
                                    (A, B.difference({feature_index})),
                                )
                            node_stack.append(
                                (child_node_ref, (A, B.difference({feature_index})))
                            )
                else:
                    # Update Shapley values based on A & B
                    NB = N.difference(B)
                    D = self.reference_data.shape[0]
                    # Compute Coalition Values [Due to linearity one could also directly use the values at the leaf node and later divide by D]
                    const_coalition = tree.values[node_id]
                    if self.debug:
                        print("-----Updating at leaf node-----")
                        print("Node ID:", node_id)
                        print("A:", A)
                        print("B:", B)
                        print("NB: ", NB)
                        print("const_coalition:", const_coalition)
                    # const_coalition /= D

                    # Obtain the necessary weight function
                    if self.index in ["SII", "SV"]:
                        weight_function = shapley_based_weight_function
                    elif self.index in ["BII", "BV"]:
                        weight_function = banzhaf_weight_function
                    elif self.index in ["CHII", "CV"]:
                        weight_function = chaining_weight_function
                    elif self.index in ["GSII", "GSV"]:
                        weight_function = partial(
                            self.general_weight_function,
                            moebius_weight_func=interaction_weight_to_moebius_weight_gv,
                        )
                    elif self.index in ["FBII"]:
                        weight_function = partial(general_weight_fbii,max_order=self.max_order)
                    else:
                        weight_function = partial(
                            self.general_weight_function,
                            moebius_weight_func=interaction_weight_to_moebius_weight,
                        )
                    self.update_values(
                            interaction_to_values=interactions_dict,
                            const_prediction=const_coalition,
                            A=A,
                            B=B,
                            NB=NB,
                            max_order=self.max_order,
                            weight_func=weight_function,
                    )
                    if self.debug:
                        print("Updated Shapley values:", interactions_dict)
                        print("-----Updating at leaf node-----")

        return interactions_dict

    def explain_function(
        self,
        x: np.ndarray,
        **_: dict,
    ) -> InteractionValues:
        """Computes the Shapley values for a single instance using interventional approach.

        Args:
            x: The instance to explain as a 1-dimensional array.
            **kwargs: Additional keyword arguments are ignored.
        """  
        if self.index not in INDICES_CII_IMPLEMENTATION_CAPABLE:
            warn(
                f"Index {self.index} not recognized. Checking if callable function was given."
            )
            if self.index_func is None:
                raise ValueError(
                    f"Index function must be provided if index {self.index} is not recognized."
                )
            print("Using custom index function provided by user.")
            interaction = self.explain_function_cii(x, **_)
            # raise ValueError(f"Index {self.index} not supported in interventional explainer.")
        elif self.index not in INDICES_C_IMPLEMENTATION_CAPABLE:
            interaction = self.explain_function_cii(x, **_)
        else:
            interaction = self.explain_function_cpp(x, **_)
        return interaction

    def explain_function_cii(
        self,
        x: np.ndarray,
        **_: dict,
    ) -> InteractionValues:
        """Computes the CII values for a single instance using interventional approach.

        Args:
            x: The instance to explain as a 1-dimensional array.
            **kwargs: Additional keyword arguments are ignored.
        """
        # Assert that x is np.float32
        x = x.astype(np.float32).flatten()

        interactions_dict: dict[tuple[int, ...], float] = {}
        for j, tree in enumerate(self.tree):
            if self.debug:
                print(f"#####Computing CII values for tree {j}#####")
            interactions = {}
            interactions = self._compute_interventional_cii_values(
                x, interactions, tree
            )
            for key, value in interactions.items():
                try:
                    interactions_dict[key] += value
                except KeyError:
                    interactions_dict[key] = value

            if self.debug:
                print(f"#####Finished Computing CII values for tree #####")
        interactions_dict[()] = self.baseline_value
        interactions_dict = dict(
            sorted(
                interactions_dict.items(),
                key=lambda item: (len(item[0]), item[0]),
            )
        )

        for key in interactions_dict.keys():
            if key != ():
                interactions_dict[key] /= self.reference_data.shape[0]

        return InteractionValues(
            interactions_dict,
            max_order=self.max_order,
            min_order=1,
            index=self.index,
            n_players=self.n_players,
            baseline_value=interactions_dict[()],
        )

    def explain_function_cpp(self, x: np.ndarray) -> dict:
        """Computes the CII values for a single instance using interventional approach with a pure C++ implementation.
        Args:
            x: The instance to explain as a 1-dimensional array.
        Returns:
            InteractionValues object containing the computed interaction values.
        """
        if not self.bool_tree:
            self.gather_e_r_statistics(x)


        interactions = {}
        # For higher order interactions we need to use the sparse implementation as the flatten one is only optimized for main effects and pairwise interactions. For main effects and pairwise interactions we can use the flatten implementation which is faster.
        if self.max_order > 2:
            interactions = compute_interactions_batched_sparse(
                self.values_list,
                self.threshold_list,
                self.features_list,
                self.children_left_list,
                self.children_right_list,
                self.children_left_default_list,
                self.reference_data.astype(np.float32),
                x.astype(np.float32).flatten(),
                self.tree[0].decision_type,
                self.index,
                self.max_order,
                self.debug,  # whether to print debug information
            )
        else:
            interactions = compute_interactions_flatten(
                self.leaf_vals_flatten,
                self.E_R_flatten,
                self.e_size_flatten,
                self.r_size_flatten,
                self.feature_in_E,
                self.leaf_id,
                self.index,
                len(self.leaf_vals_flatten),
                self.n_features,
                self.e_length,
                self.max_order,
                self.debug, # whether to print debug information
                float(self.reference_data.shape[0]), # number of reference samples for scaling the results
            )
        interactions[()] = self.baseline_value
        return InteractionValues(
            interactions,
            max_order=self.max_order,
            min_order=1,
            index=self.index,
            n_players=self.n_players,
            baseline_value=self.baseline_value,
        )