"""Most of the code is adapted from the pltreeshap package."""


import json
import numpy as np


def fullname(obj):
    """Returns the name of the class of an object including the module name."""

    module = obj.__class__.__module__
    if module is None or module == str.__class__.__module__:
        return obj.__class__.__name__
    return module + "." + obj.__class__.__name__


def _from_treeshap(explainer):
    """An auxiliary function for converting models stored by shap.TreeExplainer"""
    return {"trees": [tree.__dict__ for tree in explainer.model.trees]}


def _from_xgboost(model):
    """An auxiliary function for converting XGBoost models."""

    if hasattr(model, "get_booster"):
        model = model.get_booster()
    feature_names = model.feature_names
    model = model.get_dump(with_stats=True, dump_format="json")
    trees = []
    for tree_dump in model:
        json_tree = json.loads(tree_dump)
        children_left = []
        children_right = []
        children_default = []
        features = []
        thresholds = []
        values = []
        node_sample_weight = []
        idx_node = 0
        stack = [
            (json_tree, -1, True, True)
        ]  # stack elements: (node, idx_parent, is_left_child, is_default_child)
        while len(stack) > 0:
            node, idx_parent, is_left_child, is_default_child = stack.pop()

            # extend lists by one element:
            children_left.append(-1)
            children_right.append(-1)
            children_default.append(-1)
            features.append(-1)
            thresholds.append(0.0)
            if "cover" in node:
                node_sample_weight.append(float(node["cover"]))
            else:
                node_sample_weight.append(0.0)

            if idx_parent >= 0:
                if is_left_child:
                    children_left[idx_parent] = idx_node
                else:
                    children_right[idx_parent] = idx_node
                if is_default_child:
                    children_default[idx_parent] = idx_node
            if "split" in node:
                values.append(0.0)
                if feature_names is None:
                    try:
                        features[idx_node] = int(node["split"])
                    except ValueError:
                        features[idx_node] = int(
                            node["split"][1:]
                        )  # remove leading 'f'
                else:
                    features[idx_node] = feature_names.index(node["split"])
                if "missing" in node:
                    thresholds[idx_node] = node["split_condition"]
                    idx_left = node["yes"]
                    idx_missing = node["missing"]
                else:  # absence of 'missing' indicates boolean condition
                    thresholds[idx_node] = 0.5
                    idx_left = node["no"]
                    idx_missing = node["no"]
                for child in node["children"]:
                    idx_child = child["nodeid"]
                    stack.append(
                        (
                            child,
                            idx_node,
                            idx_child == idx_left,
                            idx_child == idx_missing,
                        )
                    )
            else:
                values.append(node["leaf"])
            idx_node += 1
        tree = {
            "children_left": np.array(children_left, dtype=np.int32),
            "children_right": np.array(children_right, dtype=np.int32),
            "children_default": np.array(children_default, dtype=np.int32),
            "features": np.array(features, dtype=np.int32),
            "thresholds": np.array(thresholds, dtype=np.float64),
            "values": np.array(values, dtype=np.float64).reshape((-1, 1)),
            "node_sample_weight": np.array(node_sample_weight, dtype=np.float64),
            "decision_type": "<",
        }
        trees.append(tree)
    return {"trees": trees}


def _from_lightgbm(model):
    """An auxiliary function for converting LightGBM models."""

    if hasattr(model, "booster_"):
        model = model.booster_
    is_linear_tree = model.params.get("linear_tree", False)
    dump = model.dump_model()
    num_features = len(dump["feature_names"])
    trees = []
    for tree_dump in dump["tree_info"]:
        decision_types = []
        num_nodes = 2 * tree_dump["num_leaves"] - 1
        children_left = np.full((num_nodes,), -1, dtype=np.int32)
        children_right = np.full((num_nodes,), -1, dtype=np.int32)
        children_default = np.full((num_nodes,), -1, dtype=np.int32)
        features = np.full((num_nodes,), -1, dtype=np.int32)
        thresholds = np.zeros((num_nodes,))
        values = np.zeros((num_nodes, 1))
        node_sample_weight = np.zeros((num_nodes,))
        if is_linear_tree:
            intercepts = np.zeros((num_nodes,))
            coeffs = np.zeros((num_nodes, num_features))

        stack = [(tree_dump["tree_structure"], -1, True, True)]
        idx_node = 0
        while len(stack) > 0:
            node, idx_parent, is_left, is_default = stack.pop()
            if idx_parent >= 0:
                if is_left:
                    children_left[idx_parent] = idx_node
                else:
                    children_right[idx_parent] = idx_node
                if is_default:
                    children_default[idx_parent] = idx_node
            if "split_index" in node:
                features[idx_node] = node["split_feature"]
                try:
                    thresholds[idx_node] = node["threshold"]
                except ValueError:
                    raise ValueError(
                        "Models with categorical splits are not supported."
                    )
                node_sample_weight[idx_node] = node["internal_count"]
                values[idx_node, 0] = node["internal_value"]
                stack.append((node["left_child"], idx_node, True, node["default_left"]))
                stack.append(
                    (node["right_child"], idx_node, False, not node["default_left"])
                )
            else:
                values[idx_node, 0] = node["leaf_value"]
                node_sample_weight[idx_node] = node["leaf_count"]
                if is_linear_tree:
                    intercepts[idx_node] = node["leaf_const"]
                    coeffs[idx_node, node["leaf_features"]] = node["leaf_coeff"]
            if "decision_type" in node:
                decision_types.append(node["decision_type"])
            idx_node += 1
        tree = {
            "children_left": children_left,
            "children_right": children_right,
            "children_default": children_default,
            "features": features,
            "thresholds": thresholds,
            "values": values,
            "node_sample_weight": node_sample_weight,
        }
        decision_type = np.unique(decision_types)
        if len(decision_type) == 1:
            tree["decision_type"] = decision_type[0]
        elif len(decision_type) > 1:
            raise ValueError("Tree contains different decision types!")
        if is_linear_tree:
            tree["intercepts"] = intercepts
            tree["coeffs"] = coeffs
        trees.append(tree)
    return {"trees": trees}


def _from_sf_modeltree(model, logit_one=1e3):
    """An auxiliary function for converting ModelTrees (https://github.com/schufa-innovationlab/model-trees).

    If `model` is of type `ModelTreeClassifier`, then the linear part (logit) of the logistic regression is exported.
    Some leaves may hold a DummyClassifier for a single class. In that case, the logit is -inf or inf. For numerical reasons,
    these values are replaced by -logit_one or logit_one, respectively. The default value is 1000.
    """

    if not fullname(model.base_estimator) in [
        "sklearn.linear_model._base.LinearRegression",
        "sklearn.linear_model._logistic.LogisticRegression",
    ]:
        raise ValueError("The base estimator should be a linear model!")

    num_features = model.n_features_
    coeffs_zeros = np.zeros((num_features,))
    children_left = []
    children_right = []
    features = []
    thresholds = []
    values = []
    node_sample_weight = []
    intercepts = []
    coeffs = []

    idx_node = 0  # for enumerating nodes as they are put on the stack
    stack = [
        (model.root_, -1, True)
    ]  # stack elements: (node, idx_parent, is_left_child)
    while len(stack) > 0:
        node, idx_parent, is_left_child = stack.pop()

        # extend lists by one element:
        children_left.append(-1)
        children_right.append(-1)
        features.append(-1)
        thresholds.append(0.0)
        values.append(float("nan"))
        node_sample_weight.append(0.0)

        if idx_parent >= 0:
            if is_left_child:
                children_left[idx_parent] = idx_node
            else:
                children_right[idx_parent] = idx_node
        if node.split is not None:
            features[idx_node] = node.split.split_feature
            thresholds[idx_node] = node.split.split_threshold
        if node.children is not None:
            coeffs.append(coeffs_zeros)
            intercepts.append(0.0)
            node_l, node_r = node.children
            stack.append((node_l, idx_node, True))
            stack.append((node_r, idx_node, False))
        else:
            estimator_name = fullname(node.estimator)
            if estimator_name == "sklearn.dummy.DummyClassifier":
                coeffs.append(coeffs_zeros)
                if node.estimator.classes_[0] == 0:
                    intercepts.append(-logit_one)
                else:
                    intercepts.append(logit_one)
            elif estimator_name == "sklearn.linear_model._logistic.LogisticRegression":
                coeffs.append(
                    node.estimator.coef_[0, :]
                )  # an (1,num_features) array (if 2 classes)
                intercepts.append(
                    node.estimator.intercept_[0]
                )  # an (1,) array (if 2 classes)
            elif estimator_name == "sklearn.linear_model._base.LinearRegression":
                coeffs.append(node.estimator.coef_)
                intercepts.append(node.estimator.intercept_)
            else:
                raise ValueError(f"Unsupported estimator: {estimator_name}")
        idx_node += 1
    tree = {
        "children_left": np.array(children_left, dtype=np.int32),
        "children_right": np.array(children_right, dtype=np.int32),
        "children_default": np.array(children_left, dtype=np.int32),
        "features": np.array(features, dtype=np.int32),
        "thresholds": np.array(thresholds, dtype=np.float64),
        "values": np.array(values, dtype=np.float64).reshape((-1, 1)),
        "node_sample_weight": np.array(node_sample_weight, dtype=np.float64),
        "intercepts": np.array(intercepts, dtype=np.float64),
        "coeffs": np.array(coeffs, dtype=np.float64),
        "decision_type": "<=",
    }
    return {"trees": [tree]}


def convert_model(model, **kwargs):
    """Parses different types of models to a common format.

    That format is a dictionary with the attribute 'trees' holding a list of trees.
    Each tree itself is a dictionary with the following attributes:
    * 'children_left' : array containing the index of left child for each node (-1 for leaf nodes)
    * 'children_right' : array containing the index of right child for each node
    * 'children_default': array containing the index of default child for each node
    * 'features' : array containing the split features for each node
    * 'thresholds' : array containing the split thresholds for each node
    * 'values' : array containing the values for each node (in case of p.w. constant tree)
    * 'node_sample_weight' : array containing the node counts for each node
    * 'intercepts : array containing the intercepts of linear models for each node
    * 'coeffs': array containing the linear coefficients of linear models for each node
    * 'decision_type' : string representing the split condition, i.e. '<', '<=', '>' or '>='
    """

    model_name = fullname(model)
    if model_name == "shap.explainers._tree.TreeExplainer":
        return _from_treeshap(model)
    elif model_name in [
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBRegressor",
        "xgboost.sklearn.XGBClassifier",
    ]:
        return _from_xgboost(model)
    elif model_name in [
        "lightgbm.basic.Booster",
        "lightgbm.sklearn.LGBMRegressor",
        "lightgbm.sklearn.LGBMClassifier",
    ]:
        return _from_lightgbm(model)
    else:
        # we employ the shapiq package for converting common tree models
        from shapiq.tree.validation import validate_tree_model
        return validate_tree_model(model, class_label=class_index)
