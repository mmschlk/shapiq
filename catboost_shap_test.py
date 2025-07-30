"""Test script to reproduce some issues with CatBoost."""

from __future__ import annotations

import numpy as np
import shap
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from shapiq.explainer.tree import TreeExplainer

# CLASSIFICATION
# data example from the CatBoost documentation
train_data_cl = np.random.randint(0, 100, size=(100, 10))  # noqa: NPY002
train_labels_cl = np.random.randint(0, 3, size=(100))  # noqa: NPY002
test_data_cl = catboost_pool = Pool(train_data_cl, train_labels_cl)

model_cl = CatBoostClassifier(iterations=2, depth=2, learning_rate=1)

# train the model
model_cl.fit(train_data_cl, train_labels_cl)

# evaluate using shap
explainer_shap_cl = shap.TreeExplainer(model_cl)
original_shap_values_cl = explainer_shap_cl(train_data_cl)

# evaluate using built in shap values
build_in_shap_values_cl = model_cl.get_feature_importance(data=test_data_cl, type="ShapValues")

# evaluate using shapiq
explainer_shapiq_cl = TreeExplainer(
    model=model_cl,
    max_order=1,
    index="SV",
    class_index=train_labels_cl,
)
# raises divide by zero warning!!
sv_shapiq_cl = explainer_shapiq_cl.explain(x=train_data_cl[0])
sv_shapiq_values_cl = sv_shapiq_cl.get_n_order_values(1)


# REGRESSION
# data example from the CatBoost documentation
train_data_reg = np.random.randint(0, 100, size=(100, 10))  # noqa: NPY002
train_labels_reg = np.random.randint(0, 1000, size=(100))  # noqa: NPY002
test_data_reg = np.random.randint(0, 100, size=(50, 10))  # noqa: NPY002
# initialize Pool
train_pool_reg = Pool(train_data_reg, train_labels_reg, cat_features=[0, 2, 5])
test_pool_reg = Pool(test_data_reg, cat_features=[0, 2, 5])

# specify the training parameters
model_reg = CatBoostRegressor(iterations=2, depth=2, learning_rate=1)
# train the model
model_reg.fit(train_pool_reg)

# evaluate using shap
explainer_shap_reg = shap.TreeExplainer(model_reg)
original_shap_values_reg = explainer_shap_reg(train_data_reg)

# evaluate using built in shap values
build_in_shap_values_reg = model_reg.get_feature_importance(data=test_pool_reg, type="ShapValues")

# evaluate using shapiq
explainer_shapiq_reg = TreeExplainer(
    model=model_reg,
    max_order=1,
    index="SV",
    class_index=train_labels_reg,
)
# raises invalid value in divide warning!!
sv_shapiq_reg = explainer_shapiq_reg.explain(x=train_data_reg[0])
sv_shapiq_values_reg = sv_shapiq_reg.get_n_order_values(1)
