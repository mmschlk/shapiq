"""
Beeswarm Plot
=============

This example demonstrates :func:`~shapiq.beeswarm_plot`, which provides a
global perspective on feature interactions by plotting interaction values
across multiple instances, colored by feature value.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import shapiq

# %%
# Train a Model
# -------------

x_data, y_data = shapiq.datasets.load_california_housing(to_numpy=False)
feature_names = list(x_data.columns)
x_data, y_data = x_data.values, y_data.values
x_train, x_test, y_train, y_test = train_test_split(
    x_data,
    y_data,
    test_size=0.2,
    random_state=42,
)
model = XGBRegressor(random_state=42, max_depth=4, n_estimators=50)
model.fit(x_train, y_train)

# %%
# Compute Explanations for Multiple Instances
# ---------------------------------------------
# We explain 20 test instances to keep the example fast.

x_explain = x_test[:20]
explainer = shapiq.TabularExplainer(
    model,
    data=x_test,
    index="FSII",
    max_order=3,
    random_state=42,
)
explanations = explainer.explain_X(x_explain, budget=200)

# %%
# Basic Beeswarm Plot
# --------------------

shapiq.beeswarm_plot(explanations, x_explain)

# %%
# With Feature Names
# -------------------

shapiq.beeswarm_plot(explanations, x_explain, feature_names=feature_names)

# %%
# Full Feature Names (no abbreviation)
# -------------------------------------

shapiq.beeswarm_plot(
    explanations,
    x_explain,
    feature_names=feature_names,
    abbreviate=False,
)

# %%
# Limit Displayed Interactions
# -----------------------------

shapiq.beeswarm_plot(
    explanations,
    x_explain,
    feature_names=feature_names,
    abbreviate=False,
    max_display=5,
)

# %%
# Adjust Row Height
# ------------------

shapiq.beeswarm_plot(
    explanations,
    x_explain,
    feature_names=feature_names,
    abbreviate=False,
    row_height=1.0,
)

# %%
# Custom Axis
# -----------

fig, ax = plt.subplots(figsize=(6, 6))
shapiq.beeswarm_plot(
    explanations,
    x_explain,
    feature_names=feature_names,
    abbreviate=False,
    ax=ax,
)
