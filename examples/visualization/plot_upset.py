"""
UpSet Plot
==========

This example demonstrates the UpSet plot for visualizing feature interactions.
The UpSet plot shows the most important interactions as vertical bars with the
interacting features in a matrix below.
"""

from __future__ import annotations

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import shapiq

# %%
# Train a Model and Compute Explanations
# ----------------------------------------

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

x_explain = x_test[2]
explainer = shapiq.TabularExplainer(
    model,
    data=x_test,
    index="FSII",
    max_order=2,
    random_state=42,
)
explanation = explainer.explain(x_explain, budget=200)
print(explanation)

# %%
# Basic UpSet Plot
# ----------------

explanation.plot_upset()

# %%
# Customization: Number of Interactions
# ---------------------------------------

explanation.plot_upset(n_interactions=10)

# %%
# Color the Matrix
# ----------------
# Color-code the matrix based on interaction sign (red = positive, blue = negative).

explanation.plot_upset(color_matrix=True)

# %%
# Feature Names
# -------------

explanation.plot_upset(feature_names=feature_names, n_interactions=15)

# %%
# Show Only Relevant Features
# ----------------------------
# Set ``all_features=False`` to show only features present in the top interactions.

explanation.plot_upset(all_features=False, n_interactions=7, feature_names=feature_names)

# %%
# Adjust Figure Size
# -------------------

explanation.plot_upset(n_interactions=5, figsize=(5, None))
