"""
SI Graph Plot
=============

This example demonstrates the SI graph plot, which visualizes Shapley
interactions as a network. Players are nodes; interactions are edges whose
color, thickness, and opacity encode strength and direction.
"""

from __future__ import annotations

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

import shapiq

# %%
# Train a Model
# -------------
# We use an XGBoost regressor on the California housing dataset.

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
# Compute Interaction Explanations
# ---------------------------------

x_explain = x_test[2]
explainer = shapiq.TabularExplainer(
    model,
    data=x_test,
    index="FSII",
    max_order=3,
    random_state=42,
)
explanation = explainer.explain(x_explain, budget=200)
print(explanation)

# %%
# Basic SI Graph
# ---------------

explanation.plot_si_graph(show=False)

# %%
# Scaling and Feature Names
# -------------------------
# Adjust node sizes and add feature names for readability.

explanation.plot_si_graph(
    feature_names=feature_names,
    size_factor=5.0,
    node_size_scaling=0.5,
)

# %%
# Filtering Interactions
# ----------------------
# Show only interactions above a threshold or the top-N strongest.

explanation.plot_si_graph(feature_names=feature_names, draw_threshold=0.05)

# %%

explanation.plot_si_graph(feature_names=feature_names, n_interactions=7)

# %%

explanation.plot_si_graph(feature_names=feature_names, interaction_direction="positive")

# %%
# Filtering by Order
# -------------------
# Show only interactions up to a certain order.

explanation.plot_si_graph(feature_names=feature_names, min_max_order=(1, 2))

# %%

explanation.plot_si_graph(feature_names=feature_names, min_max_order=(3, -1))
