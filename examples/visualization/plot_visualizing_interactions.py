"""
Visualizing Shapley Interactions
=================================

This example showcases the different visualization techniques in ``shapiq``
for Shapley interactions: force plots, waterfall plots, network plots,
and SI graph plots.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import shapiq

# %%
# Train Model
# -----------

x_data, y_data = shapiq.datasets.load_california_housing(to_numpy=False)
feature_names = list(x_data.columns)
n_features = len(feature_names)
x_data, y_data = x_data.values, y_data.values

x_train, x_test, y_train, y_test = train_test_split(
    x_data,
    y_data,
    test_size=0.2,
    random_state=42,
)
model = RandomForestRegressor(random_state=42, max_depth=15, n_estimators=15)
model.fit(x_train, y_train)
print(f"R2 Score: {model.score(x_test, y_test):.4f}")

# %%
# Compute Shapley Interactions at Different Orders
# --------------------------------------------------
# We compute SV (order 1) and k-SII (order 2) for a single instance.

x_explain = x_test[7]
y_pred = model.predict(x_explain.reshape(1, -1))[0]
print(f"True: {y_test[7]}, Predicted: {y_pred:.3f}")

explainer_sv = shapiq.TreeExplainer(model=model, max_order=1, index="SV")
sv = explainer_sv.explain(x=x_explain)

explainer_si = shapiq.TreeExplainer(model=model, max_order=2, index="k-SII")
si = explainer_si.explain(x=x_explain)

# %%
# Force Plot
# ----------
# Shows how interactions push the prediction away from the baseline.

sv.plot_force(feature_names=feature_names)

# %%

si.plot_force(feature_names=feature_names)

# %%
# Waterfall Plot
# ---------------
# Groups low-magnitude interactions into an "other" category.

sv.plot_waterfall(feature_names=feature_names)

# %%

si.plot_waterfall(feature_names=feature_names)

# %%
# Network Plot
# -------------
# Visualizes first- and second-order interactions as a graph.

si.plot_network(feature_names=feature_names)

# %%
# SI Graph Plot
# --------------
# Supports higher-order interactions via hyper-edges.

abbrev = shapiq.plot.utils.abbreviate_feature_names(feature_names)

sv.plot_si_graph(
    feature_names=abbrev,
    size_factor=2.5,
    node_size_scaling=1.5,
    plot_original_nodes=True,
)

# %%

si.plot_si_graph(
    feature_names=abbrev,
    size_factor=2.5,
    node_size_scaling=1.5,
    plot_original_nodes=True,
)

# %%
# Global Bar Plot
# ----------------
# Aggregate interaction values across 5 test instances.

explainer = shapiq.TreeExplainer(model=model, max_order=2, index="k-SII")
explanations = [explainer.explain(x=x_test[i]) for i in range(5)]
shapiq.plot.bar_plot(explanations, feature_names=feature_names)
