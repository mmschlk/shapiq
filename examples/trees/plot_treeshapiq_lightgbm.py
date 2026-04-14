"""
TreeSHAP-IQ for LightGBM
=========================

This example demonstrates :class:`~shapiq.TreeExplainer` on a LightGBM model
trained on the bike-sharing dataset. TreeSHAP-IQ computes exact Shapley
interaction values in linear time for tree ensembles.
"""

from __future__ import annotations

import lightgbm
from sklearn.model_selection import train_test_split

import shapiq

# %%
# Load Data and Train Model
# --------------------------

X, y = shapiq.load_bike_sharing()
X_train, X_test, y_train, y_test = train_test_split(
    X.values,
    y.values,
    test_size=0.25,
    random_state=42,
)
n_features = X_train.shape[1]

model = lightgbm.LGBMRegressor(
    n_estimators=100,
    max_depth=n_features,
    random_state=42,
    verbose=-1,
)
model.fit(X_train, y_train)
print(f"Train R2: {model.score(X_train, y_train):.4f}")
print(f"Test  R2: {model.score(X_test, y_test):.4f}")

# %%
# Compute Shapley Interactions
# -----------------------------
# We compute k-SII scores up to order 3 for a single instance.

explainer = shapiq.TreeExplainer(model=model, index="k-SII", min_order=1, max_order=3)
x = X_test[1234]
interaction_values = explainer.explain(x)
print(interaction_values)

# %%
# First-order Values (Shapley Values)
# -------------------------------------

print(interaction_values.get_n_order(1).dict_values)

# %%
# Visualization: Network Plot
# ----------------------------

shapiq.network_plot(interaction_values=interaction_values, feature_names=list(X.columns))

# %%
# Stacked Bar Plot (First Order)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

shapiq.stacked_bar_plot(
    interaction_values=interaction_values.get_n_order(1),
    feature_names=list(X.columns),
)

# %%
# Stacked Bar Plot (First + Second Order)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

shapiq.stacked_bar_plot(
    interaction_values=interaction_values.get_n_order(2, min_order=1),
    feature_names=list(X.columns),
)

# %%
# Force Plot
# ----------

interaction_values.plot_force(feature_names=list(X.columns), contribution_threshold=0.03)

# %%
# Global Feature Importance
# --------------------------
# Compute interaction values for 50 test instances and show global bar plot.

list_of_ivs = explainer.explain_X(X_test[:50])
shapiq.plot.bar_plot(list_of_ivs, feature_names=list(X.columns), max_display=20)
