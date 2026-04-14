"""
SHAP-IQ with scikit-learn
==========================

This example shows how to compute second-order Shapley Interaction Index (SII)
values for a scikit-learn Random Forest on the California housing dataset.
"""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

import shapiq

# %%
# Load Data and Train Model
# --------------------------

X, y = shapiq.load_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    X.values,
    y.values,
    test_size=0.25,
    random_state=42,
)
n_features = X_train.shape[1]

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=n_features,
    max_features=2 / 3,
    max_samples=2 / 3,
    random_state=42,
)
model.fit(X_train, y_train)
print(f"Train R2: {model.score(X_train, y_train):.4f}")
print(f"Test  R2: {model.score(X_test, y_test):.4f}")

# %%
# Compute Second-Order SII
# -------------------------
# :class:`~shapiq.TabularExplainer` with ``index="SII"`` and ``max_order=2``
# computes pairwise Shapley interaction values.

explainer = shapiq.TabularExplainer(model=model, data=X_train, index="SII", max_order=2)
x = X_test[24]
iv = explainer.explain(x, budget=2**n_features, random_state=0)
print(iv)

# %%
# Second-Order Interaction Matrix
# --------------------------------

print(iv.get_n_order(2).dict_values)

# %%
# Visualization: Network Plot
# ----------------------------

shapiq.network_plot(interaction_values=iv, feature_names=list(X.columns))

# %%
# Stacked Bar Plot (First Order)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

shapiq.stacked_bar_plot(iv.get_n_order(1), feature_names=list(X.columns))

# %%
# Stacked Bar Plot (All Orders)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

shapiq.stacked_bar_plot(interaction_values=iv, feature_names=list(X.columns))

# %%
# Force Plot
# ----------

iv.plot_force(feature_names=list(X.columns))
