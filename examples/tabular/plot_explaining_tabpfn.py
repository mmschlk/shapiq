"""
Explaining TabPFN
==================

`TabPFN <https://github.com/PriorLabs/TabPFN>`_ is a foundation model for
tabular data that uses **in-context learning** -- fitting is just storing the
training data, and inference contextualises new inputs against that context.

``shapiq`` provides a dedicated :class:`~shapiq.TabPFNExplainer` that exploits
this property with a **remove-and-recontextualize** strategy: instead of
imputing missing features, it simply drops feature columns from the training
*and* test data and re-fits the model. This is both faithful to the model
and inexpensive, because TabPFN's "retraining" is just an in-context forward
pass.
"""

from __future__ import annotations

import os

# Prevent OpenMP/MKL thread conflicts with TabPFN's PyTorch backend
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from sklearn.model_selection import train_test_split

import shapiq

# %%
# Prepare a Small Dataset
# -----------------------
# We use the California housing dataset with a tiny split so that TabPFN
# runs quickly on CPU.

x_data, y_data = shapiq.datasets.load_california_housing()
feature_names = list(x_data.columns)

x_train, x_test, y_train, y_test = train_test_split(
    x_data.values,
    y_data.values,
    train_size=30,
    test_size=50,
    random_state=42,
)
print(f"Train: {x_train.shape}, Test: {x_test.shape}")

# %%
# Fit TabPFN
# ----------
# We use ``TabPFNRegressor`` with ``n_estimators=1`` and
# ``fit_mode="low_memory"`` to minimise runtime. Fitting is instant --
# TabPFN just stores the training context.

import tabpfn

model = tabpfn.TabPFNRegressor(
    model_path="tabpfn-v2-regressor.ckpt",
    n_estimators=1,
    fit_mode="low_memory",
)
model.fit(x_train, y_train)

avg_pred = float(np.mean(model.predict(x_test)))
print(f"Average prediction: {avg_pred:.3f}")

# %%
# Auto-Detection of TabPFNExplainer
# -----------------------------------
# When you pass a TabPFN model to :class:`~shapiq.Explainer`, ``shapiq``
# automatically selects :class:`~shapiq.TabPFNExplainer` and sets up a
# :class:`~shapiq.TabPFNImputer` under the hood.  No special configuration
# is needed -- just pass the model, training data, and training labels.

x_explain = x_test[0]
pred = model.predict(x_explain.reshape(1, -1))[0]
print(f"Prediction for instance: {pred:.3f}, Average: {avg_pred:.3f}")

explainer = shapiq.Explainer(
    model=model,
    data=x_train,
    labels=y_train,
    index="SV",
    max_order=1,
    empty_prediction=avg_pred,
)
print(f"Auto-selected explainer: {type(explainer).__name__}")

# %%
# How Remove-and-Recontextualize Works
# --------------------------------------
# Traditional model-agnostic explanation *imputes* absent features with
# background samples (marginal or conditional imputation).  This can
# create out-of-distribution inputs that mislead the model.
#
# The :class:`~shapiq.TabPFNImputer` takes a different approach:
#
# 1. For each coalition :math:`S \subseteq \{1, \dots, d\}` of features:
# 2. **Remove** the columns *not* in :math:`S` from both training and
#    test data.
# 3. **Re-fit** the TabPFN model on the reduced training data (instant,
#    since it is just an in-context forward pass).
# 4. **Predict** on the reduced test point.
#
# This faithfully reflects what the model "knows" when only features in
# :math:`S` are available, without any distributional assumptions.

# %%
# Compute Shapley Values
# -----------------------

sv = explainer.explain(x_explain, budget=50)
print(sv)

sv.plot_force(feature_names=feature_names)

# %%
# Second-Order Interactions (FSII)
# ---------------------------------
# We can also compute Faithful Shapley Interaction Index values to see
# which pairs of features interact.

explainer_fsii = shapiq.Explainer(
    model=model,
    data=x_train,
    labels=y_train,
    index="FSII",
    max_order=2,
    empty_prediction=avg_pred,
)
fsii = explainer_fsii.explain(x_explain, budget=50)
print(fsii)

fsii.plot_force(feature_names=feature_names)

# %%
# References
# ----------
# This example uses TabPFN :footcite:t:`Hollmann.2025` with the
# remove-and-recontextualize strategy from :footcite:t:`Rundel.2024`.
#
# .. footbibliography::
