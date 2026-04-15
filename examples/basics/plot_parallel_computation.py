"""
Parallel Computation with joblib
=================================

This example shows how to speed up ``shapiq`` explanations using parallel
computation via `joblib <https://joblib.readthedocs.io/>`_. We use a simple
synthetic game to keep runtime short.
"""

from __future__ import annotations

import numpy as np

import shapiq

# %%
# Define a Synthetic Game
# -----------------------
# A lightweight callable that we can explain quickly.

rng = np.random.default_rng(42)
n_features = 8
weights = rng.standard_normal(n_features)


def synthetic_model(x: np.ndarray) -> np.ndarray:
    """Simple linear model with interaction term."""
    return x @ weights + 0.5 * x[:, 0] * x[:, 1]


# Create synthetic data
X_background = rng.standard_normal((100, n_features))
X_test = rng.standard_normal((6, n_features))

# %%
# Explain a Single Instance
# --------------------------

explainer = shapiq.Explainer(model=synthetic_model, data=X_background, random_state=0)
print(f"Explainer type: {type(explainer).__name__}")

iv = explainer.explain(X_test[0], budget=256)
print(iv)

shapiq.network_plot(interaction_values=iv, feature_names=[f"x{i}" for i in range(n_features)])

# %%
# Parallel Explanation of Multiple Instances
# --------------------------------------------
# Use ``n_jobs`` in :meth:`~shapiq.Explainer.explain_X` to parallelize.

ivs = explainer.explain_X(X_test, budget=256, n_jobs=2)
print(f"Computed {len(ivs)} explanations")

# %%
# Global Feature Importance
# --------------------------

shapiq.plot.bar_plot(ivs, feature_names=[f"x{i}" for i in range(n_features)])
