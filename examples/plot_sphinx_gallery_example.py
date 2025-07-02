""".. currentmodule:: shapiq .

Example Sphinx-Gallery
=========================

.. title:: Example Sphinx-Gallery
.. thumbnail:: _static/logo/logo_shapiq_light.png

This is an example of how to use Sphinx-Gallery to create a gallery of examples

Refer to the documentation of :class:`shapiq.explainer.Explainer` for more details.
"""

# %%
# shapiq example
# ------------------------
# This code block is an example of how to use shapiq to explain a model's predictions.
from __future__ import annotations

import shapiq

# load data
X, y = shapiq.load_california_housing(to_numpy=True)
# train a model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(X, y)
# set up an explainer with SV interaction values up to order 1
explainer = shapiq.TabularExplainer(model=model, data=X, index="SV", max_order=1)
# explain the model's prediction for the first sample
interaction_values = explainer.explain(X[0], budget=256)
# analyse interaction values
print(interaction_values)

# %%
# Second example of a shapiq explainer
# ------------------------------
#
# Another example


explainer = shapiq.Explainer(
    model=model,
    data=X,
    index="k-SII",  # k-SII interaction values
    max_order=2,  # specify any order you want
)
interaction_values = explainer.explain(X[0])
print(interaction_values)
