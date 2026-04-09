"""
TreeSHAP-IQ for Custom Tree Models
===================================

This example demonstrates how to use :class:`~shapiq.TreeExplainer` to explain
a custom tree model built from scratch using the Play Tennis dataset.
"""

from __future__ import annotations

import numpy as np

import shapiq

# %%
# The Play Tennis Dataset
# -----------------------
# A classic dataset with 4 features (Outlook, Temperature, Humidity, Wind)
# and a binary target (Play Tennis: Yes/No). Features are numerically encoded.

X = np.array(
    [
        [1, 1, 1, 1],
        [1, 1, 1, 2],
        [2, 1, 1, 1],
        [3, 2, 1, 1],
        [3, 3, 2, 1],
        [3, 3, 2, 2],
        [2, 3, 2, 2],
        [1, 2, 1, 1],
        [1, 3, 2, 1],
        [3, 2, 2, 1],
        [1, 2, 2, 2],
        [2, 2, 1, 2],
        [2, 1, 2, 1],
        [3, 2, 1, 2],
    ]
)
y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])

# %%
# Define a Custom Tree Model
# ---------------------------
# We define a 3-node decision tree:
#
# .. code-block:: text
#
#         #0: Outlook
#         /         \
#        #1: 0      #2: Humidity
#                   /           \
#                  #3: 0        #4: Wind
#                              /        \
#                             #5: 0     #6: 1

tree = shapiq.tree.TreeModel(
    children_left=np.array([1, -1, 3, -1, 5, -1, -1]),
    children_right=np.array([2, -1, 4, -1, 6, -1, -1]),
    children_missing=np.array([-1, -1, -1, -1, -1, -1, -1]),
    features=np.array([0, -2, 2, -2, 3, -2, -2]),
    thresholds=np.array([2.5, -2, 1.5, -2, 1.5, -2, -2]),
    values=np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]),
    node_sample_weight=np.array([14, 5, 9, 5, 4, 1, 3]),
)

# %%
# Explain with TreeSHAP-IQ
# -------------------------
# Compute exact Shapley values for a single instance.

explainer = shapiq.TreeExplainer(model=tree, index="SV", max_order=1)
sv = explainer.explain(X[5])
print(sv)
sv.plot_force(feature_names=["Outlook", "Temperature", "Humidity", "Wind"])
