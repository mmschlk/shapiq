'''
2D Interaction Plot
===================

This example demonstrates how to create a 2D scatter plot for visualizing 
pairwise feature interactions. The plot shows two features on x and y axes,
with points colored by their interaction values.

This addresses issue #474: Add 2-D Plot for interactions.
'''

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
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
# We explain 200 test instances to show a meaningful distribution.

x_explain = x_test[:200]
explainer = shapiq.TabularExplainer(
    model,
    data=x_test,
    index="FSII",
    max_order=2,
    random_state=42,
)
explanations = explainer.explain_X(x_explain, budget=200)

# %%
# Basic 2D Interaction Plot
# --------------------------
# Plot two features against each other, colored by their interaction value.
# This creates a scatter plot where:
# - X-axis: first feature values
# - Y-axis: second feature values  
# - Color: interaction value between the two features

fig, ax = plt.subplots(figsize=(8, 6))

# Select a pairwise interaction
interaction = ("MedInc", "Latitude")
feature_idx_1 = feature_names.index(interaction[0])
feature_idx_2 = feature_names.index(interaction[1])

# Get feature values
x_feat_1 = x_explain[:, feature_idx_1]
x_feat_2 = x_explain[:, feature_idx_2]

# Get interaction values
interaction_values = explanations[interaction]

# Create scatter plot
scatter = ax.scatter(
    x_feat_1,
    x_feat_2,
    c=interaction_values,
    cmap="RdBu_r",
    s=50,
    alpha=0.7,
    edgecolors="k",
    linewidth=0.5,
)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label(f"Interaction({interaction[0]}, {interaction[1]})", rotation=270, labelpad=20)

# Labels
ax.set_xlabel(interaction[0])
ax.set_ylabel(interaction[1])
ax.set_title(f"2D Interaction Plot: {interaction[0]} vs {interaction[1]}")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %%
# Multiple Interaction Comparisons
# ----------------------------------
# Compare different pairwise interactions side by side.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

interactions_to_plot = [("MedInc", "Latitude"), ("MedInc", "HouseAge")]

for ax, interaction in zip(axes, interactions_to_plot):
    feature_idx_1 = feature_names.index(interaction[0])
    feature_idx_2 = feature_names.index(interaction[1])
    
    x_feat_1 = x_explain[:, feature_idx_1]
    x_feat_2 = x_explain[:, feature_idx_2]
    interaction_values = explanations[interaction]
    
    scatter = ax.scatter(
        x_feat_1,
        x_feat_2,
        c=interaction_values,
        cmap="RdBu_r",
        s=40,
        alpha=0.7,
        edgecolors="k",
        linewidth=0.5,
    )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(f"Interaction Value", rotation=270, labelpad=15, fontsize=9)
    
    ax.set_xlabel(interaction[0])
    ax.set_ylabel(interaction[1])
    ax.set_title(f"{interaction[0]} vs {interaction[1]}")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
