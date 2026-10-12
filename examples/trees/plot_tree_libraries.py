"""
One Explainer for scikit-learn, XGBoost, LightGBM, and CatBoost
===============================================================

:class:`~shapiq.explainer.TreeExplainer` reads the trees of scikit-learn forests and gradient
boosting models, XGBoost, LightGBM, and CatBoost into one common format and explains all of them
with the same exact algorithms. This example trains six binary classifiers on the same data,
explains them with the same code, and shows

- which output of each model the explanations add up to (probabilities or log-odds),
- how ``class_index`` selects the explained class of a binary classifier, and
- how much the models agree on the important features.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from matplotlib.ticker import PercentFormatter
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import shapiq

# %%
# Six Tree Models on the Same Data
# --------------------------------
# The breast cancer dataset describes 569 tumors with 30 features of their cell nuclei. Class
# ``1`` is a benign tumor, class ``0`` a malignant one. We train a random forest and five gradient
# boosting models with similar sizes, each with one thread (the dataset is small).

data = load_breast_cancer()
feature_names = list(data.feature_names)
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.25, random_state=0, stratify=data.target
)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0),
    "HistGradientBoosting": HistGradientBoostingClassifier(max_depth=4, random_state=0),
    "XGBoost": XGBClassifier(n_estimators=100, max_depth=4, random_state=0, n_jobs=1),
    "LightGBM": LGBMClassifier(n_estimators=100, max_depth=4, random_state=0, n_jobs=1, verbose=-1),
    # allow_writing_files=False keeps CatBoost from writing its training logs to disk
    "CatBoost": CatBoostClassifier(
        iterations=100,
        depth=4,
        random_seed=0,
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
    ),
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"{name:>20}: test accuracy {model.score(X_test, y_test):.3f}")

# %%
# The Same Code for Every Model
# -----------------------------
# We explain the first 50 test tumors with path-dependent Shapley values. By efficiency, the
# baseline and the Shapley values of a tumor sum up to the model output that the explainer
# explains. For the random forest, this is the predicted probability of the class. For the
# gradient boosting models, it is the raw score before the sigmoid, the log-odds of class ``1``,
# because the trees add up in that space.

X_explain = X_test[:50]


def raw_output(name: str, model: object, X: np.ndarray) -> np.ndarray:
    """The output of each model that its trees add up to, for class 1."""
    if name == "RandomForest":
        return model.predict_proba(X)[:, 1]
    if name == "XGBoost":
        return model.predict(X, output_margin=True)
    if name == "LightGBM":
        return model.predict(X, raw_score=True)
    if name == "CatBoost":
        return model.predict(X, prediction_type="RawFormulaVal")
    return model.decision_function(X)  # scikit-learn gradient boosting


explanations = {}
for name, model in models.items():
    explainer = shapiq.TreeExplainer(model, index="SV", max_order=1)
    explanations[name] = explainer.explain_X(X_explain)
    totals = np.array([iv.values.sum() for iv in explanations[name]])  # baseline + values
    difference = np.max(np.abs(totals - raw_output(name, model, X_explain)))
    space = "probability" if name == "RandomForest" else "log-odds"
    print(
        f"{name:>20}: baseline {explainer.baseline_value:+.3f} ({space}), "
        f"max. difference to the model output {difference:.1e}"
    )

# %%
# XGBoost returns its predictions as single-precision (float32) numbers, which deviate from the
# exact sum by about :math:`10^{-6}`. For all other models, the explanations add up to the model
# output up to double-precision rounding.
#
# Explaining Class 0
# ------------------
# By default (``class_index=None``), the explainer explains class ``1`` of a binary classifier.
# ``class_index=0`` explains class ``0``. A binary gradient boosting model has one raw output, the
# log-odds of class ``1``, and the log-odds of class ``0`` are its negative. The probabilities of
# the random forest sum up to one. In both cases, the Shapley values of class ``0`` are exactly
# the negated ones of class ``1``; only the baselines differ.

x = X_test[0]
for name, model in models.items():
    class_1 = shapiq.TreeExplainer(model, class_index=1).explain(x)
    class_0 = shapiq.TreeExplainer(model, class_index=0).explain(x)
    difference = np.max(np.abs(class_0.get_n_order_values(1) + class_1.get_n_order_values(1)))
    print(
        f"{name:>20}: baselines {class_1.baseline_value:+.3f} (class 1), "
        f"{class_0.baseline_value:+.3f} (class 0); max. |SV_0 + SV_1| = {difference:.1e}"
    )

# %%
# Do the Models Agree?
# --------------------
# The explanations of the random forest are probabilities, and the ones of the boosting models
# are log-odds, so we compare the *shares* of the features in each model's mean absolute Shapley
# values. The heatmap shows the ten features with the largest average share. All models rely on
# the size of the tumors (``worst area``, ``worst radius``, ``worst perimeter``) and on the
# concave points of their outline, but each model spreads the importance differently over these
# correlated features.

shares = {}
for name, batch in explanations.items():
    mean_abs = np.mean([np.abs(iv.get_n_order_values(1)) for iv in batch], axis=0)
    shares[name] = mean_abs / mean_abs.sum()
share_matrix = np.array(list(shares.values()))
top = np.argsort(-share_matrix.mean(axis=0))[:10]

fig, ax = plt.subplots(figsize=(9, 4.5))
image = ax.imshow(share_matrix[:, top], cmap="Reds", vmin=0)
ax.set_xticks(np.arange(len(top)), [feature_names[i] for i in top], rotation=40, ha="right")
ax.set_yticks(np.arange(len(shares)), list(shares))
for row in range(len(shares)):
    for column, feature in enumerate(top):
        share = share_matrix[row, feature]
        color = "white" if share > 0.6 * share_matrix[:, top].max() else "black"
        ax.text(column, row, f"{share:.0%}", ha="center", va="center", color=color, fontsize=8)
fig.colorbar(image, ax=ax, label="share of mean |Shapley value|", format=PercentFormatter(1.0))
fig.tight_layout()
plt.show()

# %%
# The explanations of all libraries are :class:`~shapiq.interaction_values.InteractionValues`,
# so every plot of ``shapiq`` works for them. The beeswarm plot of the XGBoost model shows how the
# values depend on the feature values: large tumors push the log-odds towards malignant (class
# ``0``). ``explain_X`` returns a batch of explanations, which the beeswarm plot takes as a list.

shapiq.beeswarm_plot(
    list(explanations["XGBoost"]), X_explain, feature_names=feature_names, abbreviate=False
)
