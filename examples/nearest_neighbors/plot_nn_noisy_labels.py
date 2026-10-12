"""
Finding Mislabeled Training Data with KNN-Shapley
=================================================

A mislabeled training point misleads the predictions for its neighbors. Data valuation finds
such points: they get negative values, so sorting the training data by value brings them to the
front.

The value of a training point is its Shapley value in a game whose value is the performance of
the model trained on a coalition of training points :footcite:p:`Ghorbani.2019`.

For a :math:`k`-nearest-neighbor classifier, :footcite:t:`Jia.2019a` compute these Shapley
values exactly in :math:`O(n \\log n)` time for :math:`n` training points ("KNN-Shapley"), which
:class:`~shapiq.explainer.KNNExplainer` implements. This example flips 10% of the labels of a
digits dataset, values every training point, and compares how quickly KNN-Shapley and
leave-one-out values find the flipped labels.
"""

from __future__ import annotations

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from shapiq.explainer import KNNExplainer
from shapiq_games import KNNGame

# %%
# Flip 10% of the Labels
# ----------------------
# The digits dataset has 1,797 images of handwritten digits with 8 x 8 pixels. We split it into
# 1,000 training images, 300 validation images that measure the value of the training data, and
# the rest for testing. A random 10% of the training images get a wrong label.

digits = load_digits()
X, y = digits.data / 16.0, digits.target
X_train, X_rest, y_train, y_rest = train_test_split(
    X, y, train_size=1_000, random_state=0, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_rest, y_rest, train_size=300, random_state=0, stratify=y_rest
)

rng = np.random.default_rng(0)
n_train = len(y_train)
flipped = rng.choice(n_train, size=n_train // 10, replace=False)
y_noisy = y_train.copy()
y_noisy[flipped] = (y_train[flipped] + rng.integers(1, 10, size=flipped.size)) % 10
is_flipped = np.zeros(n_train, dtype=bool)
is_flipped[flipped] = True
print(f"{is_flipped.sum()} of {n_train} training labels are wrong")

# %%
# Value Every Training Point
# --------------------------
# We fit a 5-nearest-neighbor classifier on the noisy labels. For one validation point, the value
# of a coalition of training points is the share of its true class among its five nearest
# neighbors in the coalition. The value of a training point is its Shapley value averaged over
# all validation points. :class:`~shapiq.explainer.KNNExplainer` explains one class, so we create
# one explainer per class and use the one of each validation point's label.

model = KNeighborsClassifier(n_neighbors=5).fit(X_train, y_noisy)
explainers = {label: KNNExplainer(model, class_index=label) for label in model.classes_}

start = time.perf_counter()
shapley_values = np.zeros(n_train)
for x_val, label in zip(X_val, y_val, strict=True):
    shapley_values += explainers[label].explain(x_val).get_n_order_values(1)
shapley_values /= len(y_val)
print(f"KNN-Shapley values of {n_train} points: {time.perf_counter() - start:.2f} s")

# %%
# The values are efficient: they sum up to the model's average predicted probability of the
# true class on the validation data.

probability_of_true_class = model.predict_proba(X_val)[np.arange(len(y_val)), y_val]
print(f"Sum of the values: {shapley_values.sum():.6f}")
print(f"Average predicted probability of the true class: {probability_of_true_class.mean():.6f}")

# %%
# Leave-One-Out Values
# --------------------
# A simpler value of a training point is how much the validation performance drops when the point
# is left out. We evaluate it with :class:`~shapiq_games.KNNGame`, the game behind the explainer,
# on the coalitions that miss one training point each. A point only changes this value if it is
# one of the five nearest neighbors of a validation point, so most leave-one-out values are zero.

all_points = np.ones((1, n_train), dtype=bool)
leave_one_out = ~np.eye(n_train, dtype=bool)  # coalition i misses training point i
loo_values = np.zeros(n_train)
for x_val, label in zip(X_val, y_val, strict=True):
    game = KNNGame(model, x=x_val, class_index=label)
    loo_values += game(all_points)[0] - game(leave_one_out)
loo_values /= len(y_val)
print(f"Leave-one-out values that are zero: {np.mean(loo_values == 0):.0%}")

# %%
# Detection Curves
# ----------------
# We inspect the training points from the lowest to the highest value and count the flipped
# labels among them. KNN-Shapley finds almost all of them within the first 10% of the data,
# where a perfect ranking would find all. Leave-one-out stalls: the flipped points that are
# nobody's nearest neighbor all share the value zero with most correct points, and among those,
# the order is as good as random.


def detection_curve(values: np.ndarray) -> np.ndarray:
    """The share of flipped labels among the training points with the lowest values."""
    order = np.argsort(values, kind="stable")
    return np.cumsum(is_flipped[order]) / is_flipped.sum()


inspected = np.arange(1, n_train + 1) / n_train
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(inspected, np.minimum(inspected / 0.1, 1), color="gray", linestyle=":", label="perfect")
ax.plot(inspected, detection_curve(shapley_values), label="KNN-Shapley")
ax.plot(inspected, detection_curve(loo_values), label="leave-one-out")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="random order (expected)")
ax.set_xlabel("share of training points inspected (lowest value first)")
ax.set_ylabel("share of flipped labels found")
ax.legend(loc="lower right")
fig.tight_layout()
plt.show()

for share in (0.1, 0.2):
    n_inspected = int(share * n_train)
    for name, values in (("KNN-Shapley", shapley_values), ("leave-one-out", loo_values)):
        found = detection_curve(values)[n_inspected - 1]
        print(f"{name}: {found:.0%} of the flipped labels in the lowest {share:.0%}")

# %%
# The Least Valuable Images
# -------------------------
# The sixteen training images with the lowest values. All of them carry a wrong label (red;
# a correct one would be black), and the images show the digits they should have been.

lowest = np.argsort(shapley_values)[:16]
print(f"Flipped labels among the 16 lowest values: {is_flipped[lowest].sum()}")
fig, axes = plt.subplots(2, 8, figsize=(10, 3.4))
for ax, index in zip(axes.flat, lowest, strict=True):
    ax.imshow(X_train[index].reshape(8, 8), cmap="gray_r")
    if is_flipped[index]:
        title, color = f"label {y_noisy[index]}\n(true {y_train[index]})", "#ff0d57"
    else:
        title, color = f"label {y_noisy[index]}\n(correct)", "black"
    ax.set_title(title, color=color, fontsize=9)
    ax.set_axis_off()
fig.tight_layout()
plt.show()

# %%
# Cleaning the Training Data
# --------------------------
# Removing the 10% least valuable points helps any model trained on the data. A 1-nearest-neighbor
# classifier copies every wrong label to its neighborhood, so it suffers most from the noise. Its
# test accuracy after the cleaning is close to the one with the true labels.

keep = np.argsort(shapley_values)[n_train // 10 :]
one_nn = KNeighborsClassifier(n_neighbors=1)
accuracy_noisy = one_nn.fit(X_train, y_noisy).score(X_test, y_test)
accuracy_cleaned = one_nn.fit(X_train[keep], y_noisy[keep]).score(X_test, y_test)
accuracy_true = one_nn.fit(X_train, y_train).score(X_test, y_test)
print(f"1-NN test accuracy with the noisy labels: {accuracy_noisy:.3f}")
print(f"1-NN test accuracy after removing the lowest 10%: {accuracy_cleaned:.3f}")
print(f"1-NN test accuracy with the true labels: {accuracy_true:.3f}")

# %%
# References
# ----------
# .. footbibliography::
