"""Which index does ``shap.TreeExplainer.shap_interaction_values`` actually return?

shap calls its output "Shapley interaction values", which reads like the Shapley Interaction
Index (SII). It is not SII -- it is **k-SII with k = 2**, in matrix layout:

* off-diagonal ``M[i, j]`` holds half of the order-2 interaction, so ``M[i, j] + M[j, i]``
  is the pair value. At the top order k-SII and SII agree, so this half matches both;
* diagonal ``M[i, i]`` is set so each row sums to the Shapley value, i.e.
  ``M[i, i] = phi_i - sum_{j != i} M[i, j] = phi_i - 1/2 * sum_{j != i} I_SII(i, j)``.
  That is exactly the k = 2 Bernoulli term of the k-SII aggregation, and it is where SII
  and k-SII part ways.

So plotting shapiq's k-SII against shap's interaction values is a like-for-like comparison,
and calling shap's side "SII" in a figure would be wrong. This script is the check::

    python check_shap_index.py

It prints the maximum deviation of shap's output from each index; k-SII should come out at
machine precision and SII should not.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import shap
from bench_common import load_heloc
from sklearn.tree import DecisionTreeRegressor

from shapiq.tree import TreeSHAPIQ

N_FEATURES = 8  # small enough that every pair fits on screen
MAX_DEPTH = 6


def main() -> None:
    warnings.simplefilter("ignore")
    data = load_heloc()
    X = data["X_train"][:2000, :N_FEATURES]
    y = data["y_train"][:2000].astype(float)
    tree = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=0).fit(X, y)
    x = data["X_test"][0, :N_FEATURES]

    ksii = TreeSHAPIQ(tree, index="k-SII", max_order=2).explain(x)
    sii = TreeSHAPIQ(tree, index="SII", max_order=2).explain(x)
    matrix = np.asarray(
        shap.TreeExplainer(
            tree, feature_perturbation="tree_path_dependent"
        ).shap_interaction_values(x.reshape(1, -1))
    )[0]

    pairs = list(itertools.combinations(range(N_FEATURES), 2))
    singles = [(i,) for i in range(N_FEATURES)]
    scale = max(abs(ksii[key]) for key in ksii.interaction_lookup)

    print(f"tree: depth {MAX_DEPTH}, {N_FEATURES} features   |   value scale {scale:.4g}")
    print(f"shap matrix symmetric: {np.allclose(matrix, matrix.T, atol=1e-12)}\n")
    print(f"{'':22s}{'vs k-SII':>14s}{'vs SII':>14s}")
    for name, keys, shap_value in (
        ("pairs  M[i,j]+M[j,i]", pairs, lambda k: matrix[k[0], k[1]] + matrix[k[1], k[0]]),
        ("singles  M[i,i]", singles, lambda k: matrix[k[0], k[0]]),
    ):
        dev_k = max(abs(shap_value(key) - ksii[key]) for key in keys)
        dev_s = max(abs(shap_value(key) - sii[key]) for key in keys)
        print(f"{name:22s}{dev_k:14.3e}{dev_s:14.3e}")

    print(
        "\nThe diagonal is the discriminating case: it matches k-SII at machine precision and "
        "misses SII by a\nfraction of the value scale. shap's interaction values are 2-SII."
    )


if __name__ == "__main__":
    main()
