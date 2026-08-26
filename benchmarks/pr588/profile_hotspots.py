"""Why are two of the figure-1/figure-2 curves slower than they ought to be?

Two questions came out of reviewing the figures, and both turned out to be packaging around
the kernels rather than the kernels themselves:

1. In the interventional figure, ``TreeExplainer(backend="shapiq")`` (i.e.
   :class:`~shapiq.tree.interventional.computer.InterventionalTreeSHAPIQ`) runs ~60x slower
   than shap's interventional TreeSHAP. Its C kernel is not the problem: the dense "flatten"
   path re-runs ``_preprocess_tree`` for *every explained instance*, and that is a pure-Python
   double loop over (background rows x trees) allocating two numpy arrays per leaf. The sparse
   C path -- already implemented, but only selected for ``max_order > 3`` or very wide dense
   results -- does the same work in C.

2. In the bioresponse panel, shap beats LinearTreeSHAP even though LinearTreeSHAP's kernel is
   the asymptotically cheaper one. It is: the kernel wins. What loses is ``explain_function``
   building a fresh ``{(feature,): value}`` dict over *all* input features on every call, which
   costs O(n_features) Python work -- 1776 entries on bioresponse, ~6x the kernel itself.
   :class:`~shapiq.tree.quadrature.QuadratureTreeSHAP` avoids it by building its lookup once,
   over only the features the tree splits on.

3. In the same panel, QuadratureTreeSHAP's *order-2* curve loses to shap. Its kernel is not the
   problem either -- it is 0.07 ms where the call is 38 ms. The cost is the SII -> k-SII
   aggregation, which ``InteractionValues.__init__`` performs on every explanation, and which
   routes every scalar zero-test through ``np.all(...)`` (9,405 numpy dispatches per
   explanation on a depth-8 superconductivity tree). Note also that shap never performs this
   aggregation: ``shap_interaction_values`` is the Shapley interaction index, i.e. shapiq's
   ``SII`` (shap splits each pair symmetrically, so ``shap_ij == SII_ij / 2``). Compared
   like-for-like the quadrature kernel wins by 5-45x.

Run with no arguments; prints the measured split and the effect of each candidate fix.
"""

from __future__ import annotations

import statistics
import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
import scipy as sp
import shap
from bench_common import load_bioresponse, load_heloc, load_superconductivity, quiet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shapiq.game_theory import aggregation
from shapiq.interaction_values import InteractionValues
from shapiq.tree import LinearTreeSHAP, QuadratureTreeSHAP, TreeExplainer
from shapiq.utils.sets import powerset

if TYPE_CHECKING:
    from collections.abc import Callable


def median_ms(fn: Callable[[], object], *, runs: int = 5, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples) * 1e3


def interventional_explainer(model, bg, index="SV", order=1):
    explainer = TreeExplainer(
        model,
        mode="interventional",
        reference_dataset=bg,
        index=index,
        max_order=order,
        backend="shapiq",
    )
    explainer._init_explainers()  # noqa: SLF001 - the benchmark times the inner computer
    return explainer._interventional_explainer  # noqa: SLF001


def shap_interventional(model, bg):
    masker = shap.maskers.Independent(bg, max_samples=len(bg))
    return shap.TreeExplainer(model, data=masker, feature_perturbation="interventional")


def q1(model, x, train, backgrounds) -> None:
    print("Q1  interventional TreeSHAP-IQ: python preprocessing vs. the C kernel")
    print(f"      {'m':>5} {'explain':>11} {'C kernel':>11} {'python prep':>13} {'shap':>10}")
    for m in backgrounds:
        bg = train[:m]
        with quiet():
            computer = interventional_explainer(model, bg)
            full = median_ms(lambda: computer.explain_function(x), runs=3, warmup=1)
            # prime the cached E/R state, then disable the preprocessing to time the kernel alone
            computer._preprocess_tree(  # noqa: SLF001
                computer.tree[0].cast_input(np.asarray(x, dtype=np.float64))
            )
            real = computer._preprocess_tree  # noqa: SLF001
            computer._preprocess_tree = lambda *_a, **_k: None  # noqa: SLF001
            kernel = median_ms(lambda: computer.explain_function(x), runs=5, warmup=2)
            computer._preprocess_tree = real  # noqa: SLF001
            reference = median_ms(
                lambda: shap_interventional(model, bg).shap_values(
                    x.reshape(1, -1), check_additivity=False
                ),
                runs=3,
                warmup=1,
            )
        print(
            f"      {m:5d} {full:9.1f} ms {kernel:9.2f} ms {full - kernel:11.1f} ms"
            f" {reference:8.2f} ms   ({full / reference:.0f}x shap)"
        )


def q1_fix(cases) -> None:
    print()
    print("Q1  fix: route the same computation through the existing sparse C kernel")
    for label, model, train, x, index, order, m in cases:
        bg = train[:m]
        with quiet():
            dense = interventional_explainer(model, bg, index, order)
            if dense._use_sparse_path:  # noqa: SLF001
                print(f"      {label:46s} already sparse by default")
                continue
            dense_values = dense.explain_function(x)
            dense_ms = median_ms(lambda: dense.explain_function(x), runs=3, warmup=1)

            sparse = interventional_explainer(model, bg, index, order)
            sparse._use_sparse_path = True  # noqa: SLF001
            sparse._preprocess_tree_sparse_path()  # noqa: SLF001
            sparse_values = sparse.explain_function(x)
            sparse_ms = median_ms(lambda: sparse.explain_function(x), runs=3, warmup=1)
        keys = set(dense_values.interaction_lookup) | set(sparse_values.interaction_lookup)
        dev = max(abs(dense_values[k] - sparse_values[k]) for k in keys)
        print(
            f"      {label:46s} {dense_ms:8.1f} ms -> {sparse_ms:7.2f} ms"
            f"  ({dense_ms / sparse_ms:4.0f}x)  max|dev| {dev:.1e}"
        )


def q2(bio) -> None:
    print()
    print("Q2  LinearTreeSHAP on bioresponse: the kernel wins, the result packaging loses")
    for depth in (8, 32):
        tree = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(
            bio["X_train"], bio["y_train"]
        )
        x = bio["X_test"][0]
        n_features = int(x.shape[0])
        used = sorted({int(f) for f in tree.tree_.feature if f >= 0})
        with quiet():
            linear = LinearTreeSHAP(tree, class_index=1)
            quadrature = QuadratureTreeSHAP(tree, index="SV", max_order=1, class_index=1)
            reference = shap.TreeExplainer(tree, feature_perturbation="tree_path_dependent")
        baseline = float(np.sum(linear.edge_tree.empty_predictions))

        def pack(features, _linear=linear, _x=x, _n=n_features, _base=baseline):
            values = _linear.shap_values_cpp_iterative(_x.reshape(1, -1)).flatten()
            return InteractionValues(
                values={(f,): float(values[f]) for f in features},
                baseline_value=_base,
                min_order=0,
                max_order=1,
                index="SV",
                n_players=_n,
            )

        # these are tens of microseconds; a handful of runs is not enough to separate them
        fine = {"runs": 51, "warmup": 10}
        with quiet():
            kernel = median_ms(lambda: linear.shap_values_cpp_iterative(x.reshape(1, -1)), **fine)
            shipped = median_ms(lambda: linear.explain(x), **fine)
            trimmed = median_ms(lambda: pack(used), **fine)
            quad_ms = median_ms(lambda: quadrature.explain(x), **fine)
            shap_ms = median_ms(
                lambda: reference.shap_values(x.reshape(1, -1), check_additivity=False), **fine
            )
        dev = float(
            np.max(
                np.abs(
                    pack(range(n_features)).get_n_order_values(1) - pack(used).get_n_order_values(1)
                )
            )
        )
        print(
            f"      depth {depth:3d}, {len(used):4d} of {n_features} features used by the tree:\n"
            f"        LinearTreeSHAP C kernel .................. {kernel:7.3f} ms\n"
            f"        LinearTreeSHAP.explain (all-feature dict)  {shipped:7.3f} ms\n"
            f"        same, dict over the tree's features only . {trimmed:7.3f} ms  "
            f"(max|dev| {dev:.0e})\n"
            f"        QuadratureTreeSHAP.explain ............... {quad_ms:7.3f} ms\n"
            f"        shap shap_values ......................... {shap_ms:7.3f} ms"
        )


def q2_scaling(bio) -> None:
    print()
    print("      the overhead tracks the input feature space, not the tree:")
    for k in (23, 200, 1776):
        X = bio["X_train"][:, :k]
        tree = DecisionTreeClassifier(max_depth=8, random_state=0).fit(X, bio["y_train"])
        x = bio["X_test"][0, :k]
        with quiet():
            linear = LinearTreeSHAP(tree, class_index=1)
            kernel = median_ms(
                lambda: linear.shap_values_cpp_iterative(x.reshape(1, -1)), runs=51, warmup=10
            )
            shipped = median_ms(lambda: linear.explain(x), runs=51, warmup=10)
        leaves = int((tree.tree_.children_left == -1).sum())
        print(
            f"        n_features {k:5d}, {leaves:4d} leaves:  kernel {kernel:6.3f} ms"
            f"   explain {shipped:6.3f} ms   packaging {shipped - kernel:6.3f} ms"
        )


def _is_zero(value) -> bool:
    """Zero-test without a numpy dispatch on the (overwhelmingly common) scalar path."""
    return not value.any() if isinstance(value, np.ndarray) else value == 0


def faster_aggregation(interactions, index, order, min_order, baseline_value):  # noqa: ARG001
    """``aggregate_base_attributions`` with the scalar zero-test taken off ``np.all``.

    Identical output -- same keys, deviation exactly zero -- and roughly 5x faster.
    """
    bernoulli = [float(b) for b in sp.special.bernoulli(order)]
    out = {(): baseline_value}
    for base, value in interactions.items():
        size = len(base)
        for interaction in powerset(base, min_size=1, max_size=order):
            update = bernoulli[size - len(interaction)] * value
            if _is_zero(update):
                continue
            current = out.get(interaction, 0) + update
            if _is_zero(current):
                out.pop(interaction, None)
            else:
                out[interaction] = current
    return out, aggregation._change_index(index), 0  # noqa: SLF001


def q3(datasets) -> None:
    print()
    print("Q3  QuadratureTreeSHAP at order 2: the kernel is 0.1 ms, the k-SII aggregation is not")
    original = aggregation.aggregate_base_attributions
    for label, data, depth, task in datasets:
        cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
        tree = cls(max_depth=depth, random_state=0).fit(data["X_train"], data["y_train"])
        x = data["X_test"][0]
        class_index = 1 if task == "classification" else None
        with quiet():
            ksii = QuadratureTreeSHAP(tree, index="k-SII", max_order=2, class_index=class_index)
            sii = QuadratureTreeSHAP(tree, index="SII", max_order=2, class_index=class_index)
            reference = shap.TreeExplainer(tree, feature_perturbation="tree_path_dependent")
            x_relevant = ksii._trees[0].cast_input(  # noqa: SLF001
                np.asarray(x, dtype=np.float64)
            )[ksii._relevant_features]  # noqa: SLF001
            kernel = median_ms(lambda: ksii._explain_cpp(x_relevant), runs=15)  # noqa: SLF001
            shipped = median_ms(lambda: ksii.explain(x), runs=5, warmup=2)
            before = ksii.explain(x)

            aggregation.aggregate_base_attributions = faster_aggregation
            patched = median_ms(lambda: ksii.explain(x), runs=5, warmup=2)
            after = ksii.explain(x)
            aggregation.aggregate_base_attributions = original

            sii_ms = median_ms(lambda: sii.explain(x), runs=5, warmup=2)
            shap_ms = median_ms(
                lambda: reference.shap_interaction_values(x.reshape(1, -1)), runs=3, warmup=1
            )
        keys = set(before.interaction_lookup) | set(after.interaction_lookup)
        dev = max(abs(float(before[k]) - float(after[k])) for k in keys)
        n = ksii._n_features_in_tree  # noqa: SLF001
        print(
            f"      {label}, depth {depth} ({n} features in the tree,"
            f" {ksii._output_size} outputs):\n"  # noqa: SLF001
            f"        quadrature C kernel ...................... {kernel:8.3f} ms\n"
            f"        k-SII explain (shipped) .................. {shipped:8.2f} ms\n"
            f"        k-SII explain, scalar zero-test .......... {patched:8.2f} ms"
            f"  ({shipped / patched:.1f}x, max|dev| {dev:.0e})\n"
            f"        SII explain (no aggregation) ............. {sii_ms:8.2f} ms\n"
            f"        shap shap_interaction_values ............. {shap_ms:8.2f} ms"
            f"   <- computes SII, never k-SII"
        )


def q3_convention(data) -> None:
    """Shap's interaction matrix is the same index shapiq calls SII, split over both cells."""
    tree = DecisionTreeRegressor(max_depth=8, random_state=0).fit(data["X_train"], data["y_train"])
    x = data["X_test"][0]
    with quiet():
        sii = QuadratureTreeSHAP(tree, index="SII", max_order=2).explain(x)
        matrix = np.asarray(
            shap.TreeExplainer(
                tree, feature_perturbation="tree_path_dependent"
            ).shap_interaction_values(x.reshape(1, -1))
        )[0]
    pairs = [k for k in sii.interaction_lookup if len(k) == 2]
    half = max(abs(float(sii[k]) / 2 - matrix[k[0], k[1]]) for k in pairs)
    scale = max(abs(float(sii[k])) for k in pairs)
    print()
    print(
        f"      convention check: max|shapiq SII/2 - shap off-diagonal| = {half:.1e}"
        f" on a scale of {scale:.1e}"
    )


def main() -> None:
    warnings.simplefilter("ignore")
    heloc = load_heloc()
    forest = RandomForestClassifier(n_estimators=20, max_depth=8, random_state=0, n_jobs=1).fit(
        heloc["X_train"], heloc["y_train"]
    )
    x_heloc = heloc["X_test"][0]

    q1(forest, x_heloc, heloc["X_train"], (25, 100, 400))

    superconductivity = load_superconductivity()
    sc_forest = RandomForestRegressor(n_estimators=10, max_depth=6, random_state=0, n_jobs=1).fit(
        superconductivity["X_train"], superconductivity["y_train"]
    )
    sc_tree = DecisionTreeRegressor(max_depth=10, random_state=0).fit(
        superconductivity["X_train"], superconductivity["y_train"]
    )
    q1_fix(
        [
            ("heloc RF 20x8, SV", forest, heloc["X_train"], x_heloc, "SV", 1, 100),
            ("heloc RF 20x8, k-SII order 2", forest, heloc["X_train"], x_heloc, "k-SII", 2, 100),
            ("heloc RF 20x8, FSII order 2", forest, heloc["X_train"], x_heloc, "FSII", 2, 100),
            (
                "superconductivity RF 10x6 (81 features), SV",
                sc_forest,
                superconductivity["X_train"],
                superconductivity["X_test"][0],
                "SV",
                1,
                100,
            ),
            (
                "superconductivity single tree depth 10, SV",
                sc_tree,
                superconductivity["X_train"],
                superconductivity["X_test"][0],
                "SV",
                1,
                100,
            ),
        ]
    )

    bio = load_bioresponse()
    q2(bio)
    q2_scaling(bio)

    q3(
        [
            ("superconductivity", superconductivity, 8, "regression"),
            ("superconductivity", superconductivity, 12, "regression"),
            ("bioresponse", bio, 8, "classification"),
            ("bioresponse", bio, 16, "classification"),
            ("heloc", heloc, 16, "classification"),
        ]
    )
    q3_convention(superconductivity)


if __name__ == "__main__":
    main()
