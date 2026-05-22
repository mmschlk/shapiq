"""Benchmark: fused multi-output interventional kernel vs the naive per-output loop.

This script measures the central claim of the multi-output ProxySHAP extension:
explaining a single XGBoost ``multi_strategy="multi_output_tree"`` proxy with the
*fused* :class:`MultiOutputInterventionalTreeExplainer` (one tree traversal, all
``c`` outputs at once) is faster than running the existing *scalar*
:class:`InterventionalTreeExplainer` ``c`` separate times -- once per output
column -- because the expensive structural work (the E/R partition DFS and the
interaction-index weight tables) is computed once and amortized over all ``c``
outputs.

Both methods are run on the *same* fitted multi-output proxy. The scalar
baseline slices each multi-output tree into ``c`` scalar :class:`TreeModel`
objects (column ``j``) -- exactly the Oracle-A construction validated in
``tests/experimental/test_multioutput_correctness.py`` -- so the two paths are
explaining provably identical models and their results can be compared directly.

Outputs (all written next to this file):

* ``results.csv``      -- one row per (n_features, n_classes, index/order) config.
* ``speedup_vs_c.png`` -- speedup ratio vs the number of outputs ``c``.
* ``speedup_vs_nfeatures.png`` -- speedup ratio vs ``n_features``.

Run with::

    uv run python experiments/multioutput_proxyshap/benchmark.py
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from statistics import median

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

from shapiq.approximator.proxy._multioutput import (
    MultiOutputMarginalGame,
    MultiOutputProxySHAP,
)
from shapiq.approximator.proxy._multioutput.explainer import (
    MultiOutputInterventionalTreeExplainer,
)
from shapiq.approximator.proxy._multioutput.proxyshap import (
    _build_default_multioutput_proxy,
)
from shapiq.approximator.proxy._multioutput.tree import (
    MultiOutputTreeModel,
    convert_multioutput_xgboost,
)
from shapiq.tree.base import TreeModel
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer

# --------------------------------------------------------------------------- #
# experiment configuration
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
RANDOM_STATE = 0
BUDGET = 256  # fixed coalition budget used to fit every proxy
N_SAMPLES = 400  # training rows for the synthetic classifier
MAX_BACKGROUND = 40  # background rows for the interventional game
WARMUP = 1  # warmup repeats (discarded)
REPEATS = 5  # measured repeats (median reported)

# Independent sweeps. To keep order-3 dense result sizes (sum C(n, k)) and the
# total runtime reasonable, order-3 runs are capped to n_features <= ORDER3_NMAX.
N_FEATURES_SWEEP = (8, 12, 16, 20)
N_CLASSES_SWEEP = (3, 5, 10, 20)
ORDER3_NMAX = 12  # cap on n_features for the (SII, 3) configs

# index / max_order pairs probed by this experiment.
INDEX_ORDERS = (("SV", 1), ("SII", 2), ("SII", 3))

# n_features held fixed while sweeping c, and c held fixed while sweeping n.
FIXED_N_FEATURES = 12  # used for the "speedup vs c" sweep
FIXED_N_CLASSES = 5  # used for the "speedup vs n_features" sweep

ATOL = 1e-5
RTOL = 1e-5


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _scalar_tree_from_column(multi_tree: MultiOutputTreeModel, column: int) -> TreeModel:
    """Slice column ``column`` of a multi-output tree into a scalar tree.

    The result is structurally identical to the multi-output tree (same
    topology, same splits) but carries a scalar leaf value -- exactly the model
    the scalar :class:`InterventionalTreeExplainer` is the oracle for. This is
    the Oracle-A construction from ``test_multioutput_correctness.py``.

    Args:
        multi_tree: A converted multi-output tree.
        column: The output column to extract.

    Returns:
        The equivalent scalar :class:`~shapiq.tree.base.TreeModel`.
    """
    return TreeModel(
        children_left=multi_tree.children_left.astype(np.int64),
        children_right=multi_tree.children_right.astype(np.int64),
        children_missing=multi_tree.children_default.astype(np.int64),
        features=multi_tree.features.astype(np.int64),
        thresholds=multi_tree.thresholds.astype(np.float64),
        values=multi_tree.values[:, column].astype(np.float64).copy(),
        node_sample_weight=np.ones(multi_tree.n_nodes, dtype=np.float64),
        leaf_mask=multi_tree.leaf_mask.copy(),
        decision_type="<",
    )


def _fit_multioutput_proxy(
    *,
    n_features: int,
    n_classes: int,
) -> object:
    """Train a multiclass classifier and fit ONE multi-output XGBoost proxy.

    This replicates the sample+fit machinery of
    :meth:`MultiOutputProxySHAP.approximate` (steps 1-2) so the *same* fitted
    proxy can subsequently be explained by both the fused and the naive paths
    and timed independently of proxy fitting.

    Args:
        n_features: Number of players ``n``.
        n_classes: Output dimensionality ``c``.

    Returns:
        The fitted ``XGBRegressor(multi_strategy="multi_output_tree")`` proxy.
    """
    x_data, y_data = make_classification(
        n_samples=N_SAMPLES,
        n_features=n_features,
        n_informative=n_features,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=RANDOM_STATE,
    )
    clf = RandomForestClassifier(n_estimators=25, random_state=RANDOM_STATE)
    clf.fit(x_data, y_data)

    game = MultiOutputMarginalGame(
        clf,
        background_data=x_data,
        x=x_data[0],
        max_background_samples=MAX_BACKGROUND,
        random_state=RANDOM_STATE,
    )

    # Reuse MultiOutputProxySHAP's coalition sampler (steps 1-2 of approximate).
    approximator = MultiOutputProxySHAP(
        n=n_features, max_order=2, index="SII", random_state=RANDOM_STATE
    )
    approximator._sampler.sample(BUDGET)
    coalitions = approximator._sampler.coalitions_matrix
    coalition_values = np.asarray(game(coalitions), dtype=np.float64)
    coalition_values = coalition_values - coalition_values[0]

    proxy = _build_default_multioutput_proxy(RANDOM_STATE)
    proxy.fit(coalitions, coalition_values)
    return proxy


def _time_fused(
    proxy: object,
    *,
    index: str,
    max_order: int,
    n_features: int,
) -> tuple[float, float, float]:
    """Time the fused multi-output explainer on ``proxy``.

    Args:
        proxy: A fitted multi-output XGBoost proxy.
        index: Interaction index.
        max_order: Maximum interaction order.
        n_features: Number of players.

    Returns:
        ``(total_time, preprocess_time, kernel_time)`` in seconds (medians).
    """
    totals: list[float] = []
    pre: list[float] = []
    ker: list[float] = []
    for rep in range(WARMUP + REPEATS):
        t0 = time.perf_counter()
        # __init__ performs the E/R partition DFS (preprocess_boolean_trees_multi).
        explainer = MultiOutputInterventionalTreeExplainer(
            proxy, index=index, max_order=max_order, n_players=n_features
        )
        t1 = time.perf_counter()
        explainer.explain()  # the fused C kernel call.
        t2 = time.perf_counter()
        if rep >= WARMUP:
            totals.append(t2 - t0)
            pre.append(t1 - t0)
            ker.append(t2 - t1)
    return median(totals), median(pre), median(ker)


def _time_naive(
    proxy: object,
    *,
    index: str,
    max_order: int,
    n_features: int,
    n_classes: int,
) -> tuple[float, float, float]:
    """Time the naive per-output scalar explainer loop on ``proxy``.

    For each of the ``c`` output columns a scalar
    :class:`InterventionalTreeExplainer` is constructed (boolean-tree mode,
    redoing the E/R partition DFS) and called once.

    Args:
        proxy: A fitted multi-output XGBoost proxy.
        index: Interaction index.
        max_order: Maximum interaction order.
        n_features: Number of players.
        n_classes: Output dimensionality ``c``.

    Returns:
        ``(total_time, preprocess_time, kernel_time)`` in seconds (medians,
        summed over the ``c`` columns).
    """
    multi_trees = convert_multioutput_xgboost(proxy)
    reference = np.zeros((1, n_features), dtype=np.float64)
    explain_point = np.ones(n_features, dtype=np.float64)

    totals: list[float] = []
    pre: list[float] = []
    ker: list[float] = []
    for rep in range(WARMUP + REPEATS):
        rep_pre = 0.0
        rep_ker = 0.0
        t0 = time.perf_counter()
        for column in range(n_classes):
            scalar_trees = [_scalar_tree_from_column(t, column) for t in multi_trees]
            p0 = time.perf_counter()
            # __init__ with bool_tree=True runs the scalar E/R DFS per column.
            explainer = InterventionalTreeExplainer(
                scalar_trees,
                data=reference,
                index=index,
                max_order=max_order,
                bool_tree=True,
            )
            p1 = time.perf_counter()
            explainer.explain_function(explain_point)
            p2 = time.perf_counter()
            rep_pre += p1 - p0
            rep_ker += p2 - p1
        t1 = time.perf_counter()
        if rep >= WARMUP:
            totals.append(t1 - t0)
            pre.append(rep_pre)
            ker.append(rep_ker)
    return median(totals), median(pre), median(ker)


def _check_agreement(
    proxy: object,
    *,
    index: str,
    max_order: int,
    n_features: int,
    n_classes: int,
) -> None:
    """Assert the fused and naive results agree, aborting loudly if not.

    Args:
        proxy: A fitted multi-output XGBoost proxy.
        index: Interaction index.
        max_order: Maximum interaction order.
        n_features: Number of players.
        n_classes: Output dimensionality ``c``.

    Raises:
        AssertionError: If any interaction value differs beyond the tolerance.
    """
    fused = MultiOutputInterventionalTreeExplainer(
        proxy, index=index, max_order=max_order, n_players=n_features
    ).explain()
    multi_trees = convert_multioutput_xgboost(proxy)
    reference = np.zeros((1, n_features), dtype=np.float64)
    explain_point = np.ones(n_features, dtype=np.float64)

    for column in range(n_classes):
        scalar_trees = [_scalar_tree_from_column(t, column) for t in multi_trees]
        naive = InterventionalTreeExplainer(
            scalar_trees,
            data=reference,
            index=index,
            max_order=max_order,
            bool_tree=True,
        ).explain_function(explain_point)
        fused_iv = fused[column]
        for interaction, naive_value in naive.interactions.items():
            fused_value = fused_iv[interaction]
            if not np.allclose(fused_value, naive_value, atol=ATOL, rtol=RTOL):
                msg = (
                    "SANITY CHECK FAILED: fused and naive results disagree for "
                    f"output={column} interaction={interaction} "
                    f"(index={index}, max_order={max_order}, n_features={n_features}): "
                    f"fused={fused_value} naive={naive_value}"
                )
                raise AssertionError(msg)
    print(
        f"  sanity check OK: fused == naive for all {n_classes} outputs "
        f"(index={index}, max_order={max_order}, n={n_features})"
    )


# --------------------------------------------------------------------------- #
# benchmark driver
# --------------------------------------------------------------------------- #
def _build_configs() -> list[tuple[int, int]]:
    """Build the (n_features, n_classes) configuration list.

    Two independent 1-D sweeps share a common anchor point so the sweeps
    overlap: one varies ``n_classes`` at ``FIXED_N_FEATURES``, the other varies
    ``n_features`` at ``FIXED_N_CLASSES``.

    Returns:
        The sorted, de-duplicated list of ``(n_features, n_classes)`` configs.
    """
    configs: set[tuple[int, int]] = set()
    for c in N_CLASSES_SWEEP:
        configs.add((FIXED_N_FEATURES, c))
    for n in N_FEATURES_SWEEP:
        configs.add((n, FIXED_N_CLASSES))
    return sorted(configs)


def run_benchmark() -> list[dict[str, object]]:
    """Run the full benchmark sweep.

    Returns:
        A list of result-row dicts (one per config x index/order).
    """
    configs = _build_configs()
    rows: list[dict[str, object]] = []
    sanity_done = False

    for n_features, n_classes in configs:
        print(f"\n=== n_features={n_features}, n_classes={n_classes} ===")
        proxy = _fit_multioutput_proxy(n_features=n_features, n_classes=n_classes)

        for index, max_order in INDEX_ORDERS:
            if max_order == 3 and n_features > ORDER3_NMAX:
                print(
                    f"  skip (index={index}, max_order={max_order}): n_features="
                    f"{n_features} > ORDER3_NMAX={ORDER3_NMAX} (dense result too large)"
                )
                continue

            # Sanity-assert on the first config we touch, so we know the timed
            # code is correct before trusting any speedup number.
            if not sanity_done:
                _check_agreement(
                    proxy,
                    index=index,
                    max_order=max_order,
                    n_features=n_features,
                    n_classes=n_classes,
                )
                sanity_done = True

            fused_total, fused_pre, fused_ker = _time_fused(
                proxy, index=index, max_order=max_order, n_features=n_features
            )
            naive_total, naive_pre, naive_ker = _time_naive(
                proxy,
                index=index,
                max_order=max_order,
                n_features=n_features,
                n_classes=n_classes,
            )
            speedup = naive_total / fused_total if fused_total > 0 else float("nan")
            print(
                f"  index={index:3s} order={max_order}: "
                f"fused={fused_total * 1e3:8.2f} ms  "
                f"naive={naive_total * 1e3:8.2f} ms  "
                f"speedup={speedup:5.2f}x"
            )
            rows.append(
                {
                    "n_features": n_features,
                    "n_classes": n_classes,
                    "index": index,
                    "max_order": max_order,
                    "fused_time_s": fused_total,
                    "naive_time_s": naive_total,
                    "speedup": speedup,
                    "preprocess_time_s": fused_pre,
                    "kernel_time_s": fused_ker,
                    "naive_preprocess_time_s": naive_pre,
                    "naive_kernel_time_s": naive_ker,
                }
            )

    if not sanity_done:
        msg = "No config was benchmarked -- sanity check never ran."
        raise RuntimeError(msg)
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write the benchmark rows to ``path`` as CSV.

    Args:
        rows: The benchmark result rows.
        path: Destination CSV path.
    """
    fieldnames = [
        "n_features",
        "n_classes",
        "index",
        "max_order",
        "fused_time_s",
        "naive_time_s",
        "speedup",
        "preprocess_time_s",
        "kernel_time_s",
        "naive_preprocess_time_s",
        "naive_kernel_time_s",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"\nwrote {path}")


def make_plots(rows: list[dict[str, object]]) -> None:
    """Generate the speedup-vs-c and speedup-vs-n_features plots.

    Args:
        rows: The benchmark result rows.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def label(index: str, order: int) -> str:
        return f"{index}, order {order}"

    # (a) speedup vs c, at the fixed n_features anchor.
    fig, ax = plt.subplots(figsize=(6, 4))
    for index, max_order in INDEX_ORDERS:
        pts = sorted(
            (r["n_classes"], r["speedup"])
            for r in rows
            if r["index"] == index
            and r["max_order"] == max_order
            and r["n_features"] == FIXED_N_FEATURES
        )
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, marker="o", label=label(index, max_order))
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("number of outputs c (n_classes)")
    ax.set_ylabel("speedup (naive / fused)")
    ax.set_title(f"Fused vs naive speedup vs c  (n_features={FIXED_N_FEATURES})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "speedup_vs_c.png", dpi=150)
    plt.close(fig)
    print(f"wrote {HERE / 'speedup_vs_c.png'}")

    # (b) speedup vs n_features, at the fixed n_classes anchor.
    fig, ax = plt.subplots(figsize=(6, 4))
    for index, max_order in INDEX_ORDERS:
        pts = sorted(
            (r["n_features"], r["speedup"])
            for r in rows
            if r["index"] == index
            and r["max_order"] == max_order
            and r["n_classes"] == FIXED_N_CLASSES
        )
        if pts:
            xs, ys = zip(*pts, strict=True)
            ax.plot(xs, ys, marker="o", label=label(index, max_order))
    ax.axhline(1.0, color="grey", linestyle="--", linewidth=1)
    ax.set_xlabel("n_features")
    ax.set_ylabel("speedup (naive / fused)")
    ax.set_title(f"Fused vs naive speedup vs n_features  (c={FIXED_N_CLASSES})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(HERE / "speedup_vs_nfeatures.png", dpi=150)
    plt.close(fig)
    print(f"wrote {HERE / 'speedup_vs_nfeatures.png'}")


def main() -> int:
    """Run the benchmark, write the CSV and generate the plots.

    Returns:
        Process exit code (``0`` on success).
    """
    print("Multi-output ProxySHAP fused-kernel benchmark")
    print(
        f"budget={BUDGET}  repeats={REPEATS}  warmup={WARMUP}  "
        f"order-3 capped at n_features<={ORDER3_NMAX}"
    )
    rows = run_benchmark()
    write_csv(rows, HERE / "results.csv")
    make_plots(rows)
    print("\nbenchmark complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
