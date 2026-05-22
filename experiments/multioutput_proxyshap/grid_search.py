"""Grid-search exploration: where does the multi-output ProxySHAP approach break?

This script pushes the fused multi-output ProxySHAP extension to extremes to
locate the points where it degrades -- both where the *fused C kernel* breaks
(if at all) and, more importantly, where the *XGBoost multi-output proxy itself*
deteriorates.

It runs three focused studies, each writing its own CSV and PNG:

Study 1 -- Scale ``c`` toward 1000
    Sweep ``n_classes`` in {10, 50, 100, 250, 500, 1000} at fixed ``n_features``,
    small ``n_estimators`` and fixed ``budget``. Record proxy fit time, fused vs
    naive explain time, speedup and the dense result size. The fused kernel
    buffers are small so a crash is unlikely -- the question is whether fit time
    or memory explodes.

Study 2 -- Proxy fit QUALITY vs ``c``: multi_output_tree vs one_output_per_tree
    For each ``c`` fit BOTH XGBoost multi-strategies on the SAME coalition-value
    training data and measure held-out R^2 / MSE on a fresh set of sampled
    coalitions. The hypothesis: the shared-topology ``multi_output_tree`` is
    forced to use one set of splits for all ``c`` outputs, so its per-output fit
    quality degrades relative to the ``c`` independent trees of
    ``one_output_per_tree``. The gap as a function of ``c`` is the genuine
    limiting factor of the whole fused approach.

Study 3 -- XGBoost hyperparameter grid
    At a fixed moderate ``c`` grid over ``max_depth`` x ``n_estimators`` and
    record fit time, held-out R^2, fused vs naive time and the speedup, to show
    the depth / quality / speedup trade-off.

Every config is wrapped in try/except (catching ``Exception`` and
``MemoryError``); a failing or pathologically slow config is recorded as a
failure row rather than aborting the whole run.

Run with::

    uv run python experiments/multioutput_proxyshap/grid_search.py
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from math import comb
from pathlib import Path

import matplotlib
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor

from shapiq.approximator.proxy._multioutput import (
    MultiOutputMarginalGame,
    MultiOutputProxySHAP,
)
from shapiq.approximator.proxy._multioutput.explainer import (
    MultiOutputInterventionalTreeExplainer,
)
from shapiq.approximator.proxy._multioutput.tree import (
    MultiOutputTreeModel,
    convert_multioutput_xgboost,
)
from shapiq.tree.base import TreeModel
from shapiq.tree.interventional.explainer import InterventionalTreeExplainer

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
RANDOM_STATE = 0

# Fixed knobs shared across studies (kept small to bound runtime).
N_FEATURES = 12  # number of players n
N_SAMPLES = 600  # rows of the synthetic classifier (>= a few per class)
MAX_BACKGROUND = 24  # background rows for the interventional game
BUDGET = 256  # coalitions sampled to FIT the proxy
EVAL_BUDGET = 256  # FRESH coalitions sampled to score held-out fit quality
INDEX = "SII"
MAX_ORDER = 2

# Study 1 / 2 sweep over the number of outputs c.
C_SWEEP = (10, 50, 100, 250, 500, 1000)

# Study 3 hyper-parameter grid (at a fixed moderate c).
STUDY3_C = 50
DEPTH_GRID = (3, 6, 10)
N_ESTIMATORS_GRID = (10, 50)

# Default n_estimators for Study 1 / 2 (small -> fast).
DEFAULT_N_ESTIMATORS = 10

# A config taking longer than this (seconds) is flagged "pathologically slow"
# but still recorded; it does not abort the run.
SLOW_THRESHOLD_S = 120.0


# --------------------------------------------------------------------------- #
# data + proxy helpers
# --------------------------------------------------------------------------- #
def _make_data(n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Build a synthetic multiclass dataset valid for ``make_classification``.

    ``make_classification`` requires ``n_classes * n_clusters_per_class <=
    2 ** n_informative``. With ``n_clusters_per_class=1`` we therefore need
    ``n_informative >= ceil(log2(n_classes))``; we also keep at least a few
    samples per class. ``n_informative`` is capped at ``N_FEATURES``.

    Args:
        n_classes: Number of output classes ``c``.

    Returns:
        ``(x_data, y_data)`` with ``x_data`` of shape ``(n_samples, N_FEATURES)``.
    """
    n_informative = max(2, int(np.ceil(np.log2(max(2, n_classes)))))
    n_informative = min(n_informative, N_FEATURES)
    n_samples = max(N_SAMPLES, 3 * n_classes)
    x_data, y_data = make_classification(
        n_samples=n_samples,
        n_features=N_FEATURES,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=RANDOM_STATE,
    )
    return x_data, y_data


def _build_proxy(multi_strategy: str, *, n_estimators: int, max_depth: int | None) -> object:
    """Construct an unfitted XGBoost proxy with the given multi-strategy.

    Args:
        multi_strategy: Either ``"multi_output_tree"`` (shared topology, the
            fusable proxy) or ``"one_output_per_tree"`` (``c`` independent
            trees, not fusable).
        n_estimators: Number of boosting rounds.
        max_depth: Maximum tree depth (``None`` -> XGBoost default).

    Returns:
        An unfitted ``XGBRegressor``.
    """
    kwargs: dict[str, object] = {
        "multi_strategy": multi_strategy,
        "n_estimators": n_estimators,
        "random_state": RANDOM_STATE,
    }
    if max_depth is not None:
        kwargs["max_depth"] = max_depth
    return XGBRegressor(**kwargs)


def _sample_coalition_values(
    n_classes: int, budget: int, *, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a game and sample ``budget`` baseline-normalized coalition values.

    Args:
        n_classes: Number of output classes ``c``.
        budget: Number of coalitions to sample.
        seed: Random state for the coalition sampler (vary it to get a fresh,
            held-out set of coalitions for scoring).

    Returns:
        ``(coalitions, coalition_values, baseline)`` where ``coalitions`` is the
        ``(budget, n)`` binary matrix, ``coalition_values`` the
        baseline-normalized ``(budget, c)`` value matrix and ``baseline`` the
        empty-coalition ``c``-vector.
    """
    x_data, y_data = _make_data(n_classes)
    clf = RandomForestClassifier(n_estimators=25, random_state=RANDOM_STATE)
    clf.fit(x_data, y_data)

    game = MultiOutputMarginalGame(
        clf,
        background_data=x_data,
        x=x_data[0],
        max_background_samples=MAX_BACKGROUND,
        random_state=RANDOM_STATE,
    )
    approximator = MultiOutputProxySHAP(
        n=N_FEATURES, max_order=MAX_ORDER, index=INDEX, random_state=seed
    )
    approximator._sampler.sample(budget)
    coalitions = approximator._sampler.coalitions_matrix
    values = np.asarray(game(coalitions), dtype=np.float64)
    baseline = values[0].copy()
    return coalitions, values - baseline, baseline


def _r2_per_output(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute the coefficient of determination R^2 for every output column.

    Args:
        y_true: Ground-truth ``(n, c)`` matrix.
        y_pred: Predicted ``(n, c)`` matrix.

    Returns:
        A length-``c`` array of per-output R^2 scores. A column with zero
        variance yields ``1.0`` if the prediction is exact, else ``0.0``.
    """
    ss_res = np.sum((y_true - y_pred) ** 2, axis=0)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2, axis=0)
    out = np.empty(y_true.shape[1], dtype=np.float64)
    for j in range(y_true.shape[1]):
        if ss_tot[j] <= 1e-15:
            out[j] = 1.0 if ss_res[j] <= 1e-15 else 0.0
        else:
            out[j] = 1.0 - ss_res[j] / ss_tot[j]
    return out


# --------------------------------------------------------------------------- #
# explain timing helpers (reuse the benchmark.py machinery)
# --------------------------------------------------------------------------- #
def _scalar_tree_from_column(multi_tree: MultiOutputTreeModel, column: int) -> TreeModel:
    """Slice column ``column`` of a multi-output tree into a scalar tree.

    Identical to the Oracle-A construction in ``benchmark.py``: same topology
    and splits, scalar leaf values for one output column.

    Args:
        multi_tree: A converted multi-output tree.
        column: Output column to extract.

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


def _time_fused(proxy: object) -> tuple[float, float, float]:
    """Time the fused multi-output explainer on a fitted proxy.

    Args:
        proxy: A fitted ``multi_output_tree`` XGBoost proxy.

    Returns:
        ``(total_time, preprocess_time, kernel_time)`` in seconds.
    """
    t0 = time.perf_counter()
    explainer = MultiOutputInterventionalTreeExplainer(
        proxy, index=INDEX, max_order=MAX_ORDER, n_players=N_FEATURES
    )
    t1 = time.perf_counter()
    explainer.explain()
    t2 = time.perf_counter()
    return t2 - t0, t1 - t0, t2 - t1


def _time_naive(proxy: object, n_classes: int) -> tuple[float, float, float]:
    """Time the naive per-output scalar explainer loop on a fitted proxy.

    Args:
        proxy: A fitted ``multi_output_tree`` XGBoost proxy.
        n_classes: Number of output columns ``c``.

    Returns:
        ``(total_time, preprocess_time, kernel_time)`` in seconds (summed over
        the ``c`` columns).
    """
    multi_trees = convert_multioutput_xgboost(proxy)
    reference = np.zeros((1, N_FEATURES), dtype=np.float64)
    explain_point = np.ones(N_FEATURES, dtype=np.float64)

    rep_pre = 0.0
    rep_ker = 0.0
    t0 = time.perf_counter()
    for column in range(n_classes):
        scalar_trees = [_scalar_tree_from_column(t, column) for t in multi_trees]
        p0 = time.perf_counter()
        explainer = InterventionalTreeExplainer(
            scalar_trees,
            data=reference,
            index=INDEX,
            max_order=MAX_ORDER,
            bool_tree=True,
        )
        p1 = time.perf_counter()
        explainer.explain_function(explain_point)
        p2 = time.perf_counter()
        rep_pre += p1 - p0
        rep_ker += p2 - p1
    t1 = time.perf_counter()
    return t1 - t0, rep_pre, rep_ker


# --------------------------------------------------------------------------- #
# Study 1 -- scale c toward 1000
# --------------------------------------------------------------------------- #
def run_study1() -> list[dict[str, object]]:
    """Run Study 1: scale ``c`` toward 1000 and locate where it breaks.

    Returns:
        One result-row dict per ``c`` in :data:`C_SWEEP`.
    """
    print("\n" + "=" * 72)
    print("STUDY 1 -- scale c toward 1000")
    print("=" * 72)
    rows: list[dict[str, object]] = []

    for c in C_SWEEP:
        row: dict[str, object] = {
            "n_classes": c,
            "n_features": N_FEATURES,
            "n_estimators": DEFAULT_N_ESTIMATORS,
            "budget": BUDGET,
            "status": "ok",
            "error": "",
        }
        print(f"\n  c={c} ...")
        try:
            coalitions, values, _ = _sample_coalition_values(c, BUDGET, seed=RANDOM_STATE)

            t0 = time.perf_counter()
            proxy = _build_proxy(
                "multi_output_tree", n_estimators=DEFAULT_N_ESTIMATORS, max_depth=None
            )
            proxy.fit(coalitions, values)
            fit_time = time.perf_counter() - t0

            fused_total, fused_pre, fused_ker = _time_fused(proxy)
            naive_total, _naive_pre, _naive_ker = _time_naive(proxy, c)
            speedup = naive_total / fused_total if fused_total > 0 else float("nan")

            # Dense result size: c outputs x sum_{k<=order} C(n, k) entries.
            per_output_entries = sum(comb(N_FEATURES, k) for k in range(MAX_ORDER + 1))
            dense_entries = c * per_output_entries

            row.update(
                {
                    "fit_time_s": fit_time,
                    "fused_time_s": fused_total,
                    "fused_preprocess_s": fused_pre,
                    "fused_kernel_s": fused_ker,
                    "naive_time_s": naive_total,
                    "speedup": speedup,
                    "dense_result_entries": dense_entries,
                }
            )
            if fit_time > SLOW_THRESHOLD_S or naive_total > SLOW_THRESHOLD_S:
                row["status"] = "slow"
            print(
                f"    fit={fit_time:7.2f}s  fused={fused_total * 1e3:8.1f}ms  "
                f"naive={naive_total * 1e3:9.1f}ms  speedup={speedup:5.2f}x  "
                f"dense_entries={dense_entries}  [{row['status']}]"
            )
        except (Exception, MemoryError) as exc:
            row["status"] = "FAILED"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {row['error']}")
            traceback.print_exc()
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Study 2 -- proxy fit quality vs c: multi_output_tree vs one_output_per_tree
# --------------------------------------------------------------------------- #
def run_study2() -> list[dict[str, object]]:
    """Run Study 2: fit-quality of both multi-strategies vs ``c``.

    For each ``c`` both XGBoost multi-strategies are fit on the SAME training
    coalitions and scored (held-out R^2 / MSE) on a FRESH set of coalitions.

    Returns:
        One result-row dict per ``c`` in :data:`C_SWEEP`.
    """
    print("\n" + "=" * 72)
    print("STUDY 2 -- fit quality vs c: multi_output_tree vs one_output_per_tree")
    print("=" * 72)
    rows: list[dict[str, object]] = []

    for c in C_SWEEP:
        row: dict[str, object] = {
            "n_classes": c,
            "n_features": N_FEATURES,
            "n_estimators": DEFAULT_N_ESTIMATORS,
            "budget": BUDGET,
            "status": "ok",
            "error": "",
        }
        print(f"\n  c={c} ...")
        try:
            # Train coalitions (seed=RANDOM_STATE) and a FRESH held-out set.
            x_train, y_train, _ = _sample_coalition_values(c, BUDGET, seed=RANDOM_STATE)
            x_eval, y_eval, _ = _sample_coalition_values(c, EVAL_BUDGET, seed=RANDOM_STATE + 1000)

            for strategy in ("multi_output_tree", "one_output_per_tree"):
                # Build the (unfitted) estimator OUTSIDE the timed region so the
                # measured wall-clock covers only the ``.fit()`` call itself.
                proxy = _build_proxy(strategy, n_estimators=DEFAULT_N_ESTIMATORS, max_depth=None)
                t0 = time.perf_counter()
                proxy.fit(x_train, y_train)
                fit_time = time.perf_counter() - t0

                pred = np.asarray(proxy.predict(x_eval), dtype=np.float64)
                r2 = _r2_per_output(y_eval, pred)
                mse = float(np.mean((y_eval - pred) ** 2))
                tag = "mot" if strategy == "multi_output_tree" else "opt"
                row[f"{tag}_fit_time_s"] = fit_time
                row[f"{tag}_r2_mean"] = float(np.mean(r2))
                row[f"{tag}_r2_worst"] = float(np.min(r2))
                row[f"{tag}_mse"] = mse

            row["r2_gap"] = row["opt_r2_mean"] - row["mot_r2_mean"]  # type: ignore[operator]
            # Explicit ``.fit()`` wall-clock columns. ``one_output_per_tree``
            # fitting a c-column target is the right stand-in for "fit c
            # separate scalar proxies" (the cost of explaining a multivariate
            # value function today); ``multi_output_tree`` is the single fused
            # proxy that MultiOutputProxySHAP fits. The ratio quantifies the
            # fit-side speedup, which is SEPARATE from the kernel/explain ~2x.
            row["fit_time_multi_s"] = row["mot_fit_time_s"]
            row["fit_time_oneper_s"] = row["opt_fit_time_s"]
            row["fit_time_ratio_oneper_over_multi"] = (
                row["opt_fit_time_s"] / row["mot_fit_time_s"]  # type: ignore[operator]
                if row["mot_fit_time_s"]  # type: ignore[truthy-bool]
                else float("nan")
            )
            print(
                f"    multi_output_tree : R2_mean={row['mot_r2_mean']:.4f}  "
                f"R2_worst={row['mot_r2_worst']:.4f}  fit={row['mot_fit_time_s']:.4f}s"
            )
            print(
                f"    one_output_per_tree: R2_mean={row['opt_r2_mean']:.4f}  "
                f"R2_worst={row['opt_r2_worst']:.4f}  fit={row['opt_fit_time_s']:.4f}s"
            )
            print(
                f"    R2 gap (opt - mot) = {row['r2_gap']:.4f}   "
                f"fit ratio (oneper/multi) = {row['fit_time_ratio_oneper_over_multi']:.2f}x"
            )
        except (Exception, MemoryError) as exc:
            row["status"] = "FAILED"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    FAILED: {row['error']}")
            traceback.print_exc()
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Study 3 -- XGBoost hyper-parameter grid
# --------------------------------------------------------------------------- #
def run_study3() -> list[dict[str, object]]:
    """Run Study 3: max_depth x n_estimators grid at a fixed moderate ``c``.

    Returns:
        One result-row dict per (max_depth, n_estimators) cell.
    """
    print("\n" + "=" * 72)
    print(f"STUDY 3 -- XGBoost hyper-parameter grid (c={STUDY3_C})")
    print("=" * 72)
    rows: list[dict[str, object]] = []

    # Sample the train / eval coalitions ONCE -- they do not depend on the
    # hyper-parameters being gridded.
    x_train, y_train, _ = _sample_coalition_values(STUDY3_C, BUDGET, seed=RANDOM_STATE)
    x_eval, y_eval, _ = _sample_coalition_values(STUDY3_C, EVAL_BUDGET, seed=RANDOM_STATE + 1000)

    for max_depth in DEPTH_GRID:
        for n_estimators in N_ESTIMATORS_GRID:
            row: dict[str, object] = {
                "n_classes": STUDY3_C,
                "n_features": N_FEATURES,
                "max_depth": max_depth,
                "n_estimators": n_estimators,
                "budget": BUDGET,
                "status": "ok",
                "error": "",
            }
            print(f"\n  max_depth={max_depth}, n_estimators={n_estimators} ...")
            try:
                t0 = time.perf_counter()
                proxy = _build_proxy(
                    "multi_output_tree",
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                )
                proxy.fit(x_train, y_train)
                fit_time = time.perf_counter() - t0

                pred = np.asarray(proxy.predict(x_eval), dtype=np.float64)
                r2 = _r2_per_output(y_eval, pred)

                fused_total, fused_pre, fused_ker = _time_fused(proxy)
                naive_total, _naive_pre, _naive_ker = _time_naive(proxy, STUDY3_C)
                speedup = naive_total / fused_total if fused_total > 0 else float("nan")

                row.update(
                    {
                        "fit_time_s": fit_time,
                        "r2_mean": float(np.mean(r2)),
                        "r2_worst": float(np.min(r2)),
                        "fused_time_s": fused_total,
                        "fused_preprocess_s": fused_pre,
                        "fused_kernel_s": fused_ker,
                        "naive_time_s": naive_total,
                        "speedup": speedup,
                    }
                )
                if fit_time > SLOW_THRESHOLD_S:
                    row["status"] = "slow"
                print(
                    f"    fit={fit_time:6.2f}s  R2_mean={row['r2_mean']:.4f}  "
                    f"fused={fused_total * 1e3:7.1f}ms (pre={fused_pre * 1e3:.1f}/"
                    f"ker={fused_ker * 1e3:.1f})  naive={naive_total * 1e3:8.1f}ms  "
                    f"speedup={speedup:5.2f}x  [{row['status']}]"
                )
            except (Exception, MemoryError) as exc:
                row["status"] = "FAILED"
                row["error"] = f"{type(exc).__name__}: {exc}"
                print(f"    FAILED: {row['error']}")
                traceback.print_exc()
            rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    """Write result rows to ``path`` as CSV (union of all keys as header).

    Args:
        rows: Result-row dicts; may have heterogeneous keys (failure rows lack
            measurement columns).
        path: Destination CSV path.
    """
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {path}")


def _make_plots(
    s1: list[dict[str, object]],
    s2: list[dict[str, object]],
    s3: list[dict[str, object]],
) -> None:
    """Generate the three study PNGs.

    Args:
        s1: Study 1 rows (fit time + speedup vs c).
        s2: Study 2 rows (fit quality vs c, both strategies).
        s3: Study 3 rows (speedup vs max_depth).
    """
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (a) Study 1: fit time + speedup vs c.
    ok1 = [r for r in s1 if r["status"] != "FAILED"]
    if ok1:
        cs = [r["n_classes"] for r in ok1]
        fig, ax1 = plt.subplots(figsize=(6.5, 4))
        ax1.plot(cs, [r["fit_time_s"] for r in ok1], "o-", color="tab:red", label="proxy fit time")
        ax1.set_xlabel("number of outputs c")
        ax1.set_ylabel("proxy fit time (s)", color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:red")
        ax1.set_xscale("log")
        ax2 = ax1.twinx()
        ax2.plot(
            cs, [r["speedup"] for r in ok1], "s--", color="tab:blue", label="fused/naive speedup"
        )
        ax2.axhline(1.0, color="grey", linestyle=":", linewidth=1)
        ax2.set_ylabel("speedup (naive / fused)", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")
        ax1.set_title(f"Study 1: fit time + speedup vs c  (n_features={N_FEATURES})")
        fig.tight_layout()
        fig.savefig(HERE / "grid_c_scaling.png", dpi=150)
        plt.close(fig)
        print(f"wrote {HERE / 'grid_c_scaling.png'}")

    # (b) Study 2: held-out R^2 + fit time vs c for both multi-strategies.
    ok2 = [r for r in s2 if r["status"] != "FAILED"]
    if ok2:
        cs = [r["n_classes"] for r in ok2]
        fig, (ax, axf) = plt.subplots(1, 2, figsize=(11.5, 4))
        ax.plot(
            cs,
            [r["mot_r2_mean"] for r in ok2],
            "o-",
            color="tab:red",
            label="multi_output_tree (mean R2)",
        )
        ax.plot(
            cs,
            [r["mot_r2_worst"] for r in ok2],
            "o:",
            color="tab:red",
            alpha=0.5,
            label="multi_output_tree (worst R2)",
        )
        ax.plot(
            cs,
            [r["opt_r2_mean"] for r in ok2],
            "s-",
            color="tab:green",
            label="one_output_per_tree (mean R2)",
        )
        ax.plot(
            cs,
            [r["opt_r2_worst"] for r in ok2],
            "s:",
            color="tab:green",
            alpha=0.5,
            label="one_output_per_tree (worst R2)",
        )
        ax.set_xlabel("number of outputs c")
        ax.set_ylabel("held-out R^2")
        ax.set_xscale("log")
        ax.set_title("Study 2: proxy fit quality vs c")
        ax.legend(fontsize=8)

        # Fit-time panel: fit-once (multi_output_tree) vs fit-c-times
        # (one_output_per_tree) wall-clock, plus the ratio on a twin axis.
        axf.plot(
            cs,
            [r["fit_time_multi_s"] for r in ok2],
            "o-",
            color="tab:red",
            label="multi_output_tree fit (fit once)",
        )
        axf.plot(
            cs,
            [r["fit_time_oneper_s"] for r in ok2],
            "s-",
            color="tab:green",
            label="one_output_per_tree fit (proxy for fit c times)",
        )
        axf.set_xlabel("number of outputs c")
        axf.set_ylabel("proxy .fit() wall-clock (s)")
        axf.set_xscale("log")
        axf.set_title("Study 2: fitting once vs c times")
        axf2 = axf.twinx()
        axf2.plot(
            cs,
            [r["fit_time_ratio_oneper_over_multi"] for r in ok2],
            "^--",
            color="tab:blue",
            label="ratio oneper / multi",
        )
        axf2.axhline(1.0, color="grey", linestyle=":", linewidth=1)
        axf2.set_ylabel("fit-time ratio (oneper / multi)", color="tab:blue")
        axf2.tick_params(axis="y", labelcolor="tab:blue")
        lines_f, labels_f = axf.get_legend_handles_labels()
        lines_r, labels_r = axf2.get_legend_handles_labels()
        axf.legend(lines_f + lines_r, labels_f + labels_r, fontsize=8)

        fig.tight_layout()
        fig.savefig(HERE / "grid_fit_quality.png", dpi=150)
        plt.close(fig)
        print(f"wrote {HERE / 'grid_fit_quality.png'}")

    # (c) Study 3: speedup vs max_depth (one line per n_estimators).
    ok3 = [r for r in s3 if r["status"] != "FAILED"]
    if ok3:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for n_est in N_ESTIMATORS_GRID:
            pts = sorted((r["max_depth"], r["speedup"]) for r in ok3 if r["n_estimators"] == n_est)
            if pts:
                xs, ys = zip(*pts, strict=True)
                ax.plot(xs, ys, "o-", label=f"n_estimators={n_est}")
        ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
        ax.set_xlabel("max_depth")
        ax.set_ylabel("speedup (naive / fused)")
        ax.set_title(f"Study 3: fused speedup vs max_depth  (c={STUDY3_C})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(HERE / "grid_xgb_params.png", dpi=150)
        plt.close(fig)
        print(f"wrote {HERE / 'grid_xgb_params.png'}")


def main() -> int:
    """Run all three studies, write CSVs and generate the plots.

    Returns:
        Process exit code (``0`` on success).
    """
    print("Multi-output ProxySHAP grid-search exploration")
    print(
        f"n_features={N_FEATURES}  budget={BUDGET}  eval_budget={EVAL_BUDGET}  "
        f"index={INDEX}  max_order={MAX_ORDER}"
    )

    s1 = run_study1()
    s2 = run_study2()
    s3 = run_study3()

    _write_csv(s1, HERE / "grid_c_scaling.csv")
    _write_csv(s2, HERE / "grid_fit_quality.csv")
    _write_csv(s3, HERE / "grid_xgb_params.csv")
    _make_plots(s1, s2, s3)

    print("\ngrid-search complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
