"""Benchmarks 2 & 3 -- path-dependent TreeSHAP runtime as a function of tree depth.

Single-explanation runtime (one instance, one tree) of the four path-dependent algorithms

    quadrature  ``shapiq.tree.QuadratureTreeSHAP``   -- the new default of PR #588
    linear      ``shapiq.tree.LinearTreeSHAP``       -- order 1 only
    treeshapiq  ``shapiq.tree.TreeSHAPIQ``
    shap        ``shap.TreeExplainer(..., feature_perturbation="tree_path_dependent")``

at interaction order 1 (Shapley values) and order 2 (k-SII, and shap's interaction values).
Explainer construction is excluded from the reported time and recorded separately.

Two regimes, selected with ``--suite``:

``real``
    Trees fitted on TabArena datasets (superconductivity, heloc, bioresponse) at increasing
    ``max_depth``. On natural data the number of distinct features per path stays well below
    the depth, so the polynomial explainers remain inside their working range for most of the
    sweep.

``synthetic``
    The regime of shapiq issue #545: rare binary indicator features, where CART splits on a
    fresh feature at every level, so features-per-path == depth. The polynomial explainers
    degrade and then fail, while the quadrature kernel runs to depth 100 at machine precision.
    ``--ignore-guard`` also measures the polynomial explainers *past* their refusal threshold,
    so the figure can show where they actually break instead of where they stop.

Writes ``results/depth_real.json`` / ``results/depth_synthetic.json``.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import warnings

import numpy as np
import shap
from bench_common import (
    DATASETS,
    git_commit,
    make_sparse_indicator_data,
    measure,
    quiet,
    save,
    sklearn_tree_stats,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

import shapiq.tree.linear.computer as _linear_mod
import shapiq.tree.treeshapiq as _treeshapiq_mod
from shapiq.tree import LinearTreeSHAP, QuadratureTreeSHAP, TreeSHAPIQ

REAL_DEPTHS: list[int | None] = [2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, None]
SYNTHETIC_DEPTHS: list[int] = [4, 8, 12, 16, 20, 24, 28, 30, 32, 36, 40, 50, 60, 80, 100]
STOP_AFTER_S = 20.0
METHODS = ("quadrature", "linear", "treeshapiq", "shap")


def disable_precision_guard() -> None:
    """Neutralize the features-per-path guard so the polynomial explainers can be pushed past it."""
    noop = lambda *_a, **_k: None  # noqa: E731
    _treeshapiq_mod.check_features_per_path = noop
    _linear_mod.check_features_per_path = noop


def fit_tree(X: np.ndarray, y: np.ndarray, depth: int | None, task: str):
    cls = DecisionTreeClassifier if task == "classification" else DecisionTreeRegressor
    return cls(max_depth=depth, random_state=0).fit(X, y)


def _predict(model, x: np.ndarray, task: str) -> float:
    return float(
        model.predict_proba(x.reshape(1, -1))[0, 1]
        if task == "classification"
        else model.predict(x.reshape(1, -1))[0]
    )


def _efficiency_error(iv, model, x: np.ndarray, task: str) -> float:
    """``|sum(order-1 values) + baseline - prediction|`` -- 0 for an exact algorithm."""
    return float(abs(iv.get_n_order_values(1).sum() + iv.baseline_value - _predict(model, x, task)))


def _shap_efficiency_error(explainer, model, x: np.ndarray, task: str) -> float:
    """The same check for ``shap``: its path-dependent TreeSHAP is exact in exact arithmetic."""
    values = np.asarray(explainer.shap_values(x.reshape(1, -1), check_additivity=False))
    base = np.asarray(explainer.expected_value)
    if values.ndim == 3:  # sklearn classifiers: (n, features, classes)
        values, base = values[0, :, 1], float(np.ravel(base)[1])
    else:
        values, base = values[0], float(np.ravel(base)[0])
    return float(abs(values.sum() + base - _predict(model, x, task)))


def build_explainer(method: str, order: int, model, class_index: int | None):
    """Construct one explainer; ``None`` marks a (method, order) combination that does not exist."""
    if method == "quadrature":
        return QuadratureTreeSHAP(
            model,
            index="SV" if order == 1 else "k-SII",
            max_order=order,
            class_index=class_index,
        )
    if method == "treeshapiq":
        return TreeSHAPIQ(
            model,
            index="SV" if order == 1 else "k-SII",
            max_order=order,
            class_index=class_index,
        )
    if method == "linear":
        if order != 1:
            return None  # LinearTreeSHAP computes Shapley values only
        return LinearTreeSHAP(model, class_index=class_index)
    if method == "shap":
        return shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    msg = f"unknown method {method}"
    raise ValueError(msg)


def explain_call(method: str, order: int, explainer, x: np.ndarray):
    """The single-explanation call that is timed."""
    if method == "shap":
        x2d = x.reshape(1, -1)
        if order == 1:
            return lambda: explainer.shap_values(x2d, check_additivity=False)
        return lambda: explainer.shap_interaction_values(x2d)
    return lambda: explainer.explain(x)


def run_point(
    method: str,
    order: int,
    model,
    x: np.ndarray,
    class_index: int | None,
    task: str,
    repeats: int,
) -> dict:
    """Construct + time one (method, order) pair on one tree."""
    record: dict = {"method": method, "order": order}
    try:
        with warnings.catch_warnings(), quiet():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            explainer = build_explainer(method, order, model, class_index)
            record["construct_s"] = time.perf_counter() - t0
    except Exception as err:
        return {**record, "status": "refused", "error": f"{type(err).__name__}: {err}"[:200]}
    if explainer is None:
        return {**record, "status": "not_supported"}

    fn = explain_call(method, order, explainer, x)
    res = measure(fn, repeats=repeats, budget_s=STOP_AFTER_S)
    record.update(res)
    if record.get("status") == "ok" and order == 1:
        try:
            with warnings.catch_warnings(), quiet():
                warnings.simplefilter("ignore")
                record["efficiency_error"] = (
                    _shap_efficiency_error(explainer, model, x, task)
                    if method == "shap"
                    else _efficiency_error(explainer.explain(x), model, x, task)
                )
        except Exception:  # noqa: S110
            pass
    return record


def sweep(
    label: str,
    X: np.ndarray,
    y: np.ndarray,
    x_explain: np.ndarray,
    task: str,
    depths: list[int | None],
    repeats: int,
    orders: tuple[int, ...],
) -> list[dict]:
    class_index = 1 if task == "classification" else None
    records: list[dict] = []
    stopped: set[tuple[str, int]] = set()
    seen: set[tuple[int, int]] = set()
    for depth in depths:
        model = fit_tree(X, y, depth, task)
        stats = sklearn_tree_stats(model)
        key = (stats["depth"], stats["n_leaves"])
        if key in seen:  # max_depth beyond the grown depth re-fits the same tree
            continue
        seen.add(key)
        scale = abs(_predict(model, x_explain, task) - float(np.mean(y)))
        print(
            f"  [{label}] max_depth={depth!s:>4}  depth={stats['depth']:>3} "
            f"leaves={stats['n_leaves']:>6} feats/path={stats['max_features_per_path']:>3}",
            flush=True,
        )
        for order in orders:
            for method in METHODS:
                if (method, order) in stopped:
                    continue
                rec = run_point(method, order, model, x_explain, class_index, task, repeats)
                records.append(
                    {
                        "dataset": label,
                        "requested_depth": depth,
                        "prediction_scale": scale,
                        **stats,
                        **rec,
                    }
                )
                t = rec.get("median_s")
                shown = f"{t * 1000:10.3f} ms" if t is not None else f"{rec['status']:>13}"
                if rec.get("efficiency_error") is not None:
                    shown += f"   eff.err {rec['efficiency_error']:.1e}"
                print(f"      order {order} {method:11s} {shown}", flush=True)
                if (t is not None and t > STOP_AFTER_S) or rec.get(
                    "construct_s", 0.0
                ) > STOP_AFTER_S:
                    stopped.add((method, order))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("real", "synthetic"), required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--tag", default="", help="suffix for the result file, e.g. --tag pr590")
    parser.add_argument("--orders", type=int, nargs="+", default=[1, 2])
    parser.add_argument(
        "--ignore-guard",
        action="store_true",
        help="measure the polynomial explainers past their features-per-path refusal threshold",
    )
    parser.add_argument("--n-samples", type=int, default=20_000)
    parser.add_argument("--n-features", type=int, default=120)
    args = parser.parse_args()

    if args.ignore_guard:
        disable_precision_guard()

    orders = tuple(args.orders)
    records: list[dict] = []
    if args.suite == "real":
        for name in ("superconductivity", "heloc", "bioresponse"):
            ds = DATASETS[name]()
            print(f"{name}: train {ds['X_train'].shape}", flush=True)
            records += sweep(
                name,
                ds["X_train"],
                ds["y_train"],
                ds["X_test"][0],
                ds["task"],
                REAL_DEPTHS,
                args.repeats,
                orders,
            )
        meta_extra = {"datasets": ["superconductivity", "heloc", "bioresponse"]}
        out = "depth_real"
    else:
        X, y = make_sparse_indicator_data(args.n_samples, args.n_features)
        print(f"synthetic sparse indicators: {X.shape}, rate 0.02 (shapiq issue #545)", flush=True)
        records += sweep(
            "synthetic",
            X,
            y,
            X[0],
            "regression",
            SYNTHETIC_DEPTHS,
            args.repeats,
            orders,
        )
        meta_extra = {
            "n_samples": args.n_samples,
            "n_features": args.n_features,
            "indicator_rate": 0.02,
        }
        out = "depth_synthetic"

    save(
        {
            "meta": {
                "suite": args.suite,
                "mode": "pathdependent",
                "model": "single sklearn decision tree",
                "measurement": "single-explanation runtime, construction excluded",
                "orders": list(orders),
                "index": {"1": "SV", "2": "k-SII (shap: interaction values)"},
                "repeats": args.repeats,
                "stop_after_s": STOP_AFTER_S,
                "ignore_guard": args.ignore_guard,
                "tag": args.tag,
                "shapiq_commit": git_commit(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                **meta_extra,
            },
            "records": records,
        },
        out + (f"_{args.tag}" if args.tag else ""),
    )
    print(f"saved results/{out}{f'_{args.tag}' if args.tag else ''}.json")


if __name__ == "__main__":
    main()
