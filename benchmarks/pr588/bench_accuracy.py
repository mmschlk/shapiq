"""Accuracy of every path-dependent explainer, at order 1 *and* order 2.

``bench_depth.py`` only records the efficiency error for order-1 runs, which leaves the order-2
curves of the depth figures unlabelled: they look like they keep working, when in fact the
polynomial explainers return numbers with no significant digits left long before they raise.
This re-walks the same trees (same seed, same loaders, same depth grid) and records

    relative efficiency error = |sum(all values) + baseline - prediction| / |prediction - mean|

for each (dataset, depth, method, order). It is exactly 0 for an exact algorithm in exact
arithmetic, so what it measures is round-off. One explanation per point -- this is a
correctness sweep, not a timing one, so it must not run next to a benchmark.

    python bench_accuracy.py --suite synthetic
    python bench_accuracy.py --suite real

Writes ``results/accuracy_<suite>.json``.
"""

from __future__ import annotations

import argparse
import platform
import sys
import warnings

import numpy as np
from bench_common import (
    DATASETS,
    git_commit,
    make_sparse_indicator_data,
    quiet,
    save,
    sklearn_tree_stats,
)
from bench_depth import (
    REAL_DEPTHS,
    SYNTHETIC_DEPTHS,
    build_explainer,
    disable_precision_guard,
    fit_tree,
)

METHODS = ("quadrature", "linear", "treeshapiq", "shap")


def predict(model, x: np.ndarray, task: str) -> float:
    return float(
        model.predict_proba(x.reshape(1, -1))[0, 1]
        if task == "classification"
        else model.predict(x.reshape(1, -1))[0]
    )


def _shap_total(explainer, x: np.ndarray, order: int, task: str) -> float:
    """``sum(values) + baseline`` for shap, across its several output layouts."""
    x2d = x.reshape(1, -1)
    values = np.asarray(
        explainer.shap_values(x2d, check_additivity=False)
        if order == 1
        else explainer.shap_interaction_values(x2d)
    )
    base = np.ravel(explainer.expected_value)
    if task == "classification":  # sklearn classifiers append a class axis
        values, base = values[..., 1], float(base[1])
    else:
        base = float(base[0])
    return float(values.sum()) + base


def rel_error(method, order, explainer, model, x, task, scale) -> float:
    if method == "shap":
        total = _shap_total(explainer, x, order, task)
    else:
        iv = explainer.explain(x)
        # k-SII of any order is efficient: the values of every non-empty coalition sum to
        # f(x) - baseline. The empty coalition must be left out of that sum -- when
        # ``min_order == 0`` the object carries the baseline there as well, and adding both
        # double-counts it (a 1e-16 check turns into a 1e0 one).
        total = float(sum(iv[key] for key in iv.interaction_lookup if key)) + float(
            iv.baseline_value
        )
    return abs(total - predict(model, x, task)) / scale


def sweep(label, X, y, x_explain, task, depths) -> list[dict]:
    class_index = 1 if task == "classification" else None
    records, seen = [], set()
    for depth in depths:
        model = fit_tree(X, y, depth, task)
        stats = sklearn_tree_stats(model)
        key = (stats["depth"], stats["n_leaves"])
        if key in seen:  # max_depth beyond the grown depth re-fits the same tree
            continue
        seen.add(key)
        scale = max(abs(predict(model, x_explain, task) - float(np.mean(y))), 1e-12)
        line = (
            f"  [{label}] depth={stats['depth']:>3} feats/path={stats['max_features_per_path']:>3}"
        )
        for order in (1, 2):
            for method in METHODS:
                rec = {"dataset": label, "method": method, "order": order, **stats}
                try:
                    with quiet():
                        explainer = build_explainer(method, order, model, class_index)
                        if explainer is None:
                            rec["status"] = "not_supported"
                        else:
                            rec["rel_error"] = rel_error(
                                method, order, explainer, model, x_explain, task, scale
                            )
                            rec["status"] = "ok"
                except Exception as err:
                    rec.update(status="refused", error=f"{type(err).__name__}: {err}"[:200])
                records.append(rec)
                if rec.get("rel_error") is not None:
                    line += f"  {method[:4]}{order}={rec['rel_error']:.0e}"
        print(line, flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("real", "synthetic"), required=True)
    parser.add_argument(
        "--keep-guard",
        action="store_true",
        help="leave shapiq's features-per-path guard in place instead of measuring past it",
    )
    args = parser.parse_args()
    warnings.simplefilter("ignore")
    if not args.keep_guard:
        disable_precision_guard()

    records: list[dict] = []
    if args.suite == "real":
        for name in ("superconductivity", "heloc", "bioresponse"):
            ds = DATASETS[name]()
            records += sweep(
                name, ds["X_train"], ds["y_train"], ds["X_test"][0], ds["task"], REAL_DEPTHS
            )
    else:
        X, y = make_sparse_indicator_data(20_000, 120)
        records += sweep("synthetic", X, y, X[0], "regression", SYNTHETIC_DEPTHS)

    save(
        {
            "meta": {
                "suite": args.suite,
                "measure": "|sum(values) + baseline - prediction| / |prediction - mean(y)|",
                "orders": [1, 2],
                "index": {"1": "SV", "2": "k-SII (shap: interaction values)"},
                "ignore_guard": not args.keep_guard,
                "shapiq_commit": git_commit(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
            },
            "records": records,
        },
        f"accuracy_{args.suite}",
    )
    print(f"saved results/accuracy_{args.suite}.json")


if __name__ == "__main__":
    main()
