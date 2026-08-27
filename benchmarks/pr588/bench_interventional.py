"""Benchmark 1 -- interventional TreeSHAP: shapiq vs. Woodelf vs. shap.

Measures the *end-to-end* cost a user pays to explain ``n`` instances of a fitted model
against a background (reference) dataset of ``m`` rows: building the explainer plus computing
the explanations. That is the only fair basis of comparison here, because the Woodelf fast
path re-parses the model on every ``explain_X`` call and cannot amortize its setup through
the public API.

Backends
    shapiq   ``TreeExplainer(mode="interventional", backend="shapiq")`` -- InterventionalTreeSHAPIQ
    woodelf  ``TreeExplainer(mode="interventional", backend="woodelf")`` -- the vectorized fast path
    shap     ``shap.TreeExplainer(model, data=bg, feature_perturbation="interventional")``

Writes ``results/interventional.json``.
"""

from __future__ import annotations

import argparse
import platform
import sys
import warnings

import numpy as np
import shap
from bench_common import DATASETS, git_commit, measure, quiet, save
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from shapiq.tree import TreeExplainer

BACKENDS = ("shapiq", "woodelf", "shap")

N_EXPLAIN = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10_000]
N_BACKGROUND = [10, 100, 1000]
STOP_AFTER_S = 20.0  # stop extending a curve once one measurement costs this much

# The grid reaches n = 10,000 explained instances. Not every backend is cheap enough to be
# measured there: a curve stops being extended once one of its points costs more than
# ``--stop-after``, and the figure extrapolates the (dead linear) tail from the measured rate.


def shap_interventional(model, bg: np.ndarray):
    """A shap explainer that really uses the whole background.

    ``shap.TreeExplainer(model, data=bg)`` wraps ``bg`` in ``maskers.Independent(bg,
    max_samples=100)``, which silently subsamples any background beyond 100 rows -- the
    resulting values are then not the interventional values of ``bg`` (measurably so: against
    a brute-force oracle on an 8-feature forest with 400 background rows, the default is off by
    1.3e-2 and violates efficiency, while shapiq and Woodelf are exact to 1e-15). Passing the
    masker explicitly is what makes the three backends compute the same quantity.
    """
    masker = shap.maskers.Independent(bg, max_samples=len(bg))
    return shap.TreeExplainer(model, data=masker, feature_perturbation="interventional")


def build_model(ds: dict, n_estimators: int, max_depth: int):
    cls = RandomForestClassifier if ds["task"] == "classification" else RandomForestRegressor
    return cls(n_estimators=n_estimators, max_depth=max_depth, random_state=0, n_jobs=1).fit(
        ds["X_train"], ds["y_train"]
    )


def order1(iv_batch) -> np.ndarray:
    """Stack an ``InteractionValuesBatch`` into an ``(n_instances, n_features)`` array."""
    return np.array([iv.get_n_order_values(1) for iv in iv_batch])


def agreement(model, bg: np.ndarray, X: np.ndarray) -> dict[str, float]:
    """Confirm the three backends compute the same interventional Shapley values."""
    with warnings.catch_warnings(), quiet():
        warnings.simplefilter("ignore")
        a = order1(
            TreeExplainer(
                model, mode="interventional", reference_dataset=bg, backend="shapiq"
            ).explain_X(X)
        )
        b = order1(
            TreeExplainer(
                model, mode="interventional", reference_dataset=bg, backend="woodelf"
            ).explain_X(X)
        )
        c = shap_interventional(model, bg).shap_values(X, check_additivity=False)
    c = np.asarray(c)
    if c.ndim == 3:  # sklearn classifiers return (n, features, classes)
        c = c[..., 1]
    return {
        "max_abs_dev_shapiq_woodelf": float(np.max(np.abs(a - b))),
        "max_abs_dev_shapiq_shap": float(np.max(np.abs(a - c))),
        "scale": float(np.max(np.abs(a))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="heloc", choices=sorted(DATASETS))
    parser.add_argument("--n-estimators", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tag", default="", help="suffix for the result file, e.g. --tag pr590")
    parser.add_argument(
        "--n-explain", type=int, nargs="+", default=None, help="override the n grid"
    )
    parser.add_argument(
        "--n-background", type=int, nargs="+", default=None, help="override the m grid"
    )
    parser.add_argument(
        "--backends", nargs="+", default=["shapiq", "woodelf", "shap"], choices=BACKENDS
    )
    parser.add_argument("--stop-after", type=float, default=STOP_AFTER_S)
    parser.add_argument(
        "--skip-checks", action="store_true", help="the extension sweep re-uses the base run's"
    )
    args = parser.parse_args()

    n_explain = args.n_explain or N_EXPLAIN
    n_background = args.n_background or N_BACKGROUND
    stop_after = args.stop_after

    ds = DATASETS[args.dataset]()
    model = build_model(ds, args.n_estimators, args.max_depth)
    X_pool = ds["X_test"]
    while len(X_pool) < max(n_explain):
        X_pool = np.vstack([X_pool, ds["X_train"], ds["X_test"]])

    print(
        f"dataset={ds['name']} X_train={ds['X_train'].shape} model=RF"
        f"({args.n_estimators}x depth {args.max_depth})",
        flush=True,
    )
    checks = (
        {}
        if args.skip_checks
        else {
            f"m={m}": agreement(model, ds["X_train"][:m], X_pool[:5])
            for m in (20, max(n_background))
        }
    )
    for key, value in checks.items():
        print(f"backend agreement (order-1 SV) {key}: {value}", flush=True)

    records: list[dict] = []
    for n_bg in n_background:
        bg = ds["X_train"][:n_bg]
        stopped: set[str] = set()
        for n_ex in n_explain:
            X = X_pool[:n_ex]

            def run(backend: str, _X=X, _bg=bg):
                if backend == "shap":

                    def fn():
                        return shap_interventional(model, _bg).shap_values(
                            _X, check_additivity=False
                        )
                else:

                    def fn():
                        te = TreeExplainer(
                            model,
                            mode="interventional",
                            reference_dataset=_bg,
                            index="SV",
                            max_order=1,
                            backend=backend,
                        )
                        return te.explain_X(_X)

                return measure(fn, repeats=args.repeats, budget_s=stop_after)

            for backend in args.backends:
                if backend in stopped:
                    continue
                res = run(backend)
                records.append({"n_background": n_bg, "n_explain": n_ex, "backend": backend, **res})
                t = res.get("median_s")
                print(
                    f"  m={n_bg:5d} n={n_ex:5d} {backend:8s} "
                    f"{f'{t:.4f} s' if t is not None else res['status']}",
                    flush=True,
                )
                if t is not None and t > stop_after:
                    stopped.add(backend)

    save(
        {
            "meta": {
                "dataset": ds["name"],
                "n_train": int(ds["X_train"].shape[0]),
                "n_features": int(ds["X_train"].shape[1]),
                "model": f"RandomForest({args.n_estimators} trees, max_depth={args.max_depth})",
                "index": "SV",
                "mode": "interventional",
                "measurement": "end-to-end: explainer construction + explanation of n instances",
                "repeats": args.repeats,
                "stop_after_s": stop_after,
                "n_explain": n_explain,
                "n_background": n_background,
                "backends": list(args.backends),
                "tag": args.tag,
                "shapiq_commit": git_commit(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "agreement": checks,
            },
            "records": records,
        },
        "interventional" + (f"_{args.tag}" if args.tag else ""),
    )
    print(f"saved results/interventional{f'_{args.tag}' if args.tag else ''}.json")


if __name__ == "__main__":
    main()
