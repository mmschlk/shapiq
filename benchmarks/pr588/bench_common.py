"""Shared helpers for the PR #588 tree-explainer runtime benchmarks.

Every benchmark script imports this module *before* numpy so the single-thread
environment is in place: the comparison is between algorithms, not between thread
pools. Datasets are read from a local cache directory (``SHAPIQ_BENCH_DATA``,
default ``~/bench_data``) holding the raw TabArena/OpenML CSVs.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import statistics
import subprocess
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

# --- single thread, no progress bars -------------------------------------------------------
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
DATA_DIR = Path(os.environ.get("SHAPIQ_BENCH_DATA", Path.home() / "bench_data"))

RANDOM_STATE = 0


# --- datasets -------------------------------------------------------------------------------
def _split(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)


def load_superconductivity() -> dict[str, Any]:
    """21,263 x 81 regression (UCI/OpenML ``superconductivity``); 17k rows after the split."""
    frame = pd.read_csv(DATA_DIR / "superconduct.csv")
    frame = frame.drop(columns=[c for c in ("material",) if c in frame.columns])
    y = frame["critical_temp"].to_numpy(dtype=float)
    X = frame.drop(columns=["critical_temp"]).to_numpy(dtype=float)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    return {
        "name": "superconductivity",
        "task": "regression",
        "X_train": X_tr,
        "X_test": X_te,
        "y_train": y_tr,
        "y_test": y_te,
    }


def load_heloc() -> dict[str, Any]:
    """10,459 x 23 binary classification (FICO HELOC / OpenML ``heloc``); 8.4k after the split."""
    frame = pd.read_csv(DATA_DIR / "heloc.csv")
    y = (frame["RiskPerformance"].astype(str).str.strip() == "Good").to_numpy(dtype=int)
    X = frame.drop(columns=["RiskPerformance"]).to_numpy(dtype=float)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    return {
        "name": "heloc",
        "task": "classification",
        "X_train": X_tr,
        "X_test": X_te,
        "y_train": y_tr,
        "y_test": y_te,
    }


def load_bioresponse() -> dict[str, Any]:
    """3,751 x 1,776 sparse binary classification (OpenML ``Bioresponse``); 3k after the split."""
    frame = pd.read_csv(DATA_DIR / "bioresponse.csv")
    y = frame["Activity"].to_numpy(dtype=int)
    X = frame.drop(columns=["Activity"]).to_numpy(dtype=float)
    X_tr, X_te, y_tr, y_te = _split(X, y)
    return {
        "name": "bioresponse",
        "task": "classification",
        "X_train": X_tr,
        "X_test": X_te,
        "y_train": y_tr,
        "y_test": y_te,
    }


DATASETS = {
    "superconductivity": load_superconductivity,
    "heloc": load_heloc,
    "bioresponse": load_bioresponse,
}


def make_sparse_indicator_data(
    n_samples: int = 20_000, n_features: int = 120, rate: float = 0.02, seed: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    """The synthetic regime of shapiq issue #545: rare binary indicator features.

    CART keeps splitting on a fresh indicator all the way down, so the number of *distinct
    features on a root-to-leaf path* equals the tree depth -- exactly the quantity that
    breaks the polynomial explainers.
    """
    rng = np.random.default_rng(seed)
    X = (rng.random((n_samples, n_features)) < rate).astype(float)
    w = rng.normal(scale=10.0, size=n_features)
    y = X @ w + rng.normal(size=n_samples)
    return X, y


# --- tree statistics ------------------------------------------------------------------------
def sklearn_tree_stats(model: Any) -> dict[str, int]:
    """Depth, leaf count and the maximum number of *distinct* features on a root-to-leaf path."""
    t = model.tree_
    best = 0
    stack = [(0, frozenset())]
    while stack:
        node, feats = stack.pop()
        if t.children_left[node] == -1:
            best = max(best, len(feats))
            continue
        nxt = feats | {int(t.feature[node])}
        stack.append((t.children_left[node], nxt))
        stack.append((t.children_right[node], nxt))
    return {
        "depth": int(model.get_depth()),
        "n_leaves": int(model.get_n_leaves()),
        "max_features_per_path": int(best),
    }


# --- timing ---------------------------------------------------------------------------------
@contextlib.contextmanager
def quiet():
    """Silence the chatty third-party backends (woodelf prints timings and progress bars)."""
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield


def timed(
    fn: Callable[[], Any],
    *,
    repeats: int = 5,
    warmup: bool = True,
    budget_s: float = 30.0,
) -> dict[str, Any]:
    """Median wall-clock of ``fn`` with the warm-up call excluded.

    ``repeats`` is an upper bound. Measurement stops early once the accumulated time passes
    ``budget_s``, and a warm-up call that already costs more than ``budget_s`` reduces the
    sweep to a single measured run -- one slow configuration must not dominate the sweep.
    The number of runs that actually contributed to the median is reported.
    """
    warmup_s = 0.0
    if warmup:
        t0 = time.perf_counter()
        fn()
        warmup_s = time.perf_counter() - t0
        if warmup_s > budget_s:
            repeats = 1
    samples: list[float] = []
    spent = 0.0
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        dt = time.perf_counter() - t0
        samples.append(dt)
        spent += dt
        if spent > budget_s:
            break
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "n_runs": len(samples),
        "warmup_s": warmup_s,
    }


def measure(
    fn: Callable[[], Any],
    *,
    repeats: int = 5,
    warmup: bool = True,
    budget_s: float = 30.0,
    catch: tuple[type[BaseException], ...] = (Exception,),
) -> dict[str, Any]:
    """``timed`` that records a refusal/crash instead of propagating it."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with quiet():
                return {
                    "status": "ok",
                    **timed(fn, repeats=repeats, warmup=warmup, budget_s=budget_s),
                }
    except catch as err:
        return {"status": "failed", "error": f"{type(err).__name__}: {err}"[:300]}


def git_commit() -> str:
    """The shapiq commit the measurement ran against, so a result file is traceable."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=True,
            cwd=HERE,
        ).stdout.strip()
    except Exception:
        return "unknown"


def save(payload: dict[str, Any], name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_results(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / f"{name}.json").read_text())
