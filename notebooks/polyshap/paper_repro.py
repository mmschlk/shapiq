"""Reproduce the PolySHAP paper figures using the *integrated* shapiq approximators.

This module is the engine behind ``validate_polyshap_integration.ipynb``. It
validates two things at once:

1. The integration of PolySHAP into shapiq (``PolySHAPKAdd`` / ``PolySHAPPartial``)
   behaves as the paper's ``PolySHAP`` + ``ExplanationFrontierGenerator`` API did.
2. The paper's central empirical claims (Fumagalli et al., 2026, ICLR):
     * Fig. 2 - higher-order PolySHAP improves approximation quality
       (3-PolySHAP < 2-PolySHAP < 1-PolySHAP=KernelSHAP in MSE, given budget).
     * Fig. 3 - paired KernelSHAP matches 2-PolySHAP.
     * A runtime analysis and a cross-method performance table.

It is fully self-contained: it reads the *exhaustive* precomputed game tables
shipped with the original PolySHAP project (image + language games of Table 2),
computes exact Shapley ground truth directly, and never touches the original
project's code or its vendored shapiq fork.

Method -> integrated API mapping
--------------------------------
    1-PolySHAP (KernelSHAP)  -> PolySHAPKAdd(max_order=1)
    k-PolySHAP  (k=2,3,4)    -> PolySHAPKAdd(max_order=k)
    3-PolySHAP(50%)          -> PolySHAPPartial(1+d+C(d,2)+0.5*C(d,3) terms)
    PolySHAP(log)            -> PolySHAPPartial(1+d+C(d,2)+d*log(C(d,3)) terms)
    Permutation / SVARM / MSR-> PermutationSamplingSV / SVARM / UnbiasedKernelSHAP

It can also be run as a standalone CLI::

    python notebooks/polyshap/paper_repro.py --quick
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import comb
from scipy.stats import spearmanr

from shapiq import ExactComputer
from shapiq.approximator import PermutationSamplingSV, SVARM, UnbiasedKernelSHAP
from shapiq.approximator.regression.polyshap import PolySHAPKAdd, PolySHAPPartial

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Location of the exhaustive precomputed games shipped with the original project.
DEFAULT_DATA_ROOT = Path(
    r"C:\Users\Black\Documents\Studium\SoSe26\TTML Toolbox for Trustworthy "
    r"Machine Learning\PolySHAP\data\precomputed_games"
)


@dataclass(frozen=True)
class GameSpec:
    """A family of precomputed explanation games (one Table-2 row)."""

    name: str
    d: int
    domain: str
    subdir: str
    pattern: str  # "...{i}.npz", i in 1..30


GAME_SPECS: dict[str, GameSpec] = {
    "ViT9": GameSpec("ViT9", 9, "image", "ImageClassifier_Game/9", "model_name=vit_9_patches_{i}.npz"),
    "ResNet18": GameSpec(
        "ResNet18", 14, "image", "ImageClassifier_Game/14",
        "model_name=resnet_18_n_superpixel_resnet=14_{i}.npz",
    ),
    "ViT16": GameSpec("ViT16", 16, "image", "ImageClassifier_Game/16", "model_name=vit_16_patches_{i}.npz"),
    "DistilBERT": GameSpec("DistilBERT", 14, "language", "SentimentAnalysis_Game/14", "mask_strategy=mask_{i}.npz"),
}

# Method groups (paper terminology).
HIGHER_ORDER = ["1-PolySHAP", "2-PolySHAP", "3-PolySHAP", "4-PolySHAP"]
PARTIAL = ["2-PolySHAP(50%)", "3-PolySHAP(50%)", "PolySHAP(log)"]
BASELINES = ["Permutation", "SVARM", "MSR"]
ALL_METHODS = HIGHER_ORDER + PARTIAL + BASELINES

MAX_BUDGET = 20_000
N_BUDGET_STEPS = 10

# Colours sampled directly from the paper's Figure-2 legend (teal + Material
# "Deep Orange" ramp) so our plots are visually comparable to the published ones.
PAPER_COLORS: dict[str, str] = {
    "1-PolySHAP": "#009587",       # teal  (= KernelSHAP)
    "2-PolySHAP": "#ffb64d",       # amber
    "3-PolySHAP": "#e64918",       # deep orange
    "4-PolySHAP": "#bf360b",       # dark red
    "2-PolySHAP(50%)": "#ffdfb1",  # cream
    "3-PolySHAP(50%)": "#ff5621",  # orange-red
    "PolySHAP(log)": "#6d4c41",    # brown (not a paper colour; ours only)
    "Permutation": "#3949ab",      # indigo (baseline; not a paper colour)
    "SVARM": "#7e57c2",            # purple (baseline)
    "MSR": "#00838f",              # cyan   (baseline)
}

# Per-game axis windows matching the paper's Figure-2/3 panels: linear budget
# axis, log MSE axis, identical limits and ticks so the curves line up.
AXIS_SPECS: dict[str, dict] = {
    "ResNet18": {"xlim": (0, 17000), "xticks": [0, 3000, 6000, 9000, 12000, 15000], "ylim": (1e-9, 1e-1)},
    "ViT16": {"xlim": (0, 21000), "xticks": [0, 4000, 8000, 12000, 16000, 20000], "ylim": (1e-7, 1e-1)},
    "ViT9": {"xlim": (0, 512), "xticks": [0, 128, 256, 384, 512], "ylim": (1e-7, 1e0)},
    "DistilBERT": {"xlim": (0, 16384), "xticks": [0, 3000, 6000, 9000, 12000, 15000], "ylim": (1e-9, 1e-1)},
}

# --------------------------------------------------------------------------- #
# Game loading + exact ground truth
# --------------------------------------------------------------------------- #


def load_game(path: Path):
    """Load one exhaustive npz game table into a fast lookup callable.

    Returns ``(game, n)`` where ``game(X)`` maps a ``(m, n)`` boolean coalition
    matrix to an ``(m,)`` value vector.
    """
    data = np.load(path, allow_pickle=True)
    coalitions = np.asarray(data["coalitions"], dtype=bool)
    values = np.asarray(data["values"], dtype=float)
    n = int(data["n_players"])
    pow2 = (1 << np.arange(n)).astype(np.int64)
    keys = coalitions.astype(np.int64) @ pow2
    table = dict(zip(keys.tolist(), values.tolist()))

    def game(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=bool)
        k = X.astype(np.int64) @ pow2
        return np.fromiter((table[int(v)] for v in k), dtype=float, count=len(k))

    return game, n


def exact_shapley(game, n: int) -> np.ndarray:
    """Exact Shapley values (length ``n``) from the full game table."""
    ec = ExactComputer(game, n_players=n)
    sv = ec(index="SV", order=1)
    return np.asarray(sv.get_n_order_values(1), dtype=float)


# --------------------------------------------------------------------------- #
# Approximator factory  (paper name -> integrated approximator instance)
# --------------------------------------------------------------------------- #


def full_kadd_terms(n: int, k: int) -> int:
    """Number of frontier terms for a full k-additive frontier (incl. empty)."""
    return int(sum(comb(n, i, exact=True) for i in range(k + 1)))


def make_method(name: str, n: int, random_state: int, paired: bool):
    """Construct one approximator for *name* using the integrated shapiq API."""
    weights = np.ones(n + 1)  # order-1 leverage scores == uniform over subset sizes

    if name == "1-PolySHAP":
        return PolySHAPKAdd(n=n, max_order=1, sampling_weights=weights,
                            pairing_trick=paired, random_state=random_state)
    if name in ("2-PolySHAP", "3-PolySHAP", "4-PolySHAP"):
        k = int(name[0])
        return PolySHAPKAdd(n=n, max_order=k, sampling_weights=weights,
                            pairing_trick=paired, random_state=random_state)
    if name == "2-PolySHAP(50%)":
        # singletons + 50% of the pairwise interactions
        n_terms = int(1 + n + 0.5 * comb(n, 2))
        n_terms = min(n_terms, full_kadd_terms(n, 2))
        return PolySHAPPartial(n=n, n_explanation_terms=n_terms, sampling_weights=weights,
                               pairing_trick=paired, random_state=random_state)
    if name == "3-PolySHAP(50%)":
        n_terms = int(1 + n + comb(n, 2) + 0.5 * comb(n, 3))
        n_terms = min(n_terms, full_kadd_terms(n, 3))
        return PolySHAPPartial(n=n, n_explanation_terms=n_terms, sampling_weights=weights,
                               pairing_trick=paired, random_state=random_state)
    if name == "PolySHAP(log)":
        c3 = max(comb(n, 3), 2.0)
        n_terms = int(1 + n + comb(n, 2) + n * math.log(c3))
        n_terms = min(n_terms, full_kadd_terms(n, 3))
        return PolySHAPPartial(n=n, n_explanation_terms=n_terms, sampling_weights=weights,
                               pairing_trick=paired, random_state=random_state)
    if name == "Permutation":
        return PermutationSamplingSV(n=n, pairing_trick=paired, random_state=random_state)
    if name == "SVARM":
        return SVARM(n=n, pairing_trick=paired, sampling_weights=weights, random_state=random_state)
    if name == "MSR":
        return UnbiasedKernelSHAP(n=n, pairing_trick=paired, sampling_weights=weights,
                                  random_state=random_state)
    raise ValueError(f"unknown method {name!r}")


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def precision_at_k(est: np.ndarray, exact: np.ndarray, k: int) -> float:
    k = min(k, len(exact))
    top_est = set(np.argsort(-np.abs(est))[:k].tolist())
    top_exact = set(np.argsort(-np.abs(exact))[:k].tolist())
    return len(top_est & top_exact) / k


def compute_metrics(est: np.ndarray, exact: np.ndarray) -> dict[str, float]:
    mse = float(np.mean((est - exact) ** 2))
    mae = float(np.mean(np.abs(est - exact)))
    p5 = precision_at_k(est, exact, 5)
    if np.ptp(est) == 0 or np.ptp(exact) == 0:
        rho = float("nan")
    else:
        rho = float(spearmanr(est, exact).correlation)
    return {"MSE": mse, "MAE": mae, "Precision@5": p5, "Spearman": rho}


# --------------------------------------------------------------------------- #
# Sweep
# --------------------------------------------------------------------------- #


def budget_grid(n: int) -> list[int]:
    lo, hi = n + 1, min(2 ** n, MAX_BUDGET)
    grid = np.round(np.logspace(np.log10(lo), np.log10(hi), N_BUDGET_STEPS)).astype(int)
    return sorted(set(int(b) for b in grid))


def run_sweep(data_root: Path, games: list[str], instances: int, methods: list[str],
              sampling_modes: list[bool], seed: int) -> pd.DataFrame:
    rows: list[dict] = []
    total = 0
    for gname in games:
        spec = GAME_SPECS[gname]
        budgets = budget_grid(spec.d)
        total += instances * len(budgets) * len(methods) * len(sampling_modes)

    done = 0
    for gname in games:
        spec = GAME_SPECS[gname]
        gdir = Path(data_root) / spec.subdir
        budgets = budget_grid(spec.d)
        for inst in range(1, instances + 1):
            path = gdir / spec.pattern.format(i=inst)
            if not path.exists():
                continue
            game, n = load_game(path)
            exact = exact_shapley(game, n)
            for paired in sampling_modes:
                for budget in budgets:
                    for mname in methods:
                        done += 1
                        try:
                            est_obj = make_method(mname, n, seed + inst, paired)
                        except Exception as exc:  # noqa: BLE001
                            rows.append(_row(spec, inst, paired, budget, mname,
                                             status=f"ctor_error:{type(exc).__name__}"))
                            continue
                        n_vars = getattr(est_obj, "n_variables", None)
                        if n_vars is not None and n_vars > budget:
                            rows.append(_row(spec, inst, paired, budget, mname,
                                             status="skipped:budget<n_variables"))
                            continue
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                t0 = time.perf_counter()
                                iv = est_obj.approximate(budget=budget, game=game)
                                runtime = time.perf_counter() - t0
                            est = np.asarray(iv.get_n_order_values(1), dtype=float)
                            metrics = compute_metrics(est, exact)
                            rows.append(_row(spec, inst, paired, budget, mname,
                                             status="ok", runtime=runtime, **metrics))
                        except Exception as exc:  # noqa: BLE001
                            rows.append(_row(spec, inst, paired, budget, mname,
                                             status=f"error:{type(exc).__name__}"))
                    if done % 250 == 0 or done == total:
                        print(f"  [{done:>6}/{total}] {gname} inst={inst} "
                              f"paired={paired} budget={budget}", flush=True)
    return pd.DataFrame(rows)


def cached_run_sweep(data_root, games: list[str], instances: int, methods: list[str],
                     sampling_modes: list[bool], seed: int, cache_dir) -> pd.DataFrame:
    """Run :func:`run_sweep`, caching the result CSV keyed by its parameters.

    The sweep (which includes the exact-Shapley ground truth) is the only
    expensive step. The cache key is a hash of the sweep parameters, so changing
    games / instances / methods / sampling / seed produces a fresh entry while an
    unchanged configuration loads instantly from disk.
    """
    import hashlib
    import json

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = {
        "games": sorted(games), "instances": instances, "methods": sorted(methods),
        "sampling": sorted("paired" if s else "standard" for s in sampling_modes),
        "seed": seed, "budget_steps": N_BUDGET_STEPS, "max_budget": MAX_BUDGET,
    }
    keystr = json.dumps(key, sort_keys=True)
    digest = hashlib.md5(keystr.encode()).hexdigest()[:10]
    csv = cache_dir / f"sweep_{digest}.csv"
    if csv.exists():
        print(f"[cache hit ] {csv.name}")
        return pd.read_csv(csv)
    print(f"[cache miss] computing -> {csv.name}")
    df = run_sweep(data_root, games, instances, methods, sampling_modes, seed)
    df.to_csv(csv, index=False)
    (cache_dir / f"sweep_{digest}.json").write_text(keystr, encoding="utf-8")
    return df


def _row(spec: GameSpec, inst: int, paired: bool, budget: int, method: str,
         status: str, runtime: float = float("nan"), **metrics) -> dict:
    row = {
        "game": spec.name, "d": spec.d, "domain": spec.domain,
        "instance": inst, "sampling": "paired" if paired else "standard",
        "budget": budget, "method": method, "status": status, "runtime": runtime,
    }
    for key in ("MSE", "MAE", "Precision@5", "Spearman"):
        row[key] = metrics.get(key, float("nan"))
    return row


# --------------------------------------------------------------------------- #
# Plotting (used by the CLI; the notebook has its own inline panels)
# --------------------------------------------------------------------------- #

COLORS = PAPER_COLORS  # use the paper's colour-per-algorithm everywhere


def _agg(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    ok = df[df["status"] == "ok"].copy()
    g = ok.groupby(["game", "sampling", "method", "budget"])[metric]
    out = g.agg(["mean", "sem", "count"]).reset_index()
    return out


def _sampling_regime(sub: pd.DataFrame, game: str) -> pd.DataFrame:
    """Drop the full-enumeration budget (>= 2**d) - exact there, MSE ~ 0."""
    return sub[sub["budget"] < 2 ** GAME_SPECS[game].d]


def plot_higher_order(df: pd.DataFrame, outdir: Path, metric: str = "MSE") -> None:
    """Fig. 2 - higher-order PolySHAP improves approximation (standard sampling)."""
    import matplotlib.pyplot as plt

    agg = _agg(df, metric)
    agg = agg[(agg["sampling"] == "standard") & (agg["method"].isin(HIGHER_ORDER))]
    games = [g for g in GAME_SPECS if g in set(agg["game"])]
    if not games:
        return
    ncol = len(games)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 3.6), squeeze=False)
    for ax, gname in zip(axes[0], games):
        sub = _sampling_regime(agg[agg["game"] == gname], gname)
        for m in HIGHER_ORDER:
            s = sub[sub["method"] == m].sort_values("budget")
            if s.empty:
                continue
            ax.plot(s["budget"], s["mean"], marker="o", ms=3, color=COLORS[m], label=m)
            ax.fill_between(s["budget"], s["mean"] - s["sem"], s["mean"] + s["sem"],
                            color=COLORS[m], alpha=0.18, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{gname} (d={GAME_SPECS[gname].d})")
        ax.set_xlabel("Budget (m)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
    axes[0][0].set_ylabel(metric)
    axes[0][-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Higher-order PolySHAP improves approximation quality (standard sampling)")
    fig.tight_layout()
    fig.savefig(outdir / f"fig2_higher_order_{metric}.png", dpi=150)
    plt.close(fig)


def plot_paired_vs_standard(df: pd.DataFrame, outdir: Path, metric: str = "MSE") -> None:
    """Fig. 3 - paired KernelSHAP (1-PolySHAP) matches 2-PolySHAP."""
    import matplotlib.pyplot as plt

    agg = _agg(df, metric)
    methods = ["1-PolySHAP", "2-PolySHAP"]
    agg = agg[agg["method"].isin(methods)]
    games = [g for g in GAME_SPECS if g in set(agg["game"])]
    if not games:
        return
    ncol = len(games)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 3.6), squeeze=False)
    styles = {"standard": (":", "o"), "paired": ("-", "s")}
    for ax, gname in zip(axes[0], games):
        sub = _sampling_regime(agg[agg["game"] == gname], gname)
        for m in methods:
            for samp in ("standard", "paired"):
                s = sub[(sub["method"] == m) & (sub["sampling"] == samp)].sort_values("budget")
                if s.empty:
                    continue
                ls, mk = styles[samp]
                ax.plot(s["budget"], s["mean"], ls=ls, marker=mk, ms=3, color=COLORS[m],
                        label=f"{m} ({samp})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{gname} (d={GAME_SPECS[gname].d})")
        ax.set_xlabel("Budget (m)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
    axes[0][0].set_ylabel(metric)
    axes[0][-1].legend(fontsize=7, loc="upper right")
    fig.suptitle("Paired KernelSHAP matches 2-PolySHAP (dotted=standard, solid=paired)")
    fig.tight_layout()
    fig.savefig(outdir / f"fig3_paired_vs_standard_{metric}.png", dpi=150)
    plt.close(fig)


def plot_runtime(df: pd.DataFrame, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    agg = _agg(df, "runtime")
    agg = agg[agg["sampling"] == "standard"]
    games = [g for g in GAME_SPECS if g in set(agg["game"])]
    if not games:
        return
    ncol = len(games)
    fig, axes = plt.subplots(1, ncol, figsize=(4.2 * ncol, 3.6), squeeze=False)
    for ax, gname in zip(axes[0], games):
        sub = agg[agg["game"] == gname]
        for m in ALL_METHODS:
            s = sub[sub["method"] == m].sort_values("budget")
            if s.empty:
                continue
            ax.plot(s["budget"], s["mean"], marker="o", ms=3, color=COLORS[m], label=m)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{gname} (d={GAME_SPECS[gname].d})")
        ax.set_xlabel("Budget (m)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
    axes[0][0].set_ylabel("Runtime (s)")
    axes[0][-1].legend(fontsize=7, loc="upper left")
    fig.suptitle("Runtime analysis (wall-clock per approximate call, standard sampling)")
    fig.tight_layout()
    fig.savefig(outdir / "runtime_analysis.png", dpi=150)
    plt.close(fig)


def performance_table(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    """Mean MSE per (game, method) at the largest *sub-exhaustive* budget.

    At ``budget >= 2**d`` the border trick makes every method exact (MSE ~ 0),
    a degenerate comparison point. We therefore evaluate at each game's largest
    budget strictly below ``2**d`` (the hardest sampling regime that still
    separates the methods), using paired sampling.
    """
    ok = df[(df["status"] == "ok") & (df["sampling"] == "paired")].copy()
    if ok.empty:
        ok = df[df["status"] == "ok"].copy()
    ok = ok[ok["budget"] < (2 ** ok["d"])]
    if ok.empty:
        return pd.DataFrame()
    idx = ok.groupby("game")["budget"].transform("max") == ok["budget"]
    top = ok[idx]
    ref = top.groupby("game")["budget"].first().to_dict()
    table = top.pivot_table(index="method", columns="game", values="MSE", aggfunc="mean")
    table = table.reindex([m for m in ALL_METHODS if m in table.index])
    table.to_csv(outdir / "performance_table.csv")
    with pd.option_context("display.float_format", lambda v: f"{v:.3e}"):
        print("\nPerformance table - mean MSE at largest sub-exhaustive budget (paired sampling):")
        print("reference budget per game:", {g: int(b) for g, b in ref.items()})
        print(table.to_string())
    return table


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--games", default=",".join(GAME_SPECS))
    parser.add_argument("--methods", default=",".join(ALL_METHODS))
    parser.add_argument("--instances", type=int, default=15)
    parser.add_argument("--seed", type=int, default=40)
    parser.add_argument("--standard-only", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="3 instances, higher-order methods only")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).parent / "results")
    args = parser.parse_args()

    games = [g.strip() for g in args.games.split(",") if g.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    instances = args.instances
    if args.quick:
        instances = 3
        methods = HIGHER_ORDER
    sampling_modes = [False] if args.standard_only else [False, True]

    if not Path(args.data_root).exists():
        print(f"ERROR: data root not found: {args.data_root}")
        return 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.output_root / f"repro_{stamp}"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Games:    {games}")
    print(f"Methods:  {methods}")
    print(f"Instances:{instances}  sampling={'standard+paired' if len(sampling_modes)==2 else 'standard'}")
    print(f"Output:   {outdir}\n")

    df = run_sweep(args.data_root, games, instances, methods, sampling_modes, args.seed)
    df.to_csv(outdir / "results.csv", index=False)
    n_ok = int((df["status"] == "ok").sum())
    print(f"\nRan {len(df)} cells ({n_ok} ok). CSV -> {outdir / 'results.csv'}")

    plots = outdir / "plots"
    plots.mkdir(exist_ok=True)
    plot_higher_order(df, plots, "MSE")
    if len(sampling_modes) == 2:
        plot_paired_vs_standard(df, plots, "MSE")
    plot_runtime(df, plots)
    performance_table(df, outdir)
    print(f"\nPlots -> {plots}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
