"""Observed-cell weighted accuracy, paired comparisons, and complete-panel history.

Families, strata, games, budgets, then seeds receive equal conditional weight.
Elo uses a batch Bradley-Terry fit: weighted ties count as half wins, logit skills
have L2 penalty ``0.001 * sum(skill**2) / 2``, and ratings are centered at 1000.
Intervals resample independent model clusters within strata, retaining all their
points and pairing seed slots across methods and budgets. They are descriptive
percentile intervals, and require at least two independent clusters in every stratum.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

# Earliest public descriptions verified in benchmark/DESIGN.md. Unverified aliases
# and implementations intentionally stay unknown rather than receiving guessed dates.
RELEASES = {
    "StratifiedSamplingSV": ("2013-06-18", "https://arxiv.org/abs/1306.4265"),
    "KernelSHAP": ("2017-05-22", "https://arxiv.org/abs/1705.07874"),
    "OwenSamplingSV": ("2020-10-22", "https://arxiv.org/abs/2010.12082"),
    "UnbiasedKernelSHAP": ("2020-12-02", "https://arxiv.org/abs/2012.01536"),
    "kADDSHAP": ("2022-11-03", "https://arxiv.org/abs/2211.02166"),
    "SVARM": ("2023-02-01", "https://arxiv.org/abs/2302.00736"),
    "SHAPIQ": ("2023-03-02", "https://arxiv.org/abs/2303.01179"),
    "SVARMIQ": ("2024-01-24", "https://arxiv.org/abs/2401.13371"),
    "KernelSHAPIQ": ("2024-05-17", "https://arxiv.org/abs/2405.10852"),
    "LeverageSHAP": ("2024-10-02", "https://arxiv.org/abs/2410.01917"),
    "SPEX": ("2025-02-19", "https://arxiv.org/abs/2502.13870"),
    "ProxySPEX": ("2025-05-23", "https://arxiv.org/abs/2505.17495"),
    "RegressionMSR": ("2025-06-13", "https://arxiv.org/abs/2506.11849"),
    "OddSHAP": ("2026-02-01", "https://arxiv.org/abs/2602.01399"),
    "ShaplEIG": ("2026-06-01", "https://arxiv.org/abs/2606.02247"),
}
HISTORY_LABEL = (
    "Retrospective performance of current implementations, grouped by method release date."
)
L2 = 0.001


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted median, averaging neighbors at exactly half the mass."""
    positive = weights > 0
    values, weights = values[positive], weights[positive]
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order]) / math.fsum(weights)
    position = min(np.searchsorted(cumulative + 1e-14, 0.5, side="left"), len(order) - 1)
    value = float(values[order[position]])
    if abs(cumulative[position] - 0.5) <= 1e-14 and position + 1 < len(order):
        return value / 2 + float(values[order[position + 1]]) / 2
    return value


def budget_grid(budgets: list[int] | dict[str, list[int]], game: dict) -> list[int]:
    """Return the planned absolute budget grid for one game."""
    return budgets[game["id"]] if isinstance(budgets, dict) else budgets


def weights_for(
    games: list[dict], budgets: list[int] | dict[str, list[int]], seeds: list[int]
) -> tuple[list[tuple], np.ndarray]:
    """Enumerate the common panel and its fixed hierarchical weights."""
    families = sorted({game["family"] for game in games})
    cells, weights = [], []
    for family in families:
        strata = sorted({game["stratum"] for game in games if game["family"] == family})
        for stratum in strata:
            group = [
                game for game in games if (game["family"], game["stratum"]) == (family, stratum)
            ]
            for game in group:
                grid = budget_grid(budgets, game)
                weight = 1 / (len(families) * len(strata) * len(group) * len(grid) * len(seeds))
                for budget, seed in itertools.product(grid, seeds):
                    cells.append((game["id"], budget, seed))
                    weights.append(weight)
    return cells, np.array(weights)


def comparisons(
    values: np.ndarray, weights: np.ndarray, methods: list[str]
) -> tuple[list[dict], list[float] | None]:
    """Fit Elo on finite overlaps with original panel mass; require a connected pool."""
    matches, a_indices, b_indices, scores, masses = [], [], [], [], []
    neighbors = [set() for _ in methods]
    for a, b in itertools.combinations(range(len(methods)), 2):
        observed = np.isfinite(values[a]) & np.isfinite(values[b])
        if not observed.any():
            continue
        left, right, pair_weights = values[a, observed], values[b, observed], weights[observed]
        tolerance = 1e-12 + 0.01 * np.maximum(left, right)
        ties = np.abs(left - right) <= tolerance
        wins = (left < right) & ~ties
        losses = ~wins & ~ties
        win_weight, tie_weight, loss_weight = [
            float(pair_weights[mask].sum()) for mask in (wins, ties, losses)
        ]
        score = win_weight + tie_weight / 2
        matches.append(
            {
                "a": methods[a],
                "b": methods[b],
                "wins": int(wins.sum()),
                "ties": int(ties.sum()),
                "losses": int(losses.sum()),
                "n_pairs": int(observed.sum()),
                "observed_weight": float(pair_weights.sum()),
                "win_weight": win_weight,
                "tie_weight": tie_weight,
                "loss_weight": loss_weight,
                "score_a": score,
            }
        )
        a_indices.append(a)
        b_indices.append(b)
        scores.append(score)
        masses.append(float(pair_weights.sum()))
        neighbors[a].add(b)
        neighbors[b].add(a)
    if len(methods) < 2:
        return matches, None
    reached, pending = set(), [0]
    while pending:
        current = pending.pop()
        if current not in reached:
            reached.add(current)
            pending.extend(neighbors[current] - reached)
    if len(reached) != len(methods):
        return matches, None
    a_indices, b_indices, scores, masses = map(np.asarray, (a_indices, b_indices, scores, masses))

    def objective(skills: np.ndarray) -> tuple[float, np.ndarray]:
        difference = skills[a_indices] - skills[b_indices]
        loss = (
            np.sum(masses * np.logaddexp(0, difference) - scores * difference)
            + L2 * np.dot(skills, skills) / 2
        )
        residual = masses * expit(difference) - scores
        gradient = L2 * skills
        np.add.at(gradient, a_indices, residual)
        np.add.at(gradient, b_indices, -residual)
        return float(loss), gradient

    fit = minimize(
        objective,
        np.zeros(len(methods)),
        jac=True,
        method="L-BFGS-B",
        # A tighter gradient tolerance can falsely fail at line-search roundoff.
        options={"gtol": 1e-8, "ftol": 1e-12, "maxiter": 1000, "maxls": 100},
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        message = "Bradley-Terry fit did not converge."
        raise ValueError(message)
    return matches, (1000 + (fit.x - fit.x.mean()) * 400 / np.log(10)).tolist()


def release_history(rows: list[dict]) -> dict:
    """Build dated horizontal method lines and separate mean/median frontiers."""
    eligible = [row for row in rows if row.get("complete", row["eligible"])]
    methods: list[dict] = [
        {
            "method": row["method"],
            "date": RELEASES[row["method"]][0],
            "url": RELEASES[row["method"]][1],
            "date_precision": "day",
            "mean": row["mean"],
            "median": row["median"],
        }
        for row in eligible
        if row["method"] in RELEASES
    ]
    methods.sort(key=lambda item: (item["date"], item["method"]))
    history = {
        "label": HISTORY_LABEL,
        "methods": methods,
        "unknown_dates": sorted(row["method"] for row in eligible if row["method"] not in RELEASES),
    }
    for metric in ("mean", "median"):
        frontier, best = [], math.inf
        for date in sorted({item["date"] for item in methods}):
            winner = min(
                (item for item in methods if item["date"] == date),
                key=lambda item: (item[metric], item["method"]),
            )
            if winner[metric] < best:
                best = winner[metric]
                frontier.append({"date": date, "value": best, "method": winner["method"]})
        history[metric] = frontier
    return history


def bootstrap(
    games: list[dict],
    budgets: list[int] | dict[str, list[int]],
    seeds: list[int],
    cells: list[tuple],
    values: np.ndarray,
    methods: list[str],
    *,
    draws: int,
) -> tuple[dict, dict]:
    """Resample complete clusters and paired seed slots, never individual method outcomes."""
    groups = {}
    for game in games:
        key = (game["family"], game["stratum"])
        cluster = str(game.get("metadata", {}).get("cluster_id", game["stratum"]))
        groups.setdefault(key, {}).setdefault(cluster, []).append(game)
    if (
        not groups
        or not methods
        or draws < 2
        or any(len(clusters) < 2 for clusters in groups.values())
    ):
        return {}, {
            "available": False,
            "reason": "At least two independent model clusters per stratum and a complete method are required.",
            "draws": 0,
        }
    generator = np.random.default_rng(0)
    lookup = {cell: i for i, cell in enumerate(cells)}
    samples = {name: {"mean": [], "median": [], "elo": []} for name in methods}
    families = {key[0] for key in groups}
    for _ in range(draws):
        positions, weights = [], []
        for (family, _stratum), clusters in sorted(groups.items()):
            names = sorted(clusters)
            selected = generator.choice(names, size=len(names), replace=True)
            n_games = sum(len(clusters[name]) for name in selected)
            n_strata = sum(key[0] == family for key in groups)
            for cluster in selected:
                sampled_seeds = generator.choice(seeds, size=len(seeds), replace=True)
                for game in clusters[cluster]:
                    grid = budget_grid(budgets, game)
                    weight = 1 / (len(families) * n_strata * n_games * len(grid) * len(seeds))
                    for budget, seed in itertools.product(grid, sampled_seeds):
                        positions.append(lookup[(game["id"], budget, int(seed))])
                        weights.append(weight)
        weights = np.asarray(weights)
        selected_values = values[:, positions]
        _, ratings = comparisons(selected_values, weights, methods)
        for i, name in enumerate(methods):
            samples[name]["mean"].append(float(np.dot(selected_values[i], weights)))
            samples[name]["median"].append(weighted_median(selected_values[i], weights))
            if ratings is not None:
                samples[name]["elo"].append(ratings[i])
    intervals = {
        name: {
            metric: np.quantile(sample, [0.025, 0.975]).tolist() if sample else None
            for metric, sample in metrics.items()
        }
        for name, metrics in samples.items()
    }
    return intervals, {
        "available": True,
        "reason": "Paired model-cluster and seed-slot percentile bootstrap; 95% descriptive intervals.",
        "draws": draws,
    }


def summarize(data: dict, *, bootstrap_draws: int = 200) -> list[dict]:
    """Summarize observed cells; preserve complete panels for uncertainty and history."""
    methods = sorted(data["methods"])
    seeds = sorted(data["suite"]["seeds"])
    suite = data["suite"]
    budgets = sorted(suite["budgets"])
    game_grids = {
        game["id"]: sorted(suite["budgets_by_game"][game["id"]])
        if "budgets_by_game" in suite
        else budgets
        for game in data["games"]
    }
    records = {
        (row["game_id"], row["method"], row["budget"], row["seed"]): row for row in data["records"]
    }
    if len(records) != len(data["records"]):
        message = "Summary input contains duplicate result cells."
        raise ValueError(message)
    zero_games = {
        game["id"] for game in data["games"] if game.get("metadata", {}).get("zero_truth_energy")
    }
    zero_games.update(row["game_id"] for row in data["records"] if row.get("zero_truth_energy"))
    targets = sorted(
        {
            (game["index"], game["order"], bool(game.get("metadata", {}).get("synthetic")))
            for game in data["games"]
        }
    )
    summaries = []
    for index, order, synthetic in targets:
        target_games = sorted(
            [
                game
                for game in data["games"]
                if (
                    game["index"],
                    game["order"],
                    bool(game.get("metadata", {}).get("synthetic")),
                )
                == (index, order, synthetic)
            ],
            key=lambda game: game["id"],
        )
        subsets = [(None, target_games)]
        subsets.extend(
            (family, [game for game in target_games if game["family"] == family])
            for family in sorted({game["family"] for game in target_games})
        )
        seen = set()
        for family, panel_games in subsets:
            grids = [(None, {game["id"]: game_grids[game["id"]] for game in panel_games})]
            if "budgets_by_game" in suite:
                grids.extend(
                    (
                        ratio,
                        {
                            game["id"]: [math.ceil(ratio * game["n_players"])]
                            for game in panel_games
                        },
                    )
                    for ratio in sorted(suite.get("relative_budgets", []))
                )
            else:
                grids.extend(
                    (None, {game["id"]: [budget] for game in panel_games}) for budget in budgets
                )
            for relative_budget, grid in grids:
                signature = tuple((game_id, tuple(values)) for game_id, values in grid.items())
                if signature in seen:
                    continue
                seen.add(signature)
                games = [game for game in panel_games if game["id"] not in zero_games]
                cells, weights = weights_for(games, grid, seeds)
                rows, eligible, values = [], [], []
                for method in methods:
                    measured = [
                        records.get((game, method, budget, seed)) for game, budget, seed in cells
                    ]
                    sample = np.array(
                        [
                            row["nmse"]
                            if row and row["status"] == "ok" and row.get("nmse") is not None
                            else np.nan
                            for row in measured
                        ]
                    )
                    observed = np.isfinite(sample)
                    valid = int(observed.sum())
                    mass = float(weights[observed].sum())
                    complete = len(cells) > 0 and valid == len(cells)
                    rows.append(
                        {
                            "method": method,
                            "eligible": valid > 0,
                            "complete": complete,
                            "coverage_weight": mass,
                            "planned": len(cells),
                            "valid": valid,
                            "failed": sum(
                                row is not None and row["status"] == "failed" for row in measured
                            ),
                            "unsupported": sum(
                                row is not None and row["status"] == "unsupported"
                                for row in measured
                            ),
                            "missing": sum(row is None for row in measured),
                            "mean": float(np.dot(sample[observed], weights[observed]) / mass)
                            if valid
                            else None,
                            "median": weighted_median(sample[observed], weights[observed])
                            if valid
                            else None,
                            "elo": None,
                            "ci": None,
                        }
                    )
                    if valid:
                        eligible.append(method)
                        values.append(sample)
                array = np.array(values)
                _, ratings = comparisons(array, weights, eligible)
                complete_methods = [row["method"] for row in rows if row["complete"]]
                complete_values = array[[eligible.index(name) for name in complete_methods]]
                intervals, uncertainty = bootstrap(
                    games,
                    grid,
                    seeds,
                    cells,
                    complete_values,
                    complete_methods,
                    draws=bootstrap_draws,
                )
                if complete_methods != eligible and intervals:
                    for interval in intervals.values():
                        interval["elo"] = None
                    uncertainty["reason"] += (
                        " Elo intervals withheld: partial competitors are outside the complete-panel bootstrap."
                    )
                for row in rows:
                    if row["eligible"]:
                        row["elo"] = (
                            ratings[eligible.index(row["method"])] if ratings is not None else None
                        )
                        row["ci"] = intervals.get(row["method"])
                key = {
                    "index": index,
                    "order": order,
                    "family": family,
                    "game_ids": [game["id"] for game in panel_games],
                    "budgets": sorted({budget for values in grid.values() for budget in values}),
                    "game_budgets": grid,
                    "relative_budget": relative_budget,
                    "panel": "diagnostic" if synthetic else "real",
                    "seeds": seeds,
                    "methods": methods,
                }
                identity = {
                    **key,
                    "summary_protocol": "available-cells-v2",
                    "median_convention": "midpoint at exactly half the cumulative weight",
                    "snapshot_id": data.get("snapshot_id"),
                    "method_sources": data["methods"],
                }
                summaries.append(
                    {
                        **key,
                        "summary_protocol": "available-cells-v2",
                        "median_convention": "midpoint at exactly half the cumulative weight",
                        "id": hashlib.sha256(
                            json.dumps(identity, sort_keys=True).encode()
                        ).hexdigest(),
                        "excluded_zero_energy_games": sorted(
                            game["id"] for game in panel_games if game["id"] in zero_games
                        ),
                        "rows": rows,
                        "history": release_history(rows),
                        "uncertainty": uncertainty,
                        "elo_l2": L2,
                        "weighting": "equal family / stratum / game / budget / seed; renormalized over observed cells",
                        "elo_pairing": "finite overlapping cells; original panel weight; connected pool required",
                    }
                )
    return summaries
