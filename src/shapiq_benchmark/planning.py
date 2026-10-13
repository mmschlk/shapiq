"""Bound a representative first panel before scheduling the full recipe matrix."""

from __future__ import annotations

import math
from collections import Counter, defaultdict

CORE_LIMITS = {
    "recipes": 16,
    "coalition_values": 8_388_608,
    "supported_cells": 40_000,
    "reference_coordinates": 1_000_000,
    "cpu_hours": 8_000,
    "storage_bytes": 20 * 1024**3,
}


def relative_budget_grid(ratios: list[float], games: list[dict]) -> dict:
    """Derive per-game query caps and their union from player-relative budgets."""
    grids = {
        game["id"]: sorted({math.ceil(ratio * game["n_players"]) for ratio in ratios})
        for game in games
    }
    return {
        "budgets_by_game": grids,
        "budgets": sorted({budget for grid in grids.values() for budget in grid}),
    }


def workload(kind: str, spec: dict, suite: dict, catalog: dict) -> dict:
    """Conservative resource bounds, including per-cell timeout and preparation cap.

    Shared preparation across targets is intentionally not discounted for live
    structured games. Storage allows 16 bytes per cached payoff/cost and 64 bytes
    per serialized estimate coordinate plus fixed record overhead.
    """
    n = spec["n_players"]
    targets = suite["targets"] if kind == "families" else [spec]
    instances = len(suite["game_seeds"])
    cells = coordinates = 0
    record_bytes = 0
    budgets = len({math.ceil(r * n) for r in suite["relative_budgets"]})
    repeats = len(suite["seeds"])
    for target in targets:
        size = sum(math.comb(n, order) for order in range(1, target["order"] + 1))
        supported = sum(
            target["index"] in catalog[method]["indices"] for method in suite["methods"]
        )
        target_cells = instances * budgets * repeats * supported
        cells += target_cells
        coordinates += instances * size
        record_bytes += target_cells * (2048 + 64 * size)
    payoffs = instances * 2**n if kind == "families" else 0
    return {
        "recipes": 1,
        "coalition_values": payoffs,
        "supported_cells": cells,
        "reference_coordinates": coordinates,
        "cpu_hours": instances * 8 + cells * 600 / 3600,
        "storage_bytes": 16 * payoffs + record_bytes,
    }


def select_core(entries: list[tuple[str, dict]], suite: dict, catalog: dict) -> tuple[list, dict]:
    """Round-robin constructions without inspecting estimator performance.

    Within each construction, prefer underrepresented datasets, models and player
    counts; break ties with forest/boosting profiles before recipe identifiers.
    Structured constructions start with SV, then rotate unused interaction targets.
    Deferred recipes remain in the declared inventory for subsequent expansion.
    Caps are campaign-phase totals, not per-batch allowances.
    """
    groups = defaultdict(list)
    for kind, spec in entries:
        groups[spec.get("family", spec.get("oracle", kind))].append((kind, spec))
    limits = {**CORE_LIMITS, **suite.get("core_limits", {})}
    if any(
        not isinstance(value, int | float) or not math.isfinite(value) or value <= 0
        for value in limits.values()
    ):
        message = "Core resource limits must be finite and positive."
        raise ValueError(message)
    totals = dict.fromkeys(CORE_LIMITS, 0)
    selected, deferred = [], []
    usage = {field: Counter() for field in ("dataset", "model_profile", "n_players")}
    target_usage, construction_targets = Counter(), Counter()

    def priority(entry: tuple[str, dict]) -> tuple:
        kind, spec = entry
        construction = spec.get("family", spec.get("oracle", kind))
        target = (spec.get("index"), spec.get("order"))
        target_priority = (
            (
                construction_targets[construction, target],
                0 if target == ("SV", 1) else 1 if target[1] == 2 else 2,
                target_usage[target],
            )
            if kind == "games"
            else (0, 0, 0)
        )
        model = spec.get("model_profile", "")
        return (
            *target_priority,
            *(usage[field][spec.get(field)] for field in usage),
            model not in {"random_forest", "xgboost"},
            spec["n_players"],
            spec.get("dataset", ""),
            model,
            spec["id"],
        )

    while any(groups.values()):
        for name in sorted(groups):
            group = groups[name]
            # A too-large candidate must not hide a smaller qualifying recipe.
            while group:
                entry = min(group, key=priority)
                group.remove(entry)
                kind, spec = entry
                cost = workload(kind, spec, suite, catalog)
                exceeded = [key for key in totals if totals[key] + cost[key] > limits[key]]
                if exceeded:
                    deferred.append({"id": spec["id"], "limits": exceeded, "workload": cost})
                    continue
                selected.append(entry)
                if kind == "games":
                    target = (spec["index"], spec["order"])
                    construction_targets[name, target] += 1
                    target_usage[target] += 1
                for field, counts in usage.items():
                    counts[spec.get(field)] += 1
                for key in totals:
                    totals[key] += cost[key]
                break
    return selected, {
        "protocol": "bounded-core-v1",
        "selection": "construction round-robin; structured SV first then target rotation; balance dataset/model/player counts; prefer forests and boosting on ties; no estimator scores",
        "limits": limits,
        "upper_bounds": totals,
        "selected": [spec["id"] for _, spec in selected],
        "deferred": deferred,
    }
