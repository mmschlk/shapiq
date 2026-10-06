"""Translate the reviewed 384-instance manifest into existing preparation adapters.

This command writes a suite only. It never fits models, submits jobs or publishes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from shapiq_benchmark.datasets import DATASETS, feature_limit
from shapiq_benchmark.families import dataset_compatibility
from shapiq_benchmark.protocol import (
    BUDGET_MULTIPLIERS,
    CONSTRUCTIONS,
    GAME_SEEDS,
    adapter_reason,
    recipe_spec,
)
from shapiq_benchmark.quality import QUALITY_PROTOCOL
from shapiq_benchmark.runner import METHOD_NAMES, validate_suite

TARGETS = [{"index": "SV", "order": 1}] + [
    {"index": index, "order": 2} for index in ("k-SII", "SII", "STII", "FSII", "FBII")
]
METHOD_PARAMETERS = {
    "LeverageSHAP": {"ridge": 0.001, "low_budget_equal_allocation": True},
    "OddSHAP": {"ridge": 0.0, "low_budget_equal_allocation": True},
}


def build_suite(manifest: Path) -> dict:
    """Validate the fixed design and require the approved estimator constructor options."""
    with manifest.open(newline="") as stream:
        recipes = list(csv.DictReader(stream))
    if (
        Counter(row["application"] for row in recipes)
        != {
            "local": 32,
            "data": 32,
            "features": 32,
        }
        or len({row["id"] for row in recipes}) != 96
    ):
        msg = "Focused design requires 96 unique recipes, 32 per application."
        raise ValueError(msg)
    suite: dict = {
        "name": "focused-384-v1",
        "families": [],
        "games": [],
        "methods": list(METHOD_NAMES),
        "method_parameters": copy.deepcopy(METHOD_PARAMETERS),
        "targets": copy.deepcopy(TARGETS),
        "relative_budgets": list(BUDGET_MULTIPLIERS),
        "game_seeds": list(GAME_SEEDS),
        "seeds": [0],
        "min_players": 12,
        "min_signal_ratio": 1e-6,
        "cell_timeout_policy": {
            "ordinary_seconds": 30,
            "extended_seconds": 120,
            "min_players": 128,
            "min_relative_budget": 32,
        },
        "focused_design": {
            "version": 1,
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "recipes": recipes,
            "application_weights": dict.fromkeys(("local", "data", "features"), 1 / 3),
            "qualification": "pending: intended instances are not qualified instances",
        },
    }
    identities = set()
    for recipe in recipes:
        family, dataset, model = (recipe[key] for key in ("construction", "dataset", "model"))
        n = int(recipe["n_players"])
        structured = recipe["reference"] in {"tree_solver", "neighbor_solver"}
        identity = (family, dataset, model, n)
        if identity in identities:
            msg = f"Repeated game recipe: {recipe['id']}"
            raise ValueError(msg)
        identities.add(identity)
        if family not in CONSTRUCTIONS or dataset not in DATASETS:
            msg = f"Unknown construction or dataset: {recipe['id']}"
            raise ValueError(msg)
        if model not in CONSTRUCTIONS[family]["models"] or not 12 <= n <= 512:
            msg = f"Unsupported model or player count: {recipe['id']}"
            raise ValueError(msg)
        if recipe["reference"] not in {"enumeration", "tree_solver", "neighbor_solver"}:
            msg = f"Unknown exact-reference adapter: {recipe['id']}"
            raise ValueError(msg)
        if not structured and n > 20:
            msg = "Enumeration is limited to twenty players."
            raise ValueError(msg)
        if CONSTRUCTIONS[family]["player_unit"] == "feature" and n > feature_limit(family, dataset):
            msg = f"Too few original features: {recipe['id']}"
            raise ValueError(msg)
        reason = dataset_compatibility(family, dataset) or adapter_reason(
            family, dataset, model, structured=structured
        )
        if reason:
            msg = f"{recipe['id']}: {reason}"
            raise ValueError(msg)
        row = {
            "id": recipe["id"],
            "family": family,
            "dataset": dataset,
            "model_profile": model,
            "n_players": n,
            "reference": "structured" if structured else "enumeration",
        }
        for spec in recipe_spec(row, TARGETS):
            spec["quality_protocol"] = QUALITY_PROTOCOL
            if "device" in spec:
                spec["device"] = "cpu"
            if structured:
                spec["basecase_id"] = recipe["id"]
                if (dataset, model, family) in {
                    ("tabarena_bioresponse", "xgboost", "pathdependent_tree"),
                    ("tabarena_qsar_tid11", "random_forest", "interventional_tree"),
                } and n in {32, 128, 512}:
                    spec["feature_rule"] = "nested"
            suite["games" if structured else "families"].append(spec)
    # Refuse old constructors rather than silently dropping the approved fallback.
    validate_suite(suite)
    return suite


def main() -> None:
    """Generate a new suite file; never overwrite an existing campaign input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    suite = build_suite(args.manifest)
    with args.output.open("x") as stream:
        json.dump(suite, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
