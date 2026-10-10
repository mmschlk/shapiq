"""Translate a reviewed focused manifest into existing preparation adapters.

This command writes a suite only. It never fits models, submits jobs or publishes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import re
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


_REQUIRED_COLUMNS = {
    "id",
    "application",
    "subtype",
    "dataset",
    "model",
    "construction",
    "n_players",
    "reference",
}
_OPTIONAL_COLUMNS = {"input_features", "feature_rule", "training_rows", "row_selection"}
_APPLICATION_CONSTRUCTIONS = {
    "local": {
        "local_baseline",
        "local_marginal",
        "local_gaussian",
        "local_copula",
        "local_conditional",
        "pathdependent_tree",
        "interventional_tree",
    },
    "data": {"dataset_valuation", "data_valuation", "knn", "tnn"},
    "features": {"feature_selection"},
}


def build_suite(
    manifest: Path,
    *,
    min_players: int = 12,
    max_players: int = 512,
    extended_features: bool = False,
    estimator_seeds: tuple[int, ...] = (0,),
) -> dict:
    """Validate recipe semantics and require the approved estimator constructor options."""
    if not 1 <= min_players <= max_players:
        msg = "Invalid player bounds."
        raise ValueError(msg)
    with manifest.open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if (
            not _REQUIRED_COLUMNS.issubset(columns)
            or set(columns) - _REQUIRED_COLUMNS - _OPTIONAL_COLUMNS
            or len(columns) != len(set(columns))
        ):
            msg = "Focused manifest has missing, duplicate or unknown columns."
            raise ValueError(msg)
        recipes = list(reader)
    if not recipes or any(
        None in row or any(value is None for value in row.values()) for row in recipes
    ):
        msg = "Focused manifest must contain complete recipe rows."
        raise ValueError(msg)
    # Blank optional columns retain the original recipe dictionaries and defaults.
    recipes = [
        {key: value for key, value in row.items() if key not in _OPTIONAL_COLUMNS or value}
        for row in recipes
    ]
    if len({row["id"] for row in recipes}) != len(recipes) or any(
        not re.fullmatch(r"[a-zA-Z0-9_-]+", row["id"]) for row in recipes
    ):
        msg = "Focused recipes require unique, path-safe IDs."
        raise ValueError(msg)
    suite: dict = {
        "name": f"focused-{len(recipes) * len(GAME_SEEDS)}-v1",
        "families": [],
        "games": [],
        "methods": list(METHOD_NAMES),
        "method_parameters": copy.deepcopy(METHOD_PARAMETERS),
        "targets": copy.deepcopy(TARGETS),
        "relative_budgets": list(BUDGET_MULTIPLIERS),
        "game_seeds": list(GAME_SEEDS),
        "seeds": list(estimator_seeds),
        "min_players": min_players,
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
        if family not in CONSTRUCTIONS or dataset not in DATASETS:
            msg = f"Unknown construction or dataset: {recipe['id']}"
            raise ValueError(msg)
        if family not in _APPLICATION_CONSTRUCTIONS.get(recipe["application"], set()):
            msg = f"Construction does not belong to the application: {recipe['id']}"
            raise ValueError(msg)
        expected_subtype = (
            "neighbor_examples"
            if family in {"knn", "tnn"}
            else "retraining_groups"
            if family == "dataset_valuation"
            else "retraining_rows"
            if family == "data_valuation"
            else ""
        )
        if recipe["subtype"] != expected_subtype:
            msg = f"Unexpected application subtype: {recipe['id']}"
            raise ValueError(msg)
        if model not in CONSTRUCTIONS[family]["models"] or not min_players <= n <= max_players:
            msg = f"Unsupported model or player count: {recipe['id']}"
            raise ValueError(msg)
        if recipe["reference"] not in {"enumeration", "tree_solver", "neighbor_solver"}:
            msg = f"Unknown exact-reference adapter: {recipe['id']}"
            raise ValueError(msg)
        if (
            recipe["reference"] == "tree_solver"
            and family not in {"pathdependent_tree", "interventional_tree"}
        ) or (recipe["reference"] == "neighbor_solver" and family not in {"knn", "tnn"}):
            msg = f"Exact-reference adapter does not match construction: {recipe['id']}"
            raise ValueError(msg)
        options = {}
        if "input_features" in recipe:
            count = int(recipe["input_features"])
            if family not in (
                {"data_valuation", "dataset_valuation"}
                | ({"knn", "tnn"} if extended_features else set())
            ) or not (1 <= count <= DATASETS[dataset]["n_features"]):
                msg = f"Invalid valuation input-feature count: {recipe['id']}"
                raise ValueError(msg)
            options["input_features"] = count
        if "feature_rule" in recipe:
            if family not in (
                {"data_valuation", "dataset_valuation", "feature_selection"}
                | (
                    {"local_baseline", "pathdependent_tree", "interventional_tree", "knn", "tnn"}
                    if extended_features
                    else set()
                )
            ) or (recipe["feature_rule"] not in {"all", "nested"}):
                msg = f"Invalid retraining feature rule: {recipe['id']}"
                raise ValueError(msg)
            options["feature_rule"] = recipe["feature_rule"]
        if "training_rows" in recipe or "row_selection" in recipe:
            if not extended_features or family not in {"knn", "tnn"}:
                msg = f"Neighbor options require a neighbor recipe: {recipe['id']}"
                raise ValueError(msg)
            if "training_rows" not in recipe or "row_selection" not in recipe:
                msg = "Explicit neighbor pools require both size and selection rule."
                raise ValueError(msg)
            pool = int(recipe["training_rows"])
            if pool < n or recipe["row_selection"] != "nested_stratified":
                msg = f"Invalid neighbor fitting pool: {recipe['id']}"
                raise ValueError(msg)
            options.update(training_rows=pool, row_selection=recipe["row_selection"])
        # Explicit legacy defaults must not create a second identity for the same game.
        input_count = options.get("input_features", min(12, DATASETS[dataset]["n_features"]))
        identity = (
            family,
            dataset,
            model,
            n,
            input_count,
            options.get("feature_rule", "all"),
            options.get("training_rows"),
            options.get("row_selection"),
        )
        if identity in identities:
            msg = f"Repeated game recipe: {recipe['id']}"
            raise ValueError(msg)
        identities.add(identity)
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
            spec.update(options)
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
