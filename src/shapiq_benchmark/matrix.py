"""Expand a declared dataset/player matrix without constructing games or changing payoffs."""

from __future__ import annotations

import argparse
import copy
import json
import math
import re
from pathlib import Path

from shapiq_benchmark.families import (
    DATASETS,
    FAMILY_CATALOG,
    MAX_ENUMERATION_PLAYERS,
    dataset_compatibility,
)

REASON_LABELS = {
    "dataset_recipe": "Dataset/task is not supported by this bounded recipe adapter",
    "requires_class_labels": "This recipe requires class labels",
    "binary_calendar_features": "Gaussian imputation rejects Bike Sharing's binary calendar features",
    "requires_binary_target": "The product-kernel classifier requires two classes",
    "below_minimum": "Below the suite's minimum player count",
    "insufficient_features": "Fewer native features than requested players; no padding",
    "unqualified_large_adapter": "No qualified large-player ground-truth adapter for this recipe configuration",
    "exhaustive_table": "Exact enumeration of a frozen coalition table planned",
    "qualified_structured": "Existing qualified structured adapter; exact target/configuration retained",
}
COMPATIBILITY_REASONS = {
    "Requires class labels.": "requires_class_labels",
    "Gaussian imputation rejects the binary calendar features.": "binary_calendar_features",
    "The product-kernel classifier requires two classes.": "requires_binary_target",
}
DEFINITION_FIELDS = (
    "name",
    "datasets",
    "families",
    "player_counts",
    "include_native_width",
    "min_signal_ratio",
)


def expand_matrix(config: dict, base: dict) -> dict:
    """Select bounded exact-table recipes and retain explicit exclusions in suite identity."""
    if (
        set(config) != {*DEFINITION_FIELDS, "base_suite"}
        or not isinstance(config["name"], str)
        or not re.fullmatch(r"[a-zA-Z0-9_-]+", config["name"])
        or not isinstance(config["base_suite"], str)
        or type(config["include_native_width"]) is not bool
        or type(config["min_signal_ratio"]) not in (int, float)
        or not math.isfinite(config["min_signal_ratio"])
        or config["min_signal_ratio"] <= 0
    ):
        message = "Matrix configuration has invalid fields or types"
        raise ValueError(message)
    recipes = {name for name, info in FAMILY_CATALOG.items() if not info["synthetic"]} | {"tabpfn"}
    for key, registry in (("datasets", DATASETS), ("families", recipes)):
        values = config[key]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or value not in registry for value in values)
            or len(set(values)) != len(values)
        ):
            message = f"Matrix {key} must be a nonempty list of unique registered names"
            raise ValueError(message)
    counts = config["player_counts"]
    if (
        not isinstance(counts, list)
        or not counts
        or any(type(count) is not int or count <= 0 for count in counts)
        or len(set(counts)) != len(counts)
    ):
        message = "Matrix player counts must be unique positive integers"
        raise ValueError(message)
    suite = copy.deepcopy(base)
    selected, candidates = [], []
    minimum = base["min_players"]
    for family in config["families"]:
        unit = "feature" if family == "tabpfn" else FAMILY_CATALOG[family]["player_unit"]
        for dataset in config["datasets"]:
            info = DATASETS[dataset]
            compatibility = (
                (None if info["task"] == "classification" else "Requires class labels.")
                if family == "tabpfn"
                else dataset_compatibility(family, dataset)
            )
            counts = set(config["player_counts"])
            if config["include_native_width"] and unit == "feature":
                counts.add(info["n_features"])
            for count in sorted(counts):
                reason = None
                if compatibility:
                    reason = COMPATIBILITY_REASONS.get(compatibility, "dataset_recipe")
                elif count < minimum:
                    reason = "below_minimum"
                elif unit == "feature" and count > info["n_features"]:
                    reason = "insufficient_features"
                elif count > MAX_ENUMERATION_PLAYERS:
                    reason = "unqualified_large_adapter"
                row = {
                    "recipe": family,
                    "dataset": dataset,
                    "player_unit": unit,
                    "n_players": count,
                    "status": "excluded" if reason else "selected",
                    "reason": reason or "exhaustive_table",
                }
                candidates.append(row)
                if reason is None:
                    selected.append(
                        {
                            "id": f"{family}-{dataset}-{count}",
                            "family": family,
                            "dataset": dataset,
                            "n_players": count,
                        }
                    )

    # Media, causal, and synthetic settings are outside the tabular cross-product.
    extras = [spec for spec in base["families"] if spec["family"] not in recipes]
    suite["families"] = selected + extras
    suite["name"] = config["name"]
    suite["min_signal_ratio"] = config["min_signal_ratio"]
    suite["matrix_definition"] = {key: copy.deepcopy(config[key]) for key in DEFINITION_FIELDS}
    structured = []
    for spec in base["games"]:
        target = (spec["index"], spec["order"])
        supported = (
            target in (("SV", 1), ("k-SII", 2), ("SII", 2), ("STII", 2), ("FSII", 2), ("FBII", 2))
            if spec["oracle"] == "tree"
            else spec["oracle"] in ("knn", "product_kernel") and target == ("SV", 1)
        )
        if not supported or spec["dataset"] not in ("breast_cancer", "digits"):
            message = "Retained structured configuration has no qualified dataset/target adapter"
            raise ValueError(message)
        native = DATASETS[spec["dataset"]]["n_features"]
        count = spec.get("n_players", 128 if spec["oracle"] == "knn" else native)
        if type(count) is not int or count < minimum:
            message = "A retained structured configuration has an invalid player count"
            raise ValueError(message)
        if spec["oracle"] != "knn" and count != native:
            message = "Structured tree/kernel recipes require the native feature width"
            raise ValueError(message)
        structured.append(
            {
                **{key: spec[key] for key in ("id", "dataset", "oracle", "index", "order")},
                "recipe": f"structured_{spec['oracle']}",
                "n_players": count,
                "status": "selected",
                "reason": "qualified_structured",
            }
        )
    if any(
        spec["n_players"] < minimum or spec["n_players"] > MAX_ENUMERATION_PLAYERS
        for spec in extras
    ):
        message = "Non-tabular table recipes must retain the bounded player range"
        raise ValueError(message)
    if len({spec["id"] for spec in suite["families"]}) != len(suite["families"]):
        message = "The matrix contains duplicate recipe IDs"
        raise ValueError(message)
    definitions = len(base["game_seeds"]) * (
        len(suite["families"]) * len(base["targets"]) + len(base["games"])
    )
    suite["matrix_coverage"] = {
        "selection_note": f"Exact table selection is limited to {MAX_ENUMERATION_PLAYERS} players. Selected means planned, not measured; runtime qualification is recorded separately. Structured entries have distinct payoffs and target support, not replacements for excluded generic recipes.",
        "reason_labels": REASON_LABELS,
        "table_targets": base["targets"],
        "candidates": candidates,
        "structured": structured,
        "extra_settings": extras,
        "counts": {
            "requested_table_candidates": len(candidates),
            "selected_table_candidates": len(selected),
            "excluded_table_candidates": len(candidates) - len(selected),
            "non_tabular_settings": len(extras),
            "structured_settings": len(
                {(row["dataset"], row["oracle"], row["n_players"]) for row in structured}
            ),
            "structured_target_configurations": len(structured),
            "game_definitions": definitions,
        },
    }
    return suite


def main() -> None:
    """Resolve the base suite relative to the matrix config and write one frozen expansion."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    base_path = args.config.parent / config["base_suite"]
    if args.output.resolve() in (args.config.resolve(), base_path.resolve()):
        parser.error("Output must not overwrite the config or base suite")
    suite = expand_matrix(config, json.loads(base_path.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, allow_nan=False) + "\n")
    print(json.dumps(suite["matrix_coverage"]["counts"], sort_keys=True))  # noqa: T201 -- CLI summary


if __name__ == "__main__":
    main()
