"""Deterministic 24-dataset benchmark design using existing preparation adapters.

Recipe IDs encode scientific options rather than row positions. Existing campaign
results may be reused only after exact recipe and frozen-artifact comparison.
This module constructs metadata; it never loads datasets, fits models or submits jobs.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from collections import Counter
from pathlib import Path

from shapiq_benchmark import focused
from shapiq_benchmark.datasets import DATASETS

CLASSIFICATION = (
    "adult_census",
    "mushroom",
    "ionosphere",
    "tabarena_bioresponse",
    "breast_cancer",
    "digits",
    "wine",
    "tabarena_amazon_employee_access",
    "tabarena_aps_failure",
    "tabarena_anneal",
    "tabarena_splice",
    "tabarena_credit_card_default",
)
REGRESSION = (
    "wine_quality",
    "communities_and_crime",
    "nhanesi",
    "tabarena_qsar_tid11",
    "tabarena_miami_housing",
    "tabarena_superconductivity",
    "california_housing",
    "diabetes",
    "bike_sharing",
    "tabarena_airfoil_self_noise",
    "tabarena_concrete_strength",
    "tabarena_protein",
)
CORE_MODELS = ("linear", "random_forest")
GENERIC_COUNTS = (4, 8, 12)
TREE_COUNTS = (8, 16, 32)
RELATIVE_BUDGETS = (0.5, 1, 2, 4, 8, 16, 32)
COLUMNS = (
    "id",
    "application",
    "subtype",
    "dataset",
    "model",
    "construction",
    "n_players",
    "reference",
    "input_features",
    "feature_rule",
    "training_rows",
    "row_selection",
)


def recipes() -> list[dict]:
    """Return unique recipes in stable design order, without reading dataset values."""
    rows: dict[tuple, dict] = {}

    def add(
        dataset: str, model: str, construction: str, players: int, *, inputs: int | None = None
    ) -> None:
        tree = construction in {"pathdependent_tree", "interventional_tree"}
        valuation = construction == "dataset_valuation"
        application = (
            "data" if valuation else "features" if construction == "feature_selection" else "local"
        )
        identity = (dataset, model, construction, players, inputs)
        if identity in rows:
            return
        identifier = f"comprehensive-{construction}-{dataset}-{model}-p{players}"
        if inputs is not None:
            identifier += f"-f{inputs}"
        rows[identity] = dict(
            zip(
                COLUMNS,
                (
                    identifier,
                    application,
                    "retraining_groups" if valuation else "",
                    dataset,
                    model,
                    construction,
                    str(players),
                    "tree_solver" if tree else "enumeration",
                    str(inputs) if inputs is not None else "",
                    "nested",
                    "",
                    "",
                ),
                strict=True,
            )
        )

    for dataset in (*CLASSIFICATION, *REGRESSION):
        width = DATASETS[dataset]["n_features"]
        feature_counts = sorted(
            {n for n in GENERIC_COUNTS if n <= width} | ({width} if width < 12 else set())
        )
        for model in CORE_MODELS:
            for construction in ("local_baseline", "dataset_valuation", "feature_selection"):
                for players in (
                    GENERIC_COUNTS if construction == "dataset_valuation" else feature_counts
                ):
                    add(
                        dataset,
                        model,
                        construction,
                        players,
                        inputs=min(12, width) if construction == "dataset_valuation" else None,
                    )
        for construction in ("pathdependent_tree", "interventional_tree"):
            for players in TREE_COUNTS:
                if players <= width:
                    add(dataset, "random_forest", construction, players)

    return list(rows.values())


def selection_manifest() -> dict:
    """Map exact retained recipes and relative budgets for filtering frozen snapshots.

    Match a snapshot's recorded ``focused_design.recipes`` entry, rather than
    parsing its game ID. This selects design cells; reuse still requires the
    original authenticated artifacts, target, construction seed and source.
    """
    return {
        "version": 2,
        "recipes": {row["id"]: row for row in recipes()},
        "relative_budgets": list(RELATIVE_BUDGETS),
    }


def retains_cell(selection: dict, recipe: dict, budget: int) -> bool:
    """Select an unchanged recipe at an exact retained budget, including odd widths."""
    expected = selection["recipes"].get(recipe.get("id"))
    if expected is None or set(recipe) - set(COLUMNS):
        return False
    # Focused manifests omit blank optional fields; CSV rows retain them.
    if {column: str(recipe.get(column, "")) for column in COLUMNS} != expected:
        return False
    players = int(expected["n_players"])
    return type(budget) is int and budget in {
        math.ceil(ratio * players) for ratio in selection["relative_budgets"]
    }


def manifest_text() -> str:
    """Serialize the fixed design reproducibly, including a final newline."""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(recipes())
    return stream.getvalue()


def build_suite(manifest: Path) -> dict:
    """Build the full suite with strict estimator-constructor compatibility checks."""
    if manifest.read_text() != manifest_text():
        msg = "Comprehensive manifest differs from the deterministic approved design."
        raise ValueError(msg)
    suite = focused.build_suite(
        manifest, min_players=4, max_players=32, extended_features=True, estimator_seeds=(0, 1, 2)
    )
    suite["name"] = "comprehensive-24-datasets-compact-v2"
    suite["relative_budgets"] = list(RELATIVE_BUDGETS)
    suite["comprehensive_design"] = {
        "version": 2,
        "classification_datasets": list(CLASSIFICATION),
        "regression_datasets": list(REGRESSION),
        "reuse_rule": "Exact recipe identity and authenticated frozen artifacts only.",
        "nested_features": True,
    }
    return suite


def summary(suite: dict) -> dict:
    """Small design inventory, without claiming intended recipes are qualified."""
    rows = suite["focused_design"]["recipes"]
    seeds = len(suite["game_seeds"])
    targets = (len(suite["families"]) * len(suite["targets"]) + len(suite["games"])) * seeds
    return {
        "datasets": len(CLASSIFICATION) + len(REGRESSION),
        "recipes": len(rows),
        "intended_instances": len(rows) * seeds,
        "intended_targets": targets,
        "recipes_by_application": dict(Counter(row["application"] for row in rows)),
        "recipes_by_reference": dict(Counter(row["reference"] for row in rows)),
        "estimator_settings": len(suite["methods"]),
        "estimator_seeds": suite["seeds"],
        "relative_budgets": suite["relative_budgets"],
    }


def main() -> None:
    """Validate the checked-in manifest and write a new suite without overwriting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    suite = build_suite(args.manifest)
    with args.output.open("x") as stream:
        json.dump(suite, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(summary(suite), indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
