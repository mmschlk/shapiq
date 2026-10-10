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
from collections import Counter
from pathlib import Path

from shapiq_benchmark import focused
from shapiq_benchmark.datasets import DATASETS
from shapiq_benchmark.protocol import CONSTRUCTIONS

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
CONTRAST_DATASETS = (
    "adult_census",
    "breast_cancer",
    "digits",
    "tabarena_bioresponse",
    "wine_quality",
    "tabarena_miami_housing",
    "tabarena_superconductivity",
    "tabarena_qsar_tid11",
)
CORE_MODELS = ("linear", "random_forest")
GENERIC_COUNTS = (4, 8, 12, 14, 16)
TREE_COUNTS = (8, 16, 32, 64, 128, 256, 512)
NEIGHBOR_COUNTS = (32, 64, 128, 256, 512, 1024)
INPUT_COUNTS = (8, 16, 32, 64, 128)
NEIGHBOR_POOL = 5000
# These sklearn/shipped datasets predate n_samples in the expanded catalog.
_BASE_ROWS = {"breast_cancer": 569, "digits": 1797, "wine": 178}
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
        neighbor = construction in {"knn", "tnn"}
        tree = construction in {"pathdependent_tree", "interventional_tree"}
        valuation = construction == "dataset_valuation"
        application = (
            "data"
            if neighbor or valuation
            else "features"
            if construction == "feature_selection"
            else "local"
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
                    "neighbor_examples" if neighbor else "retraining_groups" if valuation else "",
                    dataset,
                    model,
                    construction,
                    str(players),
                    "neighbor_solver" if neighbor else "tree_solver" if tree else "enumeration",
                    str(inputs) if inputs is not None else "",
                    "nested",
                    str(NEIGHBOR_POOL) if neighbor else "",
                    "nested_stratified" if neighbor else "",
                ),
                strict=True,
            )
        )

    for dataset in (*CLASSIFICATION, *REGRESSION):
        width = DATASETS[dataset]["n_features"]
        feature_counts = sorted(
            {n for n in GENERIC_COUNTS if n <= width} | ({width} if width < 16 else set())
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
        for model in ("random_forest", "xgboost", "lightgbm"):
            for construction in ("pathdependent_tree", "interventional_tree"):
                for players in TREE_COUNTS:
                    if players <= width:
                        add(dataset, model, construction, players)
        if dataset in CLASSIFICATION:
            samples = DATASETS[dataset].get("n_samples", _BASE_ROWS.get(dataset))
            if samples is None:
                msg = f"Missing row count for neighbor recipe: {dataset}"
                raise ValueError(msg)
            available = min(NEIGHBOR_POOL, samples * 4 // 5)
            for construction in ("knn", "tnn"):
                for players in NEIGHBOR_COUNTS:
                    if players <= available:
                        add(dataset, construction, construction, players, inputs=min(12, width))

    for dataset in CONTRAST_DATASETS:
        width = DATASETS[dataset]["n_features"]
        for model in ("xgboost", "lightgbm", "rbf_svm", "mlp"):
            for construction in ("local_baseline", "dataset_valuation", "feature_selection"):
                if model in CONSTRUCTIONS[construction]["models"] and (
                    construction == "dataset_valuation" or width >= 12
                ):
                    add(
                        dataset,
                        model,
                        construction,
                        12,
                        inputs=min(12, width) if construction == "dataset_valuation" else None,
                    )
        for model in CORE_MODELS:
            for inputs in sorted({n for n in INPUT_COUNTS if n <= width} | {width}):
                add(dataset, model, "dataset_valuation", 12, inputs=inputs)
    return list(rows.values())


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
        manifest, min_players=4, max_players=1024, extended_features=True, estimator_seeds=(0, 1, 2)
    )
    suite["name"] = "comprehensive-24-datasets-v1"
    suite["comprehensive_design"] = {
        "version": 1,
        "classification_datasets": list(CLASSIFICATION),
        "regression_datasets": list(REGRESSION),
        "contrast_datasets": list(CONTRAST_DATASETS),
        "reuse_rule": "Exact recipe identity and authenticated frozen artifacts only.",
        "nested_features": True,
        "neighbor_training_pool": NEIGHBOR_POOL,
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
