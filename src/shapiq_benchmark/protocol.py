"""The benchmark's declared datasets, models, constructions and phased rollout.

Only phase two is executable today. Later phases are compatibility inventories:
``planned`` means an adapter or qualification is still required, never a result.
No datasets are downloaded and no models are fitted while building a manifest.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from shapiq_benchmark.datasets import DATASETS, feature_limit
from shapiq_benchmark.families import FAMILY_CATALOG, MAX_ENUMERATION_PLAYERS, dataset_compatibility
from shapiq_benchmark.media import EXTRA_CATALOG
from shapiq_benchmark.models import MODEL_PROFILES, TRAINING_PROFILE

SOURCE_ROOT = "https://github.com/rtealwitter/shapiq/blob/benchmark/"
BUDGET_MULTIPLIERS = (0.5, 1, 2, 4, 8, 16, 32, 64, 128)
GAME_SEEDS = (0, 1, 2, 3)
FIRST_DATASETS = ("adult_census", "breast_cancer", "wine_quality", "communities_and_crime")
NEXT_DATASETS = ("mushroom", "ionosphere", "tabarena_miami_housing", "tabarena_superconductivity")
DATASET_PHASES = {
    name: 2 if name in FIRST_DATASETS else 3 if name in NEXT_DATASETS else 6
    for name in DATASETS
    if name != "wine"  # The historical sklearn Wine classification game keeps its identity.
}
PREDICTION_MODELS = ("random_forest", "xgboost", "lightgbm", "mlp")
LOCAL_MODELS = (*PREDICTION_MODELS, "rbf_svm", "linear", "gaussian_process", "tabpfn_prediction")
REFIT_MODELS = (*PREDICTION_MODELS, "linear")
TREE_MODELS = ("random_forest", "xgboost", "lightgbm")
MODEL_PHASES = {
    **dict.fromkeys(("random_forest", "xgboost"), 2),
    **dict.fromkeys(("lightgbm", "mlp"), 4),
    **dict.fromkeys(
        (
            "rbf_svm",
            "linear",
            "heterogeneous_ensemble",
            "knn",
            "tnn",
            "weighted_knn",
            "binary_weighted_knn",
            "kmeans",
            "none",
        ),
        3,
    ),
    **dict.fromkeys(
        ("tabpfn", "tabpfn_prediction", "gaussian_process", "resnet18", "vit", "distilbert"), 5
    ),
}
# Each construction names its allowed models, not a blind model cross product.
_CONSTRUCTIONS = {
    "local_baseline": (2, LOCAL_MODELS),
    "local_marginal": (2, LOCAL_MODELS),
    "local_gaussian": (3, PREDICTION_MODELS),
    "local_copula": (3, PREDICTION_MODELS),
    "local_conditional": (3, PREDICTION_MODELS),
    "global_fidelity": (3, PREDICTION_MODELS),
    "feature_selection": (3, REFIT_MODELS),
    "data_valuation": (3, REFIT_MODELS),
    "dataset_valuation": (3, REFIT_MODELS),
    "ensemble": (3, ("heterogeneous_ensemble",)),
    "forest_ensemble": (3, ("random_forest",)),
    "uncertainty": (3, ("random_forest",)),
    "cluster": (3, ("kmeans",)),
    "unsupervised": (3, ("none",)),
    "pathdependent_tree": (3, TREE_MODELS),
    "interventional_tree": (3, TREE_MODELS),
    "product_kernel": (3, ("rbf_svm",)),
    "knn": (3, ("knn",)),
    "tnn": (3, ("tnn",)),
    "weighted_knn": (3, ("weighted_knn",)),
    "binary_weighted_knn": (3, ("binary_weighted_knn",)),
    "tabpfn": (5, ("tabpfn",)),
    "image": (5, ("resnet18", "vit")),
    "text": (5, ("distilbert",)),
    "causal_global": (5, ("tabpfn",)),
    "causal_local": (5, ("tabpfn",)),
    **dict.fromkeys(("unanimity", "soum", "dummy", "random"), (5, ("none",))),
}
CONSTRUCTIONS = {
    name: {
        "id": name,
        "label": name.replace("_", " ").capitalize(),
        "phase": phase,
        "models": list(models),
        "player_unit": FAMILY_CATALOG.get(name, {}).get(
            "player_unit", {"image": "image region", "text": "token"}.get(name, "feature")
        ),
        "source_url": SOURCE_ROOT + (FAMILY_CATALOG | EXTRA_CATALOG)[name]["source"],
        "targets": "All suite targets for qualified enumerated payoff tables",
        **({"variants": ["mask", "remove"]} if name == "text" else {}),
    }
    for name, (phase, models) in _CONSTRUCTIONS.items()
}
_SPECIAL = {
    "image",
    "text",
    "causal_global",
    "causal_local",
    "unanimity",
    "soum",
    "dummy",
    "random",
}
_STRUCTURED = {
    "pathdependent_tree",
    "interventional_tree",
    "product_kernel",
    "knn",
    "tnn",
    "weighted_knn",
    "binary_weighted_knn",
}


def protocol_manifest(phase: int) -> dict:
    """Describe the intended protocol; actual run metadata remains authoritative."""
    if type(phase) is not int or phase not in range(2, 8):
        message = "Phase must be an integer between 2 and 7; phase 1 qualifies models."
        raise ValueError(message)
    return {
        "phase": phase,
        "name": f"Stronger-model benchmark: phase {phase}",
        "description": "Four seeded game instances; one estimator evaluation per target and budget.",
        "source_url": SOURCE_ROOT + "benchmark/ROADMAP.md",
        "budget_multipliers": list(BUDGET_MULTIPLIERS),
        "budget_rule": "ceil(multiplier * d) oracle queries; d is the game's number of players",
        "budget_note": "The requested cap can exceed actual query use or the number of distinct coalitions.",
        "game_seeds": list(GAME_SEEDS),
        "estimator_seeds": [0],
        "minimum_players": 11,
        "maximum_enumerated_players": MAX_ENUMERATION_PLAYERS,
        "minimum_signal_ratio": 1e-6,
        "exact_reference": "Cache all 2**d coalition payoffs once; derive all qualified targets from that table. Above 20 players requires a separately qualified structured solver.",
        "timing": "Measured estimator wall time and cached-query time charges are separate; charged uncached time is an estimate.",
        "hardware": "Estimator evaluation: AMD EPYC 9754, one pinned CPU per worker. Game preparation: CPU, or an explicitly recorded GPU backend where measured faster. Concurrent timings are diagnostic; query charges retain their preparation hardware.",
        "preparation_devices": {
            "random_forest": "cpu",
            "xgboost": "cpu",
            "tabpfn": "cuda: measured contextualization speedup; float32, one worker per GPU",
            "tabpfn_prediction": "cuda planned: qualify fixed predictor adapter before scheduling",
            "other_models": "cpu until a representative GPU pilot qualifies the backend",
        },
        "datasets": [
            {
                "id": name,
                **copy.deepcopy(DATASETS[name]),
                "phase": introduced,
                "label": name.replace("_", " ").title(),
                "implementation_url": (
                    "https://scikit-learn.org/stable/modules/generated/"
                    + DATASETS[name]["source"]
                    + ".html"
                    if DATASETS[name]["source"].startswith("sklearn.")
                    else SOURCE_ROOT
                    + "src/"
                    + DATASETS[name]["source"].rsplit(".", 1)[0].replace(".", "/")
                    + "/_all.py"
                ),
            }
            for name, introduced in DATASET_PHASES.items()
        ],
        "models": [
            {
                "id": name,
                "phase": introduced,
                "label": name.replace("_", " ").title(),
                **copy.deepcopy(MODEL_PROFILES.get(name, {})),
                "status": "implemented" if name in MODEL_PROFILES else "planned",
                "source_url": SOURCE_ROOT
                + (
                    "src/shapiq_benchmark/models.py"
                    if name in MODEL_PROFILES
                    else "benchmark/ROADMAP.md"
                ),
            }
            for name, introduced in MODEL_PHASES.items()
        ],
        "training": copy.deepcopy(TRAINING_PROFILE),
        "constructions": copy.deepcopy(list(CONSTRUCTIONS.values())),
    }


def _exclusion(family: str, dataset: str, count: int) -> str | None:
    """Apply known semantic and dimensional restrictions before expensive construction."""
    info = DATASETS[dataset]
    reason = None if family == "tabpfn" else dataset_compatibility(family, dataset)
    if reason:
        return reason
    if CONSTRUCTIONS[family]["player_unit"] == "feature" and count > feature_limit(family, dataset):
        return "Too few eligible original features; padding is prohibited."
    if family in ("knn", "tnn", "weighted_knn", "binary_weighted_knn") and count < info.get(
        "n_classes", 0
    ):
        return "Too few training-example players to represent every class."
    return None


def phase_candidates(phase: int) -> list[dict]:
    """Inventory cumulative table phases, or the separate phase-seven structured extension."""
    protocol_manifest(phase)  # Validate phase without loading any datasets.
    datasets = [name for name, introduced in DATASET_PHASES.items() if introduced <= phase]
    rows = []
    for family, construction in CONSTRUCTIONS.items():
        if construction["phase"] > phase or (phase == 7 and family not in _STRUCTURED):
            continue
        for dataset in [None] if family in _SPECIAL else datasets:
            unit = construction["player_unit"]
            if phase == 7:
                counts = (
                    [DATASETS[dataset]["n_features"]]
                    if unit == "feature"
                    else [32, 64, 128, 256, 512]
                )
                counts = [count for count in counts if count > MAX_ENUMERATION_PLAYERS]
            elif family in _SPECIAL:
                counts = [12]  # Dedicated input/variant qualification remains planned.
            elif phase < 4:
                counts = [12] if phase == 2 else [11]
                if (
                    phase == 3
                    and dataset in FIRST_DATASETS
                    and family in ("local_baseline", "local_marginal")
                ):
                    counts.append(12)  # Keep the phase-two cohort in cumulative plans.
            else:
                counts = sorted(
                    {11, 12, 16, 20}
                    | (
                        {DATASETS[dataset]["n_features"]}
                        if unit == "feature" and 11 <= DATASETS[dataset]["n_features"] <= 20
                        else set()
                    )
                )
            for model in construction["models"]:
                if MODEL_PHASES[model] > phase:
                    continue
                for count in [16] if family == "image" and model == "vit" else counts:
                    reason = _exclusion(family, dataset, count) if dataset else None
                    ready = phase == 2 and reason is None
                    rows.append(
                        {
                            "id": f"{family}-{dataset or 'fixed-inputs'}-{model}-d{count}",
                            "family": family,
                            "dataset": dataset,
                            "model_profile": model,
                            "n_players": count,
                            "player_unit": unit,
                            "status": "excluded" if reason else "selected" if ready else "planned",
                            "reason": reason
                            or (
                                "Prepare and qualify the exact payoff table."
                                if ready
                                else "Requires adapter and model/output qualification before scheduling."
                            ),
                            "reference": "structured" if phase == 7 else "enumeration",
                            "target_rule": (
                                "SV only pending solver qualification"
                                if phase == 7
                                and family not in ("pathdependent_tree", "interventional_tree")
                                else "Suite targets pending exact-reference qualification"
                            ),
                        }
                    )
    return rows


def build_phase(phase: int, base: dict) -> dict:
    """Reuse estimator/target definitions, with an executable phase two and explicit later plans."""
    expected = {
        "relative_budgets": list(BUDGET_MULTIPLIERS),
        "game_seeds": list(GAME_SEEDS),
        "seeds": [0],
        "min_players": 11,
    }
    if any(base.get(key) != value for key, value in expected.items()):
        message = "Base suite disagrees with the fixed relative budgets, four game seeds or minimum players."
        raise ValueError(message)
    suite = {key: copy.deepcopy(base[key]) for key in (*expected, "methods", "targets")}
    candidates = phase_candidates(phase)
    suite.update(
        name=f"stronger-models-phase-{phase}-v1",
        games=[],
        families=[
            {key: row[key] for key in ("id", "family", "dataset", "model_profile", "n_players")}
            for row in candidates
            if row["status"] == "selected"
        ],
        min_signal_ratio=1e-6,
        protocol=protocol_manifest(phase),
        phase_plan={
            "candidates": candidates,
            "counts": {
                status: sum(row["status"] == status for row in candidates)
                for status in ("selected", "planned", "excluded")
            },
        },
    )
    return suite


def main() -> None:
    """Write a frozen phase manifest without modifying the base suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, required=True, choices=range(2, 8))
    parser.add_argument(
        "--base",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "benchmark/suites/all-families.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() == args.base.resolve():
        parser.error("Output must not overwrite the base suite")
    suite = build_phase(args.phase, json.loads(args.base.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(suite, indent=2, allow_nan=False) + "\n")
    print(json.dumps(suite["phase_plan"]["counts"], sort_keys=True))  # noqa: T201 -- CLI summary


if __name__ == "__main__":
    main()
