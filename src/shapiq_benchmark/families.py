"""Compact real-data game recipes using the public shapiq_games constructors.

Fresh preparations use ``compact-games-v2``. The published campaign is replayed
from its frozen source and artifacts. Compact feature selection explicitly keeps
the campaign's empty-zero utility instead of upstream's default dummy baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.imputer import BaselineImputer
from shapiq_benchmark.campaign_datasets import DATASETS
from shapiq_benchmark.campaign_models import coalition_model, prepare_model
from shapiq_benchmark.execution import hardware
from shapiq_benchmark.quality import QUALITY_PROTOCOL, QualityExclusion, validate_protocol
from shapiq_games.feature_selection import FeatureSelection
from shapiq_games.valuation import DatasetValuation

if TYPE_CHECKING:
    from shapiq.game import Game

MAX_ENUMERATION_PLAYERS = 12
CONSTRUCTION_PROTOCOL = "compact-games-v2"
_RECIPES = {
    "local_baseline": (
        "shapiq.imputer.baseline_imputer.BaselineImputer",
        "prediction with fixed mean baseline",
        "feature",
        "local_explanation",
    ),
    "feature_selection": (
        "shapiq_games.feature_selection.FeatureSelection",
        "held-out performance after retraining; empty zero",
        "feature",
        "feature_selection",
    ),
    "dataset_valuation": (
        "shapiq_games.valuation.DatasetValuation",
        "held-out performance after retraining on selected training groups; empty zero",
        "training group",
        "data_valuation",
    ),
    "pathdependent_tree": (
        "shapiq_games.tree.path_dependent.PathDependentTreeGame",
        "training-path-weighted tree prediction",
        "feature",
        "local_explanation",
    ),
    "interventional_tree": (
        "shapiq_games.tree.interventional.InterventionalTreeGame",
        "mean tree prediction over fixed joint background rows",
        "feature",
        "local_explanation",
    ),
    "knn": (
        "shapiq_games.nn.knn.KNNGame",
        "correct-label count among selected three nearest divided by three",
        "training example",
        "data_valuation",
    ),
    "tnn": (
        "shapiq_games.nn.threshold_nn.ThresholdNNGame",
        "correct-label proportion within the training-scaled radius",
        "training example",
        "data_valuation",
    ),
}
FAMILY_CATALOG: dict = {
    name: {
        "class": cls,
        "source": "src/" + cls.rsplit(".", 1)[0].replace(".", "/") + ".py",
        "semantics": semantics,
        "player_unit": unit,
        "application_family": application,
        "stochastic_frozen": False,
        "synthetic": False,
        "dependencies": [],
        "construction_protocol": CONSTRUCTION_PROTOCOL,
    }
    for name, (cls, semantics, unit, application) in _RECIPES.items()
}
PROFILE_CONSTRUCTIONS = {
    "linear": {"local_baseline", "feature_selection", "dataset_valuation"},
    "random_forest": {
        "local_baseline",
        "feature_selection",
        "dataset_valuation",
        "pathdependent_tree",
        "interventional_tree",
    },
}


def dataset_compatibility(name: str, dataset: str) -> str | None:
    """Explain unsupported pairs before fitting a model."""
    if dataset not in DATASETS:
        return "Unknown dataset."
    if name not in FAMILY_CATALOG:
        return "Not a compact benchmark recipe."
    if name in {"knn", "tnn"} and DATASETS[dataset]["task"] != "classification":
        return "Requires class labels."
    return None


def profile_compatibility(name: str, dataset: str, profile: str) -> str | None:
    """Validate the compact construction/model mapping."""
    if reason := dataset_compatibility(name, dataset):
        return reason
    if name not in PROFILE_CONSTRUCTIONS.get(profile, set()):
        return f"The {profile} model profile does not support {name}."
    return None


def feature_subset(x: np.ndarray, count: int | None, seed: int) -> np.ndarray:
    """Choose original feature columns without using targets or benchmark errors."""
    if count is None:
        return np.arange(x.shape[1])
    if type(count) is not int or not 1 <= count <= x.shape[1]:
        msg = "Feature count must fit the dataset width."
        raise ValueError(msg)
    if count == x.shape[1]:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(x.shape[1], count, replace=False))


def make_family(
    name: str,
    *,
    instance_seed: int = 0,
    dataset: str | None = None,
    n_players: int | None = None,
    model_profile: str | None = None,
    model_cache: str | None = None,
    device: str = "cpu",
    quality_protocol: str | None = None,
    input_features: int | None = None,
    feature_rule: str | None = None,
) -> tuple[Game, dict]:
    """Prepare one bounded compact game; native games use ``prepare_structured``."""
    validate_protocol(quality_protocol)
    dataset = dataset or "breast_cancer"
    profile = model_profile or "random_forest"
    if reason := profile_compatibility(name, dataset, profile):
        raise ValueError(reason)
    n = n_players if n_players is not None else min(8, DATASETS[dataset]["n_features"])
    if name not in {"local_baseline", "feature_selection", "dataset_valuation"}:
        msg = "Native tree and neighbor games use prepare_structured."
        raise ValueError(msg)
    if type(n) is not int or not 1 <= n <= MAX_ENUMERATION_PLAYERS:
        msg = "Enumerated compact games require one to twelve players."
        raise ValueError(msg)
    if device != "cpu":
        msg = "Compact preparation uses CPU models."
        raise ValueError(msg)
    if feature_rule not in {None, "all", "nested"}:
        msg = "Compact feature rules are all or nested."
        raise ValueError(msg)
    width = int(DATASETS[dataset]["n_features"])
    if input_features is not None and (
        name != "dataset_valuation"
        or type(input_features) is not int
        or not 1 <= input_features <= width
    ):
        msg = "Input features apply only to dataset valuation and must fit its width."
        raise ValueError(msg)
    feature_count = (input_features or min(12, width)) if name == "dataset_valuation" else n
    prepared = prepare_model(
        dataset,
        feature_count,
        instance_seed,
        profile,
        cache_dir=model_cache,
        device=device,
        feature_rule=feature_rule or "all",
        quality_protocol=quality_protocol,
    )
    metadata: dict = {**FAMILY_CATALOG[name], **prepared.metadata}
    if quality_protocol == QUALITY_PROTOCOL and not metadata["model_validation_gate"]["passed"]:
        msg = "model_not_better_than_validation_dummy"
        raise QualityExclusion(msg, metadata["model_validation_gate"])
    background = prepared.x_train[:16]
    if name == "local_baseline":
        game = BaselineImputer(
            model=prepared.predict,
            data=background,
            x=prepared.x_test[0],
            random_state=instance_seed,
        )
        metadata.update(
            background_size=len(background),
            background_indices=metadata["train_indices"][: len(background)],
            point_row=metadata["test_indices"][0],
            output_scale="class probability"
            if metadata["task"] == "classification"
            else "prediction",
        )
    else:
        options: dict = {
            "model": coalition_model(prepared),
            "x_train": prepared.x_train,
            "y_train": prepared.y_train,
            "x_test": prepared.x_test,
            "y_test": prepared.y_test,
            "task": metadata["task"],
            "metric": "accuracy" if metadata["task"] == "classification" else "neg_mse",
            "random_state": instance_seed,
        }
        if name == "dataset_valuation":
            if n > len(prepared.x_train):
                msg = "Not enough training rows for nonempty group players."
                raise ValueError(msg)
            groups = np.array_split(np.arange(len(prepared.x_train)), n)
            game = DatasetValuation(**options, groups=groups)
            metadata["group_indices"] = [
                np.asarray(metadata["train_indices"])[g].tolist() for g in groups
            ]
        else:
            game = FeatureSelection(**options, normalize=False)
            # The campaign defines f(empty)=0, not dummy-model performance.
            # These public game attributes control empty payoffs and centering.
            game.empty_value = 0.0
            game.normalization_value = 0.0
        metadata.update(
            background_size=0,
            background_indices=[],
            input_features=feature_count,
            output_scale="accuracy" if metadata["task"] == "classification" else "negative MSE",
            refit_protocol={
                "hyperparameters": "frozen before enumeration; clone per coalition",
                "empty": "zero",
                "single_class": "training majority class",
                "preprocessing": "missing-value repair frozen on fitting pool; scaler refitted per coalition",
            },
        )
    metadata.update(
        recipe=name,
        instance_seed=instance_seed,
        model_profile=profile,
        model=type(prepared.model).__name__,
        n_players=game.n_players,
        parameters={"random_state": instance_seed},
        normalize=game.normalize,
        normalization_value=float(game.normalization_value),
        preparation_hardware={"device": "cpu", "cpu_model": hardware()["cpu_model"]},
    )
    return game, metadata
