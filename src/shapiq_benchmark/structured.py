"""Qualified structured oracles from shared model profiles and portable numeric arrays."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, cast

import numpy as np

from shapiq.tree.base import TreeModel, predict_ensemble

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path
    from typing import Any

    from shapiq.game import Game
    from shapiq_benchmark.models import PreparedModel

TREE_ARRAYS = (
    "children_left",
    "children_right",
    "children_missing",
    "features",
    "thresholds",
    "values",
    "node_sample_weight",
    "cat_values",
    "cat_start",
    "cat_size",
)
TREE_SCALARS = ("decision_type", "input_precision", "empty_prediction")


class FrozenTreePredictor:
    """Predict the already converted scalar output without refitting a model."""

    def __init__(self, trees: list[TreeModel]) -> None:
        """Keep the authenticated, converted prediction trees."""
        self.trees = trees

    def predict(self, rows: np.ndarray) -> np.ndarray:
        """Evaluate the fixed scalar model output."""
        return predict_ensemble(self.trees, rows)


def _tree_game(trees: list, background: np.ndarray, point: np.ndarray, kind: str) -> Game:
    from shapiq.tree.interventional.game import InterventionalGame
    from shapiq_games.benchmark.treeshapiq_xai.base import TreeSHAPIQXAI

    if kind == "pathdependent_tree":
        return TreeSHAPIQXAI(point, trees, normalize=False, verbose=False)
    return InterventionalGame(FrozenTreePredictor(trees), background, point)


def _construct(prepared: PreparedModel, spec: dict) -> tuple:
    """Use shipped exact computers on precisely the scalar output stored in the oracle."""
    from shapiq import InteractionValues
    from shapiq.explainer.product_kernel.game import ProductKernelGame
    from shapiq.explainer.product_kernel.product_kernel import ProductKernelComputer
    from shapiq.tree import TreeExplainer
    from shapiq.tree.interventional.computer import InterventionalTreeSHAPIQ
    from shapiq.tree.validation import validate_tree_model
    from shapiq_benchmark.models import converted_svm

    model = copy.deepcopy(prepared.model)
    point, background = prepared.x_test[0], prepared.x_train[:16]
    kind, index, order = spec["oracle"], spec["index"], spec["order"]
    if kind == "product_kernel":
        if (index, order) != ("SV", 1):
            message = "Structured product kernels currently qualify SV only."
            raise ValueError(message)
        if hasattr(model, "steps"):
            point = model[:-1].transform(point[None])[0]
            model = model.steps[-1][1]
        if hasattr(model, "classes_") and len(model.classes_) != 2:
            message = "Product-kernel classification requires exactly two classes."
            raise ValueError(message)
        converted = converted_svm(model)
        if converted.gamma is None:
            message = "RBF conversion requires an explicit kernel bandwidth."
            raise ValueError(message)
        oracle = ProductKernelGame(len(point), point, converted)
        computer = ProductKernelComputer(converted)
        vectors = computer.compute_kernel_vectors(converted.X_train, point)
        scores: dict[tuple[int, ...], float] = {
            (i,): float(computer.compute_shapley_value(vectors, i)) for i in range(len(point))
        }
        truth = InteractionValues(
            values=scores,
            index="SV",
            min_order=0,
            max_order=1,
            n_players=len(point),
            estimated=False,
            baseline_value=float(converted.alpha.sum() + converted.intercept),
        )
        arrays = {"point": point, "support_vectors": converted.X_train, "alpha": converted.alpha}
        details: dict = {
            "payoff_range_upper_bound": float(np.abs(converted.alpha).sum()),
            "gamma": float(converted.gamma),
            "intercept": float(converted.intercept),
            "semantics": "RBF decision margin or regression score with omitted factors set to one",
        }
        return oracle, truth, arrays, details
    if kind not in ("tree", "pathdependent_tree") or spec["model_profile"] not in (
        "random_forest",
        "xgboost",
        "lightgbm",
    ):
        message = "Structured tree profiles require a qualified tree family."
        raise ValueError(message)
    if spec["model_profile"] == "random_forest":
        # sklearn rounds prediction inputs; converted trees must route identical values.
        point = point.astype(np.float32).astype(np.float64)
        background = background.astype(np.float32).astype(np.float64)
    if spec["model_profile"] == "xgboost":
        model._Booster = model.get_booster()[: model.best_iteration + 1]  # noqa: SLF001 -- preserve validated early stopping
    class_index = 1 if prepared.metadata["task"] == "classification" else None
    trees = validate_tree_model(model, class_label=class_index)
    for tree in trees:
        tree.values = tree.values.astype(np.float64)
        tree.node_sample_weight = tree.node_sample_weight.astype(np.float64)
        if kind == "pathdependent_tree":
            # Use the same float64 child probabilities in the shipped traversal
            # and exact solver; preserve leaf covers and all routing rules.
            stack = [(0, False)]
            while stack:
                node, visited = stack.pop()
                if tree.leaf_mask[node]:
                    continue
                left, right = tree.children_left[node], tree.children_right[node]
                if visited:
                    tree.node_sample_weight[node] = (
                        tree.node_sample_weight[left] + tree.node_sample_weight[right]
                    )
                else:
                    stack.extend(((node, True), (right, False), (left, False)))
            tree.compute_empty_prediction()
    oracle = _tree_game(trees, background, point, kind)
    if kind == "tree":
        truth = InterventionalTreeSHAPIQ(trees, background, index=index, max_order=order).explain(
            point
        )
        # Independently check conversion against the source predictor on all tested coalitions.
        from shapiq.tree.interventional.game import InterventionalGame

        source = InterventionalGame(model, background, point, class_index=class_index)
        probes = np.random.default_rng(0).integers(0, 2, (12, len(point))).astype(bool)
        converted_values, source_values = oracle(probes), source(probes)
        np.testing.assert_allclose(
            converted_values,
            source_values,
            rtol=2e-6,
            atol=2e-6 * max(1.0, float(np.std(source_values))),
        )
    else:
        computer = cast("Any", TreeExplainer)(trees, index=index, max_order=order)
        truth = computer.explain(point)
    arrays = {"point": point, "background": background}
    for i, tree in enumerate(trees):
        arrays.update({f"tree_{i}_{name}": getattr(tree, name) for name in TREE_ARRAYS})
    details: dict = {
        "oracle_precision": "float64 tree values and sample weights; path-dependent internal covers sum leaf covers",
        "payoff_range_upper_bound": float(
            sum(np.ptp(tree.values[tree.leaf_mask]) for tree in trees)
        ),
        "trees": [{name: getattr(tree, name) for name in TREE_SCALARS} for tree in trees],
        "semantics": "class-one probability"
        if class_index is not None and spec["model_profile"] == "random_forest"
        else "raw model score (boosted classifiers use margins)",
        "truth_method": "InterventionalTreeSHAPIQ" if kind == "tree" else "TreeExplainer",
    }
    if kind == "tree":
        details["source_conversion_max_error"] = float(
            np.max(np.abs(converted_values - source_values))
        )
    return oracle, truth, arrays, details


def prepare_profiled(spec: dict, output: Path) -> dict:
    """Qualify a small counterpart before freezing a native-width structured game."""
    from shapiq_benchmark.datasets import DATASETS
    from shapiq_benchmark.games import signal_metadata, truth_dict, validate_truth
    from shapiq_benchmark.models import prepare_model

    dataset, profile = spec["dataset"], spec["model_profile"]
    n = spec.get("n_players", DATASETS[dataset]["n_features"])
    if n < 11:
        message = "New structured profiles require at least eleven players."
        raise ValueError(message)
    seed = spec.get("instance_seed", 0)
    cache = output.parent / ".models"
    quality = {"quality_protocol": spec["quality_protocol"]} if "quality_protocol" in spec else {}
    small = prepare_model(dataset, 8, seed, profile, cache_dir=cache, **quality)
    small_oracle, small_truth, _, _ = _construct(small, spec)
    error = validate_truth(small_oracle, small_truth, exhaustive=True)
    prepared = prepare_model(dataset, n, seed, profile, cache_dir=cache, **quality)
    if quality and not prepared.metadata["model_validation_gate"]["passed"]:
        from shapiq_benchmark.quality import QualityExclusion

        reason = "model_not_better_than_validation_dummy"
        raise QualityExclusion(reason, prepared.metadata["model_validation_gate"])
    oracle, truth, arrays, details = _construct(prepared, spec)
    validate_truth(oracle, truth, exhaustive=False)
    details.update(signal_metadata(truth, details["payoff_range_upper_bound"]))
    artifact = output / f"{spec['id']}.npz"
    np.savez_compressed(artifact, **arrays)
    return {
        "id": spec["id"],
        "family": "local_explanation",
        "stratum": f"{dataset}_{spec['oracle']}_{n}",
        "n_players": n,
        "index": spec["index"],
        "order": spec["order"],
        "oracle": spec["oracle"],
        "artifact": artifact.name,
        "truth": truth_dict(truth),
        "metadata": {
            **prepared.metadata,
            **details,
            "structured_format": "numeric-model-v1",
            "instance_seed": seed,
            "case_id": spec.get("basecase_id", spec["id"]),
            "recipe": spec["oracle"],
            "cluster_id": f"{dataset}-{profile}-{n}-i{seed}",
            "player_unit": "feature",
            "point_row": prepared.metadata["test_indices"][0],
            "small_validation_players": 8,
            "small_validation_max_error": error,
        },
    }


def load_profiled(game: dict, arrays: Mapping[str, np.ndarray]) -> Game:
    """Reload numeric tree/kernel arrays; no pickle and no training on evaluation workers."""
    if game["oracle"] == "product_kernel":
        from shapiq.explainer.product_kernel.base import ProductKernelModel
        from shapiq.explainer.product_kernel.game import ProductKernelGame

        model = ProductKernelModel(
            X_train=arrays["support_vectors"].copy(),
            alpha=arrays["alpha"].copy(),
            n=len(arrays["alpha"]),
            d=game["n_players"],
            gamma=game["metadata"]["gamma"],
            intercept=game["metadata"]["intercept"],
        )
        return ProductKernelGame(game["n_players"], arrays["point"].copy(), model)
    trees = []
    for i, scalars in enumerate(game["metadata"]["trees"]):
        parameters: dict = {name: arrays[f"tree_{i}_{name}"].copy() for name in TREE_ARRAYS}
        trees.append(TreeModel(**parameters, **scalars))
    return _tree_game(trees, arrays["background"].copy(), arrays["point"].copy(), game["oracle"])
