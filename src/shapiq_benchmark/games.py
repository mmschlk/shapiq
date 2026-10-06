"""Prepare and reload existing structured games without exponential tables or pickle."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import re
from typing import TYPE_CHECKING, cast

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from shapiq.explainer.nn.games.knn import KNNExplainerGame
from shapiq.explainer.nn.knn import KNNExplainer
from shapiq.explainer.product_kernel import ProductKernelExplainer
from shapiq.explainer.product_kernel.base import ProductKernelModel
from shapiq.explainer.product_kernel.conversion import convert_svm
from shapiq.explainer.product_kernel.game import ProductKernelGame
from shapiq.game_theory import ExactComputer
from shapiq.tree.interventional.computer import InterventionalTreeSHAPIQ
from shapiq.tree.interventional.game import InterventionalGame
from shapiq_benchmark.exact import exact_table_truth

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from shapiq import InteractionValues
    from shapiq.typing import IndexType

TREE_PARAMETERS = {
    "n_estimators": 8,
    "max_depth": 5,
    "min_samples_leaf": 3,
    "n_jobs": 1,
    "random_state": 0,
}


def model_hash(model: RandomForestClassifier) -> str:
    """Identify the fitted forest by all prediction-relevant tree arrays."""
    hasher = hashlib.sha256()
    hasher.update(np.asarray(model.classes_).tobytes())
    for tree in model.estimators_:
        for value in (
            tree.tree_.children_left,
            tree.tree_.children_right,
            tree.tree_.feature,
            tree.tree_.threshold,
            tree.tree_.value,
        ):
            hasher.update(value.tobytes())
    return hasher.hexdigest()


def tree_game(
    x: np.ndarray,
    y: np.ndarray,
    background: np.ndarray,
    point: np.ndarray,
    parameters: dict | None = None,
    *,
    class_index: int = 1,
) -> tuple:
    """Construct the frozen recipe's probability-scale interventional game."""
    model = RandomForestClassifier(**(TREE_PARAMETERS if parameters is None else parameters)).fit(
        x, y
    )
    return model, InterventionalGame(model, background, point, class_index=class_index)


def knn_game(
    x: np.ndarray,
    y: np.ndarray,
    point: np.ndarray,
    parameters: dict | None = None,
    point_label: int = 1,
) -> tuple:
    """Construct the fixed-denominator, uniform three-neighbor utility game."""
    parameters = parameters or {
        "n_neighbors": 3,
        "weights": "uniform",
        "algorithm": "brute",
        "n_jobs": 1,
    }
    model = KNeighborsClassifier(**parameters).fit(x, y)
    matches = np.flatnonzero(model.classes_ == point_label)
    if len(matches) != 1:
        message = "The valued training subset must contain the held-out point's true class."
        raise ValueError(message)
    return model, KNNExplainerGame(model, point, class_index=int(matches[0]))


def tnn_game(
    x: np.ndarray, y: np.ndarray, point: np.ndarray, parameters: dict | None, point_label: int
) -> tuple:
    """Construct the shipped radius-neighbor utility with an explicit training-only radius."""
    from sklearn.neighbors import RadiusNeighborsClassifier

    from shapiq.explainer.nn.games.tnn import TNNExplainerGame

    if parameters is None:
        message = "Radius-neighbor reconstruction requires explicit parameters."
        raise ValueError(message)
    model = RadiusNeighborsClassifier(**parameters).fit(x, y)
    matches = np.flatnonzero(model.classes_ == point_label)
    if len(matches) != 1:
        message = "Selected radius-neighbor rows omit the held-out true class."
        raise ValueError(message)
    return model, TNNExplainerGame(model, point, class_index=int(matches[0]))


def validate_truth(oracle: Callable, truth: InteractionValues, *, exhaustive: bool) -> float:
    """Check endpoints, efficiency, determinism, and optionally every exact coefficient."""
    n = truth.n_players
    endpoints = oracle(np.array([np.zeros(n), np.ones(n)], dtype=bool))
    if not np.all(np.isfinite(truth.values)) or not np.all(np.isfinite(endpoints)):
        message = "Structured truth is nonfinite."
        raise ValueError(message)
    np.testing.assert_allclose(truth.baseline_value, endpoints[0], rtol=1e-8, atol=1e-10)
    if truth.index in ("SV", "k-SII", "STII", "FSII"):
        np.testing.assert_allclose(
            sum(value for key, value in truth.dict_values.items() if key),
            endpoints[1] - endpoints[0],
            rtol=1e-8,
            atol=1e-10,
        )
    coalitions = np.random.default_rng(0).integers(0, 2, size=(12, n)).astype(bool)
    values = oracle(coalitions)
    np.testing.assert_allclose(values[::-1], oracle(coalitions[::-1]), rtol=0, atol=0)
    np.testing.assert_allclose(
        values, np.concatenate([oracle(row[None]) for row in coalitions]), rtol=0, atol=0
    )
    if exhaustive:
        if n > 8:
            message = "Exhaustive validation is restricted to at most eight players."
            raise ValueError(message)
        if truth.index == "FSII" and truth.max_order == 2:
            # ExactComputer's finite endpoint penalty leaves a numerical floor,
            # particularly on null coefficients. Check the same exhaustive game
            # using the already-qualified derivative formula, not looser tolerances.
            masks = ((np.arange(2**n)[:, None] >> np.arange(n)) & 1).astype(bool)
            reference = exact_table_truth(
                np.asarray(oracle(masks)), n, [{"index": "FSII", "order": 2}]
            )["FSII", 2]
            expected = dict(
                zip(map(tuple, reference["coordinates"]), reference["values"], strict=True)
            )
        else:
            expected = ExactComputer(oracle, n_players=n)(
                cast("IndexType", truth.index), order=truth.max_order
            ).dict_values
        coordinates = (truth.dict_values.keys() | expected.keys()) - {()}
        errors = [abs(truth[key] - expected.get(key, 0.0)) for key in coordinates]
        np.testing.assert_allclose(
            [truth[key] for key in coordinates],
            [expected.get(key, 0.0) for key in coordinates],
            rtol=1e-8,
            atol=1e-10,
        )
        return max(errors, default=0.0)
    return 0.0


def truth_dict(truth: InteractionValues) -> dict:
    """Serialize sparse exact coefficients in explicit player order."""
    coordinates = [key for key in truth.dict_values if key]
    return {
        "coordinates": [list(key) for key in coordinates],
        "values": [float(truth[key]) for key in coordinates],
        "baseline": float(truth.baseline_value),
        "energy": float(sum(truth[key] ** 2 for key in coordinates)),
    }


def signal_metadata(truth: InteractionValues, payoff_range: float) -> dict:
    """Certify a conservative signal ratio using SD <= (maximum payoff - minimum payoff)/2."""
    if not np.isfinite(payoff_range) or payoff_range < 0:
        message = "A payoff range bound must be finite and nonnegative."
        raise ValueError(message)
    dimension = sum(math.comb(truth.n_players, degree) for degree in range(1, truth.max_order + 1))
    energy = sum(value**2 for key, value in truth.dict_values.items() if key)
    return {
        "payoff_range_upper_bound": float(payoff_range),
        "signal_ratio": float(math.sqrt(energy / dimension) / (payoff_range / 2))
        if payoff_range
        else 0.0,
        "signal_ratio_definition": "certified lower bound: RMS exact coefficients / (payoff range upper bound / 2)",
    }


def prepare_structured(specs: list[dict], output: Path) -> list[dict]:
    """Freeze seeded tree, nearest-neighbor, and product-kernel games with exact truth."""
    ids = [spec.get("id") for spec in specs]
    if (
        not ids
        or any(
            not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", name) for name in ids
        )
        or len(ids) != len(set(ids))
    ):
        message = "Structured game IDs must be nonempty, unique filename-safe names."
        raise ValueError(message)
    output.mkdir(parents=True, exist_ok=True)
    games = []
    for spec in specs:
        if "model_profile" in spec:
            from shapiq_benchmark.structured import prepare_profiled

            games.append(prepare_profiled(spec, output))
            continue
        dataset_name = spec.get("dataset", "breast_cancer")
        loaders = {"breast_cancer": load_breast_cancer, "digits": load_digits}
        if dataset_name in loaders:
            dataset = loaders[dataset_name]()
            x, y = dataset.data, dataset.target
        else:
            from shapiq_benchmark.datasets import DATASETS, load_dataset

            if (
                spec["oracle"] not in ("knn", "tnn")
                or DATASETS[dataset_name]["task"] != "classification"
            ):
                message = "New structured datasets require a model profile, or classification KNN."
                raise ValueError(message)
            x, y, _, _, _ = load_dataset(dataset_name, spec.get("instance_seed", 0))
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            message = "Structured inputs must be numeric and finite."
            raise ValueError(message)
        seed = spec.get("instance_seed", 0)
        if type(seed) is not int or not 0 <= seed < 2**32:
            message = "Structured instance_seed must be an unsigned 32-bit integer."
            raise ValueError(message)
        kind = spec["oracle"]
        population = (
            np.flatnonzero(y < 2)
            if kind == "product_kernel" and dataset_name == "digits"
            else np.arange(len(x))
        )
        train, test = train_test_split(
            population, test_size=0.2, random_state=seed, stratify=y[population]
        )
        background_indices = np.random.default_rng(seed).choice(train, 16, replace=False)
        index, order = spec["index"], spec["order"]
        if kind == "tree" and (index, order) in (
            ("SV", 1),
            ("k-SII", 2),
            ("SII", 2),
            ("STII", 2),
            ("FSII", 2),
            ("FBII", 2),
        ):
            # sklearn predicts on float32 inputs; exact tree routing must see those same values.
            tree_x = x.astype(np.float32).astype(np.float64)
            point, background = tree_x[test[0]], tree_x[background_indices]
            parameters = {**TREE_PARAMETERS, "random_state": seed}
            class_index = int(np.flatnonzero(np.unique(y[train]) == y[test[0]])[0])
            model, oracle = tree_game(
                tree_x[train], y[train], background, point, parameters, class_index=class_index
            )
            truth = InterventionalTreeSHAPIQ(
                model, background, class_index=class_index, index=index, max_order=order
            ).explain(point)
            small_model, small_oracle = tree_game(
                tree_x[train, :8],
                y[train],
                background[:, :8],
                point[:8],
                parameters,
                class_index=class_index,
            )
            small_truth = InterventionalTreeSHAPIQ(
                small_model,
                background[:, :8],
                class_index=class_index,
                index=index,
                max_order=order,
            ).explain(point[:8])
            error = validate_truth(small_oracle, small_truth, exhaustive=True)
            active = {
                int(feature)
                for tree in model.estimators_
                for feature in tree.tree_.feature
                if feature >= 0
            }
            arrays = {
                "x_train": tree_x[train],
                "y_train": y[train],
                "background": background,
                "point": point,
            }
            metadata = {
                "model": "RandomForestClassifier",
                "model_parameters": parameters,
                "class_index": class_index,
                "point_label": int(y[test[0]]),
                "background_indices": background_indices.tolist(),
                "model_sha256": model_hash(model),
                "test_accuracy": model.score(x[test], y[test]),
                "active_players": len(active),
                "active_players_definition": "features used in any forest split",
                "background_size": len(background),
                "semantics": "mean held-out true-class probability over fixed joint background rows",
                "truth_method": "InterventionalTreeSHAPIQ",
                "player_unit": "feature",
                "preprocessing": "inputs rounded to sklearn's float32 prediction precision",
            }
            n = x.shape[1]
            family = "local_explanation"
        elif kind in ("knn", "tnn") and (index, order) == ("SV", 1):
            count = spec.get("n_players", 128)
            if type(count) is not int or not 8 <= count <= len(train):
                message = "KNN n_players must be between 8 and the training split size."
                raise ValueError(message)
            scaler = StandardScaler().fit(x[train])
            scaled = scaler.transform(x)
            point = scaled[test[0]]
            selected = train[:count]
            row_selection = spec.get("row_selection", "first")
            if row_selection not in ("first", "stratified"):
                message = "KNN row_selection must be first or stratified."
                raise ValueError(message)
            if row_selection == "stratified" and count < len(train):
                selected, _ = train_test_split(
                    train, train_size=count, stratify=y[train], random_state=seed
                )
            point_label = int(y[test[0]])
            constructor, computer, parameters = knn_game, KNNExplainer, None
            if kind == "tnn":
                from scipy.spatial.distance import pdist

                from shapiq.explainer.nn import ThresholdNNExplainer

                distances = pdist(scaled[train[:512]])
                positive = distances[distances > 0]
                if not len(positive):
                    message = "Radius selection requires distinct training inputs."
                    raise ValueError(message)
                parameters = {"radius": float(np.median(positive)), "n_jobs": 1}
                constructor, computer = tnn_game, ThresholdNNExplainer
            model, oracle = constructor(
                scaled[selected], y[selected], point, parameters, point_label
            )
            truth = computer(model, class_index=oracle.class_index).explain(point)
            if kind == "tnn":
                truth.baseline_value = truth[()]
            small = selected[:8].copy()
            if point_label not in y[small]:
                small[-1] = selected[np.flatnonzero(y[selected] == point_label)[0]]
            small_model, small_oracle = constructor(
                scaled[small], y[small], point, parameters, point_label
            )
            small_truth = computer(small_model, class_index=small_oracle.class_index).explain(point)
            if kind == "tnn":
                small_truth.baseline_value = small_truth[()]
            error = validate_truth(small_oracle, small_truth, exhaustive=True)
            arrays = {
                "x_train": scaled[selected],
                "y_train": y[selected],
                "point": point,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "selected_rows": selected,
                **({"sortperm": oracle.sortperm} if kind == "knn" else {}),
            }
            metadata = {
                "model": "KNeighborsClassifier",
                "model_parameters": model.get_params(),
                "n_neighbors": 3,
                "row_selection": row_selection,
                "class_index": oracle.class_index,
                "point_label": point_label,
                "test_accuracy": model.score(scaled[test], y[test]) if kind != "tnn" else None,
                "semantics": "number of correct-label examples among coalition's three nearest / 3",
                "truth_method": computer.__name__,
                "player_unit": "training example",
            }
            if kind == "tnn":
                metadata.update(
                    model="RadiusNeighborsClassifier",
                    semantics="correct-label fraction inside the fixed radius; empty neighborhood uses class prior",
                    radius_rule="median nonzero pairwise distance on first 512 standardized training rows",
                    radius_fit_indices=train[:512].tolist(),
                )
                metadata.pop("n_neighbors")
            n, family = count, "data_valuation"
        elif kind == "product_kernel" and (index, order) == ("SV", 1):
            scaler = StandardScaler().fit(x[train])
            scaled = scaler.transform(x)
            selected, point = train[:128], scaled[test[0]]
            model = SVC(kernel="rbf", gamma="scale", random_state=seed).fit(
                scaled[selected], y[selected]
            )
            converted = convert_svm(model)
            if converted.gamma is None:
                message = "RBF conversion must preserve the fitted kernel bandwidth."
                raise ValueError(message)
            n = x.shape[1]
            oracle = ProductKernelGame(n, point, converted)
            truth = ProductKernelExplainer(model).explain(point)
            small_model = SVC(kernel="rbf", gamma="scale", random_state=seed).fit(
                scaled[selected, :8], y[selected]
            )
            small_oracle = ProductKernelGame(8, point[:8], convert_svm(small_model))
            small_truth = ProductKernelExplainer(small_model).explain(point[:8])
            error = validate_truth(small_oracle, small_truth, exhaustive=True)
            arrays = {
                "x_train": scaled[selected],
                "y_train": y[selected],
                "point": point,
                "support_vectors": converted.X_train,
                "alpha": converted.alpha,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "selected_rows": selected,
            }
            metadata = {
                "model": "SVC",
                "model_parameters": model.get_params(),
                "classes": model.classes_.tolist(),
                "gamma": float(converted.gamma),
                "intercept": float(converted.intercept),
                "test_accuracy": model.score(scaled[test], y[test]) if kind != "tnn" else None,
                "preprocessing": "StandardScaler fitted on training split only",
                "semantics": "RBF decision margin with omitted feature kernel factors set to one",
                "truth_method": "ProductKernelExplainer",
                "player_unit": "feature",
                "active_players": int(np.count_nonzero(np.ptp(scaled[train], axis=0))),
                "active_players_definition": "features varying in the training split",
            }
            family = "local_explanation"
        else:
            message = f"Unsupported structured game specification: {spec}."
            raise ValueError(message)
        validate_truth(oracle, truth, exhaustive=False)
        payoff_range = (
            1.0 if kind in ("tree", "knn", "tnn") else float(np.abs(converted.alpha).sum())
        )
        metadata.update(signal_metadata(truth, payoff_range))
        artifact = output / f"{spec['id']}.npz"
        np.savez_compressed(
            artifact, allow_pickle=False, **arrays, train_indices=train, test_indices=test
        )
        games.append(
            {
                "id": spec["id"],
                "family": family,
                "stratum": f"{dataset_name}_{kind}_{n}",
                "n_players": n,
                "index": index,
                "order": order,
                "oracle": kind,
                "artifact": artifact.name,
                "truth": truth_dict(truth),
                "metadata": {
                    **metadata,
                    "instance_seed": seed,
                    "case_id": spec.get("basecase_id", spec["id"]),
                    "recipe": spec.get("basecase_id", spec["id"]),
                    "cluster_id": f"{spec.get('basecase_id', spec['id'])}-i{seed}",
                    "dataset": dataset_name,
                    "data_sha256": hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
                    "sklearn_version": importlib.metadata.version("scikit-learn"),
                    "point_row": int(test[0]),
                    "small_validation_players": 8,
                    "nonzero_coefficients": sum(
                        key != () and value != 0 for key, value in truth.dict_values.items()
                    ),
                    "small_validation_max_error": error,
                },
            }
        )
    return games


def load_game(game: dict, root: Path) -> Callable:
    """Rebuild a live oracle from verified arrays and the qualified sklearn version."""
    from shapiq_benchmark.runner import table_game

    with np.load(root / game["artifact"], allow_pickle=False) as artifact:
        if game.get("oracle", "table") == "table":
            return table_game(
                artifact["values"].copy(),
                game["n_players"],
                artifact["evaluation_seconds"].copy() if "evaluation_seconds" in artifact else None,
            )
        if game["metadata"].get("structured_format") == "numeric-model-v1":
            from shapiq_benchmark.structured import load_profiled

            return load_profiled(game, artifact)
        if importlib.metadata.version("scikit-learn") != game["metadata"]["sklearn_version"]:
            message = "Live oracle reconstruction requires the snapshot's scikit-learn version."
            raise ValueError(message)
        x, y, point = artifact["x_train"], artifact["y_train"], artifact["point"]
        if game["oracle"] == "tree":
            model, oracle = tree_game(
                x,
                y,
                artifact["background"],
                point,
                game["metadata"]["model_parameters"],
                class_index=game["metadata"].get("class_index", 1),
            )
            if model_hash(model) != game["metadata"]["model_sha256"]:
                message = "Reconstructed tree model differs from the frozen model."
                raise ValueError(message)
        elif game["oracle"] in ("knn", "tnn"):
            constructor = knn_game if game["oracle"] == "knn" else tnn_game
            _, oracle = constructor(
                x, y, point, game["metadata"]["model_parameters"], game["metadata"]["point_label"]
            )
            if oracle.class_index != game["metadata"]["class_index"]:
                message = "Reconstructed KNN class differs from the snapshot."
                raise ValueError(message)
            if game["oracle"] == "knn":
                np.testing.assert_array_equal(oracle.sortperm, artifact["sortperm"])
        elif game["oracle"] == "product_kernel":
            converted = ProductKernelModel(
                X_train=artifact["support_vectors"].copy(),
                alpha=artifact["alpha"].copy(),
                n=len(artifact["alpha"]),
                d=game["n_players"],
                gamma=game["metadata"]["gamma"],
                intercept=game["metadata"]["intercept"],
            )
            oracle = ProductKernelGame(game["n_players"], point, converted)
        else:
            message = "Unknown oracle type."
            raise ValueError(message)
    if oracle.n_players != game["n_players"]:
        message = "Oracle player count differs from the snapshot."
        raise ValueError(message)
    return oracle
