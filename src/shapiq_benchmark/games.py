"""Prepare and reload existing structured games without exponential tables or pickle."""

from __future__ import annotations

import hashlib
import importlib.metadata
import re
from typing import TYPE_CHECKING, cast

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from shapiq.explainer.nn.games.knn import KNNExplainerGame
from shapiq.explainer.nn.knn import KNNExplainer
from shapiq.game_theory import ExactComputer
from shapiq.tree.interventional.computer import InterventionalTreeSHAPIQ
from shapiq.tree.interventional.game import InterventionalGame

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
) -> tuple:
    """Construct the frozen recipe's probability-scale interventional game."""
    model = RandomForestClassifier(**(TREE_PARAMETERS if parameters is None else parameters)).fit(
        x, y
    )
    return model, InterventionalGame(model, background, point, class_index=1)


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


def validate_truth(oracle: Callable, truth: InteractionValues, *, exhaustive: bool) -> float:
    """Check endpoints, efficiency, determinism, and optionally every exact coefficient."""
    n = truth.n_players
    endpoints = oracle(np.array([np.zeros(n), np.ones(n)], dtype=bool))
    if not np.all(np.isfinite(truth.values)) or not np.all(np.isfinite(endpoints)):
        message = "Structured truth is nonfinite."
        raise ValueError(message)
    np.testing.assert_allclose(truth.baseline_value, endpoints[0], rtol=1e-8, atol=1e-10)
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
        expected = ExactComputer(oracle, n_players=n)(
            cast("IndexType", truth.index), order=truth.max_order
        )
        coordinates = truth.dict_values.keys() | expected.dict_values.keys()
        errors = [abs(truth[key] - expected[key]) for key in coordinates]
        np.testing.assert_allclose(
            [truth[key] for key in coordinates],
            [expected[key] for key in coordinates],
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


def prepare_structured(specs: list[dict], output: Path) -> list[dict]:
    """Freeze Breast Cancer tree explanations and nearest-neighbor data valuation."""
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
        dataset_name = spec.get("dataset", "breast_cancer")
        loaders = {"breast_cancer": load_breast_cancer, "digits": load_digits}
        if dataset_name not in loaders:
            message = "Structured dataset must be breast_cancer or digits."
            raise ValueError(message)
        dataset = loaders[dataset_name]()
        x, y = dataset.data, dataset.target
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            message = "Structured inputs must be numeric and finite."
            raise ValueError(message)
        train, test = train_test_split(np.arange(len(x)), test_size=0.2, random_state=0, stratify=y)
        background_indices = np.random.default_rng(0).choice(train, 16, replace=False)
        kind = spec["oracle"]
        index, order = spec["index"], spec["order"]
        if kind == "tree" and (index, order) in (("SV", 1), ("k-SII", 2)):
            point, background = x[test[0]], x[background_indices]
            model, oracle = tree_game(x[train], y[train], background, point)
            truth = InterventionalTreeSHAPIQ(
                model, background, class_index=1, index=index, max_order=order
            ).explain(point)
            small_model, small_oracle = tree_game(
                x[train, :8], y[train], background[:, :8], point[:8]
            )
            small_truth = InterventionalTreeSHAPIQ(
                small_model, background[:, :8], class_index=1, index=index, max_order=order
            ).explain(point[:8])
            error = validate_truth(small_oracle, small_truth, exhaustive=True)
            active = {
                int(feature)
                for tree in model.estimators_
                for feature in tree.tree_.feature
                if feature >= 0
            }
            arrays = {
                "x_train": x[train],
                "y_train": y[train],
                "background": background,
                "point": point,
            }
            metadata = {
                "model": "RandomForestClassifier",
                "model_parameters": TREE_PARAMETERS,
                "model_sha256": model_hash(model),
                "test_accuracy": model.score(x[test], y[test]),
                "active_players": len(active),
                "active_players_definition": "features used in any forest split",
                "background_size": len(background),
                "semantics": "mean class-1 probability over fixed joint background rows",
                "truth_method": "InterventionalTreeSHAPIQ",
                "player_unit": "feature",
            }
            n = x.shape[1]
            family = "local_explanation"
        elif kind == "knn" and (index, order) == ("SV", 1):
            count = spec.get("n_players", 128)
            if type(count) is not int or not 8 <= count <= len(train):
                message = "KNN n_players must be between 8 and the training split size."
                raise ValueError(message)
            scaler = StandardScaler().fit(x[train])
            scaled = scaler.transform(x)
            point = scaled[test[0]]
            selected = train[:count]
            point_label = int(y[test[0]])
            model, oracle = knn_game(scaled[selected], y[selected], point, point_label=point_label)
            truth = KNNExplainer(model, class_index=oracle.class_index).explain(point)
            small = selected[:8].copy()
            if point_label not in y[small]:
                small[-1] = selected[np.flatnonzero(y[selected] == point_label)[0]]
            small_model, small_oracle = knn_game(
                scaled[small], y[small], point, point_label=point_label
            )
            small_truth = KNNExplainer(small_model, class_index=small_oracle.class_index).explain(
                point
            )
            error = validate_truth(small_oracle, small_truth, exhaustive=True)
            arrays = {
                "x_train": scaled[selected],
                "y_train": y[selected],
                "point": point,
                "scaler_mean": scaler.mean_,
                "scaler_scale": scaler.scale_,
                "selected_rows": selected,
                "sortperm": oracle.sortperm,
            }
            metadata = {
                "model": "KNeighborsClassifier",
                "model_parameters": model.get_params(),
                "n_neighbors": 3,
                "class_index": oracle.class_index,
                "point_label": point_label,
                "test_accuracy": model.score(scaled[test], y[test]),
                "semantics": "number of correct-label examples among coalition's three nearest / 3",
                "truth_method": "KNNExplainer",
                "player_unit": "training example",
            }
            n, family = count, "data_valuation"
        else:
            message = f"Unsupported structured game specification: {spec}."
            raise ValueError(message)
        validate_truth(oracle, truth, exhaustive=False)
        artifact = output / f"{spec['id']}.npz"
        np.savez_compressed(artifact, **arrays, train_indices=train, test_indices=test)
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
            return table_game(artifact["values"].copy(), game["n_players"])
        if importlib.metadata.version("scikit-learn") != game["metadata"]["sklearn_version"]:
            message = "Live oracle reconstruction requires the snapshot's scikit-learn version."
            raise ValueError(message)
        x, y, point = artifact["x_train"], artifact["y_train"], artifact["point"]
        if game["oracle"] == "tree":
            model, oracle = tree_game(
                x, y, artifact["background"], point, game["metadata"]["model_parameters"]
            )
            if model_hash(model) != game["metadata"]["model_sha256"]:
                message = "Reconstructed tree model differs from the frozen model."
                raise ValueError(message)
        elif game["oracle"] == "knn":
            _, oracle = knn_game(
                x, y, point, game["metadata"]["model_parameters"], game["metadata"]["point_label"]
            )
            if oracle.class_index != game["metadata"]["class_index"]:
                message = "Reconstructed KNN class differs from the snapshot."
                raise ValueError(message)
            np.testing.assert_array_equal(oracle.sortperm, artifact["sortperm"])
        else:
            message = "Unknown oracle type."
            raise ValueError(message)
    if oracle.n_players != game["n_players"]:
        message = "Oracle player count differs from the snapshot."
        raise ValueError(message)
    return oracle
