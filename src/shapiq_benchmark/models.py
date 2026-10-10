"""Shared, frozen prediction models for the quality-focused benchmark.

One profile/dataset/feature-set/seed produces one model, reused across game
constructions. Dataset loaders own their preprocessing; additional missing-value
repair uses fitting rows only. Held-out rows never tune a model.
"""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import Any

    from shapiq.explainer.product_kernel.base import ProductKernelModel

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, log_loss, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from shapiq.explainer.product_kernel.conversion import convert_svm
from shapiq_benchmark.datasets import DATASETS, dataset_details, load_raw_dataset
from shapiq_benchmark.quality import QUALITY_PROTOCOL, model_validation_check, validate_protocol
from shapiq_benchmark.setup import _resolve_model_builder

MODEL_PROFILES: dict = {
    "random_forest": {
        "label": "Random forest",
        "parameters": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 5, "n_jobs": 1},
    },
    "xgboost": {
        "label": "XGBoost",
        "parameters": {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.05,
            "early_stopping_rounds": 20,
            "tree_method": "hist",
            "n_jobs": 1,
        },
    },
    "lightgbm": {
        "label": "LightGBM",
        "parameters": {
            "n_estimators": 200,
            "num_leaves": 63,
            "min_child_samples": 10,
            "learning_rate": 0.05,
            "n_jobs": 1,
            "verbosity": -1,
        },
        "early_stopping_rounds": 20,
    },
    "mlp": {
        "label": "Multilayer perceptron",
        "parameters": {"hidden_layer_sizes": (128, 64), "max_iter": 200},
        "early_stopping_rounds": 20,
    },
    "rbf_svm": {
        "label": "RBF support vector machine",
        "parameters": {"kernel": "rbf"},
        "validation_grid": {"C": [0.1, 1.0, 10.0], "gamma": ["scale", 0.1]},
    },
    "linear": {"label": "Linear control", "parameters": {}},
    "gaussian_process": {
        "label": "Gaussian process",
        "parameters": {"optimizer": None},
        "fit_rows_max": 256,
        "validation_kernels": ["rbf", "matern"],
    },
    "tabpfn_prediction": {
        "label": "Fixed TabPFN predictor",
        "checkpoint_version": "v2.5",
        "prediction_batch_size": 512,
        "prediction_padding": "repeat last row to fixed batch size",
        "parameters": {"n_estimators": 1},
        "fit_rows_max": 256,
    },
}
TRAINING_PROFILE: dict = {
    "id": "quality-v1",
    "fit_rows_max": 5000,
    "validation_rows_max": 1000,
    "test_rows_max": 1000,
    "test_fraction": 0.2,
    "validation_fraction_of_remainder": 0.2,
    "missing_values": "fitting-row medians",
    "classification_output": "probability of the second sorted original class",
}


@dataclass
class PreparedModel:
    """A fitted predictor, disjoint data partitions, and public reproduction metadata."""

    model: Any
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    metadata: dict

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Use a fixed scalar output, including for multiclass classification."""
        classification = self.metadata["task"] == "classification"
        size = MODEL_PROFILES[self.metadata["model_profile"]].get("prediction_batch_size", len(x))
        if not len(x):
            return np.empty(0)
        predictions = []
        for start in range(0, len(x), size):
            part = x[start : start + size]
            inputs = part
            if self.metadata["model_profile"] == "tabpfn_prediction" and len(part) < size:
                # Fix the numerical batch shape, including endpoint/singleton calls.
                inputs = np.concatenate((part, np.repeat(part[-1:], size - len(part), axis=0)))
            values = (
                self.model.predict_proba(inputs)[:, 1]
                if classification
                else self.model.predict(inputs)
            )
            predictions.append(values[: len(part)])
        return np.concatenate(predictions)


def _tabpfn_checkpoint(task: str) -> Path:
    """Resolve a prewarmed, versioned checkpoint before authenticating a model cache."""
    loading = importlib.import_module("tabpfn.model_loading")
    paths, _, _, _ = loading.resolve_model_path(
        None,
        "classifier" if task == "classification" else "regressor",
        version=MODEL_PROFILES["tabpfn_prediction"]["checkpoint_version"],
    )
    path = paths[0]
    if not path.is_file():
        message = "Warm the declared TabPFN checkpoint cache before launching model workers."
        raise FileNotFoundError(message)
    return path


def build_model(
    profile: str,
    task: str,
    seed: int,
    *,
    parameters: dict | None = None,
    device: str = "cpu",
    quality_protocol: str | None = None,
) -> object:
    """Construct one unfitted profile; scaled models keep original feature coordinates.

    The pipeline fits its scaler on whatever training rows a coalition supplies.
    Validation-only choices are supplied through ``parameters`` when refitting.
    """
    classification = task == "classification"
    params = {**MODEL_PROFILES[profile]["parameters"], **(parameters or {})}
    if profile == "linear":
        builder = LogisticRegression if classification else Ridge
        params = {"max_iter": 1000, **params} if classification else {"alpha": 1.0, **params}
    elif profile == "rbf_svm":
        builder = SVC if classification else SVR
        if classification:
            params["probability"] = True
    elif profile == "gaussian_process":
        builder = GaussianProcessClassifier if classification else GaussianProcessRegressor
        if not classification:
            params["normalize_y"] = True
    else:
        builder = _resolve_model_builder(
            "tabpfn" if profile == "tabpfn_prediction" else profile, task
        )
    if not (profile == "rbf_svm" and not classification):
        params["random_state"] = seed
    if profile == "tabpfn_prediction":
        torch = importlib.import_module("torch")

        if device == "cuda" and not torch.cuda.is_available():
            message = "CUDA requested but unavailable; CPU fallback is forbidden."
            raise ValueError(message)
        params.update(
            device=device,
            inference_precision=torch.float32,
            n_preprocessing_jobs=1,
            model_path=_tabpfn_checkpoint(task),
        )
    estimator = builder(**params)
    if quality_protocol == QUALITY_PROTOCOL and profile == "rbf_svm" and not classification:
        estimator = TransformedTargetRegressor(regressor=estimator, transformer=StandardScaler())
    if profile in {"mlp", "rbf_svm", "linear", "gaussian_process"}:
        return Pipeline([("scaler", StandardScaler()), ("model", estimator)])
    return estimator


def converted_svm(estimator: SVC | SVR | TransformedTargetRegressor) -> ProductKernelModel:
    """Convert an SVR/SVC, restoring scaled regression coefficients to payoff units."""
    if isinstance(estimator, TransformedTargetRegressor):
        converted = convert_svm(cast("SVR", estimator.regressor_))
        transformer = cast("StandardScaler", estimator.transformer_)
        scale = float(np.asarray(transformer.scale_).item())
        offset = float(np.asarray(transformer.mean_).item())
        converted.alpha = converted.alpha * scale
        converted.intercept = float(converted.intercept * scale + offset)
        return converted
    return convert_svm(estimator)


def coalition_model(prepared: PreparedModel) -> object:
    """Clone fixed validation decisions, with no validation or tuning inside a coalition.

    Classifier callers must handle singleton classes and re-encode subset labels;
    XGBoost's objective must also match the number of labels in that coalition.
    """
    model = clone(prepared.model)
    profile = prepared.metadata["model_profile"]
    if profile == "xgboost":
        model.set_params(n_estimators=prepared.model.best_iteration + 1, early_stopping_rounds=None)
    elif profile == "lightgbm":
        model.set_params(n_estimators=prepared.model.best_iteration_ or prepared.model.n_estimators)
    elif profile == "mlp":
        model.set_params(
            model__max_iter=prepared.metadata["structure"]["selected_epochs"],
            model__early_stopping=False,
        )
    return model


def _validation_loss(model: object, x: np.ndarray, y: np.ndarray, *, classification: bool) -> float:
    model = cast("Any", model)
    if classification:
        return float(log_loss(y, model.predict_proba(x), labels=model.classes_))
    return float(mean_squared_error(y, model.predict(x)))


def _fit_predictor(
    profile: str,
    task: str,
    seed: int,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    device: str,
    quality_protocol: str | None = None,
) -> tuple[Any, dict, dict]:
    """Select on the designated validation split, never on held-out test observations."""
    classification = task == "classification"
    parameters = {**MODEL_PROFILES[profile]["parameters"], "random_state": seed}
    diagnostics: dict = {}
    if profile == "xgboost":
        multiclass = classification and len(np.unique(y_fit)) > 2
        parameters.update(
            objective="multi:softprob"
            if multiclass
            else "binary:logistic"
            if classification
            else "reg:squarederror",
            eval_metric="mlogloss" if multiclass else "logloss" if classification else "rmse",
        )
    if profile == "rbf_svm" and not classification:
        parameters.pop("random_state")
    candidates = [parameters]
    if profile == "rbf_svm":
        grid = MODEL_PROFILES[profile]["validation_grid"]
        candidates = [
            {**parameters, "C": c, "gamma": gamma} for c in grid["C"] for gamma in grid["gamma"]
        ]
    elif profile == "gaussian_process":
        candidates = [
            {**parameters, "kernel": name} for name in MODEL_PROFILES[profile]["validation_kernels"]
        ]
    best, best_loss, best_parameters = None, float("inf"), parameters
    for candidate in candidates:
        constructor_parameters = dict(candidate)
        if profile == "gaussian_process":
            kernel = RBF() if candidate["kernel"] == "rbf" else Matern(nu=1.5)
            constructor_parameters["kernel"] = kernel + WhiteKernel(noise_level=0.01)
        model = cast(
            "Any",
            build_model(
                profile,
                task,
                seed,
                parameters=constructor_parameters,
                device=device,
                quality_protocol=quality_protocol,
            ),
        )
        if profile == "xgboost":
            model.fit(x_fit, y_fit, eval_set=[(x_validation, y_validation)], verbose=False)
        elif profile == "lightgbm":
            lightgbm = importlib.import_module("lightgbm")

            model.fit(
                x_fit,
                y_fit,
                eval_set=[(x_validation, y_validation)],
                callbacks=[
                    lightgbm.early_stopping(
                        MODEL_PROFILES[profile]["early_stopping_rounds"], verbose=False
                    )
                ],
            )
        elif profile == "mlp":
            scaler, estimator = model.named_steps["scaler"], model.named_steps["model"]
            scaled = scaler.fit_transform(x_fit)
            epoch_best, epoch_loss, stale = None, float("inf"), 0
            for epoch in range(parameters["max_iter"]):
                estimator.partial_fit(
                    scaled, y_fit, **({"classes": np.unique(y_fit)} if classification else {})
                )
                loss = _validation_loss(
                    model, x_validation, y_validation, classification=classification
                )
                if loss < epoch_loss:
                    epoch_best, epoch_loss, stale = deepcopy(estimator), loss, 0
                    diagnostics["selected_epochs"] = epoch + 1
                else:
                    stale += 1
                if stale >= MODEL_PROFILES[profile]["early_stopping_rounds"]:
                    break
            if epoch_best is None:
                message = "MLP validation produced no finite score."
                raise ValueError(message)
            model.steps[-1] = ("model", epoch_best)
        else:
            model.fit(x_fit, y_fit)
        loss = _validation_loss(model, x_validation, y_validation, classification=classification)
        if loss < best_loss:
            best, best_loss, best_parameters = model, loss, candidate
    if best is None:
        message = "No model candidate produced a finite validation score."
        raise ValueError(message)
    if isinstance(best, Pipeline):
        scaler = best.named_steps["scaler"]
        diagnostics.update(
            scaling="fitting-row StandardScaler",
            mean=scaler.mean_.tolist(),
            scale=scaler.scale_.tolist(),
        )
        if isinstance(best.named_steps["model"], TransformedTargetRegressor):
            target_scaler = best.named_steps["model"].transformer_
            diagnostics["target_scaling"] = {
                "rule": "fitting-row mean and standard deviation; predictions restored to original units",
                "mean": float(target_scaler.mean_[0]),
                "scale": float(target_scaler.scale_[0]),
            }
    if len(candidates) > 1:
        diagnostics.update(
            validation_candidates=len(candidates), selected_validation_loss=best_loss
        )
    if profile == "lightgbm":
        diagnostics["selected_iterations"] = int(best.best_iteration_)
    return best, best_parameters, diagnostics


def _split(
    rows: np.ndarray, y: np.ndarray, size: float, seed: int, *, classification: bool
) -> tuple[np.ndarray, np.ndarray, str]:
    """Stratify when feasible; explicitly record a fallback for rare-class splits."""
    labels = y[rows] if classification else None
    try:
        left, right = train_test_split(rows, test_size=size, random_state=seed, stratify=labels)
    except ValueError:
        if not classification:
            raise
        left, right = train_test_split(rows, test_size=size, random_state=seed)
        return left, right, "shuffled: stratification infeasible"
    else:
        return left, right, "stratified" if classification else "shuffled"


def _bounded(
    rows: np.ndarray, y: np.ndarray, limit: int, seed: int, *, classification: bool
) -> tuple[np.ndarray, str]:
    if len(rows) <= limit:
        return rows, "all rows"
    _, selected, rule = _split(rows, y, limit, seed, classification=classification)
    return selected, rule


def _scores(
    model: object, x: np.ndarray, y: np.ndarray, classes: np.ndarray, *, classification: bool
) -> dict:
    model = cast("Any", model)
    if classification:
        return {
            "log_loss": float(log_loss(y, model.predict_proba(x), labels=classes)),
            "balanced_accuracy": float(balanced_accuracy_score(y, model.predict(x))),
        }
    return {
        "mse": float(mean_squared_error(y, model.predict(x))),
        "r2": float(r2_score(y, model.predict(x))),
    }


def _tree_diagnostics(model: object, profile: str) -> dict:
    """Describe realized structure, without claiming depth guarantees interaction strength."""
    model = cast("Any", model)
    features, depths, path_widths = set(), [], []
    trees = (
        model.estimators_
        if profile == "random_forest"
        else model.get_booster().get_dump(dump_format="json")
    )
    if profile == "xgboost":
        # The early-stopped predictor ignores trees fitted after its best round.
        classes_per_round = getattr(model, "n_classes_", 2)
        trees_per_round = classes_per_round if classes_per_round > 2 else 1
        trees = trees[: (model.best_iteration + 1) * trees_per_round]
    for estimator in trees:
        tree = estimator.tree_ if profile == "random_forest" else json.loads(estimator)
        stack: list[tuple[Any, int, frozenset]] = [
            (0 if profile == "random_forest" else tree, 0, frozenset())
        ]
        depth_max = width_max = 0
        while stack:
            node, depth, path = stack.pop()
            if profile == "random_forest":
                feature = int(tree.feature[node])
                children = (
                    [int(tree.children_left[node]), int(tree.children_right[node])]
                    if feature >= 0
                    else []
                )
            else:
                feature = int(node["split"].removeprefix("f")) if "split" in node else -1
                children = node.get("children", [])
            if children:
                features.add(feature)
                path = path | {feature}
                stack.extend((child, depth + 1, path) for child in children)
            else:
                depth_max = max(depth_max, depth)
                width_max = max(width_max, len(path))
        depths.append(depth_max)
        path_widths.append(width_max)
    return {
        "tree_count": len(trees),
        "tree_depths": depths,
        "features_used": sorted(features),
        "max_distinct_features_on_path": max(path_widths),
        "note": "Model structure only; game-level interaction diagnostics are separate.",
    }


def _fit(
    dataset: str,
    n_players: int,
    seed: int,
    profile: str,
    x: np.ndarray,
    original_y: np.ndarray,
    names: list,
    identity: dict,
) -> PreparedModel:
    classification = DATASETS[dataset]["task"] == "classification"
    training = identity["training_profile"]
    classes, encoded = np.unique(original_y, return_inverse=True)
    y = encoded if classification else original_y
    rng = np.random.default_rng(seed)
    features = np.sort(
        rng.permutation(x.shape[1])[:n_players]
        if identity.get("feature_rule") == "nested"
        else rng.choice(x.shape[1], n_players, replace=False)
    )
    fit, test, test_rule = _split(
        np.arange(len(x)), y, training["test_fraction"], seed, classification=classification
    )
    fit, validation, validation_rule = _split(
        fit,
        y,
        training["validation_fraction_of_remainder"],
        seed,
        classification=classification,
    )
    fit, fit_cap = _bounded(fit, y, training["fit_rows_max"], seed, classification=classification)
    validation, validation_cap = _bounded(
        validation, y, training["validation_rows_max"], seed, classification=classification
    )
    test, test_cap = _bounded(
        test, y, training["test_rows_max"], seed, classification=classification
    )
    if classification and len(np.unique(y[fit])) != len(classes):
        message = "Fitting partition omits a class; this seeded instance is unsupported."
        raise ValueError(message)
    if identity.get("feature_rule") == "continuous":
        categorical = cast("Any", DATASETS[dataset].get("categorical_features", []))
        eligible = [
            i
            for i, name in enumerate(names)
            if categorical != "all"
            and name not in categorical
            and len(np.unique(x[fit, i][np.isfinite(x[fit, i])])) > 2
        ]
        if len(eligible) < n_players:
            message = "Too few continuous fitting-row features for this game."
            raise ValueError(message)
        features = np.sort(np.random.default_rng(seed).choice(eligible, n_players, replace=False))
    selected = np.asarray(x[:, features], dtype=float).copy()
    if np.isinf(selected).any():
        message = "Infinite inputs cannot be repaired by the missing-value protocol."
        raise ValueError(message)
    medians = np.nanmedian(selected[fit], axis=0)
    if not np.isfinite(medians).all():
        message = "A selected feature is missing on every fitting row."
        raise ValueError(message)
    rows, columns = np.where(np.isnan(selected))
    selected[rows, columns] = medians[columns]
    model, parameters, diagnostics = _fit_predictor(
        profile,
        str(DATASETS[dataset]["task"]),
        seed,
        selected[fit],
        y[fit],
        selected[validation],
        y[validation],
        device=identity.get("device", "cpu"),
        quality_protocol=identity.get("quality_protocol"),
    )
    dummy = DummyClassifier(strategy="prior") if classification else DummyRegressor(strategy="mean")
    dummy.fit(selected[fit], y[fit])
    metrics = {}
    for label, indices in (("validation", validation), ("test", test)):
        metrics[label] = {
            "model": _scores(
                model,
                selected[indices],
                y[indices],
                np.arange(len(classes)),
                classification=classification,
            ),
            "dummy": _scores(
                dummy,
                selected[indices],
                y[indices],
                np.arange(len(classes)),
                classification=classification,
            ),
        }
    metadata = {
        **identity,
        **dataset_details(dataset),
        "model_parameters": parameters,
        "feature_indices": features.tolist(),
        "feature_names": [str(names[i]) for i in features],
        "train_indices": fit.tolist(),
        "background_indices": fit[:16].tolist(),
        "training_rows": len(fit),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "validation_indices": validation.tolist(),
        "test_indices": test.tolist(),
        "split_rules": {
            "test": test_rule,
            "validation": validation_rule,
            "fit_cap": fit_cap,
            "validation_cap": validation_cap,
            "test_cap": test_cap,
        },
        "imputation_medians": medians.tolist(),
        "output": "class_probability" if classification else "prediction",
        "classes": classes.tolist() if classification else None,
        "output_class": classes[1].item() if classification else None,
        "quality": metrics,
        "structure": (
            _tree_diagnostics(model, profile)
            if profile in {"random_forest", "xgboost"}
            else diagnostics
        ),
    }
    if profile == "tabpfn_prediction":
        from shapiq_benchmark.media import preparation_backend

        if identity.get("device") == "cuda":
            metadata["preparation_hardware"] = preparation_backend("cuda")
        metadata["oracle_precision"] = "float32"
        metadata["prediction_batch_size"] = MODEL_PROFILES[profile]["prediction_batch_size"]
    if profile == "xgboost":
        metadata["best_iteration"] = int(model.best_iteration)
    if identity.get("quality_protocol") == QUALITY_PROTOCOL:
        metadata["model_validation_gate"] = model_validation_check(metadata)
    return PreparedModel(
        model,
        selected[fit],
        y[fit],
        selected[validation],
        y[validation],
        selected[test],
        y[test],
        metadata,
    )


def prepare_model(
    dataset: str,
    n_players: int,
    seed: int,
    profile: str,
    *,
    cache_dir: str | Path | None = None,
    device: str = "cpu",
    feature_rule: str = "all",
    quality_protocol: str | None = None,
) -> PreparedModel:
    """Fit once, or authenticate and reuse a local model artifact.

    Cache keys include source, backend versions, raw data and the complete named
    training/profile configuration. A lock prevents duplicate concurrent fits.
    Joblib artifacts are private trusted-local files, never user uploads.
    """
    validate_protocol(quality_protocol)
    if profile not in MODEL_PROFILES:
        message = f"Unknown implemented model profile: {profile}"
        raise ValueError(message)
    if device not in {"cpu", "cuda"} or (device != "cpu" and profile != "tabpfn_prediction"):
        message = "CUDA is qualified only for explicit TabPFN prediction profiles."
        raise ValueError(message)
    if feature_rule not in {"all", "continuous", "nested"}:
        message = "Unknown model feature selection rule."
        raise ValueError(message)
    x, y, names = load_raw_dataset(dataset)
    if type(n_players) is not int or not 1 <= n_players <= x.shape[1]:
        message = "Model feature count must fit the dataset."
        raise ValueError(message)
    source = hashlib.sha256()
    for name in ("models.py", "datasets.py", "dataset_catalog.py", "setup.py", "quality.py"):
        source.update(Path(__file__).with_name(name).read_bytes())
    packages = ["numpy", "scikit-learn", "joblib"]
    if profile in {"xgboost", "lightgbm"}:
        packages.append(profile)
    elif profile == "tabpfn_prediction":
        packages.extend(["tabpfn", "torch"])
    training = {**TRAINING_PROFILE}
    if "fit_rows_max" in MODEL_PROFILES[profile]:
        training["id"] = f"quality-v1-{profile}"
        training["fit_rows_max"] = MODEL_PROFILES[profile]["fit_rows_max"]
    identity = {
        "dataset": dataset,
        "dataset_source": DATASETS[dataset]["source"],
        "task": DATASETS[dataset]["task"],
        "n_players": n_players,
        "instance_seed": seed,
        "model_profile": profile,
        "profile": MODEL_PROFILES[profile],
        "training_profile": training,
        "dataset_feature_names": [str(name) for name in names],
        "data_arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in (("x", np.asarray(x)), ("y", np.asarray(y)))
        },
        "data_sha256": hashlib.sha256(
            np.asarray(x).tobytes() + np.asarray(y).tobytes()
        ).hexdigest(),
        "model_source_sha256": source.hexdigest(),
        "model_packages": {name: importlib.metadata.version(name) for name in packages},
    }
    if feature_rule != "all":
        identity["feature_rule"] = feature_rule
    if quality_protocol is not None:
        identity["quality_protocol"] = quality_protocol
        training["id"] = training["id"].replace("quality-v1", quality_protocol)
    if profile == "tabpfn_prediction":
        torch = importlib.import_module("torch")

        if device == "cuda" and not torch.cuda.is_available():
            message = "CUDA requested but unavailable; CPU fallback is forbidden."
            raise ValueError(message)
        checkpoint = _tabpfn_checkpoint(str(DATASETS[dataset]["task"]))
        with checkpoint.open("rb") as stream:
            checkpoint_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        identity.update(
            checkpoint_name=checkpoint.name,
            checkpoint_sha256=checkpoint_hash,
            device=device,
            precision="float32",
            cuda_version=torch.version.cuda,
            device_name=torch.cuda.get_device_name() if device == "cuda" else "cpu",
        )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    identity["model_key"] = key
    if cache_dir is None:
        return _fit(dataset, n_players, seed, profile, x, y, names, identity)
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    artifact = directory / f"{key}.joblib"
    checksum = artifact.with_suffix(".sha256")
    with artifact.with_suffix(".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if artifact.exists() and checksum.exists():
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if digest != checksum.read_text().strip():
                message = "Cached model artifact checksum mismatch."
                raise ValueError(message)
            prepared = joblib.load(artifact)
            if any(prepared.metadata.get(name) != value for name, value in identity.items()):
                message = "Cached model identity disagrees with the requested training recipe."
                raise ValueError(message)
        else:
            prepared = _fit(dataset, n_players, seed, profile, x, y, names, identity)
            temporary = artifact.with_suffix(".tmp")
            joblib.dump(prepared, temporary)
            digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
            temporary.replace(artifact)
            checksum.write_text(digest + "\n")
        prepared.metadata["model_artifact_sha256"] = digest
        return prepared
