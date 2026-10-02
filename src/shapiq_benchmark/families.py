"""Small, explicit recipes exercising shipped games on native real-data features.

These are family representatives, not coverage of every dataset wrapper. Sampled
payoffs marked ``stochastic_frozen`` require a canonical, persisted coalition
table; exact truth then describes that realization, not a population expectation.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shapiq.explainer.product_kernel.conversion import convert_svm
from shapiq_benchmark.datasets import (
    DATASETS,
    dataset_details,
    load_dataset as _dataset,
)
from shapiq_benchmark.execution import hardware
from shapiq_benchmark.models import coalition_model, prepare_model

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from shapiq_benchmark.models import PreparedModel

MAX_ENUMERATION_PLAYERS = 20

_LEGACY = "shapiq_games.benchmark."
_NN = "shapiq.explainer.nn.games."
_IMPUTER = "shapiq.imputer."
# name: (fully qualified class, payoff, player unit, sampled realization)
_RECIPES = {
    "local_baseline": (
        _IMPUTER + "baseline_imputer.BaselineImputer",
        "prediction with fixed mean baseline",
        "feature",
        False,
    ),
    "local_baseline_forest": (
        _IMPUTER + "baseline_imputer.BaselineImputer",
        "eight-tree forest prediction with fixed mean baseline",
        "feature",
        False,
    ),
    "local_marginal": (
        _IMPUTER + "marginal_imputer.MarginalImputer",
        "mean prediction over 16 fixed joint background rows",
        "feature",
        False,
    ),
    "local_gaussian": (
        _IMPUTER + "gaussian_imputer.GaussianImputer",
        "Gaussian conditional Monte Carlo prediction",
        "feature",
        True,
    ),
    "local_copula": (
        _IMPUTER + "gaussian_copula_imputer.GaussianCopulaImputer",
        "Gaussian-copula conditional Monte Carlo prediction",
        "feature",
        True,
    ),
    "local_conditional": (
        _IMPUTER + "generative_conditional_imputer.GenerativeConditionalImputer",
        "tree-embedding conditional Monte Carlo prediction",
        "feature",
        True,
    ),
    "global_fidelity": (
        _LEGACY + "global_xai.base.GlobalExplanation",
        "MSE against full-model predictions, not observed labels",
        "feature",
        True,
    ),
    "feature_selection": (
        _LEGACY + "feature_selection.base.FeatureSelection",
        "held-out negative MSE after retraining on selected features; empty zero",
        "feature",
        False,
    ),
    "data_valuation": (
        _LEGACY + "data_valuation.base.DataValuation",
        "held-out negative MSE after training on selected rows; empty zero",
        "training example",
        False,
    ),
    "dataset_valuation": (
        _LEGACY + "dataset_valuation.base.DatasetValuation",
        "held-out negative MSE after training on selected row groups; empty zero",
        "training group",
        False,
    ),
    "ensemble": (
        _LEGACY + "ensemble_selection.base.EnsembleSelection",
        "negative MSE of average member predictions; empty zero",
        "model",
        False,
    ),
    "forest_ensemble": (
        _LEGACY + "ensemble_selection.base.RandomForestEnsembleSelection",
        "negative MSE of average tree predictions; empty zero",
        "tree",
        False,
    ),
    "uncertainty": (
        _LEGACY + "uncertainty.base.UncertaintyExplanation",
        "marginal mean of total predictive entropy in bits",
        "feature",
        False,
    ),
    "cluster": (
        _LEGACY + "unsupervised_cluster.base.ClusterExplanation",
        "Calinski-Harabasz score after KMeans on selected features",
        "feature",
        False,
    ),
    "unsupervised": (
        _LEGACY + "unsupervised_data.base.UnsupervisedData",
        "total correlation after native 20-bin discretization",
        "feature",
        False,
    ),
    "pathdependent_tree": (
        _LEGACY + "treeshapiq_xai.base.TreeSHAPIQXAI",
        "training-path-weighted tree prediction",
        "feature",
        False,
    ),
    "interventional_tree": (
        "shapiq.tree.interventional.game.InterventionalGame",
        "mean tree prediction over 16 fixed joint background rows",
        "feature",
        False,
    ),
    "product_kernel": (
        "shapiq.explainer.product_kernel.game.ProductKernelGame",
        "RBF support-vector decision function restricted to selected features",
        "feature",
        False,
    ),
    "knn": (
        _NN + "knn.KNNExplainerGame",
        "correct-label count among selected three nearest divided by three",
        "training example",
        False,
    ),
    "tnn": (
        _NN + "tnn.TNNExplainerGame",
        "correct-label proportion within fixed radius; empty neighborhood 1/classes",
        "training example",
        False,
    ),
    "weighted_knn": (
        _NN + "wknn.WeightedKNNExplainerGame",
        "average binary distance-weighted winning-vote utility",
        "training example",
        False,
    ),
    "binary_weighted_knn": (
        _NN + "wknn.BinaryWeightedKNNExplainerGame",
        "distance-weighted winning-vote utility against the smallest other class",
        "training example",
        False,
    ),
    "unanimity": (
        "shapiq_games.synthetic.soum.UnanimityGame",
        "unanimity indicator",
        "synthetic player",
        False,
    ),
    "soum": (
        "shapiq_games.synthetic.soum.SOUM",
        "seeded sum of unanimity games",
        "synthetic player",
        False,
    ),
    "dummy": (
        "shapiq_games.synthetic.dummy.DummyGame",
        "coalition size/8 plus fixed interaction",
        "synthetic player",
        False,
    ),
    "random": (
        "shapiq_games.synthetic.random_game.RandomGame",
        "canonical batch-position random realization",
        "synthetic player",
        True,
    ),
}
_APPLICATIONS = {
    **dict.fromkeys(
        [
            "local_baseline",
            "local_baseline_forest",
            "local_marginal",
            "local_gaussian",
            "local_copula",
            "local_conditional",
            "pathdependent_tree",
            "interventional_tree",
            "product_kernel",
        ],
        "local_explanation",
    ),
    **dict.fromkeys(
        ["knn", "tnn", "weighted_knn", "binary_weighted_knn", "dataset_valuation"],
        "data_valuation",
    ),
    **dict.fromkeys(["ensemble", "forest_ensemble"], "ensemble_selection"),
    **dict.fromkeys(["unanimity", "soum", "dummy", "random"], "synthetic"),
}
FAMILY_CATALOG: dict = {
    name: {
        "class": cls,
        "source": "src/" + cls.rsplit(".", 1)[0].replace(".", "/") + ".py",
        "semantics": semantics,
        "player_unit": unit,
        "stochastic_frozen": stochastic,
        "synthetic": unit == "synthetic player",
        "application_family": _APPLICATIONS.get(name, name),
        "dependencies": ["xgboost"] if name == "local_conditional" else [],
        "coverage": "one bounded representative; dataset wrappers not implied",
    }
    for name, (cls, semantics, unit, stochastic) in _RECIPES.items()
}


def dataset_compatibility(name: str, dataset: str) -> str | None:
    """Explain unsupported dataset/recipe pairs without constructing models."""
    if dataset not in DATASETS:
        return "Unknown dataset."
    if name not in FAMILY_CATALOG or FAMILY_CATALOG[name]["synthetic"]:
        return "Not a tabular recipe."
    classification = DATASETS[dataset]["task"] == "classification"
    if (
        name in ("uncertainty", "knn", "tnn", "weighted_knn", "binary_weighted_knn")
        and not classification
    ):
        return "Requires class labels."
    if name in ("local_gaussian", "local_copula") and dataset == "bike_sharing":
        return "Gaussian imputation rejects the binary calendar features."
    if name == "product_kernel" and classification and DATASETS[dataset]["n_classes"] != 2:
        return "The product-kernel classifier requires two classes."
    return None


def feature_subset(x: np.ndarray, count: int | None, seed: int) -> np.ndarray:
    """Select original columns by seed, without using targets or benchmark scores."""
    if count is None:
        return np.arange(x.shape[1])
    if type(count) is not int or not 1 <= count <= min(MAX_ENUMERATION_PLAYERS, x.shape[1]):
        message = "Feature players must be between one and the dataset width, at most twenty."
        raise ValueError(message)
    if count == x.shape[1]:
        return np.arange(count)
    return np.sort(np.random.default_rng(seed).choice(x.shape[1], count, replace=False))


def _negative_mse(y: np.ndarray, prediction: np.ndarray) -> float:
    return -float(mean_squared_error(y, prediction))


def make_family(
    name: str,
    *,
    instance_seed: int = 0,
    dataset: str | None = None,
    n_players: int | None = None,
    model_profile: str | None = None,
    model_cache: str | None = None,
    device: str = "cpu",
) -> tuple:
    """Construct a shipped game and JSON-compatible recipe/provenance metadata.

    Missing optional dependencies and broken constructors propagate to the
    preparation caller, which must preserve the failure as coverage information.
    Nothing here downloads models or substitutes a different payoff on failure.
    """
    if model_profile is not None:
        return _prediction_game(
            name, dataset, n_players, instance_seed, model_profile, model_cache, device
        )
    metadata = dict(FAMILY_CATALOG[name])
    configured = dataset is not None or n_players is not None
    module, attribute = metadata["class"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module), attribute)
    metadata.update(
        recipe=name, instance_seed=instance_seed, random_state=instance_seed, parameters={}
    )
    if n_players is not None and (
        type(n_players) is not int or not 1 <= n_players <= MAX_ENUMERATION_PLAYERS
    ):
        message = "Exhaustive recipe n_players must be an integer between one and twenty."
        raise ValueError(message)
    if metadata["synthetic"]:
        if dataset is not None:
            message = "Synthetic recipes do not accept a dataset."
            raise ValueError(message)
        n = n_players or 8
        interaction = (
            np.arange(min(3, n))
            if instance_seed == 0
            else np.sort(np.random.default_rng(instance_seed).choice(n, min(3, n), replace=False))
        )
        parameters = {
            "unanimity": {"interaction_binary": np.isin(np.arange(n), interaction).astype(int)},
            "soum": {"n": n, "n_basis_games": 12, "random_state": instance_seed},
            "dummy": {"n": n, "interaction": tuple(int(i) for i in interaction)},
            "random": {"n": n, "random_state": instance_seed},
        }[name]
        game = cls(**parameters)
        metadata.update(
            dataset="synthetic diagnostic",
            n_players=game.n_players,
            normalize=game.normalize,
            normalization_value=float(game.normalization_value),
            parameters={
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in parameters.items()
            },
        )
        return game, metadata

    neighbors = name in ("knn", "tnn", "weighted_knn", "binary_weighted_knn")
    dataset = dataset or ("iris" if neighbors or name == "uncertainty" else "california_housing")
    if message := dataset_compatibility(name, dataset):
        raise ValueError(message)
    classification = DATASETS[dataset]["task"] == "classification"
    x, y, train, test, feature_names = _dataset(dataset, instance_seed)
    data_hash = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
    features = feature_subset(
        x, n_players if metadata["player_unit"] == "feature" else None, instance_seed
    )
    if name in ("local_gaussian", "local_copula") and (
        dataset == "digits" or "categorical_features" in DATASETS[dataset]
    ):
        categorical = DATASETS[dataset].get("categorical_features", [])
        eligible = np.array(
            [
                i
                for i in range(x.shape[1])
                if categorical != "all"
                and feature_names[i] not in categorical
                and len(np.unique(x[train, i])) > 2
            ],
            dtype=int,
        )
        features = eligible[feature_subset(x[:, eligible], n_players, instance_seed)]
        metadata["parameters"]["feature_rule"] = (
            "seeded subset of noncategorical training columns with more than two unique values"
        )
    if name == "cluster" and (dataset == "digits" or "categorical_features" in DATASETS[dataset]):
        eligible = np.flatnonzero(np.ptp(x[train[:128]], axis=0) > 0)
        rule = "seeded subset of columns nonconstant on the clustering training rows"
        if "categorical_features" in DATASETS[dataset]:
            categorical = DATASETS[dataset]["categorical_features"]
            eligible = np.array(
                [
                    i
                    for i in eligible
                    if categorical != "all"
                    and feature_names[i] not in categorical
                    and len(np.unique(x[train[:128], i])) > 3
                ],
                dtype=int,
            )
            rule = "seeded subset of noncategorical clustering columns with more than three values"
        features = eligible[feature_subset(x[:, eligible], n_players, instance_seed)]
        metadata["parameters"]["feature_rule"] = rule
    x = x[:, features]
    x_train, y_train, x_test, y_test = (
        x[train].copy(),
        y[train].copy(),
        x[test].copy(),
        y[test].copy(),
    )
    metadata.update(
        dataset=dataset,
        **dataset_details(dataset),
        data_sha256=data_hash,
        feature_indices=features.tolist(),
        feature_names=[str(feature_names[i]) for i in features],
        train_indices=train.tolist(),
        test_indices=test.tolist(),
        point_row=int(test[0]),
        background_indices=train[:16].tolist(),
    )
    model_class = DecisionTreeClassifier if classification else DecisionTreeRegressor
    model = model_class(
        max_depth=4 if name == "interventional_tree" else 3,
        min_samples_leaf=5,
        random_state=instance_seed,
    ).fit(x_train, y_train)
    if name == "local_baseline_forest":
        forest_class = RandomForestClassifier if classification else RandomForestRegressor
        model = forest_class(n_estimators=8, max_depth=4, random_state=instance_seed, n_jobs=1).fit(
            x_train, y_train
        )
    if classification and (
        name.startswith("local_")
        or name in ("global_fidelity", "pathdependent_tree", "interventional_tree")
    ):
        metadata.update(class_index=1, output_scale="class probability")

    def predict_class_one(rows: np.ndarray) -> np.ndarray:
        return model.predict_proba(rows)[:, 1]

    metadata["model"] = type(model).__name__
    metadata["model_parameters"] = model.get_params()
    point, background = x_test[0], x_train[:16]
    if name.startswith("local_"):
        kwargs = {
            "model": predict_class_one if classification else model.predict,
            "data": background,
            "x": point,
            "random_state": instance_seed,
        }
        if name == "local_marginal":
            kwargs["sample_size"] = 16
        elif name in ("local_gaussian", "local_copula"):
            kwargs.update(data=x_train, sample_size=16)
            metadata["background_indices"] = train.tolist()
        elif name == "local_conditional":
            kwargs.update(data=x_train[:64], sample_size=8, conditional_budget=16)
            metadata["background_indices"] = train[:64].tolist()
        metadata["parameters"].update(
            {key: value for key, value in kwargs.items() if key not in ("model", "data", "x")}
        )
        game = cls(**kwargs)
    elif name == "global_fidelity":
        game = cls(
            data=x_train[:128],
            model=predict_class_one if classification else model.predict,
            loss_function=mean_squared_error,
            n_samples_eval=16,
            n_samples_empty=128,
            random_state=instance_seed,
        )
        metadata["parameters"] = {"n_samples_eval": 16, "n_samples_empty": 128}
        metadata["background_indices"] = train[:128].tolist()
    elif name == "feature_selection":
        game = cls(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            fit_function=model.fit,
            predict_function=model.predict,
            loss_function=accuracy_score if classification else _negative_mse,
        )
    elif name == "data_valuation":
        count = n_players or 8
        pool = np.concatenate((train[:count], test))
        game = cls(
            n_data_points=count,
            x_data=x[pool],
            y_data=y[pool],
            fit_function=model.fit,
            predict_function=model.predict,
            loss_function=accuracy_score if classification else _negative_mse,
            random_state=instance_seed,
        )
        permuted = pool[np.random.default_rng(instance_seed).permutation(len(pool))]
        metadata.update(
            train_indices=permuted[:count].tolist(), test_indices=permuted[count:].tolist()
        )
        metadata["parameters"] = {"n_data_points": count, "empty_data_value": 0}
    elif name == "dataset_valuation":
        count = n_players or 8
        groups = np.array_split(np.arange(len(train)), count)
        game = cls(
            x_train=[x_train[g] for g in groups],
            y_train=[y_train[g] for g in groups],
            x_test=x_test,
            y_test=y_test,
            fit_function=model.fit,
            predict_function=model.predict,
            loss_function=accuracy_score if classification else _negative_mse,
            random_state=instance_seed,
        )
        metadata["group_indices"] = [train[g].tolist() for g in groups]
        metadata["parameters"] = {"n_players": count, "empty_data_value": 0}
    elif name in ("ensemble", "forest_ensemble"):
        count = n_players or 8
        kwargs = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "dataset_type": "classification" if classification else "regression",
            "loss_function": accuracy_score if classification else _negative_mse,
            "verbose": False,
        }
        if name == "forest_ensemble":
            forest_class = RandomForestClassifier if classification else RandomForestRegressor
            forest = forest_class(
                n_estimators=count, max_depth=3, random_state=instance_seed, n_jobs=1
            ).fit(x_train, y_train)
            game = cls(random_forest=forest, **kwargs)
            metadata.update(model=type(forest).__name__, model_parameters=forest.get_params())
        else:
            if count < 4:
                message = "The heterogeneous ensemble requires at least four model players."
                raise ValueError(message)
            members = [
                LogisticRegression(max_iter=200, random_state=instance_seed)
                if classification
                else Ridge(alpha=1),
                SVC(random_state=instance_seed) if classification else SVR(),
                (KNeighborsClassifier if classification else KNeighborsRegressor)(
                    n_neighbors=3, n_jobs=1
                ),
                *[
                    model_class(max_depth=d, random_state=instance_seed + d)
                    for d in range(1, count - 2)
                ],
            ]
            for member in members:
                member.fit(x_train, y_train)
            game = cls(ensemble_members=members, **kwargs)
            metadata.update(
                model="heterogeneous ensemble",
                model_parameters={
                    str(i): {"class": type(m).__name__, "parameters": m.get_params()}
                    for i, m in enumerate(members)
                },
            )
    elif name == "uncertainty":
        forest = RandomForestClassifier(
            n_estimators=8, max_depth=3, random_state=instance_seed, n_jobs=1
        ).fit(x_train, y_train)
        game = cls(
            data=x_train[:20],
            model=forest,
            x=point,
            random_state=instance_seed,
            uncertainty_to_explain="total",
        )
        metadata.update(
            model=type(forest).__name__,
            model_parameters=forest.get_params(),
            background_indices=train[:20].tolist(),
        )
    elif name in ("cluster", "unsupervised"):
        scaler = StandardScaler().fit(x_train)
        data = scaler.transform(x_train[:128])
        kwargs = {"data": data}
        if name == "cluster":
            kwargs.update(
                cluster_method="kmeans",
                random_state=instance_seed,
                cluster_params={"n_clusters": 3, "n_init": 1, "max_iter": 50},
            )
        game = cls(**kwargs)
        metadata.update(
            model=None,
            model_parameters={},
            background_indices=train[:128].tolist(),
            preprocessing={
                "class": "StandardScaler",
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            },
            parameters={
                **metadata["parameters"],
                **{k: v for k, v in kwargs.items() if k != "data"},
            },
        )
    elif name == "pathdependent_tree":
        game = cls(
            x=point.astype(np.float32).astype(float) if classification else point,
            tree_model=model,
            verbose=False,
            **({"class_label": 1} if classification else {}),
        )
    elif name == "interventional_tree":
        game = cls(
            model=model,
            reference_data=background.astype(np.float32).astype(float)
            if classification
            else background,
            target_instance=point.astype(np.float32).astype(float) if classification else point,
            **({"class_index": 1} if classification else {}),
        )
    elif name == "product_kernel":
        scaler = StandardScaler().fit(x_train)
        svm_class = SVC if classification else SVR
        svm = svm_class(kernel="rbf", gamma="scale").fit(
            scaler.transform(x_train[:128]), y_train[:128]
        )
        game = cls(
            n_players=x.shape[1],
            explain_point=scaler.transform(x_test[:1])[0],
            model=convert_svm(svm),
        )
        metadata.update(
            model=type(svm).__name__,
            model_parameters=svm.get_params(),
            train_indices=train[:128].tolist(),
            preprocessing_fit_indices=train.tolist(),
            preprocessing={
                "class": "StandardScaler",
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            },
        )
        if classification:
            metadata.update(class_index=1, output_scale="binary SVC decision score")
    elif neighbors:
        scaler = StandardScaler().fit(x_train)
        if n_players is None:
            selected = np.concatenate(
                [np.flatnonzero(y_train == label)[:3] for label in (0, 1, 2)]
            )[:8]
        else:
            selected, _ = train_test_split(
                np.arange(len(train)),
                train_size=n_players,
                stratify=y_train,
                random_state=instance_seed,
            )
        point_label = y_test[0].item()
        radius = 2.0
        if name == "tnn" and configured:
            # Set geometry from training inputs alone, before inspecting any coalition scores.
            distances = pdist(scaler.transform(x_train))
            distances = distances[distances > 0]
            if not len(distances):
                message = "A data-scaled TNN radius requires distinct training inputs."
                raise ValueError(message)
            radius = float(np.median(distances))
        fitted = (
            RadiusNeighborsClassifier(radius=radius, n_jobs=1)
            if name == "tnn"
            else KNeighborsClassifier(
                n_neighbors=3,
                weights="distance" if "weighted" in name else "uniform",
                algorithm="brute",
                n_jobs=1,
            )
        )
        fitted.fit(scaler.transform(x_train[selected]), y_train[selected])
        matches = np.flatnonzero(fitted.classes_ == point_label)
        if not len(matches):
            message = "Selected neighbor training rows omit the held-out observation's class."
            raise ValueError(message)
        class_index = int(matches[0])
        kwargs = {"model": fitted, "x": scaler.transform(x_test[:1])[0], "class_index": class_index}
        if name == "binary_weighted_knn":
            kwargs["class_index_other"] = next(
                i for i in range(len(fitted.classes_)) if i != class_index
            )
        game = cls(**kwargs)
        metadata.update(
            model=type(fitted).__name__,
            model_parameters=fitted.get_params(),
            train_indices=train[selected].tolist(),
            class_index=class_index,
            point_label=point_label,
            class_labels=fitted.classes_.tolist(),
            preprocessing_fit_indices=train.tolist(),
            parameters={"class_index_other": kwargs.get("class_index_other"), "n_bits": None},
            preprocessing={
                "class": "StandardScaler",
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            },
        )
        if name == "tnn" and configured:
            metadata["parameters"].update(
                radius=radius,
                radius_rule="median nonzero pairwise Euclidean distance on standardized training rows",
            )
    else:
        message = f"No recipe for {name}"
        raise ValueError(message)
    if classification and name in (
        "feature_selection",
        "data_valuation",
        "dataset_valuation",
        "ensemble",
        "forest_ensemble",
    ):
        metadata["semantics"] = (
            "held-out accuracy of selected model majority votes; empty zero"
            if name in ("ensemble", "forest_ensemble")
            else "held-out accuracy after retraining on selected players; empty zero"
        )
        metadata["output_scale"] = "accuracy"
    metadata["n_players"] = game.n_players
    metadata["normalize"] = game.normalize
    metadata["normalization_value"] = float(game.normalization_value)
    return game, metadata


_LOCAL = {"local_baseline", "local_marginal"}
_RETRAINING = {"feature_selection", "data_valuation", "dataset_valuation"}
_PREDICTION = _LOCAL | {"local_gaussian", "local_copula", "local_conditional", "global_fidelity"}
_TREE = {"pathdependent_tree", "interventional_tree"}
PROFILE_CONSTRUCTIONS = {
    "random_forest": _PREDICTION | _RETRAINING | _TREE | {"uncertainty", "forest_ensemble"},
    "xgboost": _PREDICTION | _RETRAINING | _TREE,
    "lightgbm": _PREDICTION | _RETRAINING | _TREE,
    "mlp": _PREDICTION | _RETRAINING,
    "linear": _LOCAL | _RETRAINING,
    "rbf_svm": _LOCAL | {"product_kernel"},
    "gaussian_process": _LOCAL,
    "tabpfn_prediction": _LOCAL,
    "heterogeneous_ensemble": {"ensemble"},
    "heterogeneous_ensemble_extended": {"ensemble"},
}


def profile_compatibility(name: str, dataset: str, profile: str) -> str | None:
    """The explicit model-to-construction mapping, before expensive preparation."""
    if reason := dataset_compatibility(name, dataset):
        return reason
    if name not in PROFILE_CONSTRUCTIONS.get(profile, set()):
        return f"The {profile} model profile does not support {name}."
    return None


class _CoalitionRefit:
    """Fresh frozen-hyperparameter fits with explicit subset-class semantics.

    The shipped games retain their empty-coalition utility. A nonempty one-class
    training coalition predicts that class; other classification coalitions use
    contiguous internal labels and map predictions back to the original labels.
    """

    def __init__(self, prepared: PreparedModel) -> None:
        self.template = coalition_model(prepared)
        self.classification = prepared.metadata["task"] == "classification"
        self.profile = prepared.metadata["model_profile"]

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.model = clone(self.template)
        self.labels, encoded = np.unique(y, return_inverse=True)
        self.constant = (self.classification and len(self.labels) == 1) or (
            not self.classification and self.profile == "lightgbm" and len(y) == 1
        )
        if self.constant:
            return
        if self.classification and self.profile == "xgboost":
            binary = len(self.labels) == 2
            self.model.set_params(
                objective="binary:logistic" if binary else "multi:softprob",
                eval_metric="logloss" if binary else "mlogloss",
                num_class=None if binary else len(self.labels),
            )
        self.model.fit(x, encoded if self.classification else y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.constant:
            return np.full(len(x), self.labels[0])
        values = self.model.predict(x)
        return self.labels[np.asarray(values, dtype=int)] if self.classification else values


def _frozen_parameters(prepared: PreparedModel, model: object) -> dict:
    """Record declared settings plus frozen validation choices, without estimator objects."""
    actual = cast("Any", model).get_params()
    return {
        key: actual.get(f"model__{key}", actual.get(key, value))
        for key, value in prepared.metadata["model_parameters"].items()
    }


def _retraining_game(
    constructor: Callable[..., Any], name: str, prepared: PreparedModel, n: int, seed: int
) -> tuple:
    """Preserve shipped utilities and keep training-player rows out of the holdout."""
    refit = _CoalitionRefit(prepared)
    metadata = {}
    options = {
        "fit_function": refit.fit,
        "predict_function": refit.predict,
        "loss_function": accuracy_score if refit.classification else _negative_mse,
    }
    if name == "data_valuation":
        if n > len(prepared.x_train):
            message = "Not enough fitting rows for the declared row players."
            raise ValueError(message)
        x = np.concatenate((prepared.x_train[:n], prepared.x_test))
        y = np.concatenate((prepared.y_train[:n], prepared.y_test))
        # DataValuation permutes its inputs. Invert that exact seeded permutation
        # so its training/test pools remain the already-declared disjoint pools.
        inverse = np.argsort(np.random.default_rng(seed).permutation(len(x)))
        game = constructor(
            n_data_points=n, x_data=x[inverse], y_data=y[inverse], random_state=seed, **options
        )
        metadata["train_indices"] = prepared.metadata["train_indices"][:n]
        metadata["training_rows"] = n
    else:
        x, y = prepared.x_train, prepared.y_train
        if name == "dataset_valuation":
            if n > len(x):
                message = "Not enough fitting rows for nonempty group players."
                raise ValueError(message)
            groups = np.array_split(np.arange(len(x)), n)
            metadata["group_indices"] = [
                np.asarray(prepared.metadata["train_indices"])[group].tolist() for group in groups
            ]
            x, y = [x[group] for group in groups], [y[group] for group in groups]
            options["random_state"] = seed
        game = constructor(
            x_train=x, y_train=y, x_test=prepared.x_test, y_test=prepared.y_test, **options
        )
    metadata.update(
        semantics="held-out accuracy after retraining; empty zero"
        if refit.classification
        else "held-out negative MSE after retraining; empty zero",
        output_scale="accuracy" if refit.classification else "negative MSE",
        refit_protocol={
            "hyperparameters": "frozen before enumeration; no coalition-specific tuning",
            "early_stopping": "disabled; fitted iteration count frozen",
            "single_class": "constant prediction of the training class",
            "singleton_regression": "LightGBM predicts its sole training target; other profiles fit normally",
            "quality_scope": "validation/test scores describe the full-data parameter-selection model",
            "preprocessing": "missing-value repair frozen on the parameter-selection fitting pool; scaler refitted per coalition",
            "subset_classes": "contiguous internal labels mapped back on prediction",
            "parameters": _frozen_parameters(prepared, refit.template),
        },
    )
    return game, metadata


def _ensemble_game(
    constructor: Callable[..., Any],
    name: str,
    prepared: PreparedModel,
    n: int,
    seed: int,
    dataset: str,
    profile: str,
    cache_dir: str | None,
) -> tuple:
    """Ensemble players use explicit members with the same fitted/test split."""
    classification = prepared.metadata["task"] == "classification"
    options = {
        "x_train": prepared.x_train,
        "y_train": prepared.y_train,
        "x_test": prepared.x_test,
        "y_test": prepared.y_test,
        "loss_function": accuracy_score if classification else _negative_mse,
        "dataset_type": "classification" if classification else "regression",
        "verbose": False,
        "random_state": seed,
    }
    if name == "forest_ensemble":
        forest = clone(prepared.model).set_params(n_estimators=n)
        forest.fit(prepared.x_train, prepared.y_train)
        game = constructor(random_forest=forest, **options)
        members = [{"profile": "random_forest_tree", "tree": i} for i in range(n)]
    else:
        profiles = ["random_forest", "xgboost", "rbf_svm", "linear"]
        if profile == "heterogeneous_ensemble_extended":
            profiles += ["lightgbm", "mlp"]
        fitted, members = [], []
        for i in range(n):
            family = profiles[i % len(profiles)]
            base = prepare_model(
                dataset, prepared.x_train.shape[1], seed, family, cache_dir=cache_dir
            )
            if base.metadata["train_indices"] != prepared.metadata["train_indices"]:
                message = "Ensemble member preparation changed the shared training rows."
                raise ValueError(message)
            member = cast("Any", coalition_model(base))
            key = "model__random_state" if hasattr(member, "named_steps") else "random_state"
            if key in member.get_params():
                member.set_params(**{key: seed + i})
            member.fit(prepared.x_train, prepared.y_train)
            fitted.append(member)
            members.append(
                {
                    "profile": family,
                    "random_state": seed + i,
                    "model_key": base.metadata["model_key"],
                    "parameters": _frozen_parameters(base, member),
                }
            )
        game = constructor(ensemble_members=fitted, **options)
    return game, {
        "ensemble_members": members,
        "quality_scope": "validation/test scores describe the shared preparation model, not ensemble votes",
        "semantics": "held-out majority-vote accuracy; empty zero"
        if classification
        else "held-out negative MSE of mean member prediction; empty zero",
        "output_scale": "accuracy" if classification else "negative MSE",
        "model_parameters": {"n_members": n, "member_definitions": members},
    }


def _prediction_game(
    name: str,
    dataset: str | None,
    n_players: int | None,
    seed: int,
    profile: str,
    cache_dir: str | None,
    device: str = "cpu",
) -> tuple:
    """Construct a shipped game from shared, qualified model/data ingredients."""
    if dataset is None or type(n_players) is not int or not 1 <= n_players <= 20:
        message = "Explicit model profiles require a dataset and bounded n_players (1-20)."
        raise ValueError(message)
    if reason := profile_compatibility(name, dataset, profile):
        raise ValueError(reason)
    feature_count = (
        n_players
        if FAMILY_CATALOG[name]["player_unit"] == "feature"
        else min(12, int(DATASETS[dataset]["n_features"]))
    )
    feature_rule = "continuous" if name in ("local_gaussian", "local_copula") else "all"
    prepared = prepare_model(
        dataset,
        feature_count,
        seed,
        "random_forest" if profile.startswith("heterogeneous_ensemble") else profile,
        cache_dir=cache_dir,
        device=device,
        feature_rule=feature_rule,
    )
    metadata = {**FAMILY_CATALOG[name], **prepared.metadata}
    module, attribute = metadata["class"].rsplit(".", 1)
    constructor = getattr(importlib.import_module(module), attribute)
    model, point, background = prepared.model, prepared.x_test[0], prepared.x_train[:16]
    classification = metadata["task"] == "classification"
    options, extra = {}, {}
    if name.startswith("local_"):
        if name == "local_marginal":
            options["sample_size"] = 16
        elif name in ("local_gaussian", "local_copula"):
            background = prepared.x_train
            options["sample_size"] = 16
        elif name == "local_conditional":
            background = prepared.x_train[:64]
            options.update(sample_size=8, conditional_budget=16)
        game = constructor(
            model=prepared.predict, data=background, x=point, random_state=seed, **options
        )
    elif name == "global_fidelity":
        background = prepared.x_train[:128]
        options = {"n_samples_eval": 16, "n_samples_empty": len(background)}
        game = constructor(
            data=background,
            model=prepared.predict,
            loss_function=mean_squared_error,
            random_state=seed,
            **options,
        )
        extra["output_scale"] = "prediction fidelity (MSE relative to empty-coalition MSE)"
    elif name in _RETRAINING:
        game, extra = _retraining_game(constructor, name, prepared, n_players, seed)
    elif name in ("ensemble", "forest_ensemble"):
        game, extra = _ensemble_game(
            constructor, name, prepared, n_players, seed, dataset, profile, cache_dir
        )
    elif name == "uncertainty":
        background = prepared.x_train[:20]
        options["uncertainty_to_explain"] = "total"
        game = constructor(data=background, model=model, x=point, random_state=seed, **options)
        extra["output_scale"] = "total predictive entropy (bits)"
    elif name in _TREE:
        # Both shipped XGBoost tree paths consume every stored tree. Trim the
        # stopped tail so this is the same best-round predictor as local games.
        if profile == "xgboost":
            model = copy.deepcopy(model)
            rounds = model.best_iteration + 1
            model._Booster = model.get_booster()[:rounds]  # noqa: SLF001
            extra["tree_rounds"] = rounds
        point = point.astype(np.float32).astype(float)
        background = background.astype(np.float32).astype(float)
        if name == "pathdependent_tree":
            game = constructor(
                x=point,
                tree_model=model,
                verbose=False,
                **({"class_label": 1} if classification else {}),
            )
        else:
            game = constructor(
                model=model,
                reference_data=background,
                target_instance=point,
                **({"class_index": 1} if classification else {}),
            )
        extra["output_scale"] = (
            "class margin"
            if classification and profile != "random_forest"
            else "class probability"
            if classification
            else "prediction"
        )
    elif name == "product_kernel":
        svm = model.named_steps["model"]
        scaled_point = model.named_steps["scaler"].transform(point[None])[0]
        game = constructor(n_players=n_players, explain_point=scaled_point, model=convert_svm(svm))
        extra["output_scale"] = "binary SVC decision score" if classification else "prediction"
    else:
        message = f"No profiled construction for {name}."
        raise ValueError(message)
    metadata.update(
        recipe=name,
        model=type(model).__name__,
        model_profile=profile,
        preparation_hardware={"device": device, "cpu_model": hardware()["cpu_model"]},
        parameters={"random_state": seed, **options},
        background_size=len(background),
        background_indices=metadata["train_indices"][: len(background)],
        point_row=metadata["test_indices"][0],
        n_players=game.n_players,
        output_scale="class probability" if classification else "prediction",
        normalize=game.normalize,
        normalization_value=float(game.normalization_value),
    )
    metadata.update(extra)
    if name in _RETRAINING | {
        "ensemble",
        "forest_ensemble",
        "product_kernel",
        "pathdependent_tree",
    }:
        metadata.update(background_indices=[], background_size=0)
    if name in _RETRAINING | {"ensemble", "forest_ensemble", "global_fidelity"}:
        metadata.pop("point_row", None)
    if "output_scale" in extra:
        metadata["output"] = extra["output_scale"]
    return game, metadata
