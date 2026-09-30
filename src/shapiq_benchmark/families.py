"""Small, explicit recipes exercising shipped games on native real-data features.

These are family representatives, not coverage of every dataset wrapper. Sampled
payoffs marked ``stochastic_frozen`` require a canonical, persisted coalition
table; exact truth then describes that realization, not a population expectation.
"""

from __future__ import annotations

import hashlib
import importlib
from functools import lru_cache

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shapiq.datasets import load_bike_sharing, load_california_housing
from shapiq.explainer.product_kernel.conversion import convert_svm

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
FAMILY_CATALOG = {
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

DATASETS = {
    "california_housing": {
        "task": "regression",
        "n_features": 8,
        "source": "shapiq.datasets.load_california_housing",
    },
    "diabetes": {
        "task": "regression",
        "n_features": 10,
        "source": "sklearn.datasets.load_diabetes",
    },
    "bike_sharing": {
        "task": "regression",
        "n_features": 12,
        "source": "shapiq.datasets.load_bike_sharing",
    },
    "iris": {
        "task": "classification",
        "n_features": 4,
        "n_classes": 3,
        "source": "sklearn.datasets.load_iris",
    },
    "wine": {
        "task": "classification",
        "n_features": 13,
        "n_classes": 3,
        "source": "sklearn.datasets.load_wine",
    },
    "breast_cancer": {
        "task": "classification",
        "n_features": 30,
        "n_classes": 2,
        "source": "sklearn.datasets.load_breast_cancer",
    },
    "digits": {
        "task": "classification",
        "n_features": 64,
        "n_classes": 10,
        "source": "sklearn.datasets.load_digits",
    },
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


@lru_cache(maxsize=20)
def _dataset(name: str, instance_seed: int = 0) -> tuple:
    """Load a bundled dataset and retain explicit original row identities."""
    loaders = {
        "iris": load_iris,
        "wine": load_wine,
        "diabetes": load_diabetes,
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
    }
    if name in loaders:
        data = loaders[name]()
        x, y = data.data, data.target
        feature_names = list(data.feature_names)
    else:
        loaders = {"california_housing": load_california_housing, "bike_sharing": load_bike_sharing}
        if name not in loaders:
            message = f"Unknown benchmark dataset: {name}"
            raise ValueError(message)
        x, y = loaders[name]()
        feature_names = list(x.columns)
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    train, test = train_test_split(
        np.arange(len(x)),
        test_size=0.2,
        random_state=instance_seed,
        stratify=y if DATASETS[name]["task"] == "classification" else None,
    )
    return x, y, train[:512], test[:128], feature_names


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
    name: str, *, instance_seed: int = 0, dataset: str | None = None, n_players: int | None = None
) -> tuple:
    """Construct a shipped game and JSON-compatible recipe/provenance metadata.

    Missing optional dependencies and broken constructors propagate to the
    preparation caller, which must preserve the failure as coverage information.
    Nothing here downloads models or substitutes a different payoff on failure.
    """
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
    if name in ("local_gaussian", "local_copula") and dataset == "digits":
        eligible = np.array([i for i in range(x.shape[1]) if len(np.unique(x[train, i])) > 2])
        features = eligible[feature_subset(x[:, eligible], n_players, instance_seed)]
        metadata["parameters"]["feature_rule"] = (
            "seeded subset of training columns with more than two unique values"
        )
    if name == "cluster" and dataset == "digits":
        eligible = np.flatnonzero(np.ptp(x[train[:128]], axis=0) > 0)
        features = eligible[feature_subset(x[:, eligible], n_players, instance_seed)]
        metadata["parameters"]["feature_rule"] = (
            "seeded subset of columns nonconstant on the clustering training rows"
        )
    x = x[:, features]
    x_train, y_train, x_test, y_test = (
        x[train].copy(),
        y[train].copy(),
        x[test].copy(),
        y[test].copy(),
    )
    metadata.update(
        dataset=dataset,
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
        class_index = int(y_test[0])
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
        kwargs = {"model": fitted, "x": scaler.transform(x_test[:1])[0], "class_index": class_index}
        if name == "binary_weighted_knn":
            kwargs["class_index_other"] = next(
                int(label) for label in fitted.classes_ if label != class_index
            )
        game = cls(**kwargs)
        metadata.update(
            model=type(fitted).__name__,
            model_parameters=fitted.get_params(),
            train_indices=train[selected].tolist(),
            class_index=class_index,
            point_label=int(y_test[0]),
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
