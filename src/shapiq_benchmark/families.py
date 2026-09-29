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
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor, RadiusNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from shapiq.datasets import load_california_housing
from shapiq.explainer.product_kernel.conversion import convert_svm

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


@lru_cache(maxsize=2)
def _dataset(name: str) -> tuple:
    """Load a bundled dataset and retain explicit original row identities."""
    if name == "iris":
        data = load_iris()
        x, y = data.data, data.target
    else:
        x, y = load_california_housing()
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    train, test = train_test_split(
        np.arange(len(x)),
        test_size=0.2,
        random_state=0,
        stratify=y if name == "iris" else None,
    )
    return x, y, train[:512], test[:128]


def _negative_mse(y: np.ndarray, prediction: np.ndarray) -> float:
    return -float(mean_squared_error(y, prediction))


def make_family(name: str) -> tuple:
    """Construct a shipped game and JSON-compatible recipe/provenance metadata.

    Missing optional dependencies and broken constructors propagate to the
    preparation caller, which must preserve the failure as coverage information.
    Nothing here downloads models or substitutes a different payoff on failure.
    """
    metadata = dict(FAMILY_CATALOG[name])
    module, attribute = metadata["class"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module), attribute)
    metadata.update(recipe=name, random_state=0, parameters={})
    if metadata["synthetic"]:
        parameters = {
            "unanimity": {"interaction_binary": np.array([1, 1, 1, 0, 0, 0, 0, 0])},
            "soum": {"n": 8, "n_basis_games": 12, "random_state": 0},
            "dummy": {"n": 8, "interaction": (0, 1, 2)},
            "random": {"n": 8, "random_state": 0},
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
    dataset = "iris" if neighbors or name == "uncertainty" else "california_housing"
    x, y, train, test = _dataset(dataset)
    x_train, y_train, x_test, y_test = (
        x[train].copy(),
        y[train].copy(),
        x[test].copy(),
        y[test].copy(),
    )
    metadata.update(
        dataset=dataset,
        data_sha256=hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
        train_indices=train.tolist(),
        test_indices=test.tolist(),
        point_row=int(test[0]),
        background_indices=train[:16].tolist(),
    )
    model = DecisionTreeRegressor(
        max_depth=4 if name == "interventional_tree" else 3, min_samples_leaf=5, random_state=0
    ).fit(x_train, y_train)
    if name == "local_baseline_forest":
        model = RandomForestRegressor(n_estimators=8, max_depth=4, random_state=0, n_jobs=1).fit(
            x_train, y_train
        )
    metadata["model"] = type(model).__name__
    metadata["model_parameters"] = model.get_params()
    point, background = x_test[0], x_train[:16]
    if name.startswith("local_"):
        kwargs = {"model": model.predict, "data": background, "x": point, "random_state": 0}
        if name == "local_marginal":
            kwargs["sample_size"] = 16
        elif name in ("local_gaussian", "local_copula"):
            kwargs.update(data=x_train, sample_size=16)
            metadata["background_indices"] = train.tolist()
        elif name == "local_conditional":
            kwargs.update(data=x_train[:64], sample_size=8, conditional_budget=16)
            metadata["background_indices"] = train[:64].tolist()
        metadata["parameters"] = {
            key: value for key, value in kwargs.items() if key not in ("model", "data", "x")
        }
        game = cls(**kwargs)
    elif name == "global_fidelity":
        game = cls(
            data=x_train[:128],
            model=model.predict,
            loss_function=mean_squared_error,
            n_samples_eval=16,
            n_samples_empty=128,
            random_state=0,
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
            loss_function=_negative_mse,
        )
    elif name == "data_valuation":
        pool = np.concatenate((train[:8], test))
        game = cls(
            n_data_points=8,
            x_data=x[pool],
            y_data=y[pool],
            fit_function=model.fit,
            predict_function=model.predict,
            loss_function=_negative_mse,
            random_state=0,
        )
        permuted = pool[np.random.default_rng(0).permutation(len(pool))]
        metadata.update(train_indices=permuted[:8].tolist(), test_indices=permuted[8:].tolist())
        metadata["parameters"] = {"n_data_points": 8, "empty_data_value": 0}
    elif name == "dataset_valuation":
        groups = np.array_split(np.arange(len(train)), 8)
        game = cls(
            x_train=[x_train[g] for g in groups],
            y_train=[y_train[g] for g in groups],
            x_test=x_test,
            y_test=y_test,
            fit_function=model.fit,
            predict_function=model.predict,
            loss_function=_negative_mse,
            random_state=0,
        )
        metadata["group_indices"] = [train[g].tolist() for g in groups]
        metadata["parameters"] = {"n_players": 8, "empty_data_value": 0}
    elif name in ("ensemble", "forest_ensemble"):
        kwargs = {
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test,
            "dataset_type": "regression",
            "loss_function": _negative_mse,
            "verbose": False,
        }
        if name == "forest_ensemble":
            forest = RandomForestRegressor(
                n_estimators=8, max_depth=3, random_state=0, n_jobs=1
            ).fit(x_train, y_train)
            game = cls(random_forest=forest, **kwargs)
            metadata.update(model=type(forest).__name__, model_parameters=forest.get_params())
        else:
            members = [
                Ridge(alpha=1),
                SVR(),
                KNeighborsRegressor(n_neighbors=3, n_jobs=1),
                *[DecisionTreeRegressor(max_depth=d, random_state=d) for d in (1, 2, 3, 4, 5)],
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
        forest = RandomForestClassifier(n_estimators=8, max_depth=3, random_state=0, n_jobs=1).fit(
            x_train, y_train
        )
        game = cls(
            data=x_train[:20], model=forest, x=point, random_state=0, uncertainty_to_explain="total"
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
                random_state=0,
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
            parameters={k: v for k, v in kwargs.items() if k != "data"},
        )
    elif name == "pathdependent_tree":
        game = cls(x=point, tree_model=model, verbose=False)
    elif name == "interventional_tree":
        game = cls(model=model, reference_data=background, target_instance=point)
    elif name == "product_kernel":
        scaler = StandardScaler().fit(x_train)
        svm = SVR(kernel="rbf", gamma="scale").fit(scaler.transform(x_train[:128]), y_train[:128])
        game = cls(
            n_players=8, explain_point=scaler.transform(x_test[:1])[0], model=convert_svm(svm)
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
    elif neighbors:
        scaler = StandardScaler().fit(x_train)
        selected = np.concatenate([np.flatnonzero(y_train == label)[:3] for label in (0, 1, 2)])[:8]
        class_index = int(y_test[0])
        fitted = (
            RadiusNeighborsClassifier(radius=2.0, n_jobs=1)
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
            kwargs["class_index_other"] = next(label for label in range(3) if label != class_index)
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
    else:
        message = f"No recipe for {name}"
        raise ValueError(message)
    metadata["n_players"] = game.n_players
    metadata["normalize"] = game.normalize
    metadata["normalization_value"] = float(game.normalization_value)
    return game, metadata
