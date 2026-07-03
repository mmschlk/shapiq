"""Helpers to load datasets and models from string identifiers."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol, cast, get_args

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shapiq_games.benchmark.setup import GameBenchmarkSetup

from ._optional import require
from .bench_types import BenchmarkDataset

type AllSupportedDatasets = Literal[
    "communities_and_crime",
    "california_housing",
    "adult_census",
    "tabarena_airfoil_self_noise",
    "tabarena_amazon_employee_access",
    "tabarena_anneal",
    "tabarena_fiat_500",
    "tabarena_aps_failure",
    "tabarena_bank_marketing",
    "tabarena_bank_customer_churn",
    "tabarena_bioresponse",
    "tabarena_blood_transfusion",
    "tabarena_churn",
    "tabarena_coil2000",
    "tabarena_concrete_strength",
    "tabarena_credit_g",
    "tabarena_credit_card_default",
    "tabarena_airline_satisfaction",
    "tabarena_diabetes",
    "tabarena_diabetes130us",
    "tabarena_diamonds",
    "tabarena_ecommerce_shipping",
    "tabarena_fitness_club",
    "tabarena_food_delivery",
    "tabarena_give_me_credit",
    "tabarena_hazelnut",
    "tabarena_health_insurance",
    "tabarena_heloc",
    "tabarena_hiva_agnostic",
    "tabarena_houses",
    "tabarena_hr_analytics",
    "tabarena_coupon_recommendation",
    "tabarena_good_customer",
    "tabarena_kddcup09",
    "tabarena_marketing_campaign",
    "tabarena_maternal_health",
    "tabarena_miami_housing",
    "tabarena_online_shoppers",
    "tabarena_protein",
    "tabarena_bankruptcy",
    "tabarena_qsar_biodeg",
    "tabarena_qsar_tid11",
    "tabarena_qsar_fish_toxicity",
    "tabarena_sdss17",
    "tabarena_seismic_bumps",
    "tabarena_splice",
    "tabarena_students_dropout",
    "tabarena_superconductivity",
    "tabarena_taiwanese_bankruptcy",
    "tabarena_website_phishing",
    "tabarena_wine_quality",
    "tabarena_naticusdroid",
    "tabarena_jm1",
    "tabarena_mic",
]

type AllSupportedModels = Literal[
    "decision_tree",
    "random_forest",
    "tabpfn",
    "xgboost",
    "lightgbm",
    "mlp",
    "vit_16_patches",
    "resnet_18",
]

type SupportedModelsInterventional = Literal[
    "decision_tree", "random_forest", "xgboost", "lightgbm"
]
type SupportedModelsPathdependent = Literal["decision_tree", "random_forest", "xgboost", "lightgbm"]
type SupportedModelsLocalXAI = Literal[
    "decision_tree", "random_forest", "xgboost", "lightgbm", "mlp"
]
type SupportedModelsImage = Literal["vit_16_patches", "resnet_18"]
type SupportedModelsTabPFN = Literal["tabpfn"]


class _FitModel(Protocol):
    def fit(self, x: object, y: object) -> object: ...


ModelBuilder = Callable[..., _FitModel]


def _get_literal_args(type_alias: object) -> tuple[str, ...]:
    value = getattr(type_alias, "__value__", type_alias)
    return cast("tuple[str, ...]", get_args(value))


# scikit-learn models are always available (core shapiq dependency).
_SKLEARN_MODEL_BUILDERS: dict[tuple[str, str], ModelBuilder] = {
    ("decision_tree", "classification"): DecisionTreeClassifier,
    ("decision_tree", "regression"): DecisionTreeRegressor,
    ("random_forest", "classification"): RandomForestClassifier,
    ("random_forest", "regression"): RandomForestRegressor,
    ("mlp", "classification"): MLPClassifier,
    ("mlp", "regression"): MLPRegressor,
}

# Optional backends, resolved lazily via ``require`` so that importing this
# module does not require the ``benchmark`` extras. Maps the model name to the
# (import package, classifier attribute, regressor attribute) to look up.
_OPTIONAL_MODEL_BACKENDS: dict[str, tuple[str, str, str]] = {
    "xgboost": ("xgboost", "XGBClassifier", "XGBRegressor"),
    "lightgbm": ("lightgbm", "LGBMClassifier", "LGBMRegressor"),
    "tabpfn": ("tabpfn", "TabPFNClassifier", "TabPFNRegressor"),
}

# All supported (model, task) pairs, independent of whether the optional backend
# is installed. Used for validation and error messages so the advertised set of
# supported models does not silently shrink when an extra is missing.
_SUPPORTED_MODEL_TASKS: frozenset[tuple[str, str]] = frozenset(_SKLEARN_MODEL_BUILDERS) | {
    (name, task) for name in _OPTIONAL_MODEL_BACKENDS for task in ("classification", "regression")
}


def _resolve_model_builder(model_str: str, task: str) -> ModelBuilder:
    """Return the model builder for a ``(model, task)`` pair.

    scikit-learn models are returned directly. The optional backends
    (``xgboost`` / ``lightgbm`` / ``tabpfn``) are imported lazily and raise a
    helpful error pointing to the ``benchmark`` extra if they are not installed.
    """
    key = (model_str, task)
    if key in _SKLEARN_MODEL_BUILDERS:
        return _SKLEARN_MODEL_BUILDERS[key]
    package, classifier_attr, regressor_attr = _OPTIONAL_MODEL_BACKENDS[model_str]
    module = require(package)
    attr = classifier_attr if task == "classification" else regressor_attr
    return cast("ModelBuilder", getattr(module, attr))


def load_data_from_str(
    data_str: str,
    *,
    random_state: int | None = 42,
    test_size: float = 0.2,
) -> BenchmarkDataset:
    """Load a dataset from a string identifier.

    Args:
            data_str: Dataset identifier (e.g. "adult_census").
            random_state: Random state used for dataset shuffling and split.
            test_size: Fraction of data used for testing.

    Returns:
            A BenchmarkDataset containing train/test splits and metadata.
    """
    setup = GameBenchmarkSetup(
        dataset_name=data_str,
        model_name=None,
        verbose=False,
        random_state=random_state,
        test_size=test_size,
    )

    return BenchmarkDataset(
        x_train=setup.x_train,
        y_train=setup.y_train,
        x_test=setup.x_test,
        y_test=setup.y_test,
        data_type=cast("Literal['classification', 'regression']", setup.dataset_type),
    )


def load_and_fit_model_from_str(
    model_str: str,
    dataset: BenchmarkDataset,
    **kwargs: object,
) -> object:
    """Create a fitted model from a string identifier.

    Args:
            model_str: Model identifier (e.g. "decision_tree").
            dataset: Dataset metadata used to choose classifier vs regressor.
            random_state: Random state for the model.
            n_estimators: Number of estimators for random forest models.
            **kwargs: Additional keyword arguments passed to the model constructor.

    Returns:
            A fitted model instance.
    """
    key = (model_str.lower(), dataset.data_type.lower())
    if key not in _SUPPORTED_MODEL_TASKS:
        supported = sorted({name for name, _dtype in _SUPPORTED_MODEL_TASKS})
        msg = (
            f"Unsupported model '{model_str}' for data type '{dataset.data_type}'. "
            f"Supported models: {', '.join(supported)}"
        )
        raise ValueError(msg)

    builder = _resolve_model_builder(*key)
    model = builder(**kwargs)
    return model.fit(dataset.x_train, dataset.y_train)


def load_from_str(
    data_str: str,
    model_str: str,
    benchmark_type: str,
    random_state: int | None = 42,
    **kwargs: object,
) -> tuple[BenchmarkDataset, object]:
    """Convenience function to load both dataset and model from string identifiers."""
    allowed_data = _get_literal_args(AllSupportedDatasets)
    if data_str not in allowed_data:
        msg = (
            f"Unsupported dataset '{data_str}' for {benchmark_type}. "
            f"Supported datasets: {', '.join(allowed_data)}"
        )
        raise ValueError(msg)

    if benchmark_type == "interventional":
        allowed_models = _get_literal_args(SupportedModelsInterventional)
    elif benchmark_type == "pathdependent":
        allowed_models = _get_literal_args(SupportedModelsPathdependent)
    elif benchmark_type == "local_xai":
        allowed_models = _get_literal_args(SupportedModelsLocalXAI)
    elif benchmark_type == "tabpfn":
        allowed_models = _get_literal_args(SupportedModelsTabPFN)
    else:
        msg = f"Unsupported benchmark type '{benchmark_type}'."
        raise ValueError(msg)
    if model_str not in allowed_models:
        msg = (
            f"Unsupported model '{model_str}' for {benchmark_type}. "
            f"Supported models: {', '.join(allowed_models)}"
        )
        raise ValueError(msg)

    dataset = load_data_from_str(data_str, random_state=random_state)

    best_params = check_for_known_combination(data_str, model_str) or {}
    model_params = {**best_params, **kwargs}
    model = load_and_fit_model_from_str(model_str, dataset, **model_params)
    return dataset, model


def infer_data_type(model: object) -> Literal["classification", "regression"]:
    """Infer whether a sklearn model is a classifier or regressor based on its attributes."""
    estimator_type = getattr(model, "_estimator_type", None)
    if estimator_type == "classifier":
        return "classification"
    if estimator_type == "regressor":
        return "regression"
    if hasattr(model, "predict_proba") or hasattr(model, "classes_"):
        return "classification"
    return "regression"


def check_for_known_combination(
    data_str: str,
    model_str: str,
) -> dict[str, object] | None:
    """Return best parameters for a known dataset/model pair if available."""
    results_dir = Path(__file__).resolve().parent / "optimization" / "results"
    if not results_dir.exists():
        return None

    for result_file in results_dir.glob("*.json"):
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if (
            payload.get("dataset") == data_str
            and payload.get("model") == model_str
            and isinstance(payload.get("best_params"), dict)
        ):
            return payload["best_params"]

    return None
