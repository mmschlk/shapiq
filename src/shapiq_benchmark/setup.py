"""Helpers to load datasets and models from string identifiers."""

from __future__ import annotations
from typing import Literal, TypeAlias, get_args, Protocol, cast
from collections.abc import Callable

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from tabpfn import TabPFNClassifier, TabPFNRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

from shapiq_games.benchmark.setup import GameBenchmarkSetup

from .bench_types import BenchmarkDataset

AllSupportedDatasets: TypeAlias = Literal[
    "communities_and_crime",
    "california_housing",
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

AllSupportedModels: TypeAlias = Literal[
    "decision_tree",
    "random_forest",
    "tabpfn",
    "xgboost",
    "lightgbm",
    "vit_16_patches",
    "resnet_18",
]

SupportedModelsInterventional: TypeAlias = Literal[
    "decision_tree", "random_forest", "xgboost", "lightgbm"
]
SupportedModelsPathdependent: TypeAlias = Literal[
    "decision_tree", "random_forest", "xgboost", "lightgbm"
]
SupportedModelsLocalXAI: TypeAlias = Literal[
    "decision_tree", "random_forest", "xgboost", "lightgbm"
]
SupportedModelsImage: TypeAlias = Literal[
    "vit_16_patches",
    "resnet_18",
]
SupportedModelsTabPFN: TypeAlias = Literal["tabpfn"]


class _FitModel(Protocol):
    def fit(self, x: object, y: object) -> object: ...


ModelBuilder = Callable[[int | None, int], _FitModel]


_MODEL_BUILDERS: dict[tuple[str, str], ModelBuilder] = {
    ("decision_tree", "classification"): lambda random_state, _n_estimators: (
        DecisionTreeClassifier(random_state=random_state)
    ),
    ("decision_tree", "regression"): lambda random_state, _n_estimators: (
        DecisionTreeRegressor(random_state=random_state)
    ),
    ("random_forest", "classification"): lambda random_state, n_estimators: (
        RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    ),
    ("random_forest", "regression"): lambda random_state, n_estimators: (
        RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    ),
    ("xgboost", "classification"): lambda random_state, n_estimators: (
        XGBClassifier(n_estimators=n_estimators, random_state=random_state)
    ),
    ("xgboost", "regression"): lambda random_state, n_estimators: (
        XGBRegressor(n_estimators=n_estimators, random_state=random_state)
    ),
    ("lightgbm", "classification"): lambda random_state, n_estimators: (
        LGBMClassifier(n_estimators=n_estimators, random_state=random_state)
    ),
    ("lightgbm", "regression"): lambda random_state, n_estimators: (
        LGBMRegressor(n_estimators=n_estimators, random_state=random_state)
    ),
    ("tabpfn", "classification"): lambda random_state, _n_estimators: (
        TabPFNClassifier(random_state=random_state)
    ),
    ("tabpfn", "regression"): lambda random_state, _n_estimators: (
        TabPFNRegressor(random_state=random_state)
    ),
}


def register_model_builder(
    model_name: str, data_type: str, builder: ModelBuilder
) -> None:
    """Register a new model builder for future extensions."""
    key = (model_name.lower(), data_type.lower())
    if key in _MODEL_BUILDERS:
        msg = f"Model builder already registered for {key[0]} ({key[1]})."
        raise ValueError(msg)
    _MODEL_BUILDERS[key] = builder


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
        data_type=cast(Literal["classification", "regression"], setup.dataset_type),
    )


def load_model_from_str(
    model_str: str,
    dataset: BenchmarkDataset,
    *,
    random_state: int | None = 42,
    n_estimators: int = 10,  # TODO this somewhere else
) -> object:
    """Create an unfitted model from a string identifier.

    Args:
            model_str: Model identifier (e.g. "decision_tree").
            dataset: Dataset metadata used to choose classifier vs regressor.
            random_state: Random state for the model.
            n_estimators: Number of estimators for random forest models.

    Returns:
            A fitted model instance.
    """
    key = (model_str.lower(), dataset.data_type.lower())
    if key not in _MODEL_BUILDERS:
        supported = sorted({name for name, _dtype in _MODEL_BUILDERS})
        msg = (
            f"Unsupported model '{model_str}' for data type '{dataset.data_type}'. "
            f"Supported models: {', '.join(supported)}"
        )
        raise ValueError(msg)

    model = _MODEL_BUILDERS[key](random_state, n_estimators)
    return model.fit(dataset.x_train, dataset.y_train)


def load_from_str(
    data_str: str,
    model_str: str,
    benchmark_type: str,
    *,
    random_state: int | None = 42,
) -> tuple[BenchmarkDataset, object]:
    """Convenience function to load both dataset and model from string identifiers."""
    allowed_data = get_args(AllSupportedDatasets)
    if data_str not in allowed_data:
        msg = (
            f"Unsupported dataset '{data_str}' for {benchmark_type}. "
            f"Supported datasets: {', '.join(allowed_data)}"
        )
        raise ValueError(msg)

    if benchmark_type == "interventional":
        allowed_models = get_args(SupportedModelsInterventional)
    elif benchmark_type == "pathdependent":
        allowed_models = get_args(SupportedModelsPathdependent)
    elif benchmark_type == "local_xai":
        allowed_models = get_args(SupportedModelsLocalXAI)
    elif benchmark_type == "tabpfn":
        allowed_models = get_args(SupportedModelsTabPFN)
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
    model = load_model_from_str(model_str, dataset, random_state=random_state)
    return dataset, model


def infer_data_type(model: object) -> Literal["classification", "regression"]:
    """Infer whether a model is a classifier or regressor based on its attributes."""
    if hasattr(model, "_estimator_type"):
        if model._estimator_type == "classifier":  # type: ignore[attr-defined]
            return "classification"
        if model._estimator_type == "regressor":  # type: ignore[attr-defined]
            return "regression"
    if hasattr(model, "predict_proba") or hasattr(model, "classes_"):
        return "classification"
    return "regression"
