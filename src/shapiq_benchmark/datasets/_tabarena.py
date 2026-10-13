"""Loaders of the 51 datasets of TabArena-v0.1 (OpenML study 457).

The task type and target column of every dataset are taken from TabArena's official metadata
(``tabarena_dataset_metadata.csv`` in the TabArena dataset curation repository) instead of being
inferred from the labels. Loading requires the ``openml`` package and network access on first
use; the preprocessed data is then cached locally, under a name that changes with the OpenML id,
the target, and :data:`_PREPROCESSING_VERSION`.

Missing values are imputed with the median or mode of all rows, before any train-test split. The
test rows thus shape the imputed training values a little; this does not matter for exact game
values, but it does for model accuracy on datasets with many missing values.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pandas as pd

from shapiq_games._optional import require

from ._cache import get_data_dir, read_table, write_table
from ._preprocess import encode_categorical, impute
from ._registry import DatasetSpec, register_dataset

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq_games.typing import Task

__all__ = ["TABARENA_DATASETS"]

# name -> (OpenML dataset id, task, target column), from TabArena's official metadata.
TABARENA_DATASETS: dict[str, tuple[int, Task, str]] = {
    "airfoil_self_noise": (46904, "regression", "scaled-sound-pressure"),
    "amazon_employee_access": (46905, "classification", "ResourceApproved"),
    "anneal": (46906, "classification", "classes"),
    "fiat_500": (46907, "regression", "price"),
    "aps_failure": (46908, "classification", "AirPressureSystemFailure"),
    "bank_marketing": (46910, "classification", "SubscribeTermDeposit"),
    "bank_customer_churn": (46911, "classification", "churn"),
    "bioresponse": (46912, "classification", "MoleculeElicitsResponse"),
    "blood_transfusion": (46913, "classification", "DonatedBloodInMarch2007"),
    "churn": (46915, "classification", "CustomerChurned"),
    "coil2000": (46916, "classification", "MobileHomePolicy"),
    "concrete_strength": (46917, "regression", "ConcreteCompressiveStrength"),
    "credit_g": (46918, "classification", "good_or_bad_customer"),
    "credit_card_default": (46919, "classification", "DefaultOnPaymentNextMonth"),
    "airline_satisfaction": (46920, "classification", "satisfaction"),
    "diabetes": (46921, "classification", "TestedPositiveForDiabetes"),
    "diabetes130us": (46922, "classification", "EarlyReadmission"),
    "diamonds": (46923, "regression", "price"),
    "ecommerce_shipping": (46924, "classification", "ArrivedLate"),
    "fitness_club": (46927, "classification", "attended"),
    "food_delivery": (46928, "regression", "Time_taken(min)"),
    "give_me_credit": (46929, "classification", "FinancialDistressNextTwoYears"),
    "hazelnut": (46930, "classification", "Contaminated"),
    "health_insurance": (46931, "regression", "charges"),
    "heloc": (46932, "classification", "RiskPerformance"),
    "hiva_agnostic": (46933, "classification", "CompoundActivity"),
    "houses": (46934, "regression", "LnMedianHouseValue"),
    "hr_analytics": (46935, "classification", "LookingForJobChange"),
    "coupon_recommendation": (46937, "classification", "AcceptCoupon"),
    "good_customer": (46938, "classification", "bad_client_target"),
    "kddcup09": (46939, "classification", "appetency"),
    "marketing_campaign": (46940, "classification", "Response"),
    "maternal_health": (46941, "classification", "RiskLevel"),
    "miami_housing": (46942, "regression", "SALE_PRC"),
    "online_shoppers": (46947, "classification", "Revenue"),
    "protein": (46949, "regression", "ResidualSize"),
    "bankruptcy": (46950, "classification", "company_bankrupt"),
    "qsar_biodeg": (46952, "classification", "Biodegradable"),
    "qsar_tid11": (46953, "regression", "MEDIAN_PXC50"),
    "qsar_fish_toxicity": (46954, "regression", "LC50"),
    "sdss17": (46955, "classification", "ObjectType"),
    "seismic_bumps": (46956, "classification", "HighEnergySeismicBump"),
    "splice": (46958, "classification", "SiteType"),
    "students_dropout": (46960, "classification", "AcademicOutcome"),
    "superconductivity": (46961, "regression", "critical_temp"),
    "taiwanese_bankruptcy": (46962, "classification", "Bankrupt"),
    "website_phishing": (46963, "classification", "WebsiteType"),
    "wine_quality": (46964, "regression", "median_wine_quality"),
    "naticusdroid": (46969, "classification", "Malware"),
    "jm1": (46979, "classification", "defects"),
    "mic": (46980, "classification", "LET_IS"),
}

_TARGET_COLUMN = "__target__"

_PREPROCESSING_VERSION = 1
"""Bump when :func:`_load_tabarena` changes the data, so that old caches are not used."""


def _load_tabarena(name: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load a TabArena dataset from the local cache or OpenML.

    Missing values are imputed (median for numeric, mode for other columns), then categorical
    features are ordinal-encoded. The target is kept as is; :func:`load_dataset` encodes
    classification labels.
    """
    openml_id, _task, target = TABARENA_DATASETS[name]
    recipe = f"{openml_id}/{target}/{_PREPROCESSING_VERSION}".encode()
    path = get_data_dir() / "tabarena" / f"{name}-{hashlib.sha256(recipe).hexdigest()[:8]}.csv"
    if not path.exists():
        openml = require("openml", purpose="the TabArena datasets", extra="benchmark")
        dataset = openml.datasets.get_dataset(openml_id, download_data=True)
        x, y, _, _ = dataset.get_data(target=target, dataset_format="dataframe")
        frame = encode_categorical(impute(x))  # impute categories before encoding them
        frame[_TARGET_COLUMN] = y.astype(str) if not pd.api.types.is_numeric_dtype(y) else y
        write_table(path, frame)
    data = read_table(path)
    y = data.pop(_TARGET_COLUMN).rename("target")
    return data, y


def _make_loader(name: str) -> Callable[[], tuple[pd.DataFrame, pd.Series]]:
    def _loader() -> tuple[pd.DataFrame, pd.Series]:
        return _load_tabarena(name)

    _loader.__name__ = f"load_tabarena_{name}"
    return _loader


for _name, (_openml_id, _task, _target) in TABARENA_DATASETS.items():
    register_dataset(
        DatasetSpec(
            name=f"tabarena_{_name}",
            task=_task,
            loader=_make_loader(_name),
            source=f"TabArena-v0.1, OpenML dataset {_openml_id}",
            kind="tabarena",
        )
    )
