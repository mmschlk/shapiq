"""Loaders of the classic real-world tabular datasets.

Every dataset comes from its original source, and the preprocessing is unchanged from the
loaders that used to ship with the data:

- OpenML (``openml``): adult, amazon, bike sharing, bioresponse, leukemia, micro-mass;
- the UCI repository (``ucimlrepo``): annealing, arrhythmia, hepatitis, ionosphere, mushroom,
  nursery, soybean, thyroid, zoo;
- scikit-learn's ``fetch_california_housing``;
- the data folder of shap (checksum-pinned): NHANES I, communities and crime;
- UCI files read directly (not yet pinned by checksum): wine quality, real estate, forest fires.

The raw table of an upstream dataset is downloaded once and cached as a CSV in
``<data dir>/tabular/``. A table whose shape differs from the one the loaders were written for
is rejected, so a changed upstream fails loudly instead of silently changing a benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer as _sklearn_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from shapiq_games._optional import require

from ._cache import RemoteFile, atomic_write_bytes, fetch, get_data_dir
from ._registry import DatasetSpec, register_dataset

if TYPE_CHECKING:
    from ._registry import Task


@dataclass(frozen=True)
class _Upstream:
    """The original source of a raw table and the shape (rows, columns with target) it has."""

    source: Literal["openml", "uci", "sklearn"]
    dataset_id: int | None
    shape: tuple[int, int]


_UPSTREAM: dict[str, _Upstream] = {
    "adult_census": _Upstream("openml", 1590, (48842, 15)),
    "amazon": _Upstream("openml", 1457, (1500, 10001)),
    "bike_sharing": _Upstream("openml", 42713, (17379, 13)),
    "bioresponse": _Upstream("openml", 4134, (3751, 1777)),
    "leukemia": _Upstream("openml", 45090, (72, 7130)),
    "microresponse": _Upstream("openml", 1515, (571, 1301)),
    "annealing": _Upstream("uci", 3, (898, 39)),
    "arrhythmia": _Upstream("uci", 5, (452, 280)),
    "hepatitis": _Upstream("uci", 46, (155, 20)),
    "ionosphere": _Upstream("uci", 52, (351, 35)),
    "mushroom": _Upstream("uci", 73, (8124, 23)),
    "nursery": _Upstream("uci", 76, (12960, 9)),
    "soybean": _Upstream("uci", 90, (683, 36)),
    "thyroid": _Upstream("uci", 102, (7200, 22)),
    "zoo": _Upstream("uci", 111, (101, 17)),
    "california_housing": _Upstream("sklearn", None, (20640, 9)),
}

_SHAP_DATA_URL = "https://raw.githubusercontent.com/shap/shap/master/data/"
_SHAP_FILES: dict[str, RemoteFile] = {
    name: RemoteFile(_SHAP_DATA_URL + name, name, sha256, subdir="shap")
    for name, sha256 in {
        "CommViolPredUnnormalizedData.txt": (
            "b090072d0e8d140a936d2411704ed29b2681e4805b805a9c8c6bdb072c5416c3"
        ),
        "NHANESI_X.csv": "0fa1e524900008da254c365b501c96768e55b94858fdc4045886d887654a1ba8",
        "NHANESI_y.csv": "70c8d6ab022e7b080752b339ee8edc991f2fd203c294b5ca55b36bcb902aab93",
    }.items()
}

_UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
_UCI_FILES: dict[str, RemoteFile] = {
    "winequality-red.csv": RemoteFile(
        _UCI_URL + "wine-quality/winequality-red.csv", "winequality-red.csv", None
    ),
    "winequality-white.csv": RemoteFile(
        _UCI_URL + "wine-quality/winequality-white.csv", "winequality-white.csv", None
    ),
    "real_estate.xlsx": RemoteFile(
        _UCI_URL + "00477/Real%20estate%20valuation%20data%20set.xlsx", "real_estate.xlsx", None
    ),
    "forestfires.csv": RemoteFile(
        _UCI_URL + "forest-fires/forestfires.csv", "forestfires.csv", None
    ),
}


def _download_table(name: str) -> pd.DataFrame:
    """Download the raw table of ``name`` (features and target) from its original source."""
    upstream = _UPSTREAM[name]
    if upstream.source == "openml":
        openml = require("openml", purpose=f"the {name} dataset")
        dataset = openml.datasets.get_dataset(upstream.dataset_id, download_data=True)
        table, *_ = dataset.get_data(dataset_format="dataframe")
    elif upstream.source == "uci":
        ucimlrepo = require("ucimlrepo", purpose=f"the {name} dataset")
        data = ucimlrepo.fetch_ucirepo(id=upstream.dataset_id).data
        table = data.features.copy()
        table["target"] = data.targets.squeeze()
    else:
        from sklearn.datasets import fetch_california_housing

        home = get_data_dir() / "scikit_learn"
        table = fetch_california_housing(data_home=str(home), as_frame=True).frame
    if table.shape != upstream.shape:
        msg = (
            f"The {name} table from {upstream.source} {upstream.dataset_id or ''} has shape "
            f"{table.shape}, but the loader expects {upstream.shape}. The upstream data changed."
        )
        raise ValueError(msg)
    return table


def _read_table(name: str) -> pd.DataFrame:
    """Return the raw table of ``name``, downloading and caching it as a CSV on first use."""
    path = get_data_dir() / "tabular" / f"{name}.csv"
    if not path.exists():
        buffer = StringIO()
        _download_table(name).to_csv(buffer, index=False, float_format="%.17g")  # lossless
        atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))
    return pd.read_csv(path, low_memory=False, float_precision="round_trip")


def _read_shap_file(filename: str, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(fetch(_SHAP_FILES[filename]), **kwargs)


def _impute(x: pd.DataFrame) -> pd.DataFrame:
    """Impute numeric columns with their median and other columns with their mode."""
    for column in x.columns:
        if x[column].isna().any():
            if pd.api.types.is_numeric_dtype(x[column]):
                x[column] = x[column].fillna(x[column].median())
            else:
                x[column] = x[column].fillna(x[column].mode()[0])
    return x


def _encode_categorical(x: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode all object and category columns (unknown categories become ``-1``)."""
    categorical = x.select_dtypes(include=["object", "category"]).columns
    if len(categorical) == 0:
        return x
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    x = x.copy()
    x[categorical] = encoder.fit_transform(x[categorical])
    return x


def _encode_target(y: pd.Series) -> pd.Series:
    return pd.Series(LabelEncoder().fit_transform(y.astype(str)), name="target")


def load_california_housing() -> tuple[pd.DataFrame, pd.Series]:
    """California housing (sklearn ``fetch_california_housing``), regression."""
    dataset = _read_table("california_housing")
    y = dataset.pop("MedHouseVal")
    return dataset, y


def load_bike_sharing() -> tuple[pd.DataFrame, pd.Series]:
    """Bike sharing (OpenML 42713), regression."""
    dataset = _read_table("bike_sharing")
    num_features = [
        "hour",
        "temp",
        "feel_temp",
        "humidity",
        "windspeed",
        "year",
        "month",
        "holiday",
        "weekday",
        "workingday",
    ]
    cat_features = ["season", "weather"]
    dataset[num_features] = dataset[num_features].apply(pd.to_numeric)
    transformer = ColumnTransformer(
        [
            ("numerical", Pipeline([("scaler", RobustScaler())]), num_features),
            ("categorical", Pipeline([("ordinal_encoder", OrdinalEncoder())]), cat_features),
        ],
        remainder="passthrough",
    )
    columns = num_features + cat_features
    columns += [feature for feature in dataset.columns if feature not in columns]
    transformed = cast("np.ndarray", transformer.fit_transform(dataset))
    dataset = pd.DataFrame(transformed, columns=np.asarray(columns)).dropna()
    y = dataset.pop("count").astype(float)
    return dataset.astype(float), y


def load_adult_census() -> tuple[pd.DataFrame, pd.Series]:
    """Adult census income (OpenML 1590), binary classification of income above 50K."""
    dataset = _read_table("adult_census")
    num_features = ["age", "capital-gain", "capital-loss", "hours-per-week", "fnlwgt"]
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "education-num",
    ]
    dataset[num_features] = dataset[num_features].apply(pd.to_numeric)
    transformer = ColumnTransformer(
        [
            (
                "numerical",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
                ),
                num_features,
            ),
            ("categorical", Pipeline([("ordinal_encoder", OrdinalEncoder())]), cat_features),
        ],
        remainder="passthrough",
    )
    columns = num_features + cat_features
    columns += [feature for feature in dataset.columns if feature not in columns]
    transformed = cast("np.ndarray", transformer.fit_transform(dataset))
    dataset = pd.DataFrame(transformed, columns=np.asarray(columns)).dropna()
    y = dataset.pop("class").apply(lambda label: 1 if label == ">50K" else 0)
    return dataset.astype(float), y


def load_breast_cancer() -> tuple[pd.DataFrame, pd.Series]:
    """Breast cancer Wisconsin (bundled with scikit-learn), binary classification."""
    x, y = _sklearn_breast_cancer(return_X_y=True, as_frame=True)
    return x, y


def load_wine_quality() -> tuple[pd.DataFrame, pd.Series]:
    """Wine quality, red and white (UCI), regression of the quality score."""
    red = pd.read_csv(fetch(_UCI_FILES["winequality-red.csv"]), sep=";")
    red["type"] = "red"
    white = pd.read_csv(fetch(_UCI_FILES["winequality-white.csv"]), sep=";")
    white["type"] = "white"
    data = pd.concat([red, white], ignore_index=True)
    y = data.pop("quality").astype(float)
    x = pd.get_dummies(data, columns=["type"], drop_first=True)
    return x.astype(float), y


def load_real_estate() -> tuple[pd.DataFrame, pd.Series]:
    """Real estate valuation (UCI), regression. Requires ``openpyxl``."""
    require("openpyxl", purpose="reading the real_estate dataset")
    data = pd.read_excel(fetch(_UCI_FILES["real_estate.xlsx"]))
    data = data.drop(columns=["No"])
    data["month"] = (data["X1 transaction date"] % 1 * 12).round().astype(int)
    data["month"] = data["month"].replace({0: 1, 12: 1})
    data = data.drop(columns=["X1 transaction date"])
    data = pd.get_dummies(data, columns=["month"], drop_first=True)
    y = data.pop("Y house price of unit area").astype(float)
    return data.astype(float), y


def load_forest_fires() -> tuple[pd.DataFrame, pd.Series]:
    """Forest fires (UCI), regression of the burned area."""
    data = pd.read_csv(fetch(_UCI_FILES["forestfires.csv"]))
    y = data.pop("area").astype(float)
    data = data.drop(columns=["day"])
    seasons = {
        "dec": "winter",
        "jan": "winter",
        "feb": "winter",
        "mar": "spring",
        "apr": "spring",
        "may": "spring",
        "jun": "summer",
        "jul": "summer",
        "aug": "summer",
        "sep": "fall",
        "oct": "fall",
        "nov": "fall",
    }
    data["season"] = data.pop("month").map(seasons)
    x = pd.get_dummies(data, columns=["season"], drop_first=True)
    return x.astype(float), y


def load_nhanesi() -> tuple[pd.DataFrame, pd.Series]:
    """NHANES I survival data (as in ``shap.datasets.nhanesi``), regression."""
    x = _read_shap_file("NHANESI_X.csv", index_col=0)
    y = _read_shap_file("NHANESI_y.csv", index_col=0).squeeze()
    return x, pd.Series(np.asarray(y, dtype=float), name="target")


def load_communities_and_crime() -> tuple[pd.DataFrame, pd.Series]:
    """Communities and crime, unnormalized (as in ``shap.datasets``), regression."""
    raw = _read_shap_file("CommViolPredUnnormalizedData.txt", na_values="?")
    valid_rows = np.where(np.invert(np.isnan(raw.iloc[:, -2])))[0]
    y = pd.Series(np.array(raw.iloc[valid_rows, -2], dtype=float), name="target")
    x = raw.iloc[valid_rows, 5:-18]
    valid_columns = np.where(np.isnan(x.to_numpy()).sum(0) == 0)[0]
    return x.iloc[:, valid_columns], y


def _load_with_class_column(name: str, target: str) -> tuple[pd.DataFrame, pd.Series]:
    data = _read_table(name)
    y = data.pop(target)
    # encode the raw labels (not their string form) so numeric labels keep their numeric order
    return data, pd.Series(LabelEncoder().fit_transform(y), name="target")


def load_amazon() -> tuple[pd.DataFrame, pd.Series]:
    """Amazon commerce reviews (OpenML 1457), classification."""
    return _load_with_class_column("amazon", "Class")


def load_microresponse() -> tuple[pd.DataFrame, pd.Series]:
    """Micro-mass response (OpenML 1515), classification."""
    return _load_with_class_column("microresponse", "Class")


def load_bioresponse() -> tuple[pd.DataFrame, pd.Series]:
    """Bioresponse (OpenML 4134), binary classification."""
    data = _read_table("bioresponse")
    y = data.pop("target").rename("target")
    return data, y


def load_leukemia() -> tuple[pd.DataFrame, pd.Series]:
    """Leukemia gene expression (OpenML 45090), binary classification."""
    return _load_with_class_column("leukemia", "CLASS")


def _load_uci_classification(
    name: str,
    *,
    impute: bool = False,
    encode: bool = False,
    drop_constant: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    data = _read_table(name)
    y = data.pop("target")
    if impute:
        data = _impute(data)
    if encode:
        data = _encode_categorical(data)
    if drop_constant:
        data = data.loc[:, ~(data.iloc[0] == data).all()]
    return data, _encode_target(y)


def load_annealing() -> tuple[pd.DataFrame, pd.Series]:
    """Annealing (UCI 3), multiclass classification."""
    return _load_uci_classification("annealing", impute=True, encode=True)


def load_arrhythmia() -> tuple[pd.DataFrame, pd.Series]:
    """Arrhythmia (UCI 5), multiclass classification."""
    return _load_uci_classification("arrhythmia", impute=True, encode=True)


def load_hepatitis() -> tuple[pd.DataFrame, pd.Series]:
    """Hepatitis (UCI 46), binary classification."""
    return _load_uci_classification("hepatitis", impute=True, encode=True)


def load_ionosphere() -> tuple[pd.DataFrame, pd.Series]:
    """Ionosphere (UCI 52), binary classification."""
    return _load_uci_classification("ionosphere", drop_constant=True)


def load_mushroom() -> tuple[pd.DataFrame, pd.Series]:
    """Mushroom (UCI 73), binary classification."""
    return _load_uci_classification("mushroom", encode=True)


def load_nursery() -> tuple[pd.DataFrame, pd.Series]:
    """Nursery (UCI 76), multiclass classification."""
    return _load_uci_classification("nursery", encode=True)


def load_soybean() -> tuple[pd.DataFrame, pd.Series]:
    """Soybean, large (UCI 90), multiclass classification."""
    return _load_uci_classification("soybean", impute=True, encode=True)


def load_thyroid() -> tuple[pd.DataFrame, pd.Series]:
    """Thyroid disease (UCI 102), multiclass classification."""
    return _load_uci_classification("thyroid")


def load_zoo() -> tuple[pd.DataFrame, pd.Series]:
    """Zoo (UCI 111), multiclass classification."""
    return _load_uci_classification("zoo")


_TABULAR: list[tuple[str, Task, object, str]] = [
    ("adult_census", "classification", load_adult_census, "OpenML 1590"),
    ("amazon", "classification", load_amazon, "OpenML 1457"),
    ("annealing", "classification", load_annealing, "UCI 3 (ucimlrepo)"),
    ("arrhythmia", "classification", load_arrhythmia, "UCI 5 (ucimlrepo)"),
    ("bike_sharing", "regression", load_bike_sharing, "OpenML 42713"),
    ("bioresponse", "classification", load_bioresponse, "OpenML 4134"),
    ("breast_cancer", "classification", load_breast_cancer, "scikit-learn (bundled)"),
    ("california_housing", "regression", load_california_housing, "scikit-learn fetcher"),
    (
        "communities_and_crime",
        "regression",
        load_communities_and_crime,
        "UCI 211 via shap's data (checksum-pinned)",
    ),
    ("forest_fires", "regression", load_forest_fires, "UCI 162 (direct, not yet pinned)"),
    ("hepatitis", "classification", load_hepatitis, "UCI 46 (ucimlrepo)"),
    ("ionosphere", "classification", load_ionosphere, "UCI 52 (ucimlrepo)"),
    ("leukemia", "classification", load_leukemia, "OpenML 45090"),
    ("microresponse", "classification", load_microresponse, "OpenML 1515"),
    ("mushroom", "classification", load_mushroom, "UCI 73 (ucimlrepo)"),
    ("nhanesi", "regression", load_nhanesi, "NHANES I via shap's data (checksum-pinned)"),
    ("nursery", "classification", load_nursery, "UCI 76 (ucimlrepo)"),
    ("real_estate", "regression", load_real_estate, "UCI 477 (direct, not yet pinned)"),
    ("soybean", "classification", load_soybean, "UCI 90 (ucimlrepo)"),
    ("thyroid", "classification", load_thyroid, "UCI 102 (ucimlrepo)"),
    ("wine_quality", "regression", load_wine_quality, "UCI 186 (direct, not yet pinned)"),
    ("zoo", "classification", load_zoo, "UCI 111 (ucimlrepo)"),
]

for _name, _task, _loader, _source in _TABULAR:
    register_dataset(
        DatasetSpec(name=_name, task=_task, loader=_loader, source=_source, kind="tabular")  # type: ignore[arg-type]
    )
