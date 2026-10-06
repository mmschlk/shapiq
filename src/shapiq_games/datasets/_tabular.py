"""Loaders of the classic real-world tabular datasets.

The preprocessing is unchanged from the loaders that used to ship with the data. Files that were
previously bundled are downloaded from a pinned commit of the shapiq repository and verified by
their SHA-256 checksum. Three datasets come directly from the UCI repository and are not yet
pinned by checksum (``wine_quality``, ``real_estate``, ``forest_fires``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer as _sklearn_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from shapiq_games._optional import require

from ._cache import RemoteFile, fetch
from ._registry import DatasetSpec, register_dataset

if TYPE_CHECKING:
    from ._registry import Task

_PINNED_FILES: dict[str, str] = {
    "CommViolPredUnnormalizedData.txt": "b090072d0e8d140a936d2411704ed29b2681e4805b805a9c8c6bdb072c5416c3",
    "NHANESI_X.csv": "0fa1e524900008da254c365b501c96768e55b94858fdc4045886d887654a1ba8",
    "NHANESI_y.csv": "70c8d6ab022e7b080752b339ee8edc991f2fd203c294b5ca55b36bcb902aab93",
    "adult_census.csv": "547551a449e2453ec1b9a3044b0b74e59cf79a4c99291cd478914aa5f9efbcd0",
    "amazon.csv": "b0efede20e5942ecc8221bce5c9740d32a0f2620138c52444ef8998b17f21a11",
    "annealing.csv": "f8378ebf0ba5aca8e48615c86950d44232b4df5361d2e349cc066475509a56a1",
    "arrhythmia.csv": "345b3ee5e8b28ab76718cd4ad3027bf615f2510d79a3dddb78013053d5d2f3e5",
    "bike.csv": "28878d17e5eb141b6bcaad2f5f8fa5daf90a7ced8e60cabcce0d0e0ef0cf1d96",
    "bioresponse.csv": "63adfae758c2dffabc2a814a7dcb2c7b01000ed20826068a8663976a34b31abd",
    "california_housing.csv": "6c0573ec35c926d6fab1225b6ff5d3ee1c664ebf960168f3fd13d70b6acdc485",
    "hepatitis.csv": "95162c8fdec436cdf218bcabda8a45b9e5025c38df7097597552d190f03526c9",
    "ionosphere.csv": "66cbf3aace34dbbf88f694058644cc45a5dcd62a37531d8da4a7bd85fe524ea6",
    "leukemia.csv": "dde8de2c329bba681145ac212e6461498990a199b7f9aec41428ef3b4d62a2d2",
    "microresponse.csv": "b1dd855a05b0da4a82bbb19a2050756aac76ffdd7dbc422243b4047b572457d4",
    "mushroom.csv": "f837295c46e7dce628dcd52a769aac92b2fc9091b977051bbdd37e74bb7dcac2",
    "nursery.csv": "dbca688ed77126723c333a7eb247213020cd82d49fecc86178f0a89c24140ba7",
    "soybean.csv": "4673e9d0857cad357eef75f9900d5a7ec0f155265de5bdf09dcb0ee04e6e29b5",
    "thyroid.csv": "ced808e8a89f9e3cc75be6f38661bacb4c1d41d96ef4bb98f498ccb024158316",
    "zoo.csv": "197c9494b53e371886e75389cd896a2a8b890df77ead780b4b4b0e251d31cd7a",
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


def _read_pinned_csv(filename: str, **kwargs: Any) -> pd.DataFrame:
    remote = RemoteFile.pinned(f"datasets/data/{filename}", _PINNED_FILES[filename])
    return pd.read_csv(fetch(remote), **kwargs)


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
    dataset = _read_pinned_csv("california_housing.csv")
    y = dataset.pop("MedHouseVal")
    return dataset, y


def load_bike_sharing() -> tuple[pd.DataFrame, pd.Series]:
    """Bike sharing (OpenML 42713), regression."""
    dataset = _read_pinned_csv("bike.csv")
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
    """Adult census income (UCI), binary classification of income above 50K."""
    dataset = _read_pinned_csv("adult_census.csv")
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
    x = _read_pinned_csv("NHANESI_X.csv", index_col=0)
    y = _read_pinned_csv("NHANESI_y.csv", index_col=0).squeeze()
    return x, pd.Series(np.asarray(y, dtype=float), name="target")


def load_communities_and_crime() -> tuple[pd.DataFrame, pd.Series]:
    """Communities and crime, unnormalized (as in ``shap.datasets``), regression."""
    raw = _read_pinned_csv("CommViolPredUnnormalizedData.txt", na_values="?")
    valid_rows = np.where(np.invert(np.isnan(raw.iloc[:, -2])))[0]
    y = pd.Series(np.array(raw.iloc[valid_rows, -2], dtype=float), name="target")
    x = raw.iloc[valid_rows, 5:-18]
    valid_columns = np.where(np.isnan(x.to_numpy()).sum(0) == 0)[0]
    return x.iloc[:, valid_columns], y


def _load_with_class_column(filename: str, target: str) -> tuple[pd.DataFrame, pd.Series]:
    data = _read_pinned_csv(filename)
    y = data.pop(target)
    # encode the raw labels (not their string form) so numeric labels keep their numeric order
    return data, pd.Series(LabelEncoder().fit_transform(y), name="target")


def load_amazon() -> tuple[pd.DataFrame, pd.Series]:
    """Amazon commerce reviews (OpenML 1457), classification."""
    return _load_with_class_column("amazon.csv", "Class")


def load_microresponse() -> tuple[pd.DataFrame, pd.Series]:
    """Micro-mass response (OpenML 1515), classification."""
    return _load_with_class_column("microresponse.csv", "Class")


def load_bioresponse() -> tuple[pd.DataFrame, pd.Series]:
    """Bioresponse (OpenML 4134), binary classification."""
    data = _read_pinned_csv("bioresponse.csv")
    y = data.pop("target").rename("target")
    return data, y


def load_leukemia() -> tuple[pd.DataFrame, pd.Series]:
    """Leukemia gene expression (OpenML 45090), binary classification."""
    return _load_with_class_column("leukemia.csv", "CLASS")


def _load_uci_classification(
    filename: str,
    *,
    impute: bool = False,
    encode: bool = False,
    drop_constant: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    data = _read_pinned_csv(filename)
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
    return _load_uci_classification("annealing.csv", impute=True, encode=True)


def load_arrhythmia() -> tuple[pd.DataFrame, pd.Series]:
    """Arrhythmia (UCI 5), multiclass classification."""
    return _load_uci_classification("arrhythmia.csv", impute=True, encode=True)


def load_hepatitis() -> tuple[pd.DataFrame, pd.Series]:
    """Hepatitis (UCI 46), binary classification."""
    return _load_uci_classification("hepatitis.csv", impute=True, encode=True)


def load_ionosphere() -> tuple[pd.DataFrame, pd.Series]:
    """Ionosphere (UCI 52), binary classification."""
    return _load_uci_classification("ionosphere.csv", drop_constant=True)


def load_mushroom() -> tuple[pd.DataFrame, pd.Series]:
    """Mushroom (UCI 73), binary classification."""
    return _load_uci_classification("mushroom.csv", encode=True)


def load_nursery() -> tuple[pd.DataFrame, pd.Series]:
    """Nursery (UCI 76), multiclass classification."""
    return _load_uci_classification("nursery.csv", encode=True)


def load_soybean() -> tuple[pd.DataFrame, pd.Series]:
    """Soybean, large (UCI 90), multiclass classification."""
    return _load_uci_classification("soybean.csv", impute=True, encode=True)


def load_thyroid() -> tuple[pd.DataFrame, pd.Series]:
    """Thyroid disease (UCI 102), multiclass classification."""
    return _load_uci_classification("thyroid.csv")


def load_zoo() -> tuple[pd.DataFrame, pd.Series]:
    """Zoo (UCI 111), multiclass classification."""
    return _load_uci_classification("zoo.csv")


_TABULAR: list[tuple[str, Task, object, str]] = [
    ("adult_census", "classification", load_adult_census, "UCI adult (pinned file)"),
    ("amazon", "classification", load_amazon, "OpenML 1457 (pinned file)"),
    ("annealing", "classification", load_annealing, "UCI 3 (pinned file)"),
    ("arrhythmia", "classification", load_arrhythmia, "UCI 5 (pinned file)"),
    ("bike_sharing", "regression", load_bike_sharing, "OpenML 42713 (pinned file)"),
    ("bioresponse", "classification", load_bioresponse, "OpenML 4134 (pinned file)"),
    ("breast_cancer", "classification", load_breast_cancer, "scikit-learn (bundled)"),
    ("california_housing", "regression", load_california_housing, "scikit-learn (pinned file)"),
    (
        "communities_and_crime",
        "regression",
        load_communities_and_crime,
        "UCI 211 via shap (pinned file)",
    ),
    ("forest_fires", "regression", load_forest_fires, "UCI 162 (direct, not yet pinned)"),
    ("hepatitis", "classification", load_hepatitis, "UCI 46 (pinned file)"),
    ("ionosphere", "classification", load_ionosphere, "UCI 52 (pinned file)"),
    ("leukemia", "classification", load_leukemia, "OpenML 45090 (pinned file)"),
    ("microresponse", "classification", load_microresponse, "OpenML 1515 (pinned file)"),
    ("mushroom", "classification", load_mushroom, "UCI 73 (pinned file)"),
    ("nhanesi", "regression", load_nhanesi, "NHANES I via shap (pinned file)"),
    ("nursery", "classification", load_nursery, "UCI 76 (pinned file)"),
    ("real_estate", "regression", load_real_estate, "UCI 477 (direct, not yet pinned)"),
    ("soybean", "classification", load_soybean, "UCI 90 (pinned file)"),
    ("thyroid", "classification", load_thyroid, "UCI 102 (pinned file)"),
    ("wine_quality", "regression", load_wine_quality, "UCI 186 (direct, not yet pinned)"),
    ("zoo", "classification", load_zoo, "UCI 111 (pinned file)"),
]

for _name, _task, _loader, _source in _TABULAR:
    register_dataset(
        DatasetSpec(name=_name, task=_task, loader=_loader, source=_source, kind="tabular")  # type: ignore[arg-type]
    )
