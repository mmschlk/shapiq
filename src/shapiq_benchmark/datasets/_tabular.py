"""Loaders of the classic real-world tabular datasets.

Every dataset comes from its original source, and the preprocessing is unchanged from the
loaders that used to ship with the data:

- OpenML (``openml``): adult, amazon, arrhythmia, bike sharing, bioresponse, leukemia,
  micro-mass;
- the UCI repository through ``ucimlrepo``: annealing, hepatitis, ionosphere, mushroom, nursery,
  zoo;
- raw files of the UCI repository (checksum-pinned): soybean, thyroid, wine quality, real estate,
  forest fires;
- scikit-learn's ``fetch_california_housing``;
- the data folder of shap (checksum-pinned): NHANES I, communities and crime.

The raw table of an upstream dataset is downloaded once and cached as a lossless CSV in
``<data dir>/tabular/``. A table whose shape differs from the one the loaders were written for
is rejected, so a changed upstream fails loudly instead of silently changing a benchmark, and
tables without a header (or with upstream names that changed) get fixed column names. The
tables reproduce the files that used to ship with the package exactly.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer as _sklearn_breast_cancer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, StandardScaler

from shapiq_games._optional import require

from ._cache import RemoteFile, atomic_write_bytes, fetch, get_data_dir
from ._registry import DatasetSpec, register_dataset

if TYPE_CHECKING:
    from shapiq_games.typing import Task


def _generic_columns(n_features: int) -> tuple[str, ...]:
    """The names of a raw table without a header: ``f1, ..., fn`` and ``target``."""
    return (*(f"f{i}" for i in range(1, n_features + 1)), "target")


@dataclass(frozen=True)
class _Upstream:
    """The original source of a raw table, its shape, and the column names the loaders use.

    Attributes:
        source: ``"openml"`` (by dataset id), ``"uci"`` (``ucimlrepo``, by dataset id),
            ``"uci_files"`` (raw files of the UCI repository, concatenated), or ``"sklearn"``.
        shape: The shape of the table, target column included.
        dataset_id: The OpenML or UCI dataset id.
        files: The raw UCI files (keys of ``_UCI_FILES``).
        target_first: Whether the raw files store the target in the first column.
        separator: The column separator of the raw files.
        columns: The column names, set by position. They fix the schema of a table whose
            upstream has no header or names its columns differently from the loaders.
    """

    source: Literal["openml", "uci", "uci_files", "sklearn"]
    shape: tuple[int, int]
    dataset_id: int | None = None
    files: tuple[str, ...] = ()
    target_first: bool = False
    separator: str = ","
    columns: tuple[str, ...] | None = None


_ANNEALING_COLUMNS = (
    "family", "product-type", "steel", "carbon", "hardness", "temper_rolling", "condition",
    "formability", "strength", "non-ageing", "surface-finish", "surface-quality", "enamelability",
    "bc", "bf", "bt", "bw-me", "bl", "m", "chrom", "phos", "cbond", "marvi", "exptl", "ferro",
    "corr", "blue-bright-varn-clean", "lustre", "jurofm", "s", "p", "shape", "thick", "width",
    "len", "oil", "bore", "packing", "target",
)  # fmt: skip
_HEPATITIS_COLUMNS = (
    "age", "sex", "steroid", "antivirals", "fatigue", "malaise", "anorexia", "liver-big",
    "liver-firm", "spleen-palpable", "spiders", "ascites", "varices", "bilirubin",
    "alk-phosphate", "sgot", "albumin", "protime", "histology", "target",
)  # fmt: skip
_SOYBEAN_COLUMNS = (
    "date", "plant-stand", "precip", "temp", "hail", "crop-hist", "area-damaged", "severity",
    "seed-tmt", "germination", "plant-growth", "leaves", "leafspots-halo", "leafspots-marg",
    "leafspot-size", "leaf-shread", "leaf-malf", "leaf-mild", "stem", "lodging", "stem-cankers",
    "canker-lesion", "fruiting-bodies", "external-decay", "mycelium", "int-discolor", "sclerotia",
    "fruit-pods", "fruit-spots", "seed", "mold-growth", "seed-discolor", "seed-size", "shriveling",
    "roots", "target",
)  # fmt: skip

_UPSTREAM: dict[str, _Upstream] = {
    "adult_census": _Upstream("openml", (48842, 15), dataset_id=1590),
    "amazon": _Upstream("openml", (1500, 10001), dataset_id=1457),
    "bike_sharing": _Upstream("openml", (17379, 13), dataset_id=42713),
    "bioresponse": _Upstream("openml", (3751, 1777), dataset_id=4134),
    "leukemia": _Upstream("openml", (72, 7130), dataset_id=45090),
    "microresponse": _Upstream("openml", (571, 1301), dataset_id=1515),
    # ucimlrepo cannot export arrhythmia; OpenML 5 holds the same table
    "arrhythmia": _Upstream("openml", (452, 280), dataset_id=5, columns=_generic_columns(279)),
    "annealing": _Upstream("uci", (898, 39), dataset_id=3, columns=_ANNEALING_COLUMNS),
    "hepatitis": _Upstream("uci", (155, 20), dataset_id=46, columns=_HEPATITIS_COLUMNS),
    "ionosphere": _Upstream("uci", (351, 35), dataset_id=52, columns=_generic_columns(34)),
    "mushroom": _Upstream("uci", (8124, 23), dataset_id=73),
    "nursery": _Upstream("uci", (12960, 9), dataset_id=76),
    "zoo": _Upstream("uci", (101, 17), dataset_id=111),
    # the full soybean (large) data is its training and test file; ucimlrepo has only the first
    "soybean": _Upstream(
        "uci_files",
        (683, 36),
        files=("soybean-large.data", "soybean-large.test"),
        target_first=True,
        columns=_SOYBEAN_COLUMNS,
    ),
    # thyroid is the ANN thyroid data (UCI 102), which ucimlrepo cannot export
    "thyroid": _Upstream(
        "uci_files",
        (7200, 22),
        files=("ann-train.data", "ann-test.data"),
        separator=r"\s+",
        columns=_generic_columns(21),
    ),
    "california_housing": _Upstream("sklearn", (20640, 9)),
}

# a commit, not a branch: an upstream edit cannot break the checksums
_SHAP_DATA_URL = (
    "https://raw.githubusercontent.com/shap/shap/fc3e290e97ce12f76d1175d24c6e3023b4ca7d69/data/"
)
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
    filename: RemoteFile(_UCI_URL + path, filename, sha256, subdir="uci")
    for filename, (path, sha256) in {
        "winequality-red.csv": (
            "wine-quality/winequality-red.csv",
            "4a402cf041b025d4566d954c3b9ba8635a3a8a01e039005d97d6a710278cf05e",
        ),
        "winequality-white.csv": (
            "wine-quality/winequality-white.csv",
            "76c3f809815c17c07212622f776311faeb31e87610d52c26d87d6e361b169836",
        ),
        "real_estate.xlsx": (
            "00477/Real%20estate%20valuation%20data%20set.xlsx",
            "597d72fcc6c0539e6035a033ddb387db48fff3fb1f3c98fee31fe081c64a9059",
        ),
        "forestfires.csv": (
            "forest-fires/forestfires.csv",
            "0d6586a1fa52f55bef48578aef14eb97273f1e9330e1a53423df497a77065253",
        ),
        "soybean-large.data": (
            "soybean/soybean-large.data",
            "04b99f2728ded9f544022d2b4f6cce0ebe11ef5c9d44f10e21acd7093876507e",
        ),
        "soybean-large.test": (
            "soybean/soybean-large.test",
            "ff1c5f5c41ddc9a8e3746648a2fe5cafb6b0eef86bd49575decbb665e1b6a104",
        ),
        "ann-train.data": (
            "thyroid-disease/ann-train.data",
            "3da53a156bda36cb0c97e9f4b6b111c9226c54c4aa00230de5604b787c47e3a6",
        ),
        "ann-test.data": (
            "thyroid-disease/ann-test.data",
            "c649ea19416e78c7996cfaaa2a9e281cb597d4b075aaa68c494fc3e4ee3aa30b",
        ),
    }.items()
}


def _download_table(name: str) -> pd.DataFrame:
    """Download the raw table of ``name`` (features and target) from its original source."""
    upstream = _UPSTREAM[name]
    if upstream.source == "openml":
        openml = require("openml", purpose=f"the {name} dataset", extra="benchmark")
        dataset = openml.datasets.get_dataset(upstream.dataset_id, download_data=True)
        table, *_ = dataset.get_data(dataset_format="dataframe")
    elif upstream.source == "uci":
        ucimlrepo = require("ucimlrepo", purpose=f"the {name} dataset", extra="benchmark")
        data = ucimlrepo.fetch_ucirepo(id=upstream.dataset_id).data
        table = data.features.copy()
        table["target"] = data.targets.squeeze()
    elif upstream.source == "uci_files":
        parts = [
            pd.read_csv(fetch(_UCI_FILES[file]), header=None, sep=upstream.separator, na_values="?")
            for file in upstream.files
        ]
        table = pd.concat(parts, ignore_index=True)
        if upstream.target_first:
            table = pd.concat([table.iloc[:, 1:], table.iloc[:, 0]], axis=1)
    else:
        from sklearn.datasets import fetch_california_housing

        # a fresh folder per download: processes fetching into one folder race (one deletes the
        # archive while another opens it), and the table is cached as a CSV anyway
        parent = get_data_dir() / "scikit_learn"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as home:
            table = fetch_california_housing(data_home=home, as_frame=True).frame
    if table.shape != upstream.shape:
        msg = (
            f"The {name} table from {upstream.source} {upstream.dataset_id or ''} has shape "
            f"{table.shape}, but the loader expects {upstream.shape}. The upstream data changed."
        )
        raise ValueError(msg)
    if upstream.columns is not None:
        table.columns = list(upstream.columns)
    return table


def _read_table(name: str) -> pd.DataFrame:
    """Return the raw table of ``name``, downloading and caching it as a CSV on first use.

    A cached table of the wrong shape (e.g. truncated) is downloaded again.
    """
    path = get_data_dir() / "tabular" / f"{name}.csv"
    if path.exists():
        table = pd.read_csv(path, low_memory=False, float_precision="round_trip")
        if table.shape == _UPSTREAM[name].shape:
            return table
    table = _download_table(name)
    buffer = StringIO()
    table.to_csv(buffer, index=False, float_format="%.17g")  # lossless
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
    """Ordinal-encode all text and category columns (missing and unknown categories become ``-1``)."""
    categorical = x.select_dtypes(include=["object", "category", "string"]).columns
    if len(categorical) == 0:
        return x
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1
    )
    x = x.copy()
    values = x[categorical].astype(object)
    x[categorical] = encoder.fit_transform(values.where(values.notna(), np.nan))  # pd.NA -> NaN
    return x


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
    y = dataset.pop("class").astype(str).rename("target")  # "<=50K" and ">50K"
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
    require("openpyxl", purpose="reading the real_estate dataset", extra="benchmark")
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
    return data, data.pop(target).rename("target")  # load_dataset encodes the labels


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
    return data, y.rename("target")  # load_dataset encodes the labels


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
    ("arrhythmia", "classification", load_arrhythmia, "OpenML 5"),
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
    ("forest_fires", "regression", load_forest_fires, "UCI 162 (raw file)"),
    ("hepatitis", "classification", load_hepatitis, "UCI 46 (ucimlrepo)"),
    ("ionosphere", "classification", load_ionosphere, "UCI 52 (ucimlrepo)"),
    ("leukemia", "classification", load_leukemia, "OpenML 45090"),
    ("microresponse", "classification", load_microresponse, "OpenML 1515"),
    ("mushroom", "classification", load_mushroom, "UCI 73 (ucimlrepo)"),
    ("nhanesi", "regression", load_nhanesi, "NHANES I via shap's data (checksum-pinned)"),
    ("nursery", "classification", load_nursery, "UCI 76 (ucimlrepo)"),
    ("real_estate", "regression", load_real_estate, "UCI 477 (raw file)"),
    ("soybean", "classification", load_soybean, "UCI 90 (raw files)"),
    ("thyroid", "classification", load_thyroid, "UCI 102 (raw files)"),
    ("wine_quality", "regression", load_wine_quality, "UCI 186 (raw files)"),
    ("zoo", "classification", load_zoo, "UCI 111 (ucimlrepo)"),
]

for _name, _task, _loader, _source in _TABULAR:
    register_dataset(
        DatasetSpec(name=_name, task=_task, loader=_loader, source=_source, kind="tabular")  # type: ignore[arg-type]
    )
