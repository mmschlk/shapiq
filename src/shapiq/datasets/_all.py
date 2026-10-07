"""This module contains functions to load datasets."""

from __future__ import annotations

import os
import tempfile
import urllib.request
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

GITHUB_DATA_URL = "https://raw.githubusercontent.com/mmschlk/shapiq/main/data/"


def _datasets_folder() -> Path:
    """Return the folder of the downloaded csv files, in the user's cache, not the package.

    It is ``$SHAPIQ_DATA_DIR/core_datasets`` if the variable is set, else
    ``$XDG_CACHE_HOME/shapiq/core_datasets`` (``~/.cache/shapiq/core_datasets`` by default).
    """
    root = os.environ.get("SHAPIQ_DATA_DIR")
    if root is None:
        root = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "shapiq"
    return Path(root) / "core_datasets"


def _try_load(csv_file_name: str) -> pd.DataFrame:
    """Try to load a dataset from the local folder.

    If it does not exist, load it from GitHub and save it to the local folder.

    Args:
        csv_file_name: The name of the csv file to load.

    Returns:
        The dataset as a pandas DataFrame.

    """
    folder = _datasets_folder()
    path = folder / csv_file_name
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        with urllib.request.urlopen(GITHUB_DATA_URL + csv_file_name) as response:  # noqa: S310
            content = response.read()
        folder.mkdir(parents=True, exist_ok=True)
        # the file as downloaded, written to a temporary file and renamed, so that no process
        # reads a partial file and every call parses the same text
        fd, tmp_name = tempfile.mkstemp(dir=folder, suffix=".csv.tmp")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(content)
            Path(tmp_name).replace(path)
        finally:
            Path(tmp_name).unlink(missing_ok=True)
        return pd.read_csv(path)


def load_california_housing(
    *,
    to_numpy: bool = False,
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame] | tuple[np.ndarray, np.ndarray]:
    """Load the California housing dataset.

    Args:
        to_numpy: Return numpy objects instead of pandas. Default is ``False``.

    Returns:
        The California housing dataset as a pandas DataFrame.

    Example:
        >>> from shapiq.datasets import load_california_housing
        >>> x_data, y_data = load_california_housing()
        >>> print(x_data.shape, y_data.shape)
        ((20640, 8), (20640,))

    """
    dataset = _try_load("california_housing.csv")
    class_label = "MedHouseVal"
    y_data = dataset[class_label]
    x_data = dataset.drop(columns=[class_label])

    if to_numpy:
        return x_data.to_numpy(), y_data.to_numpy()
    return x_data, y_data


def load_bike_sharing(
    *, to_numpy: bool = False
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame] | tuple[np.ndarray, np.ndarray]:
    """Load the bike-sharing dataset from openml and preprocess it.

    Note:
        The function requires the `sklearn` package to be installed.

    Args:
        to_numpy: Return numpy objects instead of pandas. ``Default is False.``

    Returns:
        The bike-sharing dataset as a pandas DataFrame.

    Example:
        >>> from shapiq.datasets import load_bike_sharing
        >>> x_data, y_data = load_bike_sharing()
        >>> print(x_data.shape, y_data.shape)
        ((17379, 12), (17379,))

    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, RobustScaler

    dataset: pd.DataFrame = _try_load("bike.csv")
    class_label = "count"

    num_feature_names = [
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
    cat_feature_names = [
        "season",
        "weather",
    ]
    dataset[num_feature_names] = dataset[num_feature_names].apply(pd.to_numeric)
    num_pipeline = Pipeline([("scaler", RobustScaler())])
    cat_pipeline = Pipeline(
        [
            ("ordinal_encoder", OrdinalEncoder()),
        ],
    )
    column_transformer = ColumnTransformer(
        [
            ("numerical", num_pipeline, num_feature_names),
            ("categorical", cat_pipeline, cat_feature_names),
        ],
        remainder="passthrough",
    )
    col_names = num_feature_names + cat_feature_names
    col_names += [feature for feature in dataset.columns if feature not in col_names]
    transformed_data: np.ndarray = cast(
        "np.ndarray", column_transformer.fit_transform(dataset)
    )  # Transformations will always return a dense array
    dataset = pd.DataFrame(
        transformed_data,
        columns=np.asarray(col_names),
    )
    dataset = dataset.dropna()

    y_data = dataset.pop(class_label)
    x_data = dataset

    if to_numpy:
        return x_data.to_numpy(), y_data.to_numpy()
    return x_data, y_data


def load_adult_census(
    *, to_numpy: bool = False
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame] | tuple[np.ndarray, np.ndarray]:
    """Load the adult census dataset from the UCI Machine Learning Repository.

    Original source: https://archive.ics.uci.edu/ml/datasets/adult

    Note:
        The function requires the `sklearn` package to be installed.

    Args:
        to_numpy: Return numpy objects instead of pandas. Default is ``False``.

    Returns:
        The adult census dataset as a pandas DataFrame.

    Example:
        >>> from shapiq.datasets import load_adult_census
        >>> x_data, y_data = load_adult_census()
        >>> print(x_data.shape, y_data.shape)
        ((45222, 14), (45222,))

    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    dataset = _try_load("adult_census.csv")
    class_label = "class"

    num_feature_names = ["age", "capital-gain", "capital-loss", "hours-per-week", "fnlwgt"]
    cat_feature_names = [
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
    dataset[num_feature_names] = dataset[num_feature_names].apply(pd.to_numeric)
    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("std_scaler", StandardScaler())],
    )
    cat_pipeline = Pipeline(
        [
            ("ordinal_encoder", OrdinalEncoder()),
        ],
    )
    column_transformer = ColumnTransformer(
        [
            ("numerical", num_pipeline, num_feature_names),
            ("categorical", cat_pipeline, cat_feature_names),
        ],
        remainder="passthrough",
    )
    col_names = num_feature_names + cat_feature_names
    col_names += [feature for feature in dataset.columns if feature not in col_names]
    transformed_data = cast(
        "np.ndarray", column_transformer.fit_transform(dataset)
    )  # Transformations will always return a dense array
    dataset = pd.DataFrame(
        transformed_data,
        columns=np.asarray(col_names),
    )
    dataset = dataset.dropna()

    y_data = dataset.pop(class_label)
    x_data = dataset.astype(float)

    # transform '>50K' to 1 and '<=50K' to 0
    y_data = y_data.apply(lambda x: 1 if x == ">50K" else 0)

    if to_numpy:
        return x_data.to_numpy(), y_data.to_numpy()
    return x_data, y_data
