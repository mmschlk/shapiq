"""This module contains functions to load datasets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
from sklearn.datasets import load_breast_cancer

if TYPE_CHECKING:
    import numpy as np

GITHUB_DATA_URL = "https://raw.githubusercontent.com/mmschlk/shapiq/main/data/"

# csv files are located next to this file in a folder called "data"
SHAPIQ_DATASETS_FOLDER = Path(__file__).parent / "data"


def _create_folder() -> None:
    """Create the datasets folder if it does not exist."""
    Path(SHAPIQ_DATASETS_FOLDER).mkdir(parents=True, exist_ok=True)


def _try_load(csv_file_name: str) -> pd.DataFrame:
    """Try to load a dataset from the local folder.

    If it does not exist, load it from GitHub and save it to the local folder.

    Args:
        csv_file_name: The name of the csv file to load.

    Returns:
        The dataset as a pandas DataFrame.

    """
    _create_folder()
    path = Path(SHAPIQ_DATASETS_FOLDER) / csv_file_name
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        data = pd.read_csv(GITHUB_DATA_URL + csv_file_name)
        data.to_csv(path, index=False)
        return data


def load_california_housing(
    *,
    to_numpy: bool = False,
) -> tuple[pd.DataFrame, pd.Series] | tuple[np.ndarray, np.ndarray]:
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
) -> tuple[pd.DataFrame, pd.Series] | tuple[np.ndarray, np.ndarray]:
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

    dataset = _try_load("bike.csv")
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
    dataset = pd.DataFrame(column_transformer.fit_transform(dataset), columns=col_names)
    dataset = dataset.dropna()

    y_data = dataset.pop(class_label)
    x_data = dataset

    if to_numpy:
        return x_data.to_numpy(), y_data.to_numpy()
    return x_data, y_data


def load_adult_census(
    *, to_numpy: bool = False
) -> tuple[pd.DataFrame, pd.Series] | tuple[np.ndarray, np.ndarray]:
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
    dataset = pd.DataFrame(column_transformer.fit_transform(dataset), columns=col_names)
    dataset = dataset.dropna()

    y_data = dataset.pop(class_label)
    x_data = dataset.astype(float)

    # transform '>50K' to 1 and '<=50K' to 0
    y_data = y_data.apply(lambda x: 1 if x == ">50K" else 0)

    if to_numpy:
        return x_data.to_numpy(), y_data.to_numpy()
    return x_data, y_data



def breast_cancer():
    return load_breast_cancer(return_X_y=True, as_frame=True)


def load_wine_quality():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df_red = pd.read_csv(url, sep=';')
    df_red['type'] = 'red'

    url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df_white = pd.read_csv(url_white, sep=';')
    df_white['type'] = 'white'

    # Combine red and white datasets
    df = pd.concat([df_red, df_white], ignore_index=True)

    y = df['quality'].astype(float)
    X = df.drop(columns=['quality'])

    # One-hot encode the wine type
    X = pd.get_dummies(X, columns=['type'], drop_first=True)

    return X, y


def load_real_estate():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx"
    df = pd.read_excel(url)

    # Drop index column
    df = df.drop(columns=["No"])

    # Use the correct transaction date column
    df["month"] = (df["X1 transaction date"] % 1 * 12).round().astype(int)
    df["month"] = df["month"].replace({0: 1, 12: 1})  # Fix edge cases

    # Drop original date, one-hot encode month
    df = df.drop(columns=["X1 transaction date"])
    df = pd.get_dummies(df, columns=["month"], drop_first=True)

    y = df["Y house price of unit area"].astype(float)
    X = df.drop(columns=["Y house price of unit area"])

    return X, y


def load_nhanesi():
    """Load the NHANES dataset."""
    import shap
    # Load the NHANES dataset
    X, y = shap.datasets.nhanesi()
    # Convert y tp DataFrame
    y = pd.DataFrame(y, columns=["target"])
    return X, y

def load_communities_and_crime():
    """Load the Communities and Crime dataset."""
    import shap
    # Load the Communities and Crime dataset
    X, y = shap.datasets.communitiesandcrime()
    # Convert y to DataFrame
    y = pd.DataFrame(y, columns=["target"])
    return X, y


def load_forest_fires():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
    df = pd.read_csv(url)

    y = df["area"].astype(float)

    # Drop 'day' and map month to season
    df = df.drop(columns=["area", "day"])

    season_map = {
        'dec': 'winter', 'jan': 'winter', 'feb': 'winter',
        'mar': 'spring', 'apr': 'spring', 'may': 'spring',
        'jun': 'summer', 'jul': 'summer', 'aug': 'summer',
        'sep': 'fall', 'oct': 'fall', 'nov': 'fall'
    }

    df["season"] = df["month"].map(season_map)
    df = df.drop(columns=["month"])

    # One-hot encode season
    X = pd.get_dummies(df, columns=["season"], drop_first=True)

    return X, y


def load_independentlinear60():
    """Load the Independent Linear 60 dataset."""
    import shap
    # Load the Independent Linear 60 dataset
    X, y = shap.datasets.independentlinear60()
    # Convert y to DataFrame
    y = pd.DataFrame(y, columns=["target"])
    return X, y

def load_corrgroups60():
    """Load the Correlated Linear 60 dataset."""
    import shap
    # Load the Correlated Linear 60 dataset
    X, y = shap.datasets.corrgroups60()
    # Convert y to DataFrame
    y = pd.DataFrame(y, columns=["target"])
    return X, y