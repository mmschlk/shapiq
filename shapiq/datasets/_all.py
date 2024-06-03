"""This module contains functions to load datasets."""

import os

import pandas as pd

GITHUB_DATA_URL = "https://raw.githubusercontent.com/mmschlk/shapiq/main/data/"

# csv files are located next to this file in a folder called "data"
SHAPIQ_DATASETS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(SHAPIQ_DATASETS_FOLDER, exist_ok=True)


def _try_load(csv_file_name: str) -> pd.DataFrame:
    """Try to load a dataset from the local folder. If it does not exist, load it from GitHub and
    save it to the local folder.

    Args:
        csv_file_name: The name of the csv file to load.

    Returns:
        The dataset as a pandas DataFrame.
    """
    try:
        return pd.read_csv(os.path.join(SHAPIQ_DATASETS_FOLDER, csv_file_name))
    except FileNotFoundError:
        data = pd.read_csv(GITHUB_DATA_URL + csv_file_name)
        data.to_csv(os.path.join(SHAPIQ_DATASETS_FOLDER, csv_file_name), index=False)
        return data


def load_california_housing(to_numpy=False) -> tuple[pd.DataFrame, pd.Series]:
    """Load the California housing dataset.

    Args:
        to_numpy: Return numpy objects instead of pandas. Default is ``False``.

    Returns:
        The California housing dataset as a pandas DataFrame.
    """
    dataset = _try_load("california_housing.csv")
    class_label = "MedHouseVal"
    y_data = dataset[class_label]
    x_data = dataset.drop(columns=[class_label])

    if to_numpy:
        return x_data.to_numpy(), y_data.to_numpy()
    else:
        return x_data, y_data


def load_bike_sharing(to_numpy=False) -> tuple[pd.DataFrame, pd.Series]:
    """Load the bike-sharing dataset from openml.

    Args:
        to_numpy: Return numpy objects instead of pandas. ``Default is False.``

    Note:
        The function requires the `sklearn` package to be installed.

    Returns:
        The bike-sharing dataset as a pandas DataFrame.
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
        ]
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
    dataset.dropna(inplace=True)

    y_data = dataset.pop(class_label)
    x_data = dataset

    if to_numpy:
        return x_data.to_numpy(), y_data.to_numpy()
    else:
        return x_data, y_data


def load_adult_census(to_numpy=False) -> tuple[pd.DataFrame, pd.Series]:
    """Load the adult census dataset from the UCI Machine Learning Repository.

    Original source: https://archive.ics.uci.edu/ml/datasets/adult

    Args:
        to_numpy: Return numpy objects instead of pandas. Default is ``False``.

    Note:
        The function requires the `sklearn` package to be installed.

    Returns:
        The adult census dataset as a pandas DataFrame.
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
        [("imputer", SimpleImputer(strategy="median")), ("std_scaler", StandardScaler())]
    )
    cat_pipeline = Pipeline(
        [
            ("ordinal_encoder", OrdinalEncoder()),
        ]
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
    dataset.dropna(inplace=True)

    y_data = dataset.pop(class_label)
    x_data = dataset.astype(float)

    # transform '>50K' to 1 and '<=50K' to 0
    y_data = y_data.apply(lambda x: 1 if x == ">50K" else 0)

    if to_numpy:
        return x_data.to_numpy(), y_data.to_numpy()
    else:
        return x_data, y_data
