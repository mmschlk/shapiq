"""This module contains functions to load datasets."""

import os

import pandas as pd

GITHUB_DATA_URL = "https://github.com/mmschlk/shapiq/raw/main/data/"


def load_bike() -> tuple[pd.DataFrame, pd.Series]:
    """Load the bike-sharing dataset from a Kaggle competition.

    Original source: https://www.kaggle.com/c/bike-sharing-demand

    Note:
        The function and preprocessing is taken from the `sage` package.

    Returns:
        The bike-sharing dataset as a pandas DataFrame.
    """
    data = pd.read_csv(os.path.join(GITHUB_DATA_URL, "bike.csv"))
    columns = data.columns.tolist()

    # Split and remove datetime column.
    data["datetime"] = pd.to_datetime(data["datetime"])
    data["year"] = data["datetime"].dt.year
    data["month"] = data["datetime"].dt.month
    data["day"] = data["datetime"].dt.day
    data["hour"] = data["datetime"].dt.hour
    data = data.drop("datetime", axis=1)

    # Reorder and rename columns.
    data = data[["year", "month", "day", "hour"] + columns[1:]]
    data.columns = list(map(str.title, data.columns))

    # get the target column
    x_data = data.drop(columns=["Count"])
    y_data = data["Count"]

    return x_data, y_data


def _get_open_ml_dataset(open_ml_id, version=1):
    """Download a dataset from OpenML by its ID and version number.

    Note:
        The function requires the `openml` package to be installed.

    Args:
        open_ml_id: The ID of the dataset on OpenML.
        version: The version number of the dataset.

    Returns:
        The dataset as a pandas DataFrame and the name of the class label.
    """
    import openml

    dataset = openml.datasets.get_dataset(open_ml_id, version=version, download_data=True)
    class_label = dataset.default_target_attribute
    x_data = dataset.get_data()[0]
    return x_data, class_label


def load_adult_census() -> tuple[pd.DataFrame, pd.Series]:
    """Load the adult census dataset from the UCI Machine Learning Repository.

    Original source: https://archive.ics.uci.edu/ml/datasets/adult

    Note:
        The function requires the `openml` and `sklearn` packages to be installed.

    Returns:
        The adult census dataset as a pandas DataFrame.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    dataset, class_label = _get_open_ml_dataset("adult", version=2)
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

    x_data = dataset
    y_data = dataset.pop(class_label)

    return x_data, y_data
