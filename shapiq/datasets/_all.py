"""This module contains functions to load datasets."""

import pandas as pd


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


def load_bike() -> tuple[pd.DataFrame, pd.Series]:
    """Load the bike-sharing dataset from openml.

    Note:
        The function requires the `openml` and `sklearn` packages to be installed.

    Returns:
        The bike-sharing dataset as a pandas DataFrame.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, RobustScaler

    dataset, class_label = _get_open_ml_dataset(42713, version=1)
    num_feature_names = ["hour", "temp", "feel_temp", "humidity", "windspeed"]
    cat_feature_names = [
        "season",
        "year",
        "month",
        "holiday",
        "weekday",
        "workingday",
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

    x_data = dataset
    y_data = dataset.pop(class_label)

    return x_data, y_data


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
