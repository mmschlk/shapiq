"""This module contains functions to load datasets."""

import pandas as pd

GITHUB_DATA_URL = "https://raw.githubusercontent.com/mmschlk/shapiq/main/data/"


def load_california_housing() -> tuple[pd.DataFrame, pd.Series]:
    """Load the California housing dataset.

    Returns:
        The California housing dataset as a pandas DataFrame.
    """
    try:
        dataset = pd.read_csv(
            "C:\\1_Workspaces\\1_Phd_Projects\\shapiq\\data\\california_housing.csv"
        )
    except Exception:
        dataset = pd.read_csv(GITHUB_DATA_URL + "california_housing.csv")
    class_label = "MedHouseVal"
    y_data = dataset[class_label]
    x_data = dataset.drop(columns=[class_label])

    return x_data, y_data


def load_bike_sharing() -> tuple[pd.DataFrame, pd.Series]:
    """Load the bike-sharing dataset from openml.

    Note:
        The function requires the `sklearn` package to be installed.

    Returns:
        The bike-sharing dataset as a pandas DataFrame.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, RobustScaler

    try:
        dataset = pd.read_csv("C:\\1_Workspaces\\1_Phd_Projects\\shapiq\\data\\bike.csv")
    except Exception:
        dataset = pd.read_csv(GITHUB_DATA_URL + "bike.csv")
    class_label = "count"

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
        The function requires the `sklearn` package to be installed.

    Returns:
        The adult census dataset as a pandas DataFrame.
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    try:
        dataset = pd.read_csv("C:\\1_Workspaces\\1_Phd_Projects\\shapiq\\data\\adult_census.csv")
    except Exception:
        dataset = pd.read_csv(GITHUB_DATA_URL + "adult_census.csv")
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

    x_data = dataset
    y_data = dataset.pop(class_label)

    # transform '>50K' to 1 and '<=50K' to 0
    y_data = y_data.apply(lambda x: 1 if x == ">50K" else 0)

    return x_data, y_data
