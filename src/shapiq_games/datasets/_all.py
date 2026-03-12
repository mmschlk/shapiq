"""This module contains functions to load datasets."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import (
    load_breast_cancer as breast_cancer,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, RobustScaler, StandardScaler

try:
    from ucimlrepo import fetch_ucirepo as _fetch_ucirepo
except ImportError:
    _fetch_ucirepo = None  # type: ignore[assignment]

GITHUB_DATA_URL = "https://raw.githubusercontent.com/mmschlk/shapiq/main/data/"

# csv files are located next to this file in a folder called "data"
SHAPIQ_DATASETS_FOLDER = Path(__file__).parent / "data"


def _create_folder() -> None:
    """Create the datasets folder if it does not exist.

    The folder is created at the location specified by SHAPIQ_DATASETS_FOLDER.
    Uses mkdir with parents=True and exist_ok=True for safety.

    """
    Path(SHAPIQ_DATASETS_FOLDER).mkdir(parents=True, exist_ok=True)


def _try_load(csv_file_name: str, **kwargs: dict[str, any]) -> pd.DataFrame:
    """Try to load a dataset from the local folder.

    Attempts to load a CSV file from the local datasets folder. If the file
    does not exist, fetches it from GitHub and saves it locally for future use.

    Args:
        csv_file_name: The name of the CSV file to load.
        **kwargs: Additional keyword arguments forwarded to ``pandas.read_csv``.

    Returns:
        The dataset as a pandas DataFrame.

    """
    _create_folder()
    path = Path(SHAPIQ_DATASETS_FOLDER) / csv_file_name
    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError:
        data = pd.read_csv(GITHUB_DATA_URL + csv_file_name, **kwargs)
        data.to_csv(path, index=False)
        return data


def load_california_housing() -> tuple[pd.DataFrame, pd.Series]:
    """Load the California housing dataset.

    Original source: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

    Returns:
        The California housing dataset as pandas objects ``(X, y)``.

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

    return x_data, y_data


def load_bike_sharing() -> tuple[pd.DataFrame, pd.Series]:
    """Load the bike-sharing dataset from OpenML and preprocess it.

    Original source: https://www.openml.org/search?type=data&status=active&id=42713

    Note:
        The function requires the `sklearn` package to be installed.

    Returns:
        The bike-sharing dataset as pandas objects ``(X, y)``.

    Example:
        >>> from shapiq.datasets import load_bike_sharing
        >>> x_data, y_data = load_bike_sharing()
        >>> print(x_data.shape, y_data.shape)
        ((17379, 12), (17379,))

    """
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

    return x_data, y_data


def load_adult_census() -> tuple[pd.DataFrame, pd.Series]:
    """Load the adult census dataset from the UCI Machine Learning Repository.

    Original source: https://archive.ics.uci.edu/ml/datasets/adult

    Note:
        The function requires the `sklearn` package to be installed.

    Returns:
        The adult census dataset as pandas objects ``(X, y)``.

    Example:
        >>> from shapiq.datasets import load_adult_census
        >>> x_data, y_data = load_adult_census()
        >>> print(x_data.shape, y_data.shape)
        ((45222, 14), (45222,))

    """
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

    return x_data, y_data


def load_breast_cancer() -> tuple[pd.DataFrame, pd.Series]:
    """Load the breast cancer dataset from scikit-learn.

    Original source: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

    Returns:
        A tuple (X, y) where X is a DataFrame with 30 features and y is a Series
        with binary target values.

    Example:
        >>> from shapiq_games.datasets import load_breast_cancer
        >>> x_data, y_data = load_breast_cancer()
        >>> print(x_data.shape, y_data.shape)
        ((569, 30), (569,))

    """
    return breast_cancer(return_X_y=True, as_frame=True)


def load_wine_quality() -> tuple[pd.DataFrame, pd.Series]:
    """Load the wine quality dataset from the UCI Machine Learning Repository.

    Original source: https://archive.ics.uci.edu/dataset/186/wine+quality

    The loader combines the red and white wine quality datasets and one-hot encodes
    the wine type.

    Returns:
        A tuple (X, y) where X is a DataFrame with 12 features (including encoded
        wine type) and y is a Series with quality scores.

    Example:
        >>> from shapiq_games.datasets import load_wine_quality
        >>> x_data, y_data = load_wine_quality()
        >>> print(x_data.shape, y_data.shape)
        ((6497, 12), (6497,))

    """
    url = (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    df_red = pd.read_csv(url, sep=";")
    df_red["type"] = "red"

    url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    df_white = pd.read_csv(url_white, sep=";")
    df_white["type"] = "white"

    # Combine red and white datasets
    data = pd.concat([df_red, df_white], ignore_index=True)

    y = data["quality"].astype(float)
    X = data.drop(columns=["quality"])

    # One-hot encode the wine type
    X = pd.get_dummies(X, columns=["type"], drop_first=True)

    return X, y


def load_real_estate() -> tuple[pd.DataFrame, pd.Series]:
    """Load the real estate valuation dataset from the UCI repository.

    Original source: https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set

    The loader derives a month feature from the transaction date and one-hot encodes it.

    Returns:
        A tuple (X, y) where X is a DataFrame with 11 features (including encoded months)
        and y is a Series with house price unit area values.

    Example:
        >>> from shapiq_games.datasets import load_real_estate
        >>> x_data, y_data = load_real_estate()
        >>> print(x_data.shape, y_data.shape)
        ((414, 11), (414,))

    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00477/Real%20estate%20valuation%20data%20set.xlsx"
    data = pd.read_excel(url)

    # Drop index column
    data = data.drop(columns=["No"])

    # Use the correct transaction date column
    data["month"] = (data["X1 transaction date"] % 1 * 12).round().astype(int)
    data["month"] = data["month"].replace({0: 1, 12: 1})  # Fix edge cases

    # Drop original date, one-hot encode month
    data = data.drop(columns=["X1 transaction date"])
    data = pd.get_dummies(data, columns=["month"], drop_first=True)

    y = data["Y house price of unit area"].astype(float)
    X = data.drop(columns=["Y house price of unit area"])

    return X, y


def load_nhanesi() -> tuple[pd.DataFrame, pd.Series]:
    """Return a nicely packaged version of NHANES I data with survival times as labels.

    Used in survival analysis tasks. The dataset is loaded from local CSV files
    (NHANESI_X.csv and NHANESI_y.csv).

    Returns:
        A tuple (X, y) where X is a DataFrame with features and y is a Series
        with survival times as targets.

    Example:
        >>> from shapiq_games.datasets import load_nhanesi
        >>> features, survival_times = load_nhanesi()
        >>> print(features.shape, survival_times.shape)

    """
    X = _try_load("NHANESI_X.csv", index_col=0)
    y = _try_load("NHANESI_y.csv", index_col=0).squeeze()  # type: ignore[assignment]

    return X, pd.Series(y, name="target")


def load_communities_and_crime() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Communities and Crime dataset.

    Original source: https://shap.readthedocs.io/en/latest/generated/shap.datasets.communitiesandcrime.html

    The target is the total number of violent crimes per 100K population. The loader
    follows the preprocessing used in SHAP and removes features with missing values.

    Returns:
        The Communities and Crime dataset as pandas objects ``(X, y)``.

    Example:
        >>> from shapiq_games.datasets import load_communities_and_crime
        >>> x_data, y_data = load_communities_and_crime()
        >>> print(x_data.shape, y_data.shape)

    """
    raw_data = _try_load("CommViolPredUnnormalizedData.txt", na_values="?")

    # find the indices where the total violent crimes are known
    valid_inds = np.where(np.invert(np.isnan(raw_data.iloc[:, -2])))[0]

    y = pd.Series(np.array(raw_data.iloc[valid_inds, -2], dtype=float), name="target")

    # extract the predictive features and remove columns with missing values
    X = raw_data.iloc[valid_inds, 5:-18]
    valid_cols = np.where(np.isnan(X.values).sum(0) == 0)[0]
    X = X.iloc[:, valid_cols]

    return X, y


def load_forest_fires() -> tuple[pd.DataFrame, pd.Series]:
    """Load the forest fires dataset from the UCI Machine Learning Repository.

    Original source: https://archive.ics.uci.edu/dataset/162/forest+fires

    The loader drops the weekday column, maps the month to seasons, and one-hot encodes
    the season feature.

    Returns:
        A tuple (X, y) where X is a DataFrame with 12 features (including encoded seasons)
        and y is a Series with burned area values.

    Example:
        >>> from shapiq_games.datasets import load_forest_fires
        >>> x_data, y_data = load_forest_fires()
        >>> print(x_data.shape, y_data.shape)
        ((517, 12), (517,))

    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
    data = pd.read_csv(url)

    y = data["area"].astype(float)

    # Drop 'day' and map month to season
    data = data.drop(columns=["area", "day"])

    season_map = {
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

    data["season"] = data["month"].map(season_map)
    data = data.drop(columns=["month"])

    # One-hot encode season
    X = pd.get_dummies(data, columns=["season"], drop_first=True)

    return X, y


def load_independentlinear60() -> tuple[pd.DataFrame, pd.Series]:
    """Load the synthetic Independent Linear dataset with 60 features.

    Original source: https://shap.readthedocs.io/en/latest/generated/shap.datasets.independentlinear60.html

    This dataset is adapted from SHAP. It contains independent Gaussian features and a
    linear target with small Gaussian noise.

    Returns:
        The synthetic dataset as pandas objects ``(X, y)``.

    Example:
        >>> from shapiq_games.datasets import load_independentlinear60
        >>> x_data, y_data = load_independentlinear60()
        >>> print(x_data.shape, y_data.shape)
        ((1000, 60), (1000,))

    """
    # set a constant seed
    rng = np.random.default_rng(42)  # use a local RNG to avoid affecting global state

    # generate dataset with known correlation
    N, M = 1000, 60

    # set one coefficient from each group of 3 to 1
    beta = np.zeros(M)
    beta[0:30:3] = 1

    # Make sure the sample correlation is a perfect match
    X_start = rng.randn(N, M)
    X = X_start - X_start.mean(0)
    y = np.matmul(X, beta) + rng.randn(N) * 1e-2

    return pd.DataFrame(X), pd.Series(y, name="target")


def load_corrgroups60() -> tuple[pd.DataFrame, pd.Series]:
    """Load the synthetic Correlated Groups dataset with 60 features.

    Original source: https://shap.readthedocs.io/en/latest/generated/shap.datasets.corrgroups60.html

    This dataset is adapted from SHAP. It contains groups of tightly correlated Gaussian
    features and a linear target with small Gaussian noise.

    Returns:
        The synthetic dataset as pandas objects ``(X, y)``.

    Example:
        >>> from shapiq_games.datasets import load_corrgroups60
        >>> x_data, y_data = load_corrgroups60()
        >>> print(x_data.shape, y_data.shape)
        ((1000, 60), (1000,))

    """
    # set a constant seed
    rng = np.random.default_rng(42)

    # generate dataset with known correlation
    N, M = 1000, 60

    # set one coefficient from each group of 3 to 1
    beta = np.zeros(M)
    beta[0:30:3] = 1

    # build a correlation matrix with groups of 3 tightly correlated features
    C = np.eye(M)
    for i in range(0, 30, 3):
        C[i, i + 1] = C[i + 1, i] = 0.99
        C[i, i + 2] = C[i + 2, i] = 0.99
        C[i + 1, i + 2] = C[i + 2, i + 1] = 0.99

    # Make sure the sample correlation is a perfect match
    X_start = rng.randn(N, M)
    X_centered = X_start - X_start.mean(0)
    Sigma = np.matmul(X_centered.T, X_centered) / X_centered.shape[0]
    W = np.linalg.cholesky(np.linalg.inv(Sigma)).T
    X_white = np.matmul(X_centered, W.T)

    # create the final data
    X_final = np.matmul(X_white, np.linalg.cholesky(C).T)
    X = X_final
    y = np.matmul(X, beta) + rng.randn(N) * 1e-2

    return pd.DataFrame(X), pd.Series(y, name="target")


def load_amazon() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Amazon employee access dataset.

    Original source: https://www.openml.org/search?type=data&status=active&sort=runs&qualities.NumberOfFeatures=between_1000_10000&id=1457

    Note:
        The function requires the ``sklearn`` package to be installed.

    Returns:
        The Amazon dataset as pandas objects ``(X, y)``.

    Example:
        >>> from shapiq_games.datasets import load_amazon
        >>> x_data, y_data = load_amazon()
        >>> print(x_data.shape, y_data.shape)

    """
    data = _try_load("amazon.csv")
    target_name = "Class"
    X = data.drop(columns=[target_name])
    y = data[target_name]

    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), name="target")
    return X, y


def load_microresponse() -> tuple[pd.DataFrame, pd.Series]:
    """Load the MicroMass microresponse dataset.

    Original source: https://www.openml.org/search?type=data&status=active&sort=runs&qualities.NumberOfFeatures=between_1000_10000&id=1515

    Note:
        The function requires the ``sklearn`` package to be installed.

    Returns:
        The MicroMass dataset as pandas objects ``(X, y)``.

    Example:
        >>> from shapiq_games.datasets import load_microresponse
        >>> x_data, y_data = load_microresponse()
        >>> print(x_data.shape, y_data.shape)

    """
    data = _try_load("microresponse.csv")
    target_name = "Class"
    X = data.drop(columns=[target_name])
    y = data[target_name]
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), name="target")
    return X, y


def load_bioresponse() -> tuple[pd.DataFrame, pd.Series]:
    """Load the bioresponse dataset.

    Original source: https://www.openml.org/search?type=data&status=active&sort=runs&qualities.NumberOfFeatures=between_1000_10000&id=4134

    Returns:
        The bioresponse dataset as pandas objects ``(X, y)``.

    Example:
        >>> from shapiq_games.datasets import load_bioresponse
        >>> x_data, y_data = load_bioresponse()
        >>> print(x_data.shape, y_data.shape)

    """
    data = _try_load("bioresponse.csv")
    target_name = "target"
    X = data.drop(columns=[target_name])
    y = data[target_name].rename("target")
    return X, y


def load_leukemia() -> tuple[pd.DataFrame, pd.Series]:
    """Load the leukemia dataset.

    Original source: https://www.openml.org/search?type=data&status=active&id=45090

    Note:
        The function requires the ``sklearn`` package to be installed.

    Returns:
        The leukemia dataset as pandas objects ``(X, y)``.

    Example:
        >>> from shapiq_games.datasets import load_leukemia
        >>> x_data, y_data = load_leukemia()
        >>> print(x_data.shape, y_data.shape)

    """
    data = _try_load("leukemia.csv")
    target_name = "CLASS"
    X = data.drop(columns=[target_name])
    y = data[target_name]
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y), name="target")
    return X, y


def load_condind(
    n_samples: int = 1000,
    n_irrelevant: int = 3,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the conditional independence synthetic dataset.

    Generates a binary classification problem where x1 and x2 are uniformly
    distributed and the target is True if x1 + x2 > 1, with additional
    irrelevant noise features.

    Args:
        n_samples: Number of samples to generate. Defaults to 1000.
        n_irrelevant: Number of irrelevant noise features to include. Defaults to 3.
        random_state: Seed for reproducibility. Defaults to None.

    Returns:
        A tuple (X, y) where X is a DataFrame with n_irrelevant+2 features
        and y is a Series with binary labels.

    Example:
        >>> from shapiq_games.datasets import load_condind
        >>> x_data, y_data = load_condind(n_samples=500, n_irrelevant=2)
        >>> print(x_data.shape, y_data.shape)
        ((500, 4), (500,))

    """
    rng = np.random.default_rng(random_state)
    x1 = rng.uniform(0, 1, n_samples)
    x2 = rng.uniform(0, 1, n_samples)
    y = (x1 + x2 > 1).astype(int)
    data = {"x1": x1, "x2": x2}
    for i in range(1, n_irrelevant + 1):
        data[f"irr{i}"] = rng.uniform(0, 1, n_samples)
    return pd.DataFrame(data), pd.Series(y, name="y")


def load_xor(
    n_samples: int = 1000,
    n_irrelevant: int = 2,
    noise: float = 0.05,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the XOR synthetic dataset.

    Generates a binary classification problem based on XOR logic where
    y = (x1 != x2). Features are binary and optionally corrupted with label noise.

    Args:
        n_samples: Number of samples to generate. Defaults to 1000.
        n_irrelevant: Number of irrelevant noise features to include. Defaults to 2.
        noise: Fraction of labels to flip randomly. Defaults to 0.05.
        random_state: Seed for reproducibility. Defaults to None.

    Returns:
        A tuple (X, y) where X is a DataFrame with n_irrelevant+2 features
        and y is a Series with binary labels.

    Example:
        >>> from shapiq_games.datasets import load_xor
        >>> x_data, y_data = load_xor(n_samples=500, noise=0.1)
        >>> print(x_data.shape, y_data.shape)
        ((500, 4), (500,))

    """
    rng = np.random.default_rng(random_state)
    x1 = rng.integers(0, 2, n_samples)
    x2 = rng.integers(0, 2, n_samples)
    y = (x1 != x2).astype(int)
    if noise > 0:
        flip = rng.random(n_samples) < noise
        y[flip] = 1 - y[flip]
    data = {"x1": x1.astype(float), "x2": x2.astype(float)}
    for i in range(1, n_irrelevant + 1):
        data[f"irr{i}"] = rng.integers(0, 2, n_samples).astype(float)
    return pd.DataFrame(data), pd.Series(y, name="y")


def load_group(
    n_samples: int = 1000,
    n_irrelevant: int = 2,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the group interaction synthetic dataset.

    Generates a binary classification problem with four Gaussian clusters.
    Clusters are labeled with grouping interactions: (0, 0) and (2, 2) share
    label 0, while (1, 1) and (3, 3) share label 1.

    Args:
        n_samples: Number of samples to generate. Defaults to 1000.
        n_irrelevant: Number of irrelevant noise features to include. Defaults to 2.
        random_state: Seed for reproducibility. Defaults to None.

    Returns:
        A tuple (X, y) where X is a DataFrame with n_irrelevant+2 features
        and y is a Series with binary labels.

    Example:
        >>> from shapiq_games.datasets import load_group
        >>> x_data, y_data = load_group(n_samples=500)
        >>> print(x_data.shape, y_data.shape)
        ((500, 4), (500,))

    """
    rng = np.random.default_rng(random_state)
    centers = np.array([[-2, -2], [2, 2], [-2, 2], [2, -2]], dtype=float)
    cluster_labels = [0, 1, 0, 1]
    per = n_samples // 4
    counts = [per] * 4
    counts[-1] += n_samples - sum(counts)
    xs, ys = [], []
    for k, (cnt, center) in enumerate(zip(counts, centers, strict=False)):
        xs.append(rng.normal(loc=center, scale=0.8, size=(cnt, 2)))
        ys.extend([cluster_labels[k]] * cnt)
    xy = np.vstack(xs)
    y = np.array(ys)
    perm = rng.permutation(n_samples)
    xy, y = xy[perm], y[perm]
    data = {"x1": xy[:, 0], "x2": xy[:, 1]}
    for i in range(1, n_irrelevant + 1):
        data[f"irr{i}"] = rng.normal(0, 1, n_samples)
    return pd.DataFrame(data), pd.Series(y, name="y")


def load_cross(
    n_samples: int = 1000,
    a: float = 0.3,
    n_irrelevant: int = 2,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the cross synthetic dataset.

    Generates a binary classification problem where the target is True
    if (|x1| < a) XOR (|x2| < a), creating a cross-shaped decision boundary.

    Args:
        n_samples: Number of samples to generate. Defaults to 1000.
        a: Threshold parameter controlling the cross width. Defaults to 0.3.
        n_irrelevant: Number of irrelevant noise features to include. Defaults to 2.
        random_state: Seed for reproducibility. Defaults to None.

    Returns:
        A tuple (X, y) where X is a DataFrame with n_irrelevant+2 features
        and y is a Series with binary labels.

    Example:
        >>> from shapiq_games.datasets import load_cross
        >>> x_data, y_data = load_cross(n_samples=500, a=0.2)
        >>> print(x_data.shape, y_data.shape)
        ((500, 4), (500,))

    """
    rng = np.random.default_rng(random_state)
    x1 = rng.uniform(-1, 1, n_samples)
    x2 = rng.uniform(-1, 1, n_samples)
    y = ((np.abs(x1) < a) ^ (np.abs(x2) < a)).astype(int)
    data = {"x1": x1, "x2": x2}
    for i in range(1, n_irrelevant + 1):
        data[f"irr{i}"] = rng.uniform(-1, 1, n_samples)
    return pd.DataFrame(data), pd.Series(y, name="y")


def load_chess(
    n_samples: int = 1000,
    m: int = 8,
    n_irrelevant: int = 2,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the chessboard synthetic dataset.

    Generates a binary classification problem with a checkerboard pattern.
    The decision boundary is based on whether (row + col) is even or odd,
    where row and col are derived from the input features.

    Args:
        n_samples: Number of samples to generate. Defaults to 1000.
        m: Size of the chessboard grid (m x m). Defaults to 8.
        n_irrelevant: Number of irrelevant noise features to include. Defaults to 2.
        random_state: Seed for reproducibility. Defaults to None.

    Returns:
        A tuple (X, y) where X is a DataFrame with n_irrelevant+2 features
        and y is a Series with binary labels.

    Example:
        >>> from shapiq_games.datasets import load_chess
        >>> x_data, y_data = load_chess(n_samples=500, m=4)
        >>> print(x_data.shape, y_data.shape)
        ((500, 4), (500,))

    """
    rng = np.random.default_rng(random_state)
    x1 = rng.uniform(0, 1, n_samples)
    x2 = rng.uniform(0, 1, n_samples)
    row = (x1 * m).astype(int).clip(0, m - 1)
    col = (x2 * m).astype(int).clip(0, m - 1)
    y = ((row + col) % 2).astype(int)
    data = {"x1": x1, "x2": x2}
    for i in range(1, n_irrelevant + 1):
        data[f"irr{i}"] = rng.uniform(0, 1, n_samples)
    return pd.DataFrame(data), pd.Series(y, name="y")


def load_sphere(
    n_samples: int = 1000,
    radius: float = 0.7,
    n_irrelevant: int = 2,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the sphere synthetic dataset.

    Generates a binary classification problem where samples inside a sphere
    are labeled as 1 and samples outside are labeled as 0.

    Args:
        n_samples: Number of samples to generate. Defaults to 1000.
        radius: Radius of the sphere. Defaults to 0.7.
        n_irrelevant: Number of irrelevant noise features to include. Defaults to 2.
        random_state: Seed for reproducibility. Defaults to None.

    Returns:
        A tuple (X, y) where X is a DataFrame with n_irrelevant+2 features
        and y is a Series with binary labels.

    Example:
        >>> from shapiq_games.datasets import load_sphere
        >>> x_data, y_data = load_sphere(n_samples=500, radius=0.5)
        >>> print(x_data.shape, y_data.shape)
        ((500, 4), (500,))

    """
    rng = np.random.default_rng(random_state)
    x1 = rng.uniform(-1, 1, n_samples)
    x2 = rng.uniform(-1, 1, n_samples)
    y = (x1**2 + x2**2 <= radius**2).astype(int)
    data = {"x1": x1, "x2": x2}
    for i in range(1, n_irrelevant + 1):
        data[f"irr{i}"] = rng.uniform(-1, 1, n_samples)
    return pd.DataFrame(data), pd.Series(y, name="y")


def load_disjunct(
    n_samples: int = 1000,
    thresholds: list[float] | None = None,
    n_irrelevant: int = 2,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load the disjunctive synthetic dataset.

    Generates a binary classification problem where the target is True if any
    of the relevant features exceeds its corresponding threshold, creating
    a disjunctive (OR) decision boundary.

    Args:
        n_samples: Number of samples to generate. Defaults to 1000.
        thresholds: Threshold values for each relevant feature. If None, defaults
            to [0.8, 0.8, 0.8]. Defaults to None.
        n_irrelevant: Number of irrelevant noise features to include. Defaults to 2.
        random_state: Seed for reproducibility. Defaults to None.

    Returns:
        A tuple (X, y) where X is a DataFrame with (len(thresholds) + n_irrelevant)
        features and y is a Series with binary labels.

    Example:
        >>> from shapiq_games.datasets import load_disjunct
        >>> x_data, y_data = load_disjunct(n_samples=500, thresholds=[0.5, 0.5])
        >>> print(x_data.shape, y_data.shape)
        ((500, 4), (500,))

    """
    rng = np.random.default_rng(random_state)
    if thresholds is None:
        thresholds = [0.8, 0.8, 0.8]
    rel = {f"x{i + 1}": rng.uniform(0, 1, n_samples) for i in range(len(thresholds))}
    cond = np.zeros(n_samples, dtype=bool)
    for i, t in enumerate(thresholds):
        cond |= rel[f"x{i + 1}"] > t
    y = cond.astype(int)
    data = dict(rel)
    for i in range(1, n_irrelevant + 1):
        data[f"irr{i}"] = rng.uniform(0, 1, n_samples)
    return pd.DataFrame(data), pd.Series(y, name="y")


def load_random(
    n_samples: int = 1000,
    n_features: int = 5,
    p: float = 0.5,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Load a random synthetic dataset.

    Generates a binary classification problem with random features and random
    binary targets. All features are uniform random in [0, 1].

    Args:
        n_samples: Number of samples to generate. Defaults to 1000.
        n_features: Number of features to generate. Defaults to 5.
        p: Probability of target being 1 (Bernoulli parameter). Defaults to 0.5.
        random_state: Seed for reproducibility. Defaults to None.

    Returns:
        A tuple (X, y) where X is a DataFrame with n_features features
        and y is a Series with binary labels.

    Example:
        >>> from shapiq_games.datasets import load_random
        >>> x_data, y_data = load_random(n_samples=500, n_features=10)
        >>> print(x_data.shape, y_data.shape)
        ((500, 10), (500,))

    """
    rng = np.random.default_rng(random_state)
    data = {f"x{i + 1}": rng.uniform(0, 1, n_samples) for i in range(n_features)}
    y = rng.binomial(1, p, n_samples)
    return pd.DataFrame(data), pd.Series(y, name="y")


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers for UCI CSV loaders
# ─────────────────────────────────────────────────────────────────────────────


def _fetch(dataset_id: int) -> tuple[pd.DataFrame, pd.Series]:
    """Fetch a UCI dataset by ID via ucimlrepo; return (X, y) as DataFrame/Series.

    Args:
        dataset_id: The UCI Machine Learning Repository dataset ID.

    Returns:
        A tuple (X, y) where X is a DataFrame with features and y is a Series
        with targets.

    Raises:
        ImportError: If the 'ucimlrepo' package is not installed.

    """
    if _fetch_ucirepo is None:
        msg = "The 'ucimlrepo' package is required. Install it with: pip install ucimlrepo"
        raise ImportError(msg)
    ds = _fetch_ucirepo(id=dataset_id)
    X: pd.DataFrame = ds.data.features.copy()
    y: pd.Series = ds.data.targets.squeeze().rename("target")
    return X, y


def _impute(X: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values in a DataFrame.

    For numeric columns, uses median imputation. For object columns, uses mode
    imputation. Modifies the DataFrame in-place and returns it.

    Args:
        X: Input DataFrame with potential missing values.

    Returns:
        The DataFrame with missing values imputed.

    """
    for col in X.columns:
        if X[col].isna().any():
            if X[col].dtype == object:
                X[col] = X[col].fillna(X[col].mode()[0])
            else:
                X[col] = X[col].fillna(X[col].median())
    return X


def _encode_categorical(X: pd.DataFrame) -> pd.DataFrame:
    """Ordinal-encode all object/category columns in a DataFrame.

    Args:
        X: Input DataFrame with potential categorical columns.

    Returns:
        A new DataFrame with all object and category columns ordinal-encoded.
        Unknown categories are encoded as -1.

    """
    from sklearn.preprocessing import OrdinalEncoder

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) == 0:
        return X
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X = X.copy()
    X[cat_cols] = encoder.fit_transform(X[cat_cols])
    return X


def load_annealing() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Annealing dataset from the UCI Machine Learning Repository.

    Original source: https://uci-ics-mlr-prod.aws.uci.edu/dataset/3/annealing/files

    Returns:
        The Annealing dataset as pandas objects ``(X, y)``.

    Notes:
        The dataset contains 898 samples with 38 features and 6 classes.
        Mix of continuous and categorical features. The '-' values throughout
        mean "not applicable" and are treated as a valid category, not NaN.
        Categorical features are ordinal-encoded; target is label-encoded to 0-based integers.

    Example:
        >>> from shapiq_games.datasets import load_annealing
        >>> x_data, y_data = load_annealing()
        >>> print(x_data.shape, y_data.shape)
        ((898, 38), (898,))

    """
    data = _try_load("annealing.csv")
    X = data.drop(columns=["target"])
    y = data["target"]
    X = _impute(X)
    X = _encode_categorical(X)
    y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name="target")
    return X, y


def load_arrhythmia() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Arrhythmia dataset from the UCI Machine Learning Repository.

    Original source: https://archive.ics.uci.edu/dataset/5/arrhythmia

    Returns:
        The Arrhythmia dataset as pandas objects ``(X, y)``.

    Notes:
        The dataset contains 452 samples with 279 features and 16 classes.
        Includes 206 linear-valued and 73 nominal features with ~5% missing values.
        Preprocessing: impute NaN with median (numeric) / mode (categorical).

    Example:
        >>> from shapiq_games.datasets import load_arrhythmia
        >>> x_data, y_data = load_arrhythmia()
        >>> print(x_data.shape, y_data.shape)
        ((452, 279), (452,))

    """
    data = _try_load("arrhythmia.csv")
    X = data.drop(columns=["target"])
    y = data["target"]
    X = _impute(X)
    X = _encode_categorical(X)
    y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name="target")
    return X, y


def load_hepatitis() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Hepatitis dataset from the UCI Machine Learning Repository.

    Original source: https://archive.ics.uci.edu/dataset/46/hepatitis

    Returns:
        The Hepatitis dataset as pandas objects ``(X, y)``.

    Notes:
        The dataset contains 155 samples with 19 features and 2 classes (DIE/LIVE).
        Includes 13 binary categorical features and 6 continuous features with
        significant missing values (~6% overall, up to ~43% in PROTIME).
        Preprocessing: impute NaN with mean (continuous) / mode (binary).

    Example:
        >>> from shapiq_games.datasets import load_hepatitis
        >>> x_data, y_data = load_hepatitis()
        >>> print(x_data.shape, y_data.shape)
        ((155, 19), (155,))

    """
    data = _try_load("hepatitis.csv")
    X = data.drop(columns=["target"])
    y = data["target"]
    X = _impute(X)
    X = _encode_categorical(X)
    y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name="target")
    return X, y


def load_ionosphere() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Ionosphere dataset from the UCI Machine Learning Repository.

    Original source: https://archive.ics.uci.edu/dataset/52/ionosphere

    Returns:
        The Ionosphere dataset as pandas objects ``(X, y)``.

    Notes:
        The dataset contains 351 samples with 34 features and 2 classes.
        All continuous radar return features. Constant columns are dropped.
        Preprocessing: drop constant columns; target label-encoded to 0-based integers.

    Example:
        >>> from shapiq_games.datasets import load_ionosphere
        >>> x_data, y_data = load_ionosphere()
        >>> print(x_data.shape, y_data.shape)
        ((351, 34), (351,))

    """
    data = _try_load("ionosphere.csv")
    X = data.drop(columns=["target"])
    y = data["target"]
    X = X.loc[:, ~(X.iloc[0] == X).all()]  # drop constant columns
    y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name="target")
    return X, y


def load_mushroom() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Mushroom dataset from the UCI Machine Learning Repository.

    Original source: https://archive.ics.uci.edu/dataset/73/mushroom

    Returns:
        The Mushroom dataset as pandas objects ``(X, y)``.

    Notes:
        The dataset contains 8,124 samples with 22 features and 2 classes (edible/poisonous).
        All categorical string features are ordinal-encoded; target label-encoded to 0-based integers.

    Example:
        >>> from shapiq_games.datasets import load_mushroom
        >>> x_data, y_data = load_mushroom()
        >>> print(x_data.shape, y_data.shape)
        ((8124, 22), (8124,))

    """
    data = _try_load("mushroom.csv")
    X = data.drop(columns=["target"])
    y = data["target"]
    X = _encode_categorical(X)
    y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name="target")
    return X, y


def load_nursery() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Nursery dataset from the UCI Machine Learning Repository.

    Original source: https://archive.ics.uci.edu/dataset/76/nursery

    Returns:
        The Nursery dataset as pandas objects ``(X, y)``.

    Notes:
        The dataset contains 12,960 samples with 8 features and 5 classes.
        All categorical string features are ordinal-encoded; target label-encoded to 0-based integers.

    Example:
        >>> from shapiq_games.datasets import load_nursery
        >>> x_data, y_data = load_nursery()
        >>> print(x_data.shape, y_data.shape)
        ((12960, 8), (12960,))

    """
    data = _try_load("nursery.csv")
    X = data.drop(columns=["target"])
    y = data["target"]
    X = _encode_categorical(X)
    y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name="target")
    return X, y


def load_soybean() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Soybean Large dataset from the UCI Machine Learning Repository.

    Original source: https://archive.ics.uci.edu/dataset/90/soybean+large

    Returns:
        The Soybean Large dataset as pandas objects ``(X, y)``.

    Notes:
        The dataset contains 683 samples with 35 features and 19 classes.
        Missing values imputed with mode; categorical features ordinal-encoded;
        target label-encoded to 0-based integers.

    Example:
        >>> from shapiq_games.datasets import load_soybean
        >>> x_data, y_data = load_soybean()
        >>> print(x_data.shape, y_data.shape)
        ((683, 35), (683,))

    """
    data = _try_load("soybean.csv")
    X = data.drop(columns=["target"])
    y = data["target"]
    X = _impute(X)
    X = _encode_categorical(X)
    y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name="target")
    return X, y


def load_thyroid() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Thyroid Disease (ann) dataset from the UCI Machine Learning Repository.

    Original source: https://archive.ics.uci.edu/dataset/102/thyroid+disease

    Returns:
        The Thyroid Disease dataset as pandas objects ``(X, y)``.

    Notes:
        The dataset contains 7,200 samples (train + test combined) with 21 features and 3 classes.
        All features are numeric; target label-encoded to 0-based integers.

    Example:
        >>> from shapiq_games.datasets import load_thyroid
        >>> x_data, y_data = load_thyroid()
        >>> print(x_data.shape, y_data.shape)
        ((7200, 21), (7200,))

    """
    data = _try_load("thyroid.csv")
    X = data.drop(columns=["target"])
    y = data["target"]
    y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name="target")
    return X, y


def load_zoo() -> tuple[pd.DataFrame, pd.Series]:
    """Load the Zoo dataset from the UCI Machine Learning Repository.

    Original source: https://archive.ics.uci.edu/dataset/111/zoo

    Returns:
        The Zoo dataset as pandas objects ``(X, y)``.

    Notes:
        The dataset contains 101 samples with 16 features and 7 classes.
        All binary/integer features; target label-encoded to 0-based integers.

    Example:
        >>> from shapiq_games.datasets import load_zoo
        >>> x_data, y_data = load_zoo()
        >>> print(x_data.shape, y_data.shape)
        ((101, 16), (101,))

    """
    data = _try_load("zoo.csv")
    X = data.drop(columns=["target"])
    y = data["target"]
    y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name="target")
    return X, y
