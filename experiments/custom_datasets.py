import scipy.special
import shap
import numpy as np
import scipy
import itertools
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import fetch_openml


def synthetic(num_features=15):
    binary = np.zeros((2 ** num_features - 2, num_features))
    idx = 0
    for s in range(1, num_features):
        for indices in itertools.combinations(range(num_features), s):
            binary[idx, list(indices)] = 1
            idx += 1
    num_ones = np.sum(binary, axis=1)
    inv_weights = num_ones * (num_features - num_ones) * scipy.special.binom(num_features, num_ones)
    weights = 1 / inv_weights
    Z = binary * weights[:, np.newaxis]  # each row is w(||z||_1) z^T
    P = np.eye(num_features) - 1 / num_features  # projection matrix to remove all ones component
    A = Z @ P  # each row is w(||z||_1) z^T P
    xstar = np.random.randn(num_features)
    ystar = A @ xstar
    weight_prob = weights / np.sum(weights)
    leverage = 1 / scipy.special.binom(num_features, num_ones)
    leverage_prob = leverage / np.sum(leverage)
    leverage_smaller = leverage_prob < weight_prob
    # Add noise
    noise = np.random.randn(2 ** num_features - 2) * leverage_smaller
    # Convert to pandas dataframe
    X = pd.DataFrame(binary, columns=[f'Feature {i}' for i in range(num_features)])
    y = pd.Series(ystar + noise, name='Target')
    return X, y


def load_bike_sharing():
    data = fetch_openml(name="Bike_Sharing_Demand", version=2, as_frame=True)

    X = data.data
    y = data.target.astype(float)

    # Columns to treat as categorical
    categorical_cols = ['season', 'holiday', 'workingday', 'weather']

    # One-hot encode categorical columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X, y


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


import pandas as pd

import pandas as pd


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


dataset_loaders = {
    #    'IRIS' : shap.datasets.iris, # 4
    #    'California' : shap.datasets.california, #8
    #    'Diabetes' : shap.datasets.diabetes, # 10

    #    'Wine Quality' : load_wine_quality, # 12
    'Adult': shap.datasets.adult,  # 12
    'Forest Fires': load_forest_fires,  # 13
    'Real Estate': load_real_estate,  # 15
    'Bike Sharing': load_bike_sharing,  # 16
    'Breast Cancer': breast_cancer,  # 30
    #    'Correlated' : shap.datasets.corrgroups60, # 60
    'Independent': shap.datasets.independentlinear60,  # 60
    'NHANES': shap.datasets.nhanesi,  # 79
    'Communities': shap.datasets.communitiesandcrime,  # 101
}

dataset_sizes = {
    'IRIS': 4,
    'California': 8,
    'Diabetes': 10,
    'Wine Quality': 12,
    'Adult': 12,
    'Forest Fires': 13,
    'Real Estate': 15,
    'Bike Sharing': 16,
    'Breast Cancer': 30,
    'Correlated': 60,
    'Independent': 60,
    'NHANES': 79,
    'Communities': 101
}


def load_dataset(dataset_name):
    X, y = dataset_loaders[dataset_name]()
    # Remove nan values
    X = X.fillna(X.mean())
    return X, y


def load_input(X, num_runs, is_synthetic=False):
    if is_synthetic:
        baseline = np.zeros((1, X.shape[1]))
        explicand = np.ones((1, X.shape[1]))
        return baseline, explicand

    baseline = X.mean().values.reshape(1, -1)

    explicands = []
    for run_idx in range(num_runs):
        seed = run_idx * num_runs
        np.random.seed(seed)

        explicand_idx = np.random.choice(X.shape[0])
        explicand = X.iloc[explicand_idx].values.reshape(1, -1)
        for i in range(explicand.shape[1]):
            while baseline[0, i] == explicand[0, i]:
                explicand_idx = np.random.choice(X.shape[0])
                explicand[0, i] = X.iloc[explicand_idx, i]
        explicands.append(explicand)

    return baseline, explicands