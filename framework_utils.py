"""This module contains utility functions for the synthetic experiments."""

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from xgboost import XGBRegressor

from shapiq import Game
from shapiq.datasets import load_california_housing


# Function to generate the multivariate normal data
def generate_data(num_samples: int, rho: float, random_seed: int = 42, load_data=True):
    save_name = f"synthetic_data_{rho}_{num_samples}_{random_seed}.npz"
    if load_data and os.path.exists(os.path.join("game_storage", save_name)):
        data = np.load(os.path.join("game_storage", save_name))
        return data["X"]
    rng = np.random.default_rng(random_seed)
    mu = np.zeros(4)
    sigma = np.full((4, 4), rho)
    np.fill_diagonal(sigma, 1)
    X = rng.multivariate_normal(mu, sigma, size=num_samples)
    np.savez(os.path.join("game_storage", save_name), X=X)
    return X


# Linear main effect model
def linear_function(X):
    return 2 * X[:, 0] + 2 * X[:, 1] + 2 * X[:, 2]


# Interaction model
def interaction_function(X):
    second_order_interaction = 1 * X[:, 0] * X[:, 1]
    third_order_interaction = 1 * X[:, 0] * X[:, 1] * X[:, 2]
    return linear_function(X) + second_order_interaction + third_order_interaction


# Add Gaussian noise
def add_noise(Y, noise_std=0.1, random_seed: int = 42):
    rng = np.random.default_rng(random_seed)
    noise = rng.normal(0, noise_std, Y.shape)
    return Y + noise


def make_df(X):
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    return df


def _select_interaction_features(X):
    """Returns X_1, X_2 and X_1*X_3. Expects input of shape (n_samples, 14)"""
    return X[:, [0, 1, 2, 4, 10]]


def _select_linear_features(X):
    """Returns X_1 and X_2. Expects input of shape (n_samples, 4)"""
    return X[:, [0, 1, 2]]


def get_california_data_and_model(
    model_name: str,
    random_seed: Optional[int] = None,
):
    # get data
    x_data, y_data = load_california_housing(to_numpy=True)
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, random_state=random_seed, train_size=0.8
    )

    # get a model and train
    if model_name == "xgb_reg":
        model = XGBRegressor(random_state=random_seed)
    elif model_name == "rnf_reg":
        model = RandomForestRegressor(random_state=random_seed)
    else:
        raise ValueError(f"Unknown model name for california housing: {model_name}")
    model.fit(x_train, y_train)

    # predict the data and print performance
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # print the summary
    print(f"Model: {model_name}")
    print(f"MSE: {mse} R^2: {r2}")

    return model, x_data, y_data, x_train, x_test, y_train, y_test


def get_synth_data_and_model(
    model_name: str,
    num_samples: int = 10_000,
    rho: float = 0.0,
    interaction_data: bool = True,
    random_seed: Optional[int] = None,
):
    # generate the data
    x_data = generate_data(num_samples, rho, random_seed, load_data=True)
    if interaction_data:
        y_data = add_noise(interaction_function(x_data), random_seed=random_seed)
    else:
        y_data = add_noise(linear_function(x_data), random_seed=random_seed)

    # get the model
    if model_name == "lin_reg":
        if interaction_data:
            model = Pipeline(
                [
                    (
                        "poly",
                        PolynomialFeatures(degree=3, interaction_only=True, include_bias=False),
                    ),
                    ("select", FunctionTransformer(_select_interaction_features)),
                    ("lin_reg", LinearRegression()),
                ]
            )
        else:
            model = Pipeline(
                [
                    ("select", FunctionTransformer(_select_linear_features)),
                    ("lin_reg", LinearRegression()),
                ]
            )
    elif model_name == "rnf_reg":
        model = RandomForestRegressor(random_state=random_seed)
    elif model_name == "mlp_reg":
        model = MLPRegressor(random_state=random_seed)
    elif model_name == "xgb_reg":
        if interaction_data:
            interaction_constraints_int = [["f1"], ["f2"], ["f3"], ["f1", "f2"], ["f1", "f2", "f3"]]
        else:
            interaction_constraints_int = [["f1"], ["f2"], ["f3"]]
        model = XGBRegressor(interaction_constraints=interaction_constraints_int)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # train the model
    if model_name in ("xgb_reg", "xgb_clf"):
        x_data_df = make_df(x_data)
        model.fit(x_data_df, y_data)
    else:
        model.fit(x_data, y_data)

    # predict the data and print performance
    y_pred = model.predict(x_data)
    mse = mean_squared_error(y_data, y_pred)
    r2 = r2_score(y_data, y_pred)

    # print the summary
    print(f"Model: {model_name}")
    print(f"MSE: {mse} R^2: {r2}")
    if model_name == "lin_reg":
        print(f"Model: {model_name}")
        print(f"Intercept: {model.named_steps['lin_reg'].intercept_}")
        print(f"Coefficients: {model.named_steps['lin_reg'].coef_}")
    print()

    return x_data, y_data, model


def get_save_name(
    interaction_data: bool,
    model_name: str,
    random_seed: int,
    num_samples: int,
    rho: float,
    fanova: str,
    sample_size: int,
    instance_id: int,
    data_name: Optional[str] = None,
) -> str:
    _data_name = data_name
    if data_name is None:
        _data_name = "synthetic"
    _int_name = "int" if interaction_data else "lin"
    return "_".join(
        [
            _data_name,
            _int_name,
            model_name,
            str(random_seed),
            str(num_samples),
            str(rho),
            fanova,
            str(sample_size),
            str(instance_id),
        ]
    )


def get_storage_dir(model_name: str, game_type: str = "local"):
    path = os.path.join("game_storage", game_type, model_name)
    os.makedirs(path, exist_ok=True)
    return path


def load_local_games(
    model_name: str,
    interaction_data: bool,
    rho_value: float,
    fanova_setting: str,
    n_instances: int,
    random_seed: int = 42,
    num_samples: int = 10_000,
    sample_size: int = 1_000,
) -> tuple[list[Game], list[np.ndarray], list[float]]:
    """Loads a list of local games from disk."""
    game_storage_path = get_storage_dir(model_name)
    games, x_explain, y_explain = [], [], []
    x_data = generate_data(num_samples, rho_value, random_seed, load_data=True)
    if interaction_data:
        y_data = add_noise(interaction_function(x_data), random_seed=random_seed)
    else:
        y_data = add_noise(linear_function(x_data), random_seed=random_seed)
    for idx in range(n_instances):
        name = get_save_name(
            interaction_data,
            model_name,
            random_seed,
            num_samples,
            rho_value,
            fanova_setting,
            sample_size,
            idx,
        )
        save_path = os.path.join(game_storage_path, name) + ".npz"
        game = Game(path_to_values=save_path)
        games.append(game)
        x_explain.append(x_data[idx])
        y_explain.append(float(y_data[idx]))
    return games, x_explain, y_explain


if __name__ == "__main__":
    # generate the data
    _ = generate_data(10_000, 0.0, random_seed=42)
    _ = generate_data(10_000, 0.5, random_seed=42)
    _ = generate_data(10_000, 0.0, random_seed=42)
