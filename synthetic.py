"""This script conducts a synthetic experiment for the unified framework."""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures

from explanation_game import LocalExplanationGame, MultiDataExplanationGame, loss_mse


# Function to generate the multivariate normal data
def generate_data(num_samples, rho):
    mu = np.zeros(4)
    sigma = np.full((4, 4), rho)
    np.fill_diagonal(sigma, 1)
    X = RNG.multivariate_normal(mu, sigma, size=num_samples)
    return X


# Linear main effect model
def linear_function(X):
    return 2 * X[:, 0] + 2 * X[:, 1]


# Interaction model
def interaction_function(X):
    return linear_function(X) + X[:, 1] * X[:, 2]


# Add Gaussian noise
def add_noise(Y, noise_std=0.1):  # Reduced noise for testing
    noise = RNG.normal(0, noise_std, Y.shape)
    return Y + noise


def make_df(X):
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    return df


def _select_interaction_features(X):
    """Returns X_1, X_2 and X_1*X_3. Expects input of shape (n_samples, 10)"""
    return X[:, [0, 1, 7]]


def _select_linear_features(X):
    """Returns X_1 and X_2. Expects input of shape (n_samples, 10)"""
    return X[:, [0, 1]]


if __name__ == "__main__":

    game_storage_path = "game_storage"
    os.makedirs(game_storage_path, exist_ok=True)

    RANDOM_SEED = 42
    RNG = np.random.default_rng(RANDOM_SEED)

    # Experiment settings
    num_samples = 10_000
    rho_values = [0, 0.5, 0.9]
    results = {}

    # Initialize Interaction Models
    lin_reg_int = Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("select", FunctionTransformer(_select_interaction_features)),
            ("lin_reg", LinearRegression()),
        ]
    )

    for rho in rho_values:
        X = generate_data(num_samples, rho)

        # Generate targets (Y) for both models
        Y_lin = add_noise(linear_function(X))
        Y_int = add_noise(interaction_function(X))

        # Fit models on interaction data
        lin_reg_int.fit(X, Y_int)
        Y_int_pred_lin = lin_reg_int.predict(X)
        lin_reg_int_mse = mean_squared_error(Y_int, Y_int_pred_lin)
        lin_reg_int_r2 = r2_score(Y_int, Y_int_pred_lin)
        print(f"Interaction model with rho={rho}")
        print(f"MSE: {lin_reg_int_mse}")
        print(f"R^2: {lin_reg_int_r2}")

        # make a local explanation game
        local_game = LocalExplanationGame(
            fanova="b",
            model=lin_reg_int,
            x_data=X,
            x_explain=X[0],
            y_explain=Y_int[0],
            loss_function=None,
            sample_size=100,
            random_seed=RANDOM_SEED,
            normalize=False,
            verbose=True,
        )

        # pre-compute the game values
        local_game.precompute()
        local_game.save_values(os.path.join(game_storage_path, f"local_game_rho_{rho}.json"))

        # make a global explanation game
        global_game = MultiDataExplanationGame(
            sensitivity_game=False,  # global game
            fanova="b",
            model=lin_reg_int,
            X=X,
            y=Y_int,
            loss_function=loss_mse,
            sample_size=100,
            n_samples=100,
            random_seed=RANDOM_SEED,
            normalize=False,
            verbose=True,
        )

        # pre-compute the game values
        global_game.precompute()
        global_game.save_values(os.path.join(game_storage_path, f"global_game_rho_{rho}.json"))

        # make a sensitivity game
        sensitivity_game = MultiDataExplanationGame(
            sensitivity_game=True,
            fanova="b",
            model=lin_reg_int,
            X=X,
            y=Y_int,
            loss_function=loss_mse,
            sample_size=100,
            n_samples=100,
            random_seed=RANDOM_SEED,
            normalize=False,
            verbose=True,
        )

        # pre-compute the game values
        sensitivity_game.precompute()
        sensitivity_game.save_values(
            os.path.join(game_storage_path, f"sensitivity_game_rho_{rho}.json")
        )
