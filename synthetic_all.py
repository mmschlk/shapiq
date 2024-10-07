"""This script conducts a synthetic experiment for the unified framework."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures
from xgboost import XGBRegressor


# Function to generate the multivariate normal data
def generate_data(num_samples, rho):
    mu = np.zeros(4)
    sigma = np.full((4, 4), rho)
    np.fill_diagonal(sigma, 1)
    X = np.random.multivariate_normal(mu, sigma, size=num_samples)
    return X


# Linear main effect model
def linear_function(X):
    return 2 * X[:, 0] + 2 * X[:, 1]


# Interaction model
def interaction_function(X):
    return linear_function(X) + X[:, 1] * X[:, 2]


# Add Gaussian noise
def add_noise(Y, noise_std=0.1):  # Reduced noise for testing
    noise = np.random.normal(0, noise_std, Y.shape)
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

    RANDOM_SEED = 42

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
    interaction_constraints_int = [["f1"], ["f2"], ["f2", "f3"]]
    xgboost_int = XGBRegressor(interaction_constraints=interaction_constraints_int)
    rnf_reg_int = RandomForestRegressor()

    # Initialize Linear Models
    lin_reg_lin = Pipeline(
        [("select", FunctionTransformer(_select_linear_features)), ("lin_reg", LinearRegression())]
    )
    interaction_constraints_lin = [["f1"], ["f2"]]
    xgboost_lin = XGBRegressor(interaction_constraints=interaction_constraints_lin)
    rnf_reg_lin = RandomForestRegressor()

    for rho in rho_values:
        X = generate_data(num_samples, rho)

        # Generate targets (Y) for both models
        Y_lin = add_noise(linear_function(X))
        Y_int = add_noise(interaction_function(X))

        X_df = make_df(X)

        # Fit models on linear data
        lin_reg_lin.fit(X, Y_lin)
        xgboost_lin.fit(X_df, Y_lin)
        rnf_reg_lin.fit(X, Y_lin)

        # Fit models on interaction data
        lin_reg_int.fit(X, Y_int)
        xgboost_int.fit(X_df, Y_int)
        rnf_reg_int.fit(X, Y_int)

        # Predict on the same data
        Y_lin_pred_lin = lin_reg_lin.predict(X)
        Y_lin_pred_xgb = xgboost_lin.predict(X)
        Y_lin_pred_rnf = rnf_reg_lin.predict(X)

        Y_int_pred_lin = lin_reg_int.predict(X)
        Y_int_pred_xgb = xgboost_int.predict(X)
        Y_int_pred_rnf = rnf_reg_int.predict(X)

        # Calculate MSE and R2
        mse_lin_lin = mean_squared_error(Y_lin, Y_lin_pred_lin)
        mse_lin_xgb = mean_squared_error(Y_lin, Y_lin_pred_xgb)
        mse_lin_rnf = mean_squared_error(Y_lin, Y_lin_pred_rnf)

        mse_int_lin = mean_squared_error(Y_int, Y_int_pred_lin)
        mse_int_xgb = mean_squared_error(Y_int, Y_int_pred_xgb)
        mse_int_rnf = mean_squared_error(Y_int, Y_int_pred_rnf)

        r2_lin_lin = r2_score(Y_lin, Y_lin_pred_lin)
        r2_lin_xgb = r2_score(Y_lin, Y_lin_pred_xgb)
        r2_lin_rnf = r2_score(Y_lin, Y_lin_pred_rnf)

        r2_int_lin = r2_score(Y_int, Y_int_pred_lin)
        r2_int_xgb = r2_score(Y_int, Y_int_pred_xgb)
        r2_int_rnf = r2_score(Y_int, Y_int_pred_rnf)

        results[rho] = {
            "mse_lin_lin": mse_lin_lin,
            "mse_lin_xgb": mse_lin_xgb,
            "mse_lin_rnf": mse_lin_rnf,
            "mse_int_lin": mse_int_lin,
            "mse_int_xgb": mse_int_xgb,
            "mse_int_rnf": mse_int_rnf,
            "r2_lin_lin": r2_lin_lin,
            "r2_lin_xgb": r2_lin_xgb,
            "r2_lin_rnf": r2_lin_rnf,
            "r2_int_lin": r2_int_lin,
            "r2_int_xgb": r2_int_xgb,
            "r2_int_rnf": r2_int_rnf,
        }

    for rho, res in results.items():
        print(f"Results for rho={rho}")
        print("Linear Model on Linear Data:")
        print(f"MSE: {res['mse_lin_lin']}")
        print(f"R2: {res['r2_lin_lin']}")
        print("Linear Model on Interaction Data:")
        print(f"MSE: {res['mse_int_lin']}")
        print(f"R2: {res['r2_int_lin']}")
        print("XGBoost on Linear Data:")
        print(f"MSE: {res['mse_lin_xgb']}")
        print(f"R2: {res['r2_lin_xgb']}")
        print("XGBoost on Interaction Data:")
        print(f"MSE: {res['mse_int_xgb']}")
        print(f"R2: {res['r2_int_xgb']}")
        print("Random Forest on Linear Data:")
        print(f"MSE: {res['mse_lin_rnf']}")
        print(f"R2: {res['r2_lin_rnf']}")
        print("Random Forest on Interaction Data:")
        print(f"MSE: {res['mse_int_rnf']}")
        print(f"R2: {res['r2_int_rnf']}")
        print()
