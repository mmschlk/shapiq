"""This module contains utility functions for the synthetic experiments."""

import copy
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor

from framework_train_neural_nets import CaliforniaScikitWrapper
from shapiq import Game
from shapiq.datasets import load_bike_sharing, load_california_housing, load_titanic


class SynthConditionalSampler:

    def __init__(
        self, sample_size: int = 128, rho: float = 0.0, random_seed: int = 42, n_features: int = 4
    ):
        self.sample_size = sample_size
        self.n_features = n_features
        self.rho = rho
        self.random_seed = random_seed
        self.mu = np.zeros(n_features)
        self.sigma = np.full((n_features, n_features), rho)
        np.fill_diagonal(self.sigma, 1)

    def __call__(self, coalitions: np.ndarray[bool], x_to_impute: np.ndarray) -> np.ndarray:
        """Evaluate the conditional distribution for the given coalitions.

        Args:
            coalitions: A boolean array indicating which features are present (``True``) and which
                are missing (``False``). The shape of the array must be ``(n_subsets, n_features)``.
        """
        # check if the coalitions are 2 dimensional
        if coalitions.ndim != 2:
            coalitions = coalitions.reshape(1, -1)
        n_coalitions = coalitions.shape[0]
        replacement_data = np.zeros((self.sample_size, n_coalitions, self.n_features))
        for i in range(n_coalitions):
            coalition = coalitions[i]
            if sum(coalition) == 0:
                # sample from the original distribution
                rng = np.random.default_rng(self.random_seed)
                replacement_data[:, i] = rng.multivariate_normal(
                    self.mu, self.sigma, size=self.sample_size
                )
                continue
            if sum(coalition) == self.n_features:
                # all features are present
                replacement_data[:, i] = np.tile(x_to_impute, (self.sample_size, 1))
                continue
            conditioned_values = {idx: val for idx, val in enumerate(x_to_impute) if coalition[idx]}
            values = self.sample(conditioned_values, self.sample_size)
            replacement_data[:, i, ~coalition] = values
        return replacement_data

    def sample(self, conditioned_values: dict[int:float], n_samples: int) -> np.ndarray:
        """Sample from the conditional distribution.

        Args:
            conditioned_values: The conditioned values in a dictionary.
            n_samples: The number of samples to draw.

        Returns:
            The samples drawn from the conditional distribution.
        """

        # set the random seed
        rng = np.random.default_rng(self.random_seed)

        # get the indices of the conditioned and free variables
        conditioned_idx = np.array(list(conditioned_values.keys()))
        free_idx = np.array([i for i in range(self.n_features) if i not in conditioned_idx])

        # Partition the mean vector
        mu_A = copy.deepcopy(self.mu[free_idx])  # mean of the free variables
        mu_B = copy.deepcopy(self.mu[conditioned_idx])  # mean of the conditioned variables

        # Partition the covariance matrix
        sigma_AA = copy.deepcopy(self.sigma[np.ix_(free_idx, free_idx)])  # cov. of the free vars.
        sigma_AB = copy.deepcopy(
            self.sigma[np.ix_(free_idx, conditioned_idx)]
        )  # cov.: free and cond. vars.
        sigma_BB = copy.deepcopy(
            self.sigma[np.ix_(conditioned_idx, conditioned_idx)]
        )  # cov.: of cond. vars

        # condition values (X_B = x_B_star)
        x_B_star = copy.deepcopy(np.array([conditioned_values[i] for i in conditioned_idx]))

        # compute conditional mean and covariance
        sigma_BB_inv = np.linalg.inv(sigma_BB)  # inverse of the covariance of the conditioned vars
        mu_conditional = mu_A + sigma_AB @ sigma_BB_inv @ (x_B_star - mu_B)
        sigma_conditional = sigma_AA - sigma_AB @ sigma_BB_inv @ sigma_AB.T

        # sample from the conditional distribution
        samples = rng.multivariate_normal(mu_conditional, sigma_conditional, size=n_samples)
        return samples


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


def non_linear_interaction_function(X):
    x_two = X[:, 1] ** 2
    x_tree = X[:, 2] ** 3
    return 2 * X[:, 0] + x_two + x_tree


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
    x_two_order = X[:, 0] * X[:, 1]
    x_third_order = X[:, 0] * X[:, 1] * X[:, 2]
    return np.column_stack((X[:, 0], X[:, 1], X[:, 2], x_two_order, x_third_order))


def _select_non_linear_features(X):
    """Returns X_1, X_2^2 and X_3^3."""
    x_two = X[:, 1] ** 2
    x_tree = X[:, 2] ** 3
    return np.column_stack((X[:, 0], x_two, x_tree))


def _select_linear_features(X):
    """Returns X_1 and X_2. Expects input of shape (n_samples, 4)"""
    return X[:, [0, 1, 2]]


def update_results(
    _results: list,
    _explanation: dict[tuple[int, ...], float],
    _game_id: int,
    _feature_influence: str,
    _fanova_setting: str,
    _entity: str,
    _x_explain: np.ndarray,
) -> None:
    for _feature_set, _exp_val in _explanation.items():
        if len(_feature_set) == 1:
            _x_val = float(_x_explain[_feature_set])
        else:
            _x_val = float(np.prod(_x_explain[list(_feature_set)]))
        _feature_set = tuple(_feature_set)
        _results.append(
            {
                "game_id": _game_id,
                "feature_set": _feature_set,
                "feature_influence": _feature_influence,
                "fanova_setting": _fanova_setting,
                "entity": _entity,
                "explanation": _exp_val,
                "feature_value": _x_val,
                "explanation/feature_value": _exp_val / _x_val if _x_val != 0 else 0,
            }
        )


def get_ml_data(
    model_name: str,
    random_seed: Optional[int] = None,
    data_name: Optional[str] = None,
    do_k_fold: bool = False,
):
    try:
        model_name = model_name.split("_")[0]
    except IndexError:
        pass
    clf = True if data_name == "titanic" else False

    # get data
    if data_name == "california":
        x_data, y_data = load_california_housing(to_numpy=False)
    elif data_name == "bike":
        x_data, y_data = load_bike_sharing(to_numpy=False)
        if model_name == "nn":
            from sklearn.preprocessing import StandardScaler

            x_data = pd.DataFrame(StandardScaler().fit_transform(x_data), columns=x_data.columns)
            y_data = np.log(y_data + 1)
    elif data_name == "titanic":
        x_data, y_data = load_titanic(to_numpy=False)
    else:
        raise ValueError(f"Unknown data name: {data_name}")
    feature_names = x_data.columns
    x_data = x_data.to_numpy()
    y_data = y_data.to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, train_size=0.7, shuffle=True, random_state=random_seed
    )

    if model_name == "nn" and data_name == "california":
        model = CaliforniaScikitWrapper(f"california_model_{random_seed}.pth")
    elif model_name == "nn" and data_name == "bike":
        model = CaliforniaScikitWrapper(f"bike_model_{random_seed}.pth")
    else:
        # get a model and train
        if model_name == "xgb":
            if clf:
                model = XGBClassifier(seed=random_seed)
            else:
                model = XGBRegressor(seed=random_seed)
        elif model_name == "rnf":
            if clf:
                model = RandomForestClassifier(random_state=random_seed, n_estimators=50)
            else:
                model = RandomForestRegressor(random_state=random_seed, n_estimators=50)
        elif model_name == "dt":
            if clf:
                model = DecisionTreeClassifier(random_state=random_seed)
            else:
                model = DecisionTreeRegressor(random_state=random_seed)
        else:
            raise ValueError(
                f"Unknown model name for california housing: {model_name} and {data_name}"
            )

        if do_k_fold:
            first_perf, second_perf = [], []
            for i in range(5):
                x_train, x_test, y_train, y_test = train_test_split(
                    x_data, y_data, train_size=0.7, shuffle=True, random_state=i
                )
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                if clf:
                    acc = accuracy_score(y_test, y_pred)
                    f_one = f1_score(y_test, y_pred)
                    print(f"Accuracy: {acc} F1: {f_one}")
                    first_perf.append(acc), second_perf.append(f_one)
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    print(f"MSE: {mse} R^2: {r2}")
                    first_perf.append(mse), second_perf.append(r2)
            print(f"Mean performance: {np.mean(first_perf)} and {np.mean(second_perf)}")
            print(f"Std performance: {np.std(first_perf)} and {np.std(second_perf)}")

        model.fit(x_train, y_train)

    # predict the data and print performance

    y_pred = model.predict(x_test)
    print(f"Model: {model_name}")
    if not clf:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"MSE: {mse} R^2: {r2}")
    else:
        acc = accuracy_score(y_test, y_pred)
        f_one = f1_score(y_test, y_pred)
        print(f"Accuracy: {acc} F1: {f_one}")

    return model, x_data, y_data, x_train, x_test, y_train, y_test, feature_names


def get_y_synth(
    x_data: np.ndarray, interaction_data: Optional[str] = None, random_seed: Optional[int] = None
) -> np.ndarray:
    if interaction_data == "linear-interaction":
        return add_noise(interaction_function(x_data), random_seed=random_seed)
    elif interaction_data == "non-linear-interaction":
        return add_noise(non_linear_interaction_function(x_data), random_seed=random_seed)
    else:
        return add_noise(linear_function(x_data), random_seed=random_seed)


def get_synth_data_and_model(
    num_samples: int = 10_000,
    rho: float = 0.0,
    interaction_data: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, Pipeline]:
    if interaction_data not in [None, "linear-interaction", "non-linear-interaction"]:
        raise ValueError(f"Unknown interaction data: {interaction_data}")

    # generate the data
    x_data = generate_data(num_samples, rho, random_seed, load_data=True)
    x_test = generate_data(num_samples, rho, random_seed + 1, load_data=True)
    y_data = get_y_synth(x_data, interaction_data, random_seed)
    y_test = get_y_synth(x_test, interaction_data, random_seed + 1)

    # get the model
    if interaction_data == "linear-interaction":  # model for interactions
        model = Pipeline(
            [
                ("select", FunctionTransformer(_select_interaction_features)),
                ("lin_reg", LinearRegression()),
            ]
        )
    elif interaction_data == "non-linear-interaction":  # model for non-linear interactions
        model = Pipeline(
            [
                ("select", FunctionTransformer(_select_non_linear_features)),
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

    # train the model
    model.fit(x_data, y_data)

    # predict the data and print performance
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # print the summary
    print(f"Linear Model with Interaction: {interaction_data}")
    print(f"MSE: {mse} R^2: {r2}")
    print(f"Intercept: {model.named_steps['lin_reg'].intercept_}")
    print(f"Coefficients: {model.named_steps['lin_reg'].coef_}")
    print()

    return x_data, y_data, model


def get_save_name_ml(
    model_name: str,
    random_seed: int,
    fanova: str,
    instance_id: int,
    data_name: str,
) -> str:
    return "_".join([data_name, model_name, str(random_seed), fanova, str(instance_id)])


def get_save_name_synth(
    interaction_data: Optional[str],
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
    _int_name = "lin" if interaction_data is None else interaction_data
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


def load_local_games_ml(
    model_name: str, fanova_setting: str, n_instances: int, random_seed: int, data_name: str
) -> list[Game]:
    """Loads a list of local games from disk."""
    game_type = os.path.join("local", data_name)
    game_storage_path = get_storage_dir(model_name, game_type=game_type)
    games = []
    for idx in range(n_instances):
        name = get_save_name_ml(
            model_name=model_name,
            random_seed=random_seed,
            fanova=fanova_setting,
            instance_id=idx,
            data_name=data_name,
        )
        save_path = os.path.join(game_storage_path, name) + ".npz"
        game = Game(path_to_values=save_path)
        games.append(game)
    return games


def load_local_games_synth(
    model_name: str,
    interaction_data: Optional[str],
    rho_value: float,
    fanova_setting: str,
    n_instances: int,
    random_seed: int = 42,
    num_samples: int = 10_000,
    sample_size: int = 1_000,
    data_name: Optional[str] = None,
) -> tuple[list[Game], list[np.ndarray], list[float]]:
    """Loads a list of local games from disk."""
    game_storage_path = get_storage_dir(model_name, game_type="local")
    games, x_explain, y_explain = [], [], []
    x_data = generate_data(num_samples, rho_value, random_seed, load_data=True)
    y_data = get_y_synth(x_data, interaction_data, random_seed)
    for idx in range(n_instances):
        name = get_save_name_synth(
            interaction_data,
            model_name,
            random_seed,
            num_samples,
            rho_value,
            fanova_setting,
            sample_size,
            idx,
            data_name,
        )
        save_path = os.path.join(game_storage_path, name) + ".npz"
        game = Game(path_to_values=save_path)
        games.append(game)
        x_explain.append(x_data[idx])
        y_explain.append(float(y_data[idx]))
    return games, x_explain, y_explain


if __name__ == "__main__":

    random_seed = 42

    # evaluate the synth model
    for rho_value in [0.0, 0.5, 0.9]:
        _, _, _ = get_synth_data_and_model(
            num_samples=10_000, rho=0.0, interaction_data=None, random_seed=random_seed
        )

    # do k-fold monte-carlo cross validation for all ml models
    k_folds = 5

    for model_name in ["xgb", "rnf", "dt", "nn"]:
        for data_name in ["california", "bike", "titanic"]:
            try:
                print(f"Model: {model_name}, Data: {data_name}")
                _ = get_ml_data(model_name, random_seed, data_name, True)
                print()
            except Exception as e:
                print(f"Error: {e}")  # some models might not work or are not specified
                continue
