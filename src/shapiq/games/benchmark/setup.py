"""This module contains a setup for the tabular benchmark games."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

# data needs to be normalized for the neural network
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shapiq.datasets import load_adult_census, load_bike_sharing, load_california_housing
from shapiq.utils import shuffle_data

if TYPE_CHECKING:
    from shapiq.typing import Model

AVAILABLE_DATASETS = ["adult_census", "bike_sharing", "california_housing"]


class GameBenchmarkSetup:
    """Class to load and prepare models and datasets for the benchmark games.

    This class is used to load and prepare the models and datasets for the benchmark games. It can
    be used with a variaty of datasets and models and is typically used to set up inside subclasses
    the benchmark games. The class loads the dataset and the model, splits the dataset into a
    training and test set, and prepares the model for training. The class also provides a number of
    attributes to access the dataset and model information (e.g. number of features, feature
    names, model name, etc.).


    Note:
        Depending on the models, this game requires the ``scikit-learn`` or ``torch`` packages to be
        installed.

    Attributes:
        dataset_name: The name of the dataset.
        feature_names: The names of the features in the dataset.
        n_features: The number of features in the dataset.
        model_name: The name of the loaded model.
        x_data: The whole feature set of the dataset.
        y_data: The target variable of the dataset.
        x_train: The training data used to fit the model.
        y_train: The training labels used to fit the model.
        x_test: The test data used to evaluate the model.
        y_test: The test labels used to evaluate the model.
        n_data: The number of samples in the whole dataset.
        n_train: The number of samples in the training set.
        n_test: The number of samples in the test set.
        model: The loaded model object.
        fit_function: The function that fits the model to the training data as a callable expecting
            the training data and labels as input in form of numpy arrays.
        score_function: The function that scores the model's performance on the test data as a
            callable expecting the test data and labels as input in form of numpy arrays.
        predict_function: The function that predicts the test labels given the test data as a
            callable expecting the test data as input in form of numpy arrays.
        loss_function: A sensible loss function that computes the loss between the predicted and
            true test labels as a callable expecting the true and predicted test labels as input in
            form of numpy arrays.

    Raises:
        ValueError: If an invalid dataset name is provided.
        ValueError: If an invalid model name is provided for the dataset.

    Examples:
        >>> from shapiq.games.benchmark.setup import GameBenchmarkSetup
        >>> setup = GameBenchmarkSetup(dataset_name='adult_census', model_name='decision_tree')
        >>> setup.n_features
        14
        >>> setup.fit_function # returns a callable
    """

    def __init__(
        self,
        dataset_name: str,
        *,
        model_name: str | None = None,
        loss_function: str | None = None,
        verbose: bool = True,
        test_size: float = 0.2,
        random_state: int | None = 42,
        random_forest_n_estimators: int = 10,
    ) -> None:
        """Initializes the GameBenchmarkSetup class.

        Args:
            dataset_name: The dataset to load the models for. Available datasets are
                ``'adult_census'``,``'bike_sharing'``, and ``'california_housing'``.

            model_name: If specified, the name of the model to load. Defaults to ``None``, which
                means that no model will be loaded. Available models for the datasets are the
                following:
                - ``'adult_census'``: '``decision_tree'``, ``'random_forest'``,
                    ``'gradient_boosting'``
                - ``'bike_sharing'``: ``'decision_tree'``, ``'random_forest'``,
                    ``'gradient_boosting'``
                - ``'california_housing'``: ``'decision_tree'``, ``'random_forest'``,
                    ``'gradient_boosting'``, ``'neural_network'``

            loss_function: If specified, the loss function to use for the game (as a string).
                Defaults to ``None``, which means ``'r2_score'`` for regression and
                ``'accuracy_score'`` for classification. Available loss functions are:
                - ``'mean_squared_error'``
                - ``'mean_absolute_error'``
                - ``'log_loss'``
                - ``'r2_score'``
                - ``'accuracy_score'``
                - ``'roc_auc_score'``
                - ``'f1_score'``

            verbose: Whether to print the predicted class and score. Defaults to True.

            test_size: The size of the validation set. Defaults to 0.2.

            random_state: The random state to use for all random operations. Defaults to ``42``.

            random_forest_n_estimators: The number of estimators to use for the random forest model
                if the model is a random forest. Defaults to ``10``.
        """
        self.random_state = random_state

        # load the dataset
        self.dataset_type = "regression"
        if dataset_name == "adult_census":
            x_data, y_data = load_adult_census()
            self.feature_names: list = list(x_data.columns)
            self.dataset_type = "classification"
        elif dataset_name == "bike_sharing":
            x_data, y_data = load_bike_sharing()
            self.feature_names: list = list(x_data.columns)
        elif dataset_name == "california_housing":
            x_data, y_data = load_california_housing()
            self.feature_names: list = list(x_data.columns)
        else:
            msg = (
                f"Invalid dataset name {dataset_name}. Available datasets are 'adult_census', "
                "'bike_sharing', 'california_housing'."
            )
            raise ValueError(msg)

        self.dataset_name: str = dataset_name

        # prepare the data
        x_data, y_data = x_data.values, y_data.values
        x_data, y_data = shuffle_data(x_data, y_data, random_state=random_state)
        self.x_data: np.ndarray = x_data
        self.y_data: np.ndarray = y_data
        self.n_data: int = self.x_data.shape[0]
        self.n_features: int = len(self.feature_names)
        self.n_test = int(test_size * self.n_data)
        self.n_train = self.n_data - self.n_test
        self.x_train: np.ndarray = copy.deepcopy(x_data[: self.n_train])
        self.y_train: np.ndarray = copy.deepcopy(y_data[: self.n_train])
        self.x_test: np.ndarray = copy.deepcopy(x_data[self.n_train :])
        self.y_test: np.ndarray = copy.deepcopy(y_data[self.n_train :])

        self.model_name = model_name
        self._random_forest_n_estimators = random_forest_n_estimators

        # to be set in the model initialization
        self.model: Model | None = None
        self.fit_function = None
        self.score_function = None
        self.predict_function = None
        self.loss_function = None

        # load the model
        if dataset_name == "adult_census":  # adult census dataset
            if model_name == "decision_tree":
                self.init_decision_tree_classifier()
            if model_name == "random_forest":
                self.init_random_forest_classifier()
            if model_name == "gradient_boosting":
                self.init_gradient_boosting_classifier()
        if dataset_name == "bike_sharing":  # bike sharing dataset
            if model_name == "decision_tree":
                self.init_decision_tree_regressor()
            if model_name == "random_forest":
                self.init_random_forest_regressor()
            if model_name == "gradient_boosting":
                self.init_gradient_boosting_regressor()
        if dataset_name == "california_housing":
            if model_name == "decision_tree":
                self.init_decision_tree_regressor()
            if model_name == "random_forest":
                self.init_random_forest_regressor()
            if model_name == "gradient_boosting":
                self.init_gradient_boosting_regressor()
            if model_name == "neural_network":
                self.init_california_neural_network()

        # check if the model is loaded
        if self.model is None and model_name is not None:
            msg = f"Invalid model name {model_name} for the {dataset_name} dataset."
            raise ValueError(msg)

        # set up the functions
        if self.dataset_type == "classification" and model_name is not None:
            self.loss_function = _accuracy  # custom accuracy function
            self.score_function = self.model.score
            self.fit_function = self.model.fit
            self.predict_function = self.model.predict_proba
        if self.dataset_type == "regression" and model_name is not None:
            self.loss_function = r2_score
            self.score_function = self.model.score
            self.fit_function = self.model.fit
            self.predict_function = self.model.predict

        # update loss function if specified
        if loss_function is not None:
            if loss_function == "mean_squared_error":
                self.loss_function = mean_squared_error
            elif loss_function == "mean_absolute_error":
                self.loss_function = mean_absolute_error
            elif loss_function == "log_loss":
                self.loss_function = log_loss
            elif loss_function == "r2_score":
                self.loss_function = r2_score
            elif loss_function == "accuracy_score":
                self.loss_function = _accuracy  # custom accuracy function
            elif loss_function == "f1_score":
                self.loss_function = f1_score
            elif loss_function == "roc_auc_score":
                self.loss_function = roc_auc_score

        # print the performance of the model on the test data
        if verbose and model_name is not None:
            self.print_train_performance()

    def print_train_performance(self) -> None:
        """Prints the performance of the model on the test data."""

    def init_decision_tree_classifier(self) -> None:
        """Initializes and trains a decision tree model for a classification dataset."""
        self.model = DecisionTreeClassifier(random_state=self.random_state)
        self.model.fit(self.x_train, self.y_train)

    def init_random_forest_classifier(self) -> None:
        """Initializes and trains a random forest model for a classification dataset."""
        self.model = RandomForestClassifier(
            n_estimators=self._random_forest_n_estimators,
            random_state=self.random_state,
        )
        self.model.fit(self.x_train, self.y_train)

    def init_gradient_boosting_classifier(self) -> None:
        """Initializes and trains a gradient boosting model for a classification dataset."""
        from xgboost import XGBClassifier

        self.model = XGBClassifier(random_state=self.random_state, n_jobs=1)
        self.model.fit(self.x_train, self.y_train)

    def init_decision_tree_regressor(self) -> None:
        """Initializes and trains a decision tree model for a regression dataset."""
        self.model = DecisionTreeRegressor(random_state=self.random_state)
        self.model.fit(self.x_train, self.y_train)

    def init_random_forest_regressor(self) -> None:
        """Initializes and trains a random forest model for a regression dataset."""
        self.model = RandomForestRegressor(n_estimators=10, random_state=self.random_state)
        self.model.fit(self.x_train, self.y_train)

    def init_gradient_boosting_regressor(self) -> None:
        """Initializes and trains a gradient boosting model for a regression dataset."""
        from xgboost import XGBRegressor

        self.model = XGBRegressor(random_state=self.random_state, n_jobs=1)
        self.model.fit(self.x_train, self.y_train)

    def init_california_neural_network(self) -> None:
        """Initializes a neural network model for the California Housing dataset."""
        from ._setup._california_torch_setup import CaliforniaHousingTorchModel

        self.model = CaliforniaHousingTorchModel()

        scaler = StandardScaler()
        self.x_train = scaler.fit_transform(self.x_train)
        self.x_test = scaler.transform(self.x_test)
        self.x_data = scaler.transform(self.x_data)

        # y_test and y_train need to be log transformed
        self.y_train = np.log10(self.y_train)
        self.y_test = np.log10(self.y_test)
        self.y_data = np.log10(self.y_data)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Returns the accuracy score of the model."""
    if y_true.ndim > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1:
        y_pred = np.argmax(y_pred, axis=1)

    return accuracy_score(y_true, y_pred)


def get_x_explain(x: np.ndarray | int | None, x_set: np.ndarray) -> np.ndarray:
    """Returns the data point to explain given the input.

    Args:
        x: The data point to explain. Can be an index of the background data or a 1d matrix of shape
            (n_features).
        x_set: The data set to select the data point from. Should be a 2d matrix of shape
            (n_samples, n_features).

    Returns:
        The data point to explain as a numpy array.

    """
    if x is None:
        rng = np.random.default_rng()
        idx = rng.choice(x_set.shape[0])
        x = x_set[idx]
    if isinstance(x, int):
        x = x_set[x]
    return x
