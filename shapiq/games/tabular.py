"""This module contains all tabular machine learning games."""
import os
from typing import Any, Callable, Optional, Union

import numpy as np

from .base import Game
from .imputer import MarginalImputer


class LocalExplanation(Game):
    """The LocalExplanation game class.

    The `LocalExplanation` game is a game that performs local explanation of a model at a specific
    data point as a coalition game. The game evaluates the model's prediction on feature subsets
    around a specific data point. Therein, marginal imputation is used to impute the missing values
    of the data point (for more information see `MarginalImputer`).

    Args:
        x_data: The background data used to fit the imputer. Should be a 2d matrix of shape
            (n_samples, n_features).
        model: The model to explain as a callable function expecting data points as input and
            returning the model's predictions. The input should be a 2d matrix of shape
            (n_samples, n_features) and the output a 1d matrix of shape (n_samples).
        x_explain: The data point to explain. Can be an index of the background data or a 1d matrix
            of shape (n_features).
        random_state: The random state to use for the imputer. Defaults to `None`.
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`.

    Attributes:
        x_explain: The data point to explain.
        empty_prediction: The model's prediction on an empty data point (all features missing).

    Examples:
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> from sklearn.datasets import make_regression
        >>> from shapiq.games.tabular import LocalExplanation
        >>> # create a regression dataset and fit the model
        >>> x_data, y_data = make_regression(n_samples=100, n_features=10, noise=0.1)
        >>> model = DecisionTreeRegressor(max_depth=4)
        >>> model.fit(x_data, y_data)
        >>> # create a LocalExplanation game
        >>> x_explain = x_data[0]
        >>> game = LocalExplanation(x_explain=x_explain,x_data=x_data,model=model.predict)
        >>> # evaluate the game on a specific coalition
        >>> coalition = np.zeros(shape=(1, 10), dtype=bool)
        >>> coalition[0][0, 1, 2] = True
        >>> value = game(coalition)
        >>> # precompute the game (if needed)
        >>> game.precompute()
        >>> # save and load the game
        >>> game.save("game.pkl")
        >>> new_game = LocalExplanation.load("game.pkl")
    """

    def __init__(
        self,
        x_explain: Union[np.ndarray, int],
        x_data: Optional[np.ndarray] = None,
        model: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        imputer: Optional[MarginalImputer] = None,
        random_state: Optional[int] = None,
        normalize: bool = True,
    ) -> None:
        # set attributes # TODO refactor imputer to be used without model and x_data
        self._model = model
        self._x_data = x_data

        # set explanation point
        if isinstance(x_explain, int):
            x_explain = self._x_data[x_explain]
        self.x_explain = x_explain

        # init the imputer which serves as the workhorse of this Game
        self._imputer = imputer
        if self._imputer is None:
            self._imputer = MarginalImputer(
                model=model,
                background_data=self._x_data,
                x_explain=x_explain,
                random_state=random_state,
                normalize=False,
            )

        self.empty_prediction: float = self._imputer.empty_prediction

        # init the base game
        super().__init__(
            x_data.shape[1],
            normalize=normalize,
            normalization_value=self._imputer.empty_prediction,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Calls the model and returns the prediction.

        Args:
            coalitions: The coalitions as a one-hot matrix for which the game is to be evaluated.

        Returns:
            The output of the model on feature subsets.
        """
        return self._imputer(coalitions)


class FeatureSelectionGame(Game):

    """The FeatureSelection game.

    The `FeatureSelectionGame` is a game that evaluates the goodness of fit of a model on a subset
    of features. The goodness of fit is determined by a score or loss function that compares the
    model's test set performance.

    Args:
        x_train: The training data used to fit the model. Should be a 2d matrix of shape
            (n_samples, n_features).
        y_train: The training labels used to fit the model. Can be a 1d or 2d matrix of shape
            (n_samples, n_outputs).
        x_test: The test data used to evaluate the model. Should be the same shape as `x_train`.
        y_test: The test labels used to evaluate the model. Should be the same shape as `y_train`.
        fit_function: The function that fits the model to the training data. It should take the
            training data and labels as input.
        score_function: The function that scores the model's performance on the test data. It should
            take the test data and labels as input. If not provided, then `predict_function` and
            `loss_function` must be provided.
        predict_function: The function that predicts the test labels given the test data. It should
            take the test data as input. If not provided, then `score_function` must be provided.
        loss_function: The function that computes the loss between the predicted and true test
            labels. It should take the true and predicted test labels as input. If not provided, then
            `score_function` must be provided.
        empty_value: The value to return when the subset of features is empty. Defaults to 0.0.
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`.

    Attributes:
        empty_value: The value to return when the subset of features is empty.

    Examples:
        >>> from sklearn.tree import DecisionTreeRegressor
        >>> from sklearn.datasets import make_regression
        >>> from sklearn.model_selection import train_test_split
        >>> from shapiq.games.tabular import FeatureSelectionGame
        >>> # create a regression dataset
        >>> x_data, y_data = make_regression(n_samples=100, n_features=10, noise=0.1)
        >>> x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
        >>> # create a decision tree regressor
        >>> model = DecisionTreeRegressor(max_depth=4)
        >>> # create a FeatureSelection game
        >>> game = FeatureSelectionGame(
        ...     x_train=x_train,
        ...     x_test=x_test,
        ...     y_train=y_train,
        ...     y_test=y_test,
        ...     fit_function=model.fit,
        ...     score_function=model.score,
        ... )
        >>> # evaluate the game on a specific coalition
        >>> coalition = np.zeros(shape=(1, 10), dtype=bool)
        >>> coalition[0][0, 1, 2] = True
        >>> value = game(coalition)
        >>> # precompute the game (if needed)
        >>> game.precompute()
        >>> # save and load the game
        >>> game.save("game.pkl")
        >>> new_game = FeatureSelectionGame.load("game.pkl")
    """

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        fit_function: Callable[[np.ndarray, np.ndarray], Any],
        score_function: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        predict_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        loss_function: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
        empty_value: float = 0.0,
        normalize: bool = True,
    ) -> None:
        super().__init__(x_train.shape[1], normalization_value=empty_value, normalize=normalize)

        # set datasets
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test

        # sanity check on  input params
        if score_function is None:
            if loss_function is None or predict_function is None:
                raise ValueError(
                    "If score function is not provided, then 'predict_function' and 'loss_function'"
                    " must be provided."
                )

        # setup callables
        self._fit_function = fit_function
        self._predict_function = predict_function
        self._loss_function = loss_function
        self._score_function = score_function

        # set empty value
        self.empty_value = empty_value

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Trains and evaluates the model on a coalition (subset) of features. The output of the
            value function is the value of the specified loss function (goodness of fit metric).

        Args:
            coalitions: A one-hot 2d matrix of coalitions denoting the feature selection to train
                and evaluate the model on.

        Returns:
            A vector of loss function values given the subset of features.
        """
        scores = np.zeros(shape=coalitions.shape[0], dtype=float)
        for i in range(len(coalitions)):
            coalition = coalitions[i]  # get coalition
            if sum(coalition) == 0:  # if empty subset then set to empty prediction
                scores[i] = self.empty_value
                continue
            x_train, x_test = self._x_train[:, coalition], self._x_test[:, coalition]
            self._fit_function(x_train, self._y_train)  # fit model
            if self._score_function is not None:
                score = self._score_function(x_test, self._y_test)
            else:
                y_pred = self._predict_function(x_test)  # get y hat prediction
                score = self._loss_function(self._y_test, y_pred)  # compare prediction with gt
            scores[i] = score
        return scores


class CaliforniaHousing(LocalExplanation):

    """The CaliforniaHousing dataset as a LocalExplanation game.

    The CaliforniaHousing dataset is a regression dataset that contains data on housing prices in
    California. For this game a default model is trained if no model is provided by the user. The
    value function of the game is the model's prediction on feature.

    Note:
        This game requires the `sklearn` package to be installed.

    Args:
        x_explain: The data point to explain. Can be an index of the background data or a 1d matrix
            of shape (n_features).
        model: The model to explain as a string or a callable function. If a string is provided it
            should be one of the following:
            - "sklearn_gbt": A gradient boosting regressor from the `sklearn` package is fitted.
            - "torch_nn": A simple neural network model is trained using PyTorch.
            If a callable function is provided, then it should be expecting data points as input and
            returning the model's predictions. The input should be a 2d matrix of shape (n_samples,
            n_features) and the output a 1d matrix of shape (n_samples). Defaults to `None`.
        imputer: The imputer to use for the game. If not provided, then a default imputer is
            created. Defaults to `None`.
        random_state: The random state to use for the imputer. Defaults to `None`.
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`.
        verbose: A flag to print the validation score of the model if trained. Defaults to `True`.

    Attributes:
        feature_names: The names of the features in the dataset in the order they appear.
    """

    def __init__(
        self,
        x_explain: Optional[Union[np.ndarray, int]] = None,
        model: Union[Callable[[np.ndarray], np.ndarray], str] = "sklearn_gbt",
        imputer: Optional[MarginalImputer] = None,
        random_state: Optional[int] = None,
        normalize: bool = True,
        verbose: bool = True,
    ) -> None:
        # validate the input
        if isinstance(model, str) and model not in ["sklearn_gbt", "torch_nn"]:
            raise ValueError(
                "Invalid model string provided. Should be one of 'sklearn_gbt' or 'torch_nn'."
            )

        # do the imports for this class
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        # load the dataset
        data = fetch_california_housing()
        self.feature_names = list(data.feature_names)

        x_data, y_data = data.data, data.target

        # split the data
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, train_size=0.7, shuffle=True, random_state=42
        )

        # create a model if none is provided
        if isinstance(model, str) and model == "sklearn_gbt":
            model = self._get_sklearn_model(x_train, x_test, y_train, y_test, verbose)
        self._torch_model = None
        if isinstance(model, str) and model == "torch_nn":
            self._load_torch_model()
            model = self._torch_model_call
            # scale the data
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)

        # get x_explain
        if x_explain is None:  # get a random data point from the test set
            x_explain = x_test[np.random.randint(0, x_test.shape[0])]
        # if x_explain is an index then get the data point
        if isinstance(x_explain, int):
            x_explain = x_test[x_explain]

        # call the super constructor
        super().__init__(
            x_explain=x_explain,
            x_data=x_train,
            model=model,
            imputer=imputer,
            random_state=random_state,
            normalize=normalize,
        )

    @staticmethod
    def _get_sklearn_model(
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Creates a default model for the CaliforniaHousing dataset and returns it as a callable
            function.

        Args:
            x_train: The training data used to fit the model. Should be a 2d matrix of shape
                (n_samples, n_features).
            x_test: The test data used to evaluate the model. Should be the same shape as `x_train`.
            y_train: The training labels used to fit the model. Can be a 1d or 2d matrix of shape
                (n_samples, n_outputs).
            y_test: The test labels used to evaluate the model. Should be the same shape as
                `y_train`.
            verbose: A flag to print the validation score of the model if trained. Defaults to
                `True`.

        Returns:
            A callable function that predicts the output given the input data.
        """
        from sklearn.ensemble import GradientBoostingRegressor

        # create a random forest regressor
        model = GradientBoostingRegressor(n_estimators=50, random_state=42)
        model.fit(x_train, y_train)

        if verbose:
            validation_score = model.score(x_test, y_test)
            print(f"Validation score of fitted GradientBoostingRegressor: {validation_score:.4f}")

        return model.predict

    def _load_torch_model(self) -> None:
        """Loads a pre-trained neural network model for the CaliforniaHousing dataset."""
        import torch
        from torch import nn

        # create a simple neural network
        class SmallNeuralNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Sequential(
                    nn.Linear(8, 50),
                    nn.ReLU(),
                    nn.Linear(50, 100),
                    nn.ReLU(),
                    nn.Linear(100, 5),
                    nn.Linear(5, 1),
                )

            def forward(self, x):
                x = self.model(x)
                return x

        # instantiate the model
        model = SmallNeuralNetwork()

        # load model from file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(
            current_dir, "precomputed", "models", "california_nn_0.812511_0.076331.weights"
        )
        model.load_state_dict(torch.load(model_path))
        self._torch_model = model

    def _torch_model_call(self, x: np.ndarray) -> np.ndarray:
        """A wrapper function to call the pre-trained neural network model on the numpy input data.

        Args:
            x: The input data to predict on.

        Returns:
            The model's prediction on the input data.
        """
        import torch

        x = torch.tensor(x.astype(float), dtype=torch.float32)
        return self._torch_model(x).flatten().detach().numpy()
