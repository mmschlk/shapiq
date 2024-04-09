"""This module contains tabular benchmark games for local explanation."""

import os
from typing import Callable, Optional, Union

import numpy as np

from .tabular import LocalExplanation, get_x_explain


class AdultCensus(LocalExplanation):

    """The AdultCensus dataset as a local explanation game.

    This class represents the AdultCensus dataset as a local explanation game. The dataset is a
    classification dataset that contains data on the income of individuals. The game evaluates the
    model's prediction on feature subsets. The value function of the game is the model's predicted
    class probability on feature subsets.

    Note:
        This game requires the `openml` and `sklearn` packages to be installed.

    Args:
        path_to_values: The path to the pre-computed game values to load. If provided, then the game
            is loaded from the file and no other parameters are used. Defaults to `None`.
        class_to_explain: The class label to explain. Should be either 0 or 1. Defaults to `1`.
        x_explain: The data point to explain. Can be an index of the background data or a 1d matrix
            of shape (n_features).
        model: The model to explain as a string or a callable function. If a string is provided it
            should be 'sklearn_rf'. If a callable function is provided, then it should be expecting
            data points as input and returning the model's predictions. The input should be a 2d
            matrix of shape (n_samples, n_features) and the output a 1d matrix of shape (n_samples).
            Defaults to 'sklearn_rf'.
        random_state: The random state to use for the imputer. Defaults to `None`.
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`.
        verbose: A flag to print the validation score of the model if trained. Defaults to `True`.

    Attributes:
        feature_names: The names of the features in the dataset in the order they appear.
        class_to_explain: The class label to explain.

    Examples:
        >>> game = AdultCensus(x_explain=0)
        >>> game.n_players
        14
        >>> # precompute the game (if needed)
        >>> game.precompute()
        >>> # save and load the game
        >>> game.save_values("adult_income.npz")
        >>> new_game = AdultCensus(path_to_values="adult_income.npz")
    """

    def __init__(
        self,
        *,
        path_to_values: Optional[str] = None,
        class_to_explain: int = 1,
        x_explain: Optional[Union[np.ndarray, int]] = None,
        model: Union[Callable[[np.ndarray], np.ndarray], str] = "sklearn_rf",
        random_state: Optional[int] = None,
        normalize: bool = True,
        verbose: bool = True,
    ) -> None:
        # check if path is provided
        if path_to_values is not None:
            super().__init__(path_to_values=path_to_values)
            return

        # validate the inputs
        if isinstance(model, str) and model != "sklearn_rf":
            raise ValueError(
                f"Invalid model string provided. Should be 'sklearn_rf' but got '{model}'."
            )
        if class_to_explain not in [0, 1]:
            raise ValueError(
                f"Invalid class label provided. Should be 0 or 1 but got {class_to_explain}."
            )

        # import necessary packages
        from sklearn.model_selection import train_test_split

        from shapiq.datasets import load_adult_census

        # get data
        x_data, y_data = load_adult_census()
        self.feature_names = list(x_data.columns)
        self.class_to_explain = class_to_explain

        # transform to numpy
        x_data, y_data = x_data.to_numpy(), y_data.to_numpy()

        # split the data
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, train_size=0.7, shuffle=True, random_state=42
        )

        # create a model if none is provided
        if isinstance(model, str) and model == "sklearn_rf":
            model = self._get_sklearn_model(
                x_train, x_test, y_train, y_test, verbose, self.class_to_explain
            )

        # get x_explain
        x_explain = get_x_explain(x_explain, x_test)

        # call the super constructor
        super().__init__(
            x_explain=x_explain,
            x_data=x_train,
            model=model,
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
        class_to_explain: int = 1,
    ):
        """Creates a default model for the AdultIncome dataset and returns it as a callable
            function.

        Args:
            x_train: The training data used to fit the model. Should be a 2d matrix of shape
                (n_samples, n_features).
            x_test: The test data used to evaluate the model. Should be the same shape as `x_train`.
            y_train: The training labels used to fit the model. Can be a 1d or 2d matrix of shape
                (n_samples, n_outputs).
            y_test: The test labels used to evaluate the model. Should be the same shape as
                `y_train`.
            class_to_explain: The class to explain. Defaults to `1`.
            verbose: A flag to print the validation score of the model if trained. Defaults to
                `True`.

        Returns:
            A callable function that predicts the output given the input data.
        """
        from sklearn.ensemble import RandomForestClassifier

        # create a random forest classifier
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(x_train, y_train)

        if verbose:
            validation_score = model.score(x_test, y_test)
            print(f"Validation score of fitted RandomForestClassifier: {validation_score:.4f}")

        return lambda x: model.predict_proba(x)[:, class_to_explain]


class BikeRegression(LocalExplanation):

    """The BikeRental dataset as a local explanation game.

    This class represents the BikeRental regression dataset as a local explanation game. The dataset
    contains data on bike rentals in a city given various features such as the weather or time of
    day. The value function of the game is the model's prediction on feature subsets. Missing
    features are removed using marginal imputation.

    Note:
        This game requires the `sklearn` package to be installed.
        For the default model, the game requires the `xgboost` package to be installed.

    Args:
        path_to_values: The path to the pre-computed game values to load. If provided, then the game
            is loaded from the file and no other parameters are used. Defaults to `None`.
        x_explain: The data point to explain. Can be an index of the background data or a 1d matrix
            of shape (n_features).
        model: The model to explain as a string or a callable function. If a string is provided it
            should be 'xgboost'. If a callable function is provided, then it should be expecting
            data points as input and returning the model's predictions. The input should be a 2d
            matrix of shape (n_samples, n_features) and the output a 1d matrix of shape (n_samples).
            Defaults to 'xgboost'.
        random_state: The random state to use for the imputer. Defaults to `None`.
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`.
        verbose: A flag to print the validation score of the model if trained. Defaults to `True`.

    Attributes:
        feature_names: The names of the features in the dataset in the order they appear.

    Examples:
        >>> game = BikeRegression(x_explain=1)
        >>> game.n_players
        14
        >>> # call the game with a coalition
        >>> coalition = np.ones(14, dtype=bool)
        >>> game(coalition)
        0.28  # [0.72, 0.28] for x_explain=1 (would be different for other x_explain)
        >>> # precompute the game (if needed)
        >>> game.precompute()
        >>> # save and load the game
        >>> game.save_values("bike_rental.npz")
        >>> new_game = BikeRegression(path_to_values="bike_rental.npz")

    """

    def __init__(
        self,
        *,
        path_to_values: Optional[str] = None,
        x_explain: Optional[Union[np.ndarray, int]] = None,
        model: Union[Callable[[np.ndarray], np.ndarray], str] = "xgboost",
        random_state: Optional[int] = None,
        normalize: bool = True,
        verbose: bool = True,
    ) -> None:
        if path_to_values is not None:
            super().__init__(path_to_values=path_to_values)
            return

        # validate the inputs
        if isinstance(model, str) and model != "xgboost":
            raise ValueError(
                f"Invalid model string provided. Should be 'xgboost' but got '{model}'."
            )

        from sklearn.model_selection import train_test_split

        from shapiq.datasets import load_bike

        x_data, y_data = load_bike()
        self.feature_names = list(x_data.columns)

        # transform to numpy
        x_data, y_data = x_data.to_numpy(), y_data.to_numpy()

        # get test/train
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, train_size=0.7, random_state=42
        )

        # create a model if none is provided
        if isinstance(model, str) and model == "xgboost":
            model = self._get_xgboost_model(x_train, x_test, y_train, y_test, verbose)

        # get x_explain
        x_explain = get_x_explain(x_explain, x_test)

        # call the super constructor
        super().__init__(
            x_explain=x_explain,
            x_data=x_train,
            model=model,
            random_state=random_state,
            normalize=normalize,
        )

    @staticmethod
    def _get_xgboost_model(
        x_train: np.ndarray,
        x_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        verbose: bool = True,
    ) -> Callable[[np.ndarray], np.ndarray]:
        """Creates a default XGBoost model for the BikeRental dataset and returns it as a callable
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
        from xgboost import XGBRegressor

        # create a xgboost regressor
        model: XGBRegressor = XGBRegressor(random_state=42)
        model.fit(x_train, y_train)

        if verbose:
            validation_score = model.score(x_test, y_test)
            print(f"Validation score of fitted XGBRegressor: {validation_score:.4f}")

        return model.predict


class CaliforniaHousing(LocalExplanation):

    """The CaliforniaHousing dataset as a LocalExplanation game.

    The CaliforniaHousing dataset is a regression dataset that contains data on housing prices in
    California. For this game a default model is trained if no model is provided by the user. The
    value function of the game is the model's prediction on feature.

    Note:
        This game requires the `sklearn` package to be installed.

    Args:
        path_to_values: The path to the pre-computed game values to load. If provided, then the game
            is loaded from the file and no other parameters are used. Defaults to `None`.
        x_explain: The data point to explain. Can be an index of the background data or a 1d matrix
            of shape (n_features).
        model: The model to explain as a string or a callable function. If a string is provided it
            should be one of the following:
            - "sklearn_gbt": A gradient boosting regressor from the `sklearn` package is fitted.
            - "torch_nn": A simple neural network model is loaded using PyTorch.
            If a callable function is provided, then it should be expecting data points as input and
            returning the model's predictions. The input should be a 2d matrix of shape (n_samples,
            n_features) and the output a 1d matrix of shape (n_samples). Defaults to 'sklearn_gbt'.
        random_state: The random state to use for the imputer. Defaults to `None`.
        normalize: A flag to normalize the game values. If `True`, then the game values are
            normalized and centered to be zero for the empty set of features. Defaults to `True`.
        verbose: A flag to print the validation score of the model if trained. Defaults to `True`.

    Attributes:
        feature_names: The names of the features in the dataset in the order they appear.
        scaler: The scaler used to normalize the data (only fitted for the neural network model).

    Examples:
        >>> game = CaliforniaHousing(x_explain=0)
        >>> game.n_players
        8
        >>> # call the game with a coalition
        >>> coalition = np.ones(8, dtype=bool)
        >>> game(coalition)
        ... # some value
        >>> # precompute the game (if needed)
        >>> game.precompute()
        >>> # save and load the game
        >>> game.save_values("california_housing.npz")
        >>> new_game = CaliforniaHousing(path_to_values="california_housing.npz")
        >>> # load the game with a torch model
        >>> game = CaliforniaHousing(model="torch_nn")
        >>> game(coalition)
        ... # some value but this time the values are logarithmic
    """

    def __init__(
        self,
        *,
        path_to_values: Optional[str] = None,
        x_explain: Optional[Union[np.ndarray, int]] = None,
        model: Union[Callable[[np.ndarray], np.ndarray], str] = "sklearn_gbt",
        random_state: Optional[int] = None,
        normalize: bool = True,
        verbose: bool = True,
    ) -> None:
        # check if path is provided
        if path_to_values is not None:
            super().__init__(path_to_values=path_to_values)
            return

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

        self.scaler: StandardScaler = StandardScaler()

        # create a model if none is provided
        if isinstance(model, str) and model == "sklearn_gbt":
            model = self._get_sklearn_model(x_train, x_test, y_train, y_test, verbose)
        self._torch_model = None
        if isinstance(model, str) and model == "torch_nn":
            self._load_torch_model()
            model = self._torch_model_call
            # scale the data
            x_train = self.scaler.fit_transform(x_train)
            x_test = self.scaler.transform(x_test)

        # get x_explain
        x_explain = get_x_explain(x_explain, x_test)

        # call the super constructor
        super().__init__(
            x_explain=x_explain,
            x_data=x_train,
            model=model,
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
