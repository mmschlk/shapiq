"""This module contains the ExplanationGame class for the unified framework."""

from typing import Any, Callable, Optional, Union

import numpy as np

from shapiq import Game
from shapiq.games.imputer import ConditionalImputer, MarginalImputer


def loss_mse(y_true: Union[np.ndarray, float], y_pred: Union[np.ndarray, float]) -> float:
    """Mean squared error loss function."""
    return np.mean((y_true - y_pred) ** 2)


def loss_mae(y_true: Union[np.ndarray, float], y_pred: Union[np.ndarray, float]) -> float:
    """Mean absolute error loss function."""
    return np.mean(np.abs(y_true - y_pred))


def loss_r2(y_true: Union[np.ndarray, float], y_pred: Union[np.ndarray, float]) -> float:
    """R2 score loss function."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def loss_cross_entropy(y_true: Union[np.ndarray, float], y_pred: Union[np.ndarray, float]) -> float:
    """Cross entropy loss function."""
    return -np.sum(y_true * np.log(y_pred))


class LocalExplanationGame(Game):

    def __init__(
        self,
        fanova: str,
        model: Any,
        x_data: np.ndarray,
        x_explain: np.ndarray,
        y_explain: Optional[np.ndarray],
        loss_function: Optional[Callable] = None,
        sample_size: int = 100,
        random_seed: Optional[int] = None,
        normalize: bool = True,
        *args,
        **kwargs,
    ) -> None:
        self.fanova = fanova
        self.sample_size = sample_size
        self.random_seed = random_seed

        predict_function = model.predict_proba if hasattr(model, "predict_proba") else model.predict

        # select the imputer based on the fanova
        if self.fanova == "b":
            imputer = MarginalImputer(
                model=predict_function,
                data=x_data,
                x=x_explain,
                sample_replacements=False,  # will be using baseline imputation with mean
                random_state=random_seed,
                normalize=False,
            )
        elif self.fanova == "m":
            imputer = MarginalImputer(
                model=predict_function,
                data=x_data,
                x=x_explain,
                sample_replacements=True,  # will be using marginal imputation with samples
                random_state=random_seed,
                normalize=False,
                sample_size=sample_size,
            )
        elif self.fanova == "c":
            imputer = ConditionalImputer(
                model=predict_function,
                data=x_data,
                x=x_explain,
                random_state=random_seed,
                normalize=False,
            )
        else:
            raise ValueError(f"Invalid fanova value: {fanova}. Available: 'b', 'm', 'c'.")
        self.imputer = imputer

        self.loss_function = loss_function
        empty_prediction = self.imputer.empty_prediction
        if self.loss_function is not None:
            empty_prediction = loss_function(empty_prediction, y_explain)

        n_players = x_data.shape[1]
        self.y_explain = y_explain
        self.model = model

        super().__init__(
            n_players=n_players,
            normalization_value=empty_prediction,
            normalize=normalize,
            *args,
            **kwargs,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate the model and imputer on the given coalitions."""
        outputs = self.imputer.value_function(coalitions)
        if self.loss_function is not None:
            return self.loss_function(outputs, self.y_explain)
        return outputs


class MultiDataExplanationGame(Game):
    """Global and Sensitivity Explanation games.


    The global explanation games is defined as the average over local explanation games. The
    sensitivity explanation game is defined as the variance over local explanation games.

    """

    def __init__(
        self,
        fanova: str,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        loss_function: Optional[Callable] = None,
        sample_size: int = 100,
        n_samples: int = 100,
        random_seed: Optional[int] = None,
        normalize: bool = True,
        sensitivity_game: bool = False,
        *args,
        **kwargs,
    ) -> None:
        self._rng = np.random.default_rng(random_seed)

        self.sensitivity_game = sensitivity_game
        if sensitivity_game is False and loss_function is None:
            raise ValueError("A loss function must be provided for the global explanation game.")

        self.fanova = fanova
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.loss_function = loss_function
        self.X = X
        self.y = y

        # get local games
        self.local_games = []
        n_samples = min(n_samples, X.shape[0])
        idx = self._rng.choice(X.shape[0], n_samples, replace=False)
        for i in idx:
            local_game = LocalExplanationGame(
                fanova=fanova,
                model=model,
                x_data=X,
                x_explain=X[i],
                y_explain=y[i],
                loss_function=loss_function,
                sample_size=sample_size,
                random_seed=random_seed,
                normalize=False,
            )
            self.local_games.append(local_game)

        n_players = X.shape[1]
        empty_prediction = np.mean(y)
        if self.loss_function is not None:
            empty_prediction = loss_function(empty_prediction, y)

        super().__init__(
            n_players=n_players,
            normalization_value=empty_prediction,
            normalize=normalize,
            *args,
            **kwargs,
        )

    def global_value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate the model and imputer on the given coalitions."""
        outputs = np.zeros(coalitions.shape[0])
        for game in self.local_games:
            outputs += game.value_function(coalitions)
        outputs /= len(self.local_games)
        return outputs

    def sensitivity_value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate the model and imputer on the given coalitions."""
        outputs = np.zeros((len(self.local_games), coalitions.shape[0]))
        for i, game in enumerate(self.local_games):
            outputs[i] = game.value_function(coalitions)
        outputs = np.var(outputs, axis=0)
        return outputs

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate the model and imputer on the given coalitions."""
        if self.sensitivity_game:
            return self.sensitivity_value_function(coalitions)
        return self.global_value_function(coalitions)
