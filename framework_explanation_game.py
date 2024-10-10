"""This module contains the ExplanationGame class for the unified framework."""

from typing import Any, Callable, Optional, Union

import numpy as np

from shapiq import ConditionalImputer, Game
from shapiq.games.imputer import MarginalImputer


def loss_mse(y_true: Union[np.ndarray, float], y_pred: Union[np.ndarray, float]) -> float:
    """Mean squared error loss function."""
    return float(np.mean((y_true - y_pred) ** 2))


def loss_mae(y_true: Union[np.ndarray, float], y_pred: Union[np.ndarray, float]) -> float:
    """Mean absolute error loss function."""
    return float(np.mean(np.abs(y_true - y_pred)))


def loss_r2(y_true: Union[np.ndarray, float], y_pred: Union[np.ndarray, float]) -> float:
    """R2 score loss function."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


class LocalExplanationGame(Game):
    """Local Explanation Game.

    The local explanation game is defined as the model's prediction on the given coalition.
    Optionally, a loss function can be provided to evaluate the model's prediction.

    Args:
        fanova: The type of functional ANOVA to apply for accounting for the feature distribution.
            Defaults to 'm'. Available options are 'b' for baseline, 'm' for marginal, and 'c' for
            conditional.
        model: The model to explain as a callable function expecting data points as input and
            returning the model's predictions.
        x_data: The background data to use for the explainer as a 2-dimensional array with shape
            `(n_samples, n_features)`.
        x_explain: The data point to explain as a 1-dimensional array with shape `(n_features,)`.
        y_explain: An optional target value for the data point to explain. A target value is only
            required if a loss function is provided. Defaults to `None`.
        sample_size: The number of samples to use for integration in the fanova. Defaults to 100.
        random_seed: The random state to use for sampling. Defaults to `None`.
        normalize: Whether to normalize the game values. Defaults to `False`.

    Raises:
        ValueError: If an invalid fanova value is provided.
    """

    def __init__(
        self,
        model: Any,
        x_data: np.ndarray,
        x_explain: np.ndarray,
        y_explain: Optional[np.ndarray] = None,
        fanova: str = "m",
        sample_size: int = 100,
        random_seed: Optional[int] = None,
        normalize: bool = False,
        cond_sampler: Optional[Callable] = None,
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
                joint_marginal_distribution=True,
            )
        elif self.fanova == "c" and cond_sampler is not None:
            imputer = MarginalImputer(
                model=predict_function,
                data=x_data,
                x=x_explain,
                sample_replacements=True,  # will be using marginal imputation with samples
                random_state=random_seed,
                normalize=False,
                sample_size=sample_size,
                joint_marginal_distribution=True,
                cond_sampler=cond_sampler,
            )
        elif self.fanova == "c" and cond_sampler is None:
            imputer = ConditionalImputer(
                model=predict_function,
                data=x_data,
                x=x_explain,
                sample_size=sample_size,
                random_state=random_seed,
                normalize=False,
            )
        else:
            raise ValueError(f"Invalid fanova value: {fanova}. Available: 'b', 'm', 'c'.")
        self.imputer = imputer

        empty_prediction = self.imputer.empty_prediction

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
        """Evaluate the model and imputer on the given coalitions.

        The value function is the model's prediction (or loss) on the given coalition. Features
        not in the coalition are imputed using the selected imputer to approximate the fanova.

        Args:
            coalitions: The coalitions to evaluate as a 2-dimensional array with shape
                `(n_samples, n_features)`.

        Returns:
            The model's prediction (or loss) on the given coalitions as a 1-dimensional
            array with shape `(n_samples,)`.
        """
        outputs = self.imputer(coalitions)
        return outputs


class MultiDataExplanationGame(Game):
    """Global and Sensitivity Explanation games.

    This class defines the global and sensitivity explanation games. The global explanation game
    is defined as the average of the local explanation games. The sensitivity explanation game is
    defined as the variance of the local explanation games.

    Args:
        model: The model to explain as a callable function expecting a data points as input and
            returning the model's predictions.
        x_data: The background data to use for the explainer as a 2-dimensional array with shape
            `(n_samples, n_features)`.
        y_data: The target values for the background data as a 1-dimensional array with shape
            `(n_samples,)`.
        sensitivity_game: Whether to select the sensitivity explanation game (variance) or the
            global explanation game (average). Defaults to `False` for the global explanation game.
        fanova: The type of functional ANOVA to apply for accounting for the feature distribution.
            Defaults to 'm'. Available options are 'b' for baseline, 'm' for marginal, and 'c' for
            conditional.
        loss_function: The loss function to evaluate the model's prediction. Required if the
            sensitivity game is `False`.
        sample_size: The number of samples to use for integration in the fanova. Defaults to 100.
        n_samples: The number of samples to use for the local explanation games. Defaults to 100.
        random_seed: The random state to use for sampling. Defaults to `None`.
        normalize: Whether to normalize the game values. Defaults to `False`.

    Raises:
        ValueError: If a loss function is not provided for the global explanation game.
    """

    def __init__(
        self,
        local_games: Optional[list[Game]] = None,
        y_targets: Optional[np.ndarray] = None,
        model: Optional[Any] = None,
        x_data: Optional[np.ndarray] = None,
        y_data: Optional[np.ndarray] = None,
        sensitivity_game: bool = False,
        fanova: str = "m",
        loss_function: Optional[Callable] = None,
        sample_size: int = 100,
        n_samples: int = 100,
        random_seed: Optional[int] = None,
        normalize: bool = False,
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
        self.X = x_data
        self.y = y_data

        # get local games
        self.local_games = local_games
        self.y_targets = y_targets
        if local_games is None:
            self.local_games = []
            self.y_targets = np.zeros(n_samples)
            n_samples = min(n_samples, x_data.shape[0])
            idx = self._rng.choice(x_data.shape[0], n_samples, replace=False)
            for i in idx:
                local_game = LocalExplanationGame(
                    fanova=fanova,
                    model=model,
                    x_data=x_data,
                    x_explain=x_data[i],
                    y_explain=y_data[i],
                    sample_size=sample_size,
                    random_seed=random_seed,
                    normalize=False,
                )
                self.local_games.append(local_game)
                self.y_targets[i] = y_data[i]
        n_players = local_games[0].n_players

        if sensitivity_game:
            empty_predictions = [game.empty_coalition_value for game in self.local_games]
            empty_prediction = float(np.var(empty_predictions))
        else:
            # empty_predictions = [game.grand_coalition_value for game in self.local_games]
            empty_predictions = [game.empty_coalition_value for game in self.local_games]
            empty_losses = [
                loss_function(y_target, empty_prediction)
                for y_target, empty_prediction in zip(self.y_targets, empty_predictions)
            ]
            empty_prediction = float(np.mean(empty_losses))

        super().__init__(
            n_players=n_players,
            normalization_value=empty_prediction,
            normalize=normalize,
            *args,
            **kwargs,
        )

    def global_value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate the model and imputer on the global explanation game.

        The global explanation game is defined as the average of the local explanation games.

        Args:
            coalitions: The coalitions to evaluate as a 2-dimensional array with shape
                `(n_samples, n_features)`.

        Returns:
            The average of the local explanation games as a 1-dimensional array with shape
            `(n_samples,)`.
        """
        outputs = np.zeros(coalitions.shape[0])
        for i, game in enumerate(self.local_games):
            marginal_predictions = game(coalitions)
            y_target = np.full(marginal_predictions.shape, self.y_targets[i])
            loss = self.loss_function(marginal_predictions, y_target)
            outputs += loss
        outputs /= len(self.local_games)
        return outputs

    def sensitivity_value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate the model and imputer on the sensitivity explanation game.

        The sensitivity explanation game is defined as the variance of the local explanation games.

        Args:
            coalitions: The coalitions to evaluate as a 2-dimensional array with shape
                `(n_samples, n_features)`.

        Returns:
            The variance of the local explanation games as a 1-dimensional array with shape
            `(n_samples,)`.
        """
        outputs = np.zeros((len(self.local_games), coalitions.shape[0]))
        for i, game in enumerate(self.local_games):
            outputs[i] = game(coalitions)
        outputs = np.var(outputs, axis=0)
        return outputs

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate the model and imputer on the given coalitions."""
        if self.sensitivity_game:
            return self.sensitivity_value_function(coalitions)
        return self.global_value_function(coalitions)
