"""This module contains the base class for the ensemble selection games."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from shapiq.games.base import Game

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.utils.custom_types import Model


class EnsembleSelection(Game):
    """The Ensemble Selection game.

    The ensemble selection game models ensemble selection problems as cooperative games. The players
    are ensemble members and the value of a coalition is the performance of the ensemble on a
    test set.

    Note:
        Depending on the ensemble members, this game requires the ``scikit-learn`` and ``xgboost``
            packages.

    """

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        loss_function: Callable[[np.ndarray, np.ndarray], float],
        *,
        dataset_type: str = "classification",
        available_ensemble_members: list[str] | None = None,
        ensemble_members: list[str] | list[Model] | None = None,
        n_members: int = 10,
        verbose: bool = True,
        normalize: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the EnsembleSelection game.

        Args:
            x_train: The training data as a numpy array of shape ``(n_samples, n_features)``.

            y_train: The training labels as a numpy array of shape ``(n_samples,)``.

            x_test: The test data as a numpy array of shape ``(n_samples, n_features)``.

            y_test: The test labels as a numpy array of shape ``(n_samples,)``.

            loss_function: The loss function to use for the ensemble members as a callable expecting
                two arguments: ``y_true`` and ``y_pred`` and returning a ``float``.

            dataset_type: The type of dataset. Available dataset types are ``'classification'`` and
                ``'regression'``. Defaults to ``'classification'``.

            ensemble_members: A optional list of ensemble members to use. Defaults to ``None``. If
                ``None``, then the ensemble members are determined by the game. Available ensemble
                members are:
                - ``'regression'`` (will use linear regression for regression datasets and logistic
                    regression for classification datasets)
                - ``'decision_tree'``
                - ``'random_forest'``
                - ``'gradient_boosting'``
                - ``'knn'``
                - ``'svm'``

            available_ensemble_members:  An optional list of available ensemble members to select
                from. Defaults to ``None``. If ``None``, then the available ensemble members are
                determined by the game.

            n_members: The number of ensemble members to use. Defaults to ``10``. Ignored if
                ``ensemble_members`` is not ``None``.

            verbose: A flag to enable verbose output. Defaults to ``False``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            random_state: The random state to use for the ensemble selection game. Defaults to
                ``42``, which is the same random state used in the other benchmark games with this
                model type for this dataset.

        """
        if dataset_type not in ["classification", "regression"]:
            msg = (
                f"Invalid dataset type provided. Got {dataset_type} but expected one of "
                f"['classification', 'regression']."
            )
            raise ValueError(msg)
        self.dataset_type: str = dataset_type
        self.random_state: int | None = random_state
        self._rng = np.random.default_rng(seed=random_state)

        # set the loss function
        self.loss_function: Callable[[np.ndarray, np.ndarray], float] = loss_function
        if self.loss_function is None:
            msg = "No loss function provided."
            raise ValueError(msg)
        self._empty_coalition_value: float = 0.0  # is set to 0 for all games

        # set the ensemble members attribute
        self.ensemble_members: dict[int, Model] = {}

        # create the sanitized ensemble members list
        self.available_members: list[str] = available_ensemble_members
        if available_ensemble_members is None:
            self.available_members: list[str] = [
                "regression",
                "decision_tree",
                "random_forest",
                "svm",
                "knn",
                "gradient_boosting",
            ]
        if ensemble_members is None:
            ensemble_members = []
            for _ in range(n_members):
                # sample a random ensemble member
                ensemble_member = str(self._rng.choice(self.available_members, size=1)[0])
                ensemble_members.append(ensemble_member)

        # get the ensemble member models
        if any(isinstance(member, str) for member in ensemble_members):
            for member in ensemble_members:
                if member not in self.available_members:
                    msg = (
                        f"Invalid ensemble member provided. Got {member} but expected one of "
                        f"{self.available_members}."
                    )
                    raise ValueError(msg)
            self.player_names: list[str] = ensemble_members
            self.ensemble_members = self._init_ensemble_members()  # initialize the ensemble members
            for member in self.ensemble_members.values():  # fit the ensemble members
                if verbose:
                    pass
                member.fit(x_train, y_train)
        else:
            self.player_names: list[str] = [str(member) for member in ensemble_members]
            self.ensemble_members = dict(enumerate(ensemble_members))

        # setup base game and attributes
        self.player_names: list[str] = ensemble_members
        n_players: int = len(ensemble_members)
        super().__init__(
            n_players=n_players,
            normalize=normalize,
            normalization_value=self._empty_coalition_value,  # is set to 0 for all games
            verbose=verbose,
        )

        # compute the predictions of the ensemble members
        self.predictions: np.ndarray = np.zeros((n_players, y_test.shape[0]))
        for member_id, member in self.ensemble_members.items():
            self.predictions[member_id] = member.predict(x_test)

        # store the test labels
        self._y_test: np.ndarray = y_test

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Computes the worth of the coalition for the ensemble selection game.

        The worth of a coalition is the performance of the ensemble on the test set as measured by
        a goodness_of_fit function.

        Args:
            coalitions: The coalitions as a binary matrix of shape (n_coalitions, n_players).

        Returns:
            The worth of the coalition.

        """
        worth = np.zeros(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            if sum(coalition) == 0:
                worth[i] = self._empty_coalition_value
                continue
            if self.dataset_type == "regression":
                coalition_predictions = self.predictions[coalition].mean(axis=0)
            else:
                coalition_predictions = self.predictions[coalition]
                coalition_predictions = mode(coalition_predictions, axis=0)[0].ravel()
            worth[i] = self.loss_function(self._y_test, coalition_predictions)
        return worth

    def _init_ensemble_members(self) -> dict[int, Model]:
        """Initializes the ensemble members."""
        ensemble_members: dict[int, Model] = {}
        for member_id, member in enumerate(self.player_names):
            if member == "regression":
                if self.dataset_type == "classification":
                    model = LogisticRegression(random_state=self.random_state + member_id, n_jobs=1)
                else:
                    model = LinearRegression()
            elif member == "decision_tree":
                if self.dataset_type == "classification":
                    model = DecisionTreeClassifier(random_state=self.random_state + member_id)
                else:
                    model = DecisionTreeRegressor(random_state=self.random_state + member_id)
            elif member == "random_forest":
                if self.dataset_type == "classification":
                    model = RandomForestClassifier(
                        n_estimators=10,
                        random_state=self.random_state + member_id,
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=10,
                        random_state=self.random_state + member_id,
                    )
            elif member == "knn":
                if self.dataset_type == "classification":
                    model = KNeighborsClassifier(n_neighbors=3)
                else:
                    model = KNeighborsRegressor(n_neighbors=3)
            elif member == "svm":
                if self.dataset_type == "classification":
                    model = SVC(random_state=self.random_state + member_id)
                else:
                    model = SVR()
            elif member == "gradient_boosting":
                from xgboost import XGBClassifier, XGBRegressor

                if self.dataset_type == "classification":
                    model = XGBClassifier(random_state=self.random_state + member_id, n_jobs=1)
                else:
                    model = XGBRegressor(random_state=self.random_state + member_id, n_jobs=1)
            else:
                msg = (
                    f"Invalid ensemble member provided. Got {member} but expected one of "
                    f"{self.available_members}."
                )
                raise ValueError(
                    msg,
                )

            ensemble_members[member_id] = model
        return ensemble_members


class RandomForestEnsembleSelection(EnsembleSelection):
    """The RandomForest Ensemble Selection game.

    The RandomForest ensemble selection game models ensemble selection problems as a cooperative
    games. The players are trees of a random forest and the value of a coalition is the performance
    of the ensemble on a test set.

    Note:
        Depending on the ensemble members, this game requires the ``scikit-learn`` and ``xgboost``
            packages.

    """

    def __init__(
        self,
        random_forest: Model,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        loss_function: Callable[[np.ndarray, np.ndarray], float],
        *,
        dataset_type: str = "classification",
        verbose: bool = True,
        normalize: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the RandomForestEnsembleSelection game.

        Args:
            random_forest: The random forest model to use for the game.

            x_train: The training data as a numpy array of shape ``(n_samples, n_features)``.

            y_train: The training labels as a numpy array of shape ``(n_samples,)``.

            x_test: The test data as a numpy array of shape ``(n_samples, n_features)``.

            y_test: The test labels as a numpy array of shape ``(n_samples,)``.

            loss_function: The loss function to use for the ensemble members as a callable expecting
                two arguments: ``y_true`` and ``y_pred`` and returning a ``float``.

            dataset_type: The type of dataset. Available dataset types are ``'classification'`` and
                ``'regression'``. Defaults to ``'classification'``.

            verbose: Whether to print information about the game and the ensemble members. Defaults
                to ``True``.

            normalize: Whether to normalize the game values. Defaults to ``True``. If ``True``, then
                the game values are normalized and centered to be zero for the empty player set.

            random_state: The random state to use for the ensemble members. Defaults to ``42``.
        """
        # check if the random forest is a scikit-learn random forest
        if not isinstance(random_forest, RandomForestClassifier) and not isinstance(
            random_forest,
            RandomForestRegressor,
        ):
            msg = (
                "Invalid random forest provided. Expected a RandomForestClassifier or "
                "RandomForestRegressor as provided by the scikit-learn package."
            )
            raise TypeError(msg)

        # get the ensemble members
        ensemble_members = random_forest.estimators_
        ensemble_members = list(ensemble_members)

        super().__init__(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            loss_function=loss_function,
            dataset_type=dataset_type,
            ensemble_members=ensemble_members,
            n_members=len(ensemble_members),
            verbose=verbose,
            normalize=normalize,
            random_state=random_state,
        )
