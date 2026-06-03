"""This module contains tabular benchmark games for local explanation."""

from __future__ import annotations

import numpy as np

from shapiq_games.benchmark.local_xai.base import LocalExplanation
from shapiq_games.benchmark.setup import GameBenchmarkSetup, get_x_explain


def _init_classification_game(
    game: LocalExplanation,
    *,
    dataset_name: str,
    class_to_explain: int | None,
    x: np.ndarray | int | None,
    model_name: str,
    imputer: str,
    normalize: bool,
    verbose: bool,
    random_state: int | None,
) -> None:
    """Initialize a classification LocalExplanation game from a benchmark dataset."""
    game.setup = GameBenchmarkSetup(
        dataset_name=dataset_name,
        model_name=model_name,
        verbose=verbose,
        random_state=random_state,
    )

    x_explain = get_x_explain(x, game.setup.x_test)

    if class_to_explain is None:
        class_to_explain = int(np.argmax(game.setup.predict_function(x_explain.reshape(1, -1))))

    def predict_function(x_data: np.ndarray) -> np.ndarray:
        """Predict function for the selected class."""
        return game.setup.predict_function(x_data)[:, class_to_explain]

    LocalExplanation.__init__(
        game,
        x=x_explain,
        data=game.setup.x_train,
        model=predict_function,
        imputer=imputer,
        random_state=random_state,
        normalize=normalize,
        verbose=verbose,
    )


def _init_regression_game(
    game: LocalExplanation,
    *,
    dataset_name: str,
    x: np.ndarray | int | None,
    model_name: str,
    imputer: str,
    normalize: bool,
    verbose: bool,
    random_state: int | None,
) -> None:
    """Initialize a regression LocalExplanation game from a benchmark dataset."""
    game.setup = GameBenchmarkSetup(
        dataset_name=dataset_name,
        model_name=model_name,
        verbose=verbose,
        random_state=random_state,
    )

    x_explain = get_x_explain(x, game.setup.x_test)

    LocalExplanation.__init__(
        game,
        x=x_explain,
        data=game.setup.x_test,
        model=game.setup.predict_function,
        imputer=imputer,
        random_state=random_state,
        normalize=normalize,
        verbose=verbose,
    )


class AdultCensus(LocalExplanation):
    """The AdultCensus dataset as a local explanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the AdultCensus LocalExplanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select a random data point
                from the background data.

            class_to_explain: The class label to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'``, ``'random_forest'``, and ``'gradient_boosting'``.

            imputer: The imputer to use. Defaults to 'marginal'. Available imputers are
                ``'marginal'`` and ``'conditional'``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        # validate the inputs
        if isinstance(class_to_explain, int) and class_to_explain not in [0, 1]:
            msg = f"Invalid class label provided. Should be 0 or 1 but got {class_to_explain}."
            raise ValueError(msg)

        _init_classification_game(
            self,
            dataset_name="adult_census",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class BikeSharing(LocalExplanation):
    """The BikeSharing dataset as a Local Explanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the BikeSharing LocalExplanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select a random data point
                from the background data.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'``, ``'random_forest'``, and ``'gradient_boosting'``.

            imputer: The imputer to use. Defaults to 'marginal'. Available imputers are
                ``'marginal'`` and ``'conditional'``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        _init_regression_game(
            self,
            dataset_name="bike_sharing",
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class CaliforniaHousing(LocalExplanation):
    """The CaliforniaHousing dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the CaliforniaHousing LocalExplanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select a random data point
                from the background data.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'``, ``'random_forest'``, ``'gradient_boosting'``, and
                ``'neural_network'``.

            imputer: The imputer to use. Defaults to 'marginal'. Available imputers are
                ``'marginal'`` and ``'conditional'``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        _init_regression_game(
            self,
            dataset_name="california_housing",
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Mushroom(LocalExplanation):
    """The Mushroom dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Mushroom LocalExplanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select a random data point
                from the background data.

            class_to_explain: The class label to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'``, ``'random_forest'``, and ``'gradient_boosting'``.

            imputer: The imputer to use. Defaults to 'marginal'. Available imputers are
                ``'marginal'`` and ``'conditional'``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``False``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        _init_classification_game(
            self,
            dataset_name="mushroom",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Soybean(LocalExplanation):
    """The Soybean dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Soybean LocalExplanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select a random data point
                from the background data.

            class_to_explain: The class label to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'``, ``'random_forest'``, and ``'gradient_boosting'``.

            imputer: The imputer to use. Defaults to 'marginal'. Available imputers are
                ``'marginal'`` and ``'conditional'``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``False``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        _init_classification_game(
            self,
            dataset_name="soybean",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Thyroid(LocalExplanation):
    """The Thyroid dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Thyroid LocalExplanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select a random data point
                from the background data.

            class_to_explain: The class label to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'``, ``'random_forest'``, and ``'gradient_boosting'``.

            imputer: The imputer to use. Defaults to 'marginal'. Available imputers are
                ``'marginal'`` and ``'conditional'``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``False``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        _init_classification_game(
            self,
            dataset_name="thyroid",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Annealing(LocalExplanation):
    """The Annealing dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Annealing LocalExplanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select a random data point
                from the background data.

            class_to_explain: The class label to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'``, ``'random_forest'``, and ``'gradient_boosting'``.

            imputer: The imputer to use. Defaults to ``'marginal'``. Available imputers are
                ``'marginal'`` and ``'conditional'``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``False``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        _init_classification_game(
            self,
            dataset_name="annealing",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Arrhythmia(LocalExplanation):
    """The Arrhythmia dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Arrhythmia LocalExplanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select a random data point
                from the background data.

            class_to_explain: The class label to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'``, ``'random_forest'``, and ``'gradient_boosting'``.

            imputer: The imputer to use. Defaults to ``'marginal'``. Available imputers are
                ``'marginal'`` and ``'conditional'``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``False``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        _init_classification_game(
            self,
            dataset_name="arrhythmia",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class BreastCancer(LocalExplanation):
    """The BreastCancer dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the BreastCancer LocalExplanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select a random data point
                from the background data.

            class_to_explain: The class label to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'``, ``'random_forest'``, and ``'gradient_boosting'``.

            imputer: The imputer to use. Defaults to ``'marginal'``. Available imputers are
                ``'marginal'`` and ``'conditional'``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``False``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        _init_classification_game(
            self,
            dataset_name="breast_cancer",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Hepatitis(LocalExplanation):
    """The Hepatitis dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Hepatitis LocalExplanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select a random data point
                from the background data.

            class_to_explain: The class label to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'``, ``'random_forest'``, and ``'gradient_boosting'``.

            imputer: The imputer to use. Defaults to ``'marginal'``. Available imputers are
                ``'marginal'`` and ``'conditional'``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``False``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        _init_classification_game(
            self,
            dataset_name="hepatitis",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Ionosphere(LocalExplanation):
    """The Ionosphere dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Ionosphere LocalExplanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select a random data point
                from the background data.

            class_to_explain: The class label to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'``, ``'random_forest'``, and ``'gradient_boosting'``.

            imputer: The imputer to use. Defaults to ``'marginal'``. Available imputers are
                ``'marginal'`` and ``'conditional'``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``False``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        _init_classification_game(
            self,
            dataset_name="ionosphere",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Nursery(LocalExplanation):
    """The Nursery dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Nursery LocalExplanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select a random data point
                from the background data.

            class_to_explain: The class label to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'``, ``'random_forest'``, and ``'gradient_boosting'``.

            imputer: The imputer to use. Defaults to ``'marginal'``. Available imputers are
                ``'marginal'`` and ``'conditional'``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``False``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        _init_classification_game(
            self,
            dataset_name="nursery",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Zoo(LocalExplanation):
    """The Zoo dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Zoo LocalExplanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select a random data point
                from the background data.

            class_to_explain: The class label to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'``, ``'random_forest'``, and ``'gradient_boosting'``.

            imputer: The imputer to use. Defaults to ``'marginal'``. Available imputers are
                ``'marginal'`` and ``'conditional'``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``False``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        _init_classification_game(
            self,
            dataset_name="zoo",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class ForestFires(LocalExplanation):
    """The ForestFires dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the ForestFires LocalExplanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select a random data point
                from the background data.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'``, ``'random_forest'``, and ``'gradient_boosting'``.

            imputer: The imputer to use. Defaults to ``'marginal'``. Available imputers are
                ``'marginal'`` and ``'conditional'``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``False``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        _init_regression_game(
            self,
            dataset_name="forest_fires",
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Independent(LocalExplanation):
    """The Independent dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Independent LocalExplanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select a random data point
                from the background data.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'``, ``'random_forest'``, and ``'gradient_boosting'``.

            imputer: The imputer to use. Defaults to ``'marginal'``. Available imputers are
                ``'marginal'`` and ``'conditional'``.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features. Defaults to
                ``True``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``False``.

            random_state: The random state to use for the imputer. Defaults to ``42``.
        """
        _init_regression_game(
            self,
            dataset_name="independentlinear60",
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Amazon(LocalExplanation):
    """The Amazon dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Amazon LocalExplanation game."""
        _init_classification_game(
            self,
            dataset_name="amazon",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Bioresponse(LocalExplanation):
    """The Bioresponse dataset as a LocalExplanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Bioresponse LocalExplanation game."""
        _init_classification_game(
            self,
            dataset_name="bioresponse",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class CommunitiesAndCrime(LocalExplanation):
    """The CommunitiesAndCrime dataset as a LocalExplanation game."""

    def __init__(
        self,
        *,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the CommunitiesAndCrime LocalExplanation game."""
        _init_regression_game(
            self,
            dataset_name="communities_and_crime",
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Chess(LocalExplanation):
    """The Chess dataset as a LocalExplanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Chess LocalExplanation game."""
        _init_classification_game(
            self,
            dataset_name="chess",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Condind(LocalExplanation):
    """The Condind dataset as a LocalExplanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Condind LocalExplanation game."""
        _init_classification_game(
            self,
            dataset_name="condind",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Corrgroups60(LocalExplanation):
    """The Corrgroups60 dataset as a LocalExplanation game."""

    def __init__(
        self,
        *,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Corrgroups60 LocalExplanation game."""
        _init_regression_game(
            self,
            dataset_name="corrgroups60",
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Cross(LocalExplanation):
    """The Cross dataset as a LocalExplanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Cross LocalExplanation game."""
        _init_classification_game(
            self,
            dataset_name="cross",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Disjunct(LocalExplanation):
    """The Disjunct dataset as a LocalExplanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Disjunct LocalExplanation game."""
        _init_classification_game(
            self,
            dataset_name="disjunct",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Sphere(LocalExplanation):
    """The Sphere dataset as a LocalExplanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Sphere LocalExplanation game."""
        _init_classification_game(
            self,
            dataset_name="sphere",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Nhanesi(LocalExplanation):
    """The Nhanesi dataset as a LocalExplanation game."""

    def __init__(
        self,
        *,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Nhanesi LocalExplanation game."""
        _init_regression_game(
            self,
            dataset_name="nhanesi",
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class RealEstate(LocalExplanation):
    """The RealEstate dataset as a LocalExplanation game."""

    def __init__(
        self,
        *,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the RealEstate LocalExplanation game."""
        _init_regression_game(
            self,
            dataset_name="real_estate",
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Microresponse(LocalExplanation):
    """The Microresponse dataset as a LocalExplanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Microresponse LocalExplanation game."""
        _init_classification_game(
            self,
            dataset_name="microresponse",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Group(LocalExplanation):
    """The Group dataset as a LocalExplanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Group LocalExplanation game."""
        _init_classification_game(
            self,
            dataset_name="group",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Leukemia(LocalExplanation):
    """The Leukemia dataset as a LocalExplanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Leukemia LocalExplanation game."""
        _init_classification_game(
            self,
            dataset_name="leukemia",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class WineQuality(LocalExplanation):
    """The WineQuality dataset as a LocalExplanation game."""

    def __init__(
        self,
        *,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the WineQuality LocalExplanation game."""
        _init_regression_game(
            self,
            dataset_name="wine_quality",
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class Xor(LocalExplanation):
    """The Xor dataset as a LocalExplanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Xor LocalExplanation game."""
        _init_classification_game(
            self,
            dataset_name="xor",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


def _init_tabarena_game(
    game: LocalExplanation,
    *,
    dataset_name: str,
    class_to_explain: int | None,
    x: np.ndarray | int | None,
    model_name: str,
    imputer: str,
    normalize: bool,
    verbose: bool,
    random_state: int | None,
) -> None:
    """Initialize a TabArena LocalExplanation game, auto-detecting task type."""
    game.setup = GameBenchmarkSetup(
        dataset_name=dataset_name,
        model_name=model_name,
        verbose=verbose,
        random_state=random_state,
    )
    x_explain = get_x_explain(x, game.setup.x_test)
    if game.setup.dataset_type == "classification":
        if class_to_explain is None:
            class_to_explain = int(np.argmax(game.setup.predict_function(x_explain.reshape(1, -1))))

        def predict_function(x_data: np.ndarray) -> np.ndarray:
            return game.setup.predict_function(x_data)[:, class_to_explain]

        LocalExplanation.__init__(
            game,
            x=x_explain,
            data=game.setup.x_train,
            model=predict_function,
            imputer=imputer,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )
    else:
        LocalExplanation.__init__(
            game,
            x=x_explain,
            data=game.setup.x_test,
            model=game.setup.predict_function,
            imputer=imputer,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )


class TabArenaAirfoilSelfNoise(LocalExplanation):
    """The TabArena airfoil_self_noise dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaAirfoilSelfNoise LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_airfoil_self_noise",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaAmazonEmployeeAccess(LocalExplanation):
    """The TabArena amazon_employee_access dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaAmazonEmployeeAccess LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_amazon_employee_access",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaAnneal(LocalExplanation):
    """The TabArena anneal dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaAnneal LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_anneal",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaFiat500(LocalExplanation):
    """The TabArena fiat_500 dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaFiat500 LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_fiat_500",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaApsFailure(LocalExplanation):
    """The TabArena aps_failure dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaApsFailure LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_aps_failure",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaBankMarketing(LocalExplanation):
    """The TabArena bank_marketing dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaBankMarketing LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_bank_marketing",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaBankCustomerChurn(LocalExplanation):
    """The TabArena bank_customer_churn dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaBankCustomerChurn LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_bank_customer_churn",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaBioresponse(LocalExplanation):
    """The TabArena bioresponse dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaBioresponse LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_bioresponse",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaBloodTransfusion(LocalExplanation):
    """The TabArena blood_transfusion dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaBloodTransfusion LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_blood_transfusion",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaChurn(LocalExplanation):
    """The TabArena churn dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaChurn LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_churn",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaCoil2000(LocalExplanation):
    """The TabArena coil2000 dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaCoil2000 LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_coil2000",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaConcreteStrength(LocalExplanation):
    """The TabArena concrete_strength dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaConcreteStrength LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_concrete_strength",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaCreditG(LocalExplanation):
    """The TabArena credit_g dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaCreditG LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_credit_g",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaCreditCardDefault(LocalExplanation):
    """The TabArena credit_card_default dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaCreditCardDefault LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_credit_card_default",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaAirlineSatisfaction(LocalExplanation):
    """The TabArena airline_satisfaction dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaAirlineSatisfaction LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_airline_satisfaction",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaDiabetes(LocalExplanation):
    """The TabArena diabetes dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaDiabetes LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_diabetes",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaDiabetes130us(LocalExplanation):
    """The TabArena diabetes130us dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaDiabetes130us LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_diabetes130us",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaDiamonds(LocalExplanation):
    """The TabArena diamonds dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaDiamonds LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_diamonds",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaEcommerceShipping(LocalExplanation):
    """The TabArena ecommerce_shipping dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaEcommerceShipping LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_ecommerce_shipping",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaFitnessClub(LocalExplanation):
    """The TabArena fitness_club dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaFitnessClub LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_fitness_club",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaFoodDelivery(LocalExplanation):
    """The TabArena food_delivery dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaFoodDelivery LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_food_delivery",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaGiveMeCredit(LocalExplanation):
    """The TabArena give_me_credit dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaGiveMeCredit LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_give_me_credit",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaHazelnut(LocalExplanation):
    """The TabArena hazelnut dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaHazelnut LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_hazelnut",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaHealthInsurance(LocalExplanation):
    """The TabArena health_insurance dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaHealthInsurance LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_health_insurance",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaHeloc(LocalExplanation):
    """The TabArena heloc dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaHeloc LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_heloc",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaHivaAgnostic(LocalExplanation):
    """The TabArena hiva_agnostic dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaHivaAgnostic LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_hiva_agnostic",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaHouses(LocalExplanation):
    """The TabArena houses dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaHouses LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_houses",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaHrAnalytics(LocalExplanation):
    """The TabArena hr_analytics dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaHrAnalytics LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_hr_analytics",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaCouponRecommendation(LocalExplanation):
    """The TabArena coupon_recommendation dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaCouponRecommendation LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_coupon_recommendation",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaGoodCustomer(LocalExplanation):
    """The TabArena good_customer dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaGoodCustomer LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_good_customer",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaKddcup09(LocalExplanation):
    """The TabArena kddcup09 dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaKddcup09 LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_kddcup09",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaMarketingCampaign(LocalExplanation):
    """The TabArena marketing_campaign dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaMarketingCampaign LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_marketing_campaign",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaMaternalHealth(LocalExplanation):
    """The TabArena maternal_health dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaMaternalHealth LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_maternal_health",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaMiamiHousing(LocalExplanation):
    """The TabArena miami_housing dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaMiamiHousing LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_miami_housing",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaOnlineShoppers(LocalExplanation):
    """The TabArena online_shoppers dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaOnlineShoppers LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_online_shoppers",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaProtein(LocalExplanation):
    """The TabArena protein dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaProtein LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_protein",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaBankruptcy(LocalExplanation):
    """The TabArena bankruptcy dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaBankruptcy LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_bankruptcy",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaQsarBiodeg(LocalExplanation):
    """The TabArena qsar_biodeg dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaQsarBiodeg LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_qsar_biodeg",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaQsarTid11(LocalExplanation):
    """The TabArena qsar_tid11 dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaQsarTid11 LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_qsar_tid11",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaQsarFishToxicity(LocalExplanation):
    """The TabArena qsar_fish_toxicity dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaQsarFishToxicity LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_qsar_fish_toxicity",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaSdss17(LocalExplanation):
    """The TabArena sdss17 dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaSdss17 LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_sdss17",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaSeismicBumps(LocalExplanation):
    """The TabArena seismic_bumps dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaSeismicBumps LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_seismic_bumps",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaSplice(LocalExplanation):
    """The TabArena splice dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaSplice LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_splice",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaStudentsDropout(LocalExplanation):
    """The TabArena students_dropout dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaStudentsDropout LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_students_dropout",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaSuperconductivity(LocalExplanation):
    """The TabArena superconductivity dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaSuperconductivity LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_superconductivity",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaTaiwaneseBankruptcy(LocalExplanation):
    """The TabArena taiwanese_bankruptcy dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaTaiwaneseBankruptcy LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_taiwanese_bankruptcy",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaWebsitePhishing(LocalExplanation):
    """The TabArena website_phishing dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaWebsitePhishing LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_website_phishing",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaWineQuality(LocalExplanation):
    """The TabArena wine_quality dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaWineQuality LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_wine_quality",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaNaticusdroid(LocalExplanation):
    """The TabArena naticusdroid dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaNaticusdroid LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_naticusdroid",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaJm1(LocalExplanation):
    """The TabArena jm1 dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaJm1 LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_jm1",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )


class TabArenaMic(LocalExplanation):
    """The TabArena mic dataset as a local explanation game."""

    def __init__(
        self,
        *,
        class_to_explain: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        imputer: str = "marginal",
        normalize: bool = True,
        verbose: bool = False,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the TabArenaMic LocalExplanation game."""
        _init_tabarena_game(
            self,
            dataset_name="tabarena_mic",
            class_to_explain=class_to_explain,
            x=x,
            model_name=model_name,
            imputer=imputer,
            normalize=normalize,
            verbose=verbose,
            random_state=random_state,
        )
