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
