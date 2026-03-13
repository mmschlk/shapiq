"""This module contains the Interventional TreeSHAPIQ explanation benchmark games."""

from __future__ import annotations

import numpy as np

from shapiq_games.benchmark.interventionaltreeshapiq_xai.base import InterventionalGame
from shapiq_games.benchmark.setup import GameBenchmarkSetup, get_x_explain


def _init_classification_game(
    game: InterventionalGame,
    *,
    dataset_name: str,
    class_index: int | None,
    x: np.ndarray | int | None,
    model_name: str,
    verbose: bool,
    random_state: int | None,
) -> None:
    """Initialize a classification InterventionalGame from a benchmark dataset."""
    game.setup = GameBenchmarkSetup(
        dataset_name=dataset_name,
        model_name=model_name,
        verbose=verbose,
        random_state=random_state,
    )

    x_explain = get_x_explain(x, game.setup.x_test)

    if class_index is None:
        # Default to the class with the highest probability
        proba = game.setup.predict_function(x_explain.reshape(1, -1))
        class_index = int(np.argmax(proba))

    InterventionalGame.__init__(
        game,
        model=game.setup.model,
        reference_data=game.setup.x_train,
        target_instance=x_explain,
        class_index=class_index,
    )


def _init_regression_game(
    game: InterventionalGame,
    *,
    dataset_name: str,
    x: np.ndarray | int | None,
    model_name: str,
    verbose: bool,
    random_state: int | None,
) -> None:
    """Initialize a regression InterventionalGame from a benchmark dataset."""
    game.setup = GameBenchmarkSetup(
        dataset_name=dataset_name,
        model_name=model_name,
        verbose=verbose,
        random_state=random_state,
    )

    x_explain = get_x_explain(x, game.setup.x_test)

    InterventionalGame.__init__(
        game,
        model=game.setup.model,
        reference_data=game.setup.x_train,
        target_instance=x_explain,
        class_index=None,
    )


class AdultCensus(InterventionalGame):
    """The Adult Census dataset as an Interventional TreeSHAP-IQ explanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_index: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Adult Census Interventional explanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select the first data point
                from the test set.

            class_index: The class index to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the model. Defaults to ``42``.
        """
        if isinstance(class_index, int) and class_index not in [0, 1]:
            msg = f"Invalid class index provided. Should be 0 or 1 but got {class_index}."
            raise ValueError(msg)

        if model_name not in ["decision_tree", "random_forest"]:
            msg = "Model name must be either 'decision_tree' or 'random_forest'."
            raise ValueError(msg)

        _init_classification_game(
            self,
            dataset_name="adult_census",
            class_index=class_index,
            x=x,
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )


class BikeSharing(InterventionalGame):
    """The Bike Sharing dataset as an Interventional TreeSHAP-IQ explanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Bike Sharing Interventional explanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select the first data point
                from the test set.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the model. Defaults to ``42``.
        """
        if model_name not in ["decision_tree", "random_forest"]:
            msg = "Model name must be either 'decision_tree' or 'random_forest'."
            raise ValueError(msg)

        _init_regression_game(
            self,
            dataset_name="bike_sharing",
            x=x,
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )


class CaliforniaHousing(InterventionalGame):
    """The California Housing dataset as an Interventional TreeSHAP-IQ explanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the California Housing Interventional explanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select the first data point
                from the test set.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the model. Defaults to ``42``.
        """
        if model_name not in ["decision_tree", "random_forest"]:
            msg = "Model name must be either 'decision_tree' or 'random_forest'."
            raise ValueError(msg)

        _init_regression_game(
            self,
            dataset_name="california_housing",
            x=x,
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )


class SynthData(InterventionalGame):
    """A synthetic dataset as an Interventional TreeSHAP-IQ explanation game."""

    def __init__(
        self,
        *,
        x: int = 0,
        n_features: int = 30,
        classification: bool = True,
        model_name: str = "decision_tree",
        class_index: int | None = None,
        verbose: bool = True,  # noqa: ARG002
        random_state: int | None = 42,
    ) -> None:
        """Initializes the synthetic dataset Interventional explanation game.

        Args:
            x: The index of the data point to be explained. Defaults to ``0``.

            n_features: The number of features in the synthetic dataset. Defaults to ``30``.

            classification: A flag to indicate if the dataset is a classification or regression
                task. Defaults to ``True`` (classification).

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            class_index: The class index to explain (for classification only). If ``None``, then the
                class with the highest probability is used. Defaults to ``None``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the model. Defaults to ``42``.
        """
        from sklearn.datasets import make_classification, make_regression
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        n_samples = 10_000
        n_informative = int(n_features * 2 / 3)
        n_redundant = 0

        if classification:
            x_data, y_data = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_clusters_per_class=3,
                random_state=random_state,
            )
            x_train = x_data[: int(0.8 * n_samples)]
            y_train = y_data[: int(0.8 * n_samples)]
            x_test = x_data[int(0.8 * n_samples) :]
        else:
            x_data, y_data = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_informative,
                n_targets=1,
                random_state=random_state,
            )
            x_train = x_data[: int(0.8 * n_samples)]
            y_train = y_data[: int(0.8 * n_samples)]
            x_test = x_data[int(0.8 * n_samples) :]

        if model_name == "decision_tree":
            if classification:
                model = DecisionTreeClassifier(random_state=random_state, max_depth=15)
            else:
                model = DecisionTreeRegressor(random_state=random_state, max_depth=15)
        elif classification:
            model = RandomForestClassifier(
                random_state=random_state,
                n_estimators=10,
                max_depth=15,
            )
        else:
            model = RandomForestRegressor(
                random_state=random_state,
                n_estimators=10,
                max_depth=15,
            )

        model.fit(x_train, y_train)

        # Get x_explain
        x_explain = get_x_explain(x, x_test)

        # Determine class_index if not provided and classification
        if classification and class_index is None:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_explain.reshape(1, -1))
                class_index = int(np.argmax(proba))
            else:
                class_index = 0

        InterventionalGame.__init__(
            self,
            model=model,
            reference_data=x_train,
            target_instance=x_explain,
            class_index=class_index if classification else None,
        )


class Mushroom(InterventionalGame):
    """The Mushroom dataset as an Interventional TreeSHAP-IQ explanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_index: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Mushroom Interventional explanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select the first data point
                from the test set.

            class_index: The class index to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the model. Defaults to ``42``.
        """
        if model_name not in ["decision_tree", "random_forest"]:
            msg = "Model name must be either 'decision_tree' or 'random_forest'."
            raise ValueError(msg)

        _init_classification_game(
            self,
            dataset_name="mushroom",
            class_index=class_index,
            x=x,
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )


class Soybean(InterventionalGame):
    """The Soybean dataset as an Interventional TreeSHAP-IQ explanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_index: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Soybean Interventional explanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select the first data point
                from the test set.

            class_index: The class index to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the model. Defaults to ``42``.
        """
        if model_name not in ["decision_tree", "random_forest"]:
            msg = "Model name must be either 'decision_tree' or 'random_forest'."
            raise ValueError(msg)

        _init_classification_game(
            self,
            dataset_name="soybean",
            class_index=class_index,
            x=x,
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )


class Thyroid(InterventionalGame):
    """The Thyroid dataset as an Interventional TreeSHAP-IQ explanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_index: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Thyroid Interventional explanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select the first data point
                from the test set.

            class_index: The class index to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the model. Defaults to ``42``.
        """
        if model_name not in ["decision_tree", "random_forest"]:
            msg = "Model name must be either 'decision_tree' or 'random_forest'."
            raise ValueError(msg)

        _init_classification_game(
            self,
            dataset_name="thyroid",
            class_index=class_index,
            x=x,
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )


class Annealing(InterventionalGame):
    """The Annealing dataset as an Interventional TreeSHAP-IQ explanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_index: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Annealing Interventional explanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select the first data point
                from the test set.

            class_index: The class index to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the model. Defaults to ``42``.
        """
        if model_name not in ["decision_tree", "random_forest"]:
            msg = "Model name must be either 'decision_tree' or 'random_forest'."
            raise ValueError(msg)

        _init_classification_game(
            self,
            dataset_name="annealing",
            class_index=class_index,
            x=x,
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )


class Arrhythmia(InterventionalGame):
    """The Arrhythmia dataset as an Interventional TreeSHAP-IQ explanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_index: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Arrhythmia Interventional explanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select the first data point
                from the test set.

            class_index: The class index to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the model. Defaults to ``42``.
        """
        if model_name not in ["decision_tree", "random_forest"]:
            msg = "Model name must be either 'decision_tree' or 'random_forest'."
            raise ValueError(msg)

        _init_classification_game(
            self,
            dataset_name="arrhythmia",
            class_index=class_index,
            x=x,
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )


class BreastCancer(InterventionalGame):
    """The Breast Cancer dataset as an Interventional TreeSHAP-IQ explanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_index: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Breast Cancer Interventional explanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select the first data point
                from the test set.

            class_index: The class index to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the model. Defaults to ``42``.
        """
        if model_name not in ["decision_tree", "random_forest"]:
            msg = "Model name must be either 'decision_tree' or 'random_forest'."
            raise ValueError(msg)

        _init_classification_game(
            self,
            dataset_name="breast_cancer",
            class_index=class_index,
            x=x,
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )


class Hepatitis(InterventionalGame):
    """The Hepatitis dataset as an Interventional TreeSHAP-IQ explanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_index: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Hepatitis Interventional explanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select the first data point
                from the test set.

            class_index: The class index to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the model. Defaults to ``42``.
        """
        if model_name not in ["decision_tree", "random_forest"]:
            msg = "Model name must be either 'decision_tree' or 'random_forest'."
            raise ValueError(msg)

        _init_classification_game(
            self,
            dataset_name="hepatitis",
            class_index=class_index,
            x=x,
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )


class Ionosphere(InterventionalGame):
    """The Ionosphere dataset as an Interventional TreeSHAP-IQ explanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_index: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Ionosphere Interventional explanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select the first data point
                from the test set.

            class_index: The class index to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the model. Defaults to ``42``.
        """
        if model_name not in ["decision_tree", "random_forest"]:
            msg = "Model name must be either 'decision_tree' or 'random_forest'."
            raise ValueError(msg)

        _init_classification_game(
            self,
            dataset_name="ionosphere",
            class_index=class_index,
            x=x,
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )


class Nursery(InterventionalGame):
    """The Nursery dataset as an Interventional TreeSHAP-IQ explanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_index: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Nursery Interventional explanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select the first data point
                from the test set.

            class_index: The class index to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the model. Defaults to ``42``.
        """
        if model_name not in ["decision_tree", "random_forest"]:
            msg = "Model name must be either 'decision_tree' or 'random_forest'."
            raise ValueError(msg)

        _init_classification_game(
            self,
            dataset_name="nursery",
            class_index=class_index,
            x=x,
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )


class Zoo(InterventionalGame):
    """The Zoo dataset as an Interventional TreeSHAP-IQ explanation game.

    Attributes:
        setup: The :class:`~shapiq_games.benchmark.setup.GameBenchmarkSetup` object.
    """

    def __init__(
        self,
        *,
        class_index: int | None = None,
        x: np.ndarray | int | None = None,
        model_name: str = "decision_tree",
        verbose: bool = True,
        random_state: int | None = 42,
    ) -> None:
        """Initializes the Zoo Interventional explanation game.

        Args:
            x: The data point to explain. Can be an index of the background data or a 1d matrix of
                shape ``(n_features,)``. Defaults to ``None`` which will select the first data point
                from the test set.

            class_index: The class index to explain. If ``None``, then the class with the
                highest probability is used. Defaults to ``None``.

            model_name: The model to explain as a string. Defaults to ``'decision_tree'``. Available
                models are ``'decision_tree'`` and ``'random_forest'``.

            verbose: A flag to print the validation score of the model if trained. Defaults to
                ``True``.

            random_state: The random state to use for the model. Defaults to ``42``.
        """
        if model_name not in ["decision_tree", "random_forest"]:
            msg = "Model name must be either 'decision_tree' or 'random_forest'."
            raise ValueError(msg)

        _init_classification_game(
            self,
            dataset_name="zoo",
            class_index=class_index,
            x=x,
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )
