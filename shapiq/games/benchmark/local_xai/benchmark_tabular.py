"""This module contains tabular benchmark games for local explanation."""

from __future__ import annotations

import numpy as np

from shapiq.games.benchmark.local_xai.base import LocalExplanation
from shapiq.games.benchmark.setup import GameBenchmarkSetup, get_x_explain


class AdultCensus(LocalExplanation):
    """The AdultCensus dataset as a local explanation game.

    Attributes:
        setup: The :class:`~shapiq.games.benchmark.setup.GameBenchmarkSetup` object.
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

        self.setup = GameBenchmarkSetup(
            dataset_name="adult_census",
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )

        # get x_explain
        x = get_x_explain(x, self.setup.x_test)

        # get class_to_explain
        if class_to_explain is None:
            class_to_explain = int(np.argmax(self.setup.predict_function(x.reshape(1, -1))))

        def predict_function(x: np.ndarray) -> np.ndarray:
            """Predict function for the model."""
            return self.setup.predict_function(x)[:, class_to_explain]

        # call the super constructor
        super().__init__(
            x=x,
            data=self.setup.x_train,
            model=predict_function,
            imputer=imputer,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )


class BikeSharing(LocalExplanation):
    """The BikeSharing dataset as a Local Explanation game.

    Attributes:
        setup: The :class:`~shapiq.games.benchmark.setup.GameBenchmarkSetup` object.
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
        self.setup = GameBenchmarkSetup(
            dataset_name="bike_sharing",
            model_name=model_name,
            verbose=False,
            random_state=random_state,
        )

        # get x_explain
        x = get_x_explain(x, self.setup.x_test)

        predict_function = self.setup.predict_function

        # call the super constructor
        super().__init__(
            x=x,
            data=self.setup.x_test,
            model=predict_function,
            imputer=imputer,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )


class CaliforniaHousing(LocalExplanation):
    """The CaliforniaHousing dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq.games.benchmark.setup.GameBenchmarkSetup` object.
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
        self.setup = GameBenchmarkSetup(
            dataset_name="california_housing",
            model_name=model_name,
            verbose=False,
            random_state=random_state,
        )

        # get x_explain
        x = get_x_explain(x, self.setup.x_test)

        predict_function = self.setup.predict_function

        # call the super constructor
        super().__init__(
            x=x,
            data=self.setup.x_test,
            model=predict_function,
            imputer=imputer,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )


class ForestFires(LocalExplanation):
    """The ForestFires dataset as a LocalExplanation game.
    Attributes:
        setup: The :class:`~shapiq.games.benchmark.setup.GameBenchmarkSetup` object.
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
        """Initializes the ForestFires LocalExplanation game.
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
        self.setup = GameBenchmarkSetup(
            dataset_name="forest_fires",
            model_name=model_name,
            verbose=verbose,
            random_state=random_state,
        )

        # get x_explain
        x = get_x_explain(x, self.setup.x_test)

        predict_function = self.setup.predict_function

        # call the super constructor
        super().__init__(
            x=x,
            data=self.setup.x_train,
            model=predict_function,
            imputer=imputer,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )


class RealEstate(LocalExplanation):
    """The RealEstate dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq.games.benchmark.setup.GameBenchmarkSetup` object.
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
        """Initializes the RealEstate LocalExplanation game.

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
        self.setup = GameBenchmarkSetup(
            dataset_name="real_estate",
            model_name=model_name,
            verbose=False,
            random_state=random_state,
        )

        # get x_explain
        x = get_x_explain(x, self.setup.x_test)

        predict_function = self.setup.predict_function

        # call the super constructor
        super().__init__(
            x=x,
            data=self.setup.x_test,
            model=predict_function,
            imputer=imputer,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )

class NHANESI(LocalExplanation):
    """The NHANESI dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq.games.benchmark.setup.GameBenchmarkSetup` object.
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
        """Initializes the NHANESI LocalExplanation game.

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
        self.setup = GameBenchmarkSetup(
            dataset_name="nhanesi",
            model_name=model_name,
            verbose=False,
            random_state=random_state,
        )

        # get x_explain
        x = get_x_explain(x, self.setup.x_test)

        predict_function = self.setup.predict_function

        # call the super constructor
        super().__init__(
            x=x,
            data=self.setup.x_test,
            model=predict_function,
            imputer=imputer,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )

class BreastCancer(LocalExplanation):
    """The BreastCancer dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq.games.benchmark.setup.GameBenchmarkSetup` object.
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
        self.setup = GameBenchmarkSetup(
            dataset_name="breast_cancer",
            model_name=model_name,
            verbose=False,
            random_state=random_state,
        )

        # get x_explain
        x = get_x_explain(x, self.setup.x_test)

        # get class_to_explain
        if class_to_explain is None:
            class_to_explain = int(np.argmax(self.setup.predict_function(x.reshape(1, -1))))

        def predict_function(x: np.ndarray) -> np.ndarray:
            """Predict function for the model."""
            return self.setup.predict_function(x)[:, class_to_explain]

        # call the super constructor
        super().__init__(
            x=x,
            data=self.setup.x_test,
            model=predict_function,
            imputer=imputer,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )

class CommunitiesAndCrime(LocalExplanation):
    """The CommunitiesAndCrime dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq.games.benchmark.setup.GameBenchmarkSetup` object.
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
        """Initializes the CommunitiesAndCrime LocalExplanation game.

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
        self.setup = GameBenchmarkSetup(
            dataset_name="communities_and_crime",
            model_name=model_name,
            verbose=False,
            random_state=random_state,
        )

        # get x_explain
        x = get_x_explain(x, self.setup.x_test)

        predict_function = self.setup.predict_function

        # call the super constructor
        super().__init__(
            x=x,
            data=self.setup.x_test,
            model=predict_function,
            imputer=imputer,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )

class WineQuality(LocalExplanation):
    """The WineQuality dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq.games.benchmark.setup.GameBenchmarkSetup` object.
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
        """Initializes the WineQuality LocalExplanation game.

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
        self.setup = GameBenchmarkSetup(
            dataset_name="wine_quality",
            model_name=model_name,
            verbose=False,
            random_state=random_state,
        )

        # get x_explain
        x = get_x_explain(x, self.setup.x_test)

        predict_function = self.setup.predict_function

        # call the super constructor
        super().__init__(
            x=x,
            data=self.setup.x_test,
            model=predict_function,
            imputer=imputer,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )

class IndependentLinear60(LocalExplanation):
    """The IndependetLinear60 dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq.games.benchmark.setup.GameBenchmarkSetup` object.
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
        """Initializes the IndependetLinear60 LocalExplanation game."""
        self.setup = GameBenchmarkSetup(
            dataset_name="independentlinear60",
            model_name=model_name,
            verbose=False,
            random_state=random_state,
        )

        # get x_explain
        x = get_x_explain(x, self.setup.x_test)

        predict_function = self.setup.predict_function

        # call the super constructor
        super().__init__(
            x=x,
            data=self.setup.x_test,
            model=predict_function,
            imputer=imputer,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )

class Corrgroups60(LocalExplanation):
    """The Corrgroups60 dataset as a LocalExplanation game.

    Attributes:
        setup: The :class:`~shapiq.games.benchmark.setup.GameBenchmarkSetup` object.
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
        """Initializes the Corrgroups60 LocalExplanation game."""
        self.setup = GameBenchmarkSetup(
            dataset_name="corrgroups60",
            model_name=model_name,
            verbose=False,
            random_state=random_state,
        )

        # get x_explain
        x = get_x_explain(x, self.setup.x_test)

        predict_function = self.setup.predict_function

        # call the super constructor
        super().__init__(
            x=x,
            data=self.setup.x_test,
            model=predict_function,
            imputer=imputer,
            random_state=random_state,
            normalize=normalize,
            verbose=verbose,
        )