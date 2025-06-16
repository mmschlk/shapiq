"""The base Explainer classes for the shapiq package."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from tqdm.auto import tqdm

from .utils import (
    get_explainers,
    get_predict_function_and_model_type,
    print_class,
)
from .validation import validate_data_predict_function, validate_index_and_max_order

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    import numpy as np

    from shapiq.approximator.base import Approximator
    from shapiq.game_theory import ExactComputer
    from shapiq.games.base import Game
    from shapiq.games.imputer.base import Imputer
    from shapiq.interaction_values import InteractionValues
    from shapiq.utils import Model

    from .custom_types import ExplainerIndices


class Explainer:
    """The main Explainer class for a simpler user interface.

    shapiq.Explainer is a simplified interface for the ``shapiq`` package. It detects between
    :class:`~shapiq.explainer.tabular.TabularExplainer`,
    :class:`~shapiq.explainer.tree.TreeExplainer`,
    and :class:`~shapiq.explainer.tabpfn.TabPFNExplainer`. For a detailed description of the
    different explainers, see the respective classes.
    """

    approximator: Approximator | None
    """The approximator which may be used for the explanation."""

    exact_computer: ExactComputer | None
    """An exact computer which computes the :class:`~shapiq.interaction_values.InteractionValues`
    exactly (without the need for approximations). Note that this only works for small number of
    features as the number of coalitions grows exponentially with the number of features.
    """

    imputer: Imputer | None
    """An imputer which is used to impute missing values in computing the interaction values."""

    model: Model | Game | Callable[[np.ndarray], np.ndarray]
    """The model to be explained, either as a Model instance or a callable function."""

    def __init__(
        self,
        model: Model | Game | Callable[[np.ndarray], np.ndarray],
        data: np.ndarray | None = None,
        class_index: int | None = None,
        index: ExplainerIndices = "k-SII",
        max_order: int = 2,
        **kwargs: Any,
    ) -> None:
        """Initialize the Explainer class.

        Args:
            model: The model object to be explained.

            data: A background dataset to be used for imputation in
                :class:`~shapiq.explainer.tabular.TabularExplainer` or
                :class:`~shapiq.explainer.tabpfn.TabPFNExplainer`. This is a 2-dimensional
                NumPy array with shape ``(n_samples, n_features)``. Can be ``None`` for the
                :class:`~shapiq.explainer.tree.TreeExplainer`, which does not require background
                data.

            class_index: The class index of the model to explain. Defaults to ``None``, which will
                set the class index to ``1`` per default for classification models and is ignored
                for regression models. Note, it is important to specify the class index for your
                classification model.

            index: The type of Shapley interaction index to use. Defaults to ``"k-SII"``, which
                computes the k-Shapley Interaction Index. If ``max_order`` is set to 1, this
                corresponds to the Shapley value (``index="SV"``). Options are:
                - ``"SV"``: Shapley value
                - ``"k-SII"``: k-Shapley Interaction Index
                - ``"FSII"``: Faithful Shapley Interaction Index
                - ``"FBII"``: Faithful Banzhaf Interaction Index (becomes ``BV`` for order 1)
                - ``"STII"``: Shapley Taylor Interaction Index
                - ``"SII"``: Shapley Interaction Index

            max_order: The maximum interaction order to be computed. Defaults to ``2``. Set to
                ``1`` for no interactions (single feature attribution).

            **kwargs: Additional keyword-only arguments passed to the specific explainer classes.

        """
        # If Explainer is instantiated directly, dynamically dispatch to the appropriate subclass
        if self.__class__ is Explainer:
            model_class = print_class(model)
            _, model_type = get_predict_function_and_model_type(model, model_class, class_index)
            explainer_classes = get_explainers()
            if model_type in explainer_classes:
                explainer_cls = explainer_classes[model_type]
                self.__class__ = explainer_cls
                explainer_cls.__init__(
                    self,
                    model=model,
                    data=data,
                    class_index=class_index,
                    index=index,
                    max_order=max_order,
                    **kwargs,
                )
                return  # avoid continuing in base Explainer
            msg = f"Model '{model_class}' with type '{model_type}' is not supported by shapiq.Explainer."
            raise TypeError(msg)

        # proceed with the base Explainer initialization
        self._model_class = print_class(model)
        self._shapiq_predict_function, self._model_type = get_predict_function_and_model_type(
            model, self._model_class, class_index
        )

        # validate the model and data
        self.model = model
        if data is not None:
            validate_data_predict_function(data, predict_function=self.predict, raise_error=False)
        self._data: np.ndarray | None = data

        # validate index and max_order and set them as attributes
        self._index, self._max_order = validate_index_and_max_order(index, max_order)

        # set the class attributes
        self.approximator = None
        self.exact_computer = None
        self.imputer = None

    @property
    def index(self) -> ExplainerIndices:
        """The type of Shapley interaction index the explainer is using."""
        return self._index

    @property
    def max_order(self) -> int:
        """The maximum interaction order the explainer is using."""
        return self._max_order

    def explain(self, x: np.ndarray | None = None, **kwargs: Any) -> InteractionValues:
        """Explain a single prediction in terms of interaction values.

        Args:
            x: A numpy array of a data point to be explained.
            **kwargs: Additional keyword-only arguments passed to the specific explainer's
                ``explain_function`` method.

        Returns:
            The interaction values of the prediction.

        """
        return self.explain_function(x=x, **kwargs)

    def set_random_state(self, random_state: int | None = None) -> None:
        """Set the random state for the explainer and its components.

        Note:
            Setting the random state in the explainer will also overwrite the random state
            in the approximator and imputer, if they are set.

        Args:
            random_state: The random state to set. If ``None``, no random state is set.

        """
        if random_state is None:
            return

        if self.approximator is not None:
            self.approximator.set_random_state(random_state=random_state)

        if self.imputer is not None:
            self.imputer.set_random_state(random_state=random_state)

    @abstractmethod
    def explain_function(self, x: np.ndarray, *args: Any, **kwargs: Any) -> InteractionValues:
        """Explain a single prediction in terms of interaction values.

        Args:
            x: A numpy array of a data point to be explained.
            *args: Additional positional arguments passed to the explainer.
            **kwargs: Additional keyword-only arguments passed to the explainer.

        Returns:
            The interaction values of the prediction.

        """
        msg = "The method `explain` must be implemented in a subclass."
        raise NotImplementedError(msg)

    def explain_X(
        self,
        X: np.ndarray,
        *,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list[InteractionValues]:
        """Explain multiple predictions at once.

        This method is a wrapper around the ``explain`` method. It allows to explain multiple
        predictions at once. It is a convenience method that uses the ``joblib`` library to
        parallelize the computation of the interaction values.

        Args:
            X: A 2-dimensional matrix of inputs to be explained with shape (n_samples, n_features).

            n_jobs: Number of jobs for ``joblib.Parallel``. Defaults to ``None``, which will
                use no parallelization. If set to ``-1``, all available cores will be used.

            random_state: The random state to re-initialize Imputer and Approximator with. Defaults
                to ``None``.

            verbose: Whether to print a progress bar. Defaults to ``False``.

            **kwargs: Additional keyword-only arguments passed to the explainer's
                ``explain_function`` method.

        Returns:
            A list of interaction values for each prediction in the input matrix ``X``.

        """
        if len(X.shape) != 2:
            msg = "The `X` must be a 2-dimensional matrix."
            raise TypeError(msg)

        self.set_random_state(random_state=random_state)

        if n_jobs:  # parallelization with joblib
            import joblib

            parallel = joblib.Parallel(n_jobs=n_jobs)
            ivs = parallel(
                joblib.delayed(self.explain)(X[i, :], **kwargs) for i in range(X.shape[0])
            )
        else:
            ivs = []
            pbar = tqdm(total=X.shape[0], desc="Explaining") if verbose else None
            for i in range(X.shape[0]):
                ivs.append(self.explain(X[i, :], **kwargs))
                if pbar is not None:
                    pbar.update(1)
        return ivs

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Provides a unified prediction interface for the explainer.

        Args:
            x: An instance/point/sample/observation to be explained.

        Returns:
            The model's prediction for the given data point as a vector.
        """
        return self._shapiq_predict_function(self.model, x)
