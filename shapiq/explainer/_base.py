"""The base Explainer classes for the shapiq package."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from warnings import warn

import numpy as np
from tqdm.auto import tqdm

from shapiq.explainer.utils import get_explainers, get_predict_function_and_model_type, print_class

if TYPE_CHECKING:
    from typing import Any

    from shapiq.interaction_values import InteractionValues
    from shapiq.utils import Model


class Explainer:
    """The main Explainer class for a simpler user interface.

    shapiq.Explainer is a simplified interface for the ``shapiq`` package. It detects between
    :class:`~shapiq.explainer.tabular.TabularExplainer`,
    :class:`~shapiq.explainer.tree.TreeExplainer`,
    and :class:`~shapiq.explainer.tabpfn.TabPFNExplainer`. For a detailed description of the
    different explainers, see the respective classes.

    Attributes:
        model: The model object to be explained.
        data: A background data to use for the explainer.

    """

    def __init__(
        self,
        model: Model,
        data: np.ndarray | None = None,
        class_index: int | None = None,
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

            **kwargs: Additional keyword-only arguments passed to the specific explainer classes.

        """
        self._model_class = print_class(model)
        self._shapiq_predict_function, self._model_type = get_predict_function_and_model_type(
            model,
            self._model_class,
            class_index,
        )
        self.model = model

        if data is not None and self._model_type != "tabpfn":
            self._validate_data(data, raise_error=False)
        self.data = data

        # if the class was not run as super
        if self.__class__ is Explainer and self._model_type in list(get_explainers()):
            _explainer = get_explainers()[self._model_type]
            self.__class__ = _explainer
            _explainer.__init__(self, model=model, data=data, class_index=class_index, **kwargs)

    def _validate_data(self, data: np.ndarray, *, raise_error: bool = False) -> None:
        """Validate the data for compatibility with the model.

        Args:
            data: A 2-dimensional matrix of inputs to be explained.
            raise_error: Whether to raise an error if the data is not compatible with the model or
                only print a warning. Defaults to ``False``.

        Raises:
            TypeError: If the data is not a NumPy array.

        """
        message = ""

        # check input data type
        if not isinstance(data, np.ndarray):
            message += " The `data` must be a NumPy array."

        try:
            data_to_pred = data[0:1, :]
        except Exception as e:
            message += " The `data` must have at least one sample and be 2-dimensional."
            raise TypeError(message) from e

        try:
            pred = self.predict(data_to_pred)
        except Exception as e:
            message += f" The model's prediction failed with the following error: {e}."
            raise TypeError(message) from e

        if isinstance(pred, np.ndarray):
            if len(pred.shape) != 1:
                message += " The model's prediction must be a 1-dimensional array."
        else:
            message += " The model's prediction must be a NumPy array."

        if message != "":
            message = "The `data` and the model must be compatible." + message
            if raise_error:
                raise TypeError(message)
            warn(message, stacklevel=2)

    def explain(self, x: np.ndarray, **kwargs: Any) -> InteractionValues:
        """Explain a single prediction in terms of interaction values.

        Args:
            x: A numpy array of a data point to be explained.
            **kwargs: Additional keyword-only arguments passed to the explainer.

        Returns:
            The interaction values of the prediction.

        """
        return self.explain_function(x=x, **kwargs)

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

        if random_state is not None:
            if hasattr(self, "_imputer"):
                self._imputer._rng = np.random.default_rng(random_state)  # noqa: SLF001
            if hasattr(self, "_approximator"):
                self._approximator._rng = np.random.default_rng(random_state)  # noqa: SLF001
                if hasattr(self._approximator, "_sampler"):
                    self._approximator._sampler._rng = np.random.default_rng(random_state)  # noqa: SLF001

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
