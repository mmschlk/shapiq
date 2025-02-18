"""The base Explainer classes for the shapiq package."""

from abc import abstractmethod
from warnings import warn

import numpy as np
from tqdm.auto import tqdm

from ..explainer.utils import get_explainers, get_predict_function_and_model_type, print_class
from ..interaction_values import InteractionValues


class Explainer:
    """The main Explainer class for a simpler user interface.

    shapiq.Explainer is a simplified interface for the ``shapiq`` package. It detects between
    :class:`~shapiq.explainer.tabular.TabularExplainer`,
    :class:`~shapiq.explainer.tree.TreeExplainer`,
    and :class:`~shapiq.explainer.tabpfn.TabPFNExplainer`. For a detailed description of the
    different explainers, see the respective classes.

    Args:
        model: The model object to be explained.
        data: A background dataset to be used for imputation in ``TabularExplainer``.
        class_index: The class index of the model to explain. Defaults to ``None``, which will set
            the class index to ``1`` per default for classification models and is ignored for
            regression models.
        **kwargs: Additional keyword-only arguments passed to ``TabularExplainer`` or ``TreeExplainer``.

    Attributes:
        model: The model object to be explained.
        data: A background data to use for the explainer.
    """

    def __init__(
        self, model, data: np.ndarray | None = None, class_index: int | None = None, **kwargs
    ) -> None:

        self._model_class = print_class(model)
        self._shapiq_predict_function, self._model_type = get_predict_function_and_model_type(
            model, self._model_class, class_index
        )
        self.model = model

        if data is not None:
            if self._model_type != "tabpfn":
                self._validate_data(data)
        self.data = data

        # not super()
        if self.__class__ is Explainer:
            if self._model_type in list(get_explainers()):
                _explainer = get_explainers()[self._model_type]
                self.__class__ = _explainer
                _explainer.__init__(self, model=model, data=data, class_index=class_index, **kwargs)

    def _validate_data(self, data: np.ndarray, raise_error: bool = False) -> None:
        """Validate the data for compatibility with the model.

        Args:
            data: A 2-dimensional matrix of inputs to be explained.
            raise_error: Whether to raise an error if the data is not compatible with the model or
                only print a warning. Defaults to ``False``.

        Raises:
            TypeError: If the data is not a NumPy array.
        """
        message = "The `data` and the model must be compatible."
        if not isinstance(data, np.ndarray):
            message += " The `data` must be a NumPy array."
            raise TypeError(message)
        try:
            # TODO (mmschlk): This can take a long time for large datasets and slow models
            pred = self.predict(data)
            if isinstance(pred, np.ndarray):
                if len(pred.shape) > 1:
                    message += " The model's prediction must be a 1-dimensional array."
                    raise ValueError()
            else:
                message += " The model's prediction must be a NumPy array."
                raise ValueError()
        except Exception as e:
            if raise_error:
                raise ValueError(message) from e
            else:
                warn(message)

    def explain(self, x: np.ndarray, *args, **kwargs) -> InteractionValues:
        """Explain a single prediction in terms of interaction values.

        Args:
            x: A numpy array of a data point to be explained.
            *args: Additional positional arguments passed to the explainer.
            **kwargs: Additional keyword-only arguments passed to the explainer.

        Returns:
            The interaction values of the prediction.
        """
        explanation = self.explain_function(x=x, *args, **kwargs)
        if explanation.min_order == 0:
            explanation[()] = explanation.baseline_value
        return explanation

    @abstractmethod
    def explain_function(self, x: np.ndarray, *args, **kwargs) -> InteractionValues:
        """Explain a single prediction in terms of interaction values.

        Args:
            x: A numpy array of a data point to be explained.
            *args: Additional positional arguments passed to the explainer.
            **kwargs: Additional keyword-only arguments passed to the explainer.

        Returns:
            The interaction values of the prediction.
        """
        raise NotImplementedError("The method `explain` must be implemented in a subclass.")

    def explain_X(
        self, X: np.ndarray, n_jobs=None, random_state=None, **kwargs
    ) -> list[InteractionValues]:
        """Explain multiple predictions in terms of interaction values.

        Args:
            X: A 2-dimensional matrix of inputs to be explained.
            n_jobs: Number of jobs for ``joblib.Parallel``.
            random_state: The random state to re-initialize Imputer and Approximator with. Defaults to ``None``.
        """
        assert len(X.shape) == 2
        if random_state is not None:
            if hasattr(self, "_imputer"):
                self._imputer._rng = np.random.default_rng(random_state)
            if hasattr(self, "_approximator"):
                self._approximator._rng = np.random.default_rng(random_state)
                if hasattr(self._approximator, "_sampler"):
                    self._approximator._sampler._rng = np.random.default_rng(random_state)
        if n_jobs:
            import joblib

            parallel = joblib.Parallel(n_jobs=n_jobs)
            ivs = parallel(
                joblib.delayed(self.explain)(X[i, :], **kwargs) for i in range(X.shape[0])
            )
        else:
            ivs = []
            for i in tqdm(range(X.shape[0])):
                ivs.append(self.explain(X[i, :], **kwargs))
        return ivs

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Provides a unified prediction interface.

        Args:
            x: An instance/point/sample/observation to be explained.
        """
        return self._shapiq_predict_function(self.model, x)
