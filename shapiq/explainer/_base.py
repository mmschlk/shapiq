"""The base Explainer classes for the shapiq package."""

from abc import abstractmethod

import numpy as np
from tqdm.auto import tqdm

from ..explainer.utils import get_explainers, get_predict_function_and_model_type, print_class
from ..interaction_values import InteractionValues
from .validation import set_random_state, validate_data, validate_index, validate_max_order


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
        self,
        model,
        data: np.ndarray | None = None,
        class_index: int | None = None,
        index: str = "k-SII",
        max_order: int = 2,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:

        # validate index/max_order and set some useful attributes
        self.index = validate_index(index, max_order)
        self._max_order = validate_max_order(index, max_order)  # todo maybe private
        self._random_state = random_state
        self.verbose = verbose

        # validate the model and data
        self._model_class = print_class(model)
        self._shapiq_predict_function, self._model_type = get_predict_function_and_model_type(
            model, self._model_class, class_index
        )
        self.model = model

        # validate and set the data
        if data is not None:
            if self._model_type != "tabpfn":
                validate_data(data, predict_function=self.predict)
        self.data = data

        # if the class is Explainer, set the class to the respective child explainer and init it
        if self.__class__ is Explainer:
            if self._model_type in list(get_explainers()):
                _explainer = get_explainers()[self._model_type]
                self.__class__ = _explainer
                _explainer.__init__(self, model=model, data=data, class_index=class_index, **kwargs)

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
    def explain_function(
        self,
        x: np.ndarray,
        budget: int | None = None,
        random_state: int | None = None,
        *args,
        **kwargs,
    ) -> InteractionValues:
        """Explain a single prediction in terms of interaction values.

        Args:
            x: A numpy array of a data point to be explained.
            budget: The budget to use for the approximation. Defaults to `None`, which will
                set the budget to 2**n_features based on the number of features.
            random_state: The random state to re-initialize Imputer and Approximator with.
                Defaults to ``None``.
            *args: Additional positional arguments passed to the explainer.
            **kwargs: Additional keyword-only arguments passed to the explainer.

        Returns:
            The interaction values of the prediction.
        """
        raise NotImplementedError("`explain_function` must be implemented in a subclass.")

    def explain_X(
        self, X: np.ndarray, n_jobs: int | None = None, random_state: int | None = None, **kwargs
    ) -> list[InteractionValues]:
        """Explain multiple predictions in terms of interaction values.

        Args:
            X: A 2-dimensional matrix of inputs to be explained.
            n_jobs: Number of jobs for ``joblib.Parallel``.
            random_state: The random state to re-initialize Imputer and Approximator with.
            Defaults to ``None``.
        """
        assert len(X.shape) == 2
        set_random_state(random_state, self)
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

        Returns:
            The prediction of the model.
        """
        return self._shapiq_predict_function(self.model, x)
