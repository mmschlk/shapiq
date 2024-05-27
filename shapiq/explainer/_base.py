"""The base Explainer classes for the shapiq package."""

import numpy as np

from shapiq.interaction_values import InteractionValues

from . import utils


class Explainer:
    """The main Explainer class for a simpler user interface.
    
    shapiq.Explainer is a simplified interface for the `shapiq` package. It detects between
    TabularExplainer and TreeExplainer based on the model class.

    Args:
        model: The model object to be explained.
        data: A background dataset to be used in TabularExplainer.
        **kwargs: Additional keyword-only arguments passed to TabularExplainer or TreeExplainer.

    Attributes:
        model: The model object to be explained.
        data: A background data to use for the explainer.
    """

    def __init__(
            self, 
            model, 
            data: np.ndarray = None, 
            **kwargs
        ) -> None:

        self._model_class = utils.print_class(model)
        self._predict_function, self._model_type = utils.get_predict_function_and_model_type(
            model, self._model_class
        )
        self.model = model

        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("`data` must be a NumPy array.")
            try:
                pred = self.predict(data)
                if isinstance(pred, np.ndarray):
                    if len(pred.shape) > 1:
                        raise ValueError()
                else:
                    raise ValueError()
            except Exception as e:
                print(f"Error: The `data` provided is not compatible with the model. {e}")
                pass
        self.data = data

        # not super()
        if self.__class__ is Explainer:
            if self._model_type in list(utils.get_explainers()):
                _explainer = utils.get_explainers()[self._model_type]
                self.__class__ = _explainer
                _explainer.__init__(self, model=model, data=data, **kwargs)

    def explain(self, x: np.ndarray) -> InteractionValues:
        """Explain the model's prediction in terms of interaction values.
        
        Args:
            x: An instance/point/sample/observation to be explained.
        """
        return {}

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Provides a unified prediction interface.

        Args:
            x: An instance/point/sample/observation to be explained.
        """
        return self._predict_function(self.model, x)
