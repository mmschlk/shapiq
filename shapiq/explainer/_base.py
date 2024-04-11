"""This module contains the base explainer classes for the shapiq package."""

import numpy as np

from shapiq.interaction_values import InteractionValues

from . import utils


class Explainer:
    """The main Explainer class for a simpler user interface."""

    def __init__(self, model, data=None, **kwargs) -> None:
        self.model = model
        self.data = data

        self._model_class = utils.print_class(model)
        self._predict_function, self._model_type = utils.get_predict_function_and_model_type(
            model, self._model_class
        )

        if data is not None:
            try:
                x = self.predict(data)
                if isinstance(x, np.ndarray):
                    if x.size > 1:
                        raise ValueError()
                else:
                    raise ValueError()
            except Exception as e:
                print(f"Error: The data provided is not compatible with the model. {e}")
                pass

        # not super()
        if self.__class__ is Explainer:
            if self._model_type in list(utils.get_explainers()):
                _explainer = utils.get_explainers()[self._model_type]
                self.__class__ = _explainer
                _explainer.__init__(self, model=model, data=data, **kwargs)

    def explain(self, x: np.ndarray) -> InteractionValues:
        """Explain the model's prediction in terms of interaction values."""
        return {}

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Provides a unified prediction interface."""
        return self._predict_function(self.model, x)
