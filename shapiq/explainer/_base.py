"""This module contains the base explainer classes for the shapiq package."""

import re

import numpy as np

from .. import explainer
from shapiq.interaction_values import InteractionValues


class Explainer:
    """The main Explainer class for a simple user interface."""
    def __init__(
        self,
        model,
        data=None,
        **kwargs
    ) -> None:
        
        self.model = model
        self.data = data

        _model_type = "tree" # tree / base etc.

        # not super()
        if self.__class__ is Explainer:
            EXPLAINERS = {'tree': explainer.TreeExplainer, 'tabular': explainer.TabularExplainer}
            EXPLAINERS_NICELY = [".".join([re.search("(?<=<class ').*(?='>)", str(v))[0].split(".")[i] for i in (0, -1)])\
                                for _, v in EXPLAINERS.items()]
            if _model_type in list(EXPLAINERS):
                _explainer = EXPLAINERS[_model_type]
                self.__class__ = _explainer
                _explainer.__init__(self, model=model, data=data, **kwargs)
            else:
                raise TypeError(f'`model` is of unsupported type: {type(model)}.\
                                Please, raise a new issue at https://github.com/mmschlk/shapiq/issues if\
                                you want this model type to be handled automatically by shapiq.Explainer.\
                                Otherwise, use one of the supported explainers: {EXPLAINERS_NICELY}.')


    def explain(self, x: np.ndarray) -> InteractionValues:
        """Explain the model's prediction in terms of interaction values."""
        return {}
            

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Provides a unified prediction interface."""
        return self.model(x)