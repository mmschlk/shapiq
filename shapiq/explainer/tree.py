"""This module contains the TreeSHAP-IQ explainer for computing exact any order Shapley interactions
for trees and tree ensembles."""
import numpy as np
from approximator._interaction_values import InteractionValues
from explainer._base import Explainer

__all__ = ["TreeExplainer"]


class TreeExplainer(Explainer):
    def __init__(self) -> None:
        raise NotImplementedError(
            "The TreeExplainer is not yet implemented. An initial version can be found here: "
            "'https://github.com/mmschlk/TreeSHAP-IQ'."
        )

    def explain(self, x_explain: np.ndarray) -> InteractionValues:
        pass
