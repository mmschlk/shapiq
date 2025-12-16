from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from .base import NNExplainerBase

if TYPE_CHECKING:
    from sklearn.neighbors import KNeighborsClassifier


class SupportedKNNWeights(Enum):
    """Enumeration of all supported weights types for sklearn KNN models."""

    uniform = "uniform"
    distance = "distance"


# TODO(Zaphoood): Consider removing this class entirely.  # noqa: TD003
#  - The automagic dispatch will be handled by the base Explainer
#  - Extracting the model and parameter k can be done by each of the *KNNExplainers individually
class _CommonKNNExplainer(NNExplainerBase):
    """Helper class that performs functionality shared between NormalKNNExplainer and WeightedKNNExplainer."""

    def __init__(
        self,
        model: KNeighborsClassifier,
        class_index: int | None = None,
    ) -> None:
        """Initializes the class.

        This method extracts the training data as well as the parameter :math:`k` from the provided KNN model and stores them as class members.

        Args:
            model: The KNN model to explain. The model must not use multi-output classification, i.e. the ``y`` value provided to ``model.fit(X, y)`` must be a 1D vector.
            data: This parameter is currently ignored but may be used in future versions.
            labels: This parameter is currently ignored but may be used in future versions.
            class_index: The class index of the model to explain. Defaults to ``1``.

        Raises:
            sklearn.exceptions.NotFittedError: The constructor was called with a model that hasn't been fitted.
            shapiq_student.explainer.knn.exceptions.MultiOutputKNNError: The constructor was called with a model that uses multi-output classification.
        """
        super().__init__(model, class_index=class_index)

        # The type of the superclass's `model` attribute is to broad, since it also allows for other KNN explainers
        # To circumvent this, we store the model separately in an attribute with a narrower type
        self.knn_model = model
        self.k: int = self.knn_model.n_neighbors  # type: ignore[attr-defined]
