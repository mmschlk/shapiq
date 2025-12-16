"""Implementation of NNExplainerBase."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, cast

from sklearn.utils.validation import check_is_fitted

from shapiq import Explainer

from .exceptions import MultiOutputKNNError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt
    from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier


class NNExplainerBase(Explainer):
    """Base class for nearest-neighbor Explainers.

    Based on the model passed to the constructor, the class automatically detects between
    :class:`~shapiq_student.explainer.knn.NormalKNNExplainer` and
    :class:`~shapiq_student.explainer.knn.WeightedKNNExplainer` for (weighted) :math:`k`-nearest neighbor models,
    as well as :class:`~shapiq_student.explainer.knn.ThresholdNNExplainer` for threshold nearest neighbor models.

    For a detailed description of the different explainers, see the respective classes.
    """

    X_train: npt.NDArray[np.floating]
    """Training data features extracted from the model."""

    y_train_indices: npt.NDArray[np.integer]
    """Training data labels as indices into the classes array."""

    y_train_classes: npt.NDArray[np.object_]
    """Class labels from the model's training data."""

    def __init__(
        self,
        model: KNeighborsClassifier | RadiusNeighborsClassifier,
        class_index: int | None = None,
    ) -> None:
        """Initializes the class.

        This method extracts the training data from the provided model and stores it in a class member.

        Args:
            model: The KNN model to explain. Must be an instance of ``sklearn.neighbors.KNeighborsClassifier`` or ``sklearn.neighbors.RadiusNeighborsClassifier``.
                The model must not use multi-output classification, i.e. the ``y`` value provided to ``model.fit(X, y)`` must be a 1D vector.
            data: This parameter is currently ignored but may be used in future versions.
            labels: This parameter is currently ignored but may be used in future versions.
            class_index: The class index of the model to explain. Defaults to ``1``.

        Raises:
            sklearn.exceptions.NotFittedError: The constructor was called with a model that hasn't been fitted.
            shapiq_student.explainer.knn.exceptions.MultiOutputKNNError: The constructor was called with a model that uses multi-output classification.
        """
        super().__init__(model, data=None, class_index=class_index, index="SV", max_order=1)
        check_is_fitted(model)

        self.X_train = model._fit_X  # type: ignore[union-attr] # noqa: SLF001
        self.y_train_indices = cast("npt.NDArray[np.integer]", model._y)  # type: ignore[union-attr] # noqa: SLF001
        if self.y_train_indices.ndim != 1:
            raise MultiOutputKNNError
        self.y_train_classes = cast("npt.NDArray[np.object_]", model.classes_)

        # This is highly sketchy. We are relying on `shapiq` to handle `class_index == None` analogously, but there is no way to check, since they don't set `class_index` as an attribute of `shapiq.Explainer`
        if class_index is None:
            class_index = 1
        self.class_index = class_index

    # TODO(Zaphoood): Consider removing this property  # noqa: TD003
    @property
    @abstractmethod
    def mode(self) -> str:
        """The mode in which the Explainer operates."""
        msg = f"Each subclass of {self.__class__.__name__} must implement the mode() property"
        raise NotImplementedError(msg)
