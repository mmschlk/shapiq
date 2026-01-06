"""Implementation of NNExplainerBase."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from sklearn.utils.validation import check_is_fitted

from shapiq import Explainer

if TYPE_CHECKING:
    import numpy.typing as npt
    from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier


class NNExplainerBase(Explainer):
    """Base class for nearest-neighbor Explainers."""

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

        This method extracts the training data from the provided model.

        Args:
            model: The NN model to explain. Must be an instance of ``sklearn.neighbors.KNeighborsClassifier`` or ``sklearn.neighbors.RadiusNeighborsClassifier``.
                The model must not use multi-output classification, i.e. the ``y`` value provided to ``model.fit(X, y)`` must be a 1D vector.
            class_index: The class index of the model to explain. Defaults to ``1``.

        Raises:
            sklearn.exceptions.NotFittedError: The constructor was called with a model that hasn't been fitted.
        """
        super().__init__(model, data=None, class_index=class_index, index="SV", max_order=1)
        check_is_fitted(model)

        X_train = _sklearn_model_get_private_attribute(model, "_fit_X")
        if not isinstance(X_train, np.ndarray):
            msg = f"Expected model's training features (model._fit_X) to be numpy array but got {type(X_train)}"
            raise TypeError(msg)
        if not (
            np.issubdtype(X_train.dtype, np.floating) or np.issubdtype(X_train.dtype, np.integer)
        ):
            msg = f"Expected dtype of model's training features (model._fit_X) to be a subtype of np.floating or np.integer, but got {X_train.dtype}"
            raise TypeError(msg)
        if np.issubdtype(X_train.dtype, np.integer):
            X_train = X_train.astype(np.float32)
        self.X_train = X_train

        y_train_indices = _sklearn_model_get_private_attribute(model, "_y")
        if not isinstance(y_train_indices, np.ndarray):
            msg = f"Expected model's training class indices (model._y) to be numpy array but got {type(y_train_indices)}"
            raise TypeError(msg)
        if not np.issubdtype(y_train_indices.dtype, np.integer):
            msg = f"Expected dtype of model's training class indices (model._y) to be a subtype of np.integer, but got {y_train_indices.dtype}"
            raise TypeError(msg)
        if y_train_indices.ndim != 1:
            msg = "Multi-output nearest-neighbor classifiers are not supported. Make sure to pass the training labels as a 1D vector when calling `model.fit()`."
            raise ValueError(msg)
        self.y_train_indices = y_train_indices

        if not isinstance(model.classes_, np.ndarray):
            msg = f"Expected model's training classes (model.classes_) to be numpy array but got {type(model.classes_)}"
            raise TypeError(msg)
        self.y_train_classes = cast("npt.NDArray[np.object_]", model.classes_)

        # TODO(Zaphoood): Fix this sketchiness  # noqa: TD003
        # This is highly sketchy. We are relying on `shapiq` to handle `class_index == None` analogously, but there is no way to check, since they don't set `class_index` as an attribute of `shapiq.Explainer`
        if class_index is None:
            class_index = 1
        self.class_index = class_index


def _sklearn_model_get_private_attribute(
    model: KNeighborsClassifier | RadiusNeighborsClassifier, attribute: str
) -> object:
    if not attribute.startswith("_"):
        msg = f"Name of private attribute must start with underscore, but got '{attribute}'"
        raise ValueError(msg)

    try:
        return model.__getattribute__(attribute)
    except AttributeError as e:
        msg = (
            f"Failed to access private attribute '{attribute}' of sklearn model. This may be caused by a change to the "
            "implementation of the sklearn library. Please report this problem at https://github.com/mmschlk/shapiq/issues"
        )
        raise AttributeError(msg) from e
