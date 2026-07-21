"""Imputer for vision models."""

from __future__ import annotations

import numpy as np

from shapiq.imputer.base import Imputer

from .architecture import ModelArchitecture
from .utils import ImageLike, as_hwc_array, tensor_to_numpy

try:
    import torch
except ImportError as err:
    from ._error import _vision_import_error

    raise _vision_import_error from err


class ImageImputer(Imputer):
    """Imputer for images: creates masked versions of the input image based on player coalitions and returns model predictions.

    Requires pytorch to be installed for tensor operations.
    Converts images to numpy arrays internally, and to tensors for model inference.
    """

    def __init__(
        self,
        model: ModelArchitecture,
        image: ImageLike,
        *,
        normalize: bool = True,
        batch_size: int = 32,
        class_index: int | None = None,
    ) -> None:
        """Initialise the imputer for a specific image and model architecture.

        Args:
            model: A configured
                :class:`~shapiq.vision.architecture.ModelArchitecture`
                (e.g. :class:`~shapiq.vision.architecture.ClassificationArchitecture` or
                :class:`~shapiq.vision.architecture.ViTClassificationArchitecture`).
                This object owns the model, the player strategy, and the masking
                strategy. Sensible defaults are chosen automatically if no custom
                strategies are passed to the architecture constructor.
            image: The image to explain. Accepts a PIL Image, numpy array
                ``(H, W, C)`` or ``(C, H, W)``, or a PyTorch tensor.
            normalize: If ``True``, the empty-coalition prediction is used as
                the normalization baseline for interaction values.
            batch_size: Maximum number of coalitions to evaluate in a single
                model forward pass.
            class_index: Optional index of the class to explain. If not provided,
            the class with the highest logit is used.

        Raises:
            TypeError: If the model is not a ModelArchitecture instance.
        """
        if not isinstance(model, ModelArchitecture):
            msg = "ImageImputer expects a ModelArchitecture instance for the model."
            raise TypeError(msg)
        self.architecture = model
        self._batch_size = batch_size
        self._normalize = normalize

        self._image: np.ndarray = as_hwc_array(image)
        self.architecture.prepare(self._image, class_index)
        self.n_features = self.architecture.n_players

        dummy_data = np.zeros((1, self.n_features))
        super().__init__(model=model.model, data=dummy_data)

        self.empty_prediction = self.calc_empty_prediction()
        if self._normalize:
            self.normalization_value = self.empty_prediction

    def fit(self, x: ImageLike) -> ImageImputer:
        """Fits the imputer to a new image.

        Replaces the current image, re-runs player and masking strategy
        preparation, and resets the empty prediction baseline.

        Args:
            x: A new image to explain. Accepts PIL Image, numpy array
            (H, W, C) or (C, H, W), or a torch Tensor.

        Returns:
            The fitted imputer (self).
        """
        self._image = as_hwc_array(x)
        self.architecture.prepare(self._image)
        self.n_features = self.architecture.n_players

        self.n_players = self.n_features
        self.empty_coalition = np.zeros(self.n_players, dtype=bool)
        self.grand_coalition = np.ones(self.n_players, dtype=bool)
        self._empty_coalition_value_property = None
        self._grand_coalition_value_property = None

        self._x = np.zeros((1, self.n_features), dtype=bool)  # dummy data to satisfy base class

        self.empty_prediction = self.calc_empty_prediction()
        if self._normalize:
            self.normalization_value = self.empty_prediction

        return self

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate the model for a batch of player coalitions.

        Converts ``coalitions`` to a boolean PyTorch tensor, splits it into
        mini-batches of at most :attr:`batch_size` rows, and issues one
        ``model`` forward call per mini-batch.

        Args:
            coalitions: Boolean array of shape ``(n_coalitions, n_players)``

        Returns:
            Float numpy array of shape ``(n_coalitions,)`` containing the
            scalar model output (logit or probability depending on
            :attr:`architecture`) for each coalition.

        """
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)

        n = len(coalitions)
        if n <= self._batch_size:
            coalitions_t = torch.from_numpy(coalitions).bool()
            return tensor_to_numpy(self.architecture.value_function(coalitions_t))

        chunks = [
            tensor_to_numpy(
                self.architecture.value_function(
                    torch.from_numpy(coalitions[start : start + self._batch_size]).bool()
                )
            )
            for start in range(0, n, self._batch_size)
        ]
        return np.concatenate(chunks, axis=0)

    def calc_empty_prediction(self) -> float:
        """Evaluate the model with all players absent to obtain the baseline prediction.

        Returns:
            The scalar model output when no players are present.
        """
        return float(self.value_function(np.zeros((1, self.n_features), dtype=bool))[0])

    @property
    def image(self) -> np.ndarray:
        """Returns the current explanation image as an HWC numpy array."""
        return self._image.copy()

    @property
    def player_masks(self) -> np.ndarray:
        """Spatial masks per player as a ``(n_players, H, W)`` boolean numpy array.

        Returns:
            Boolean numpy array of shape ``(n_players, H, W)``.
        """
        return tensor_to_numpy(self.architecture.player_masks)
