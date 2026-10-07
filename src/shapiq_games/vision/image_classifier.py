"""Image classification games: the class probability with regions of the image removed."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

import numpy as np
from PIL import Image

from shapiq.game import Game
from shapiq_games._base import ConfigMixin, as_bool_coalitions
from shapiq_games.datasets import load_example_image

from ._superpixels import get_superpixels
from ._vit import VIT_PATCH_GRIDS, ViTPatchModel

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["ImageClassifier"]

type BuiltinModel = Literal[
    "vit_9_patches", "vit_16_patches", "vit_36_patches", "vit_144_patches", "resnet_18"
]
_VIT_MODELS = {f"vit_{n}_patches": n for n in VIT_PATCH_GRIDS}


def _as_rgb_array(image: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image, str | Path):
        with Image.open(image) as img:
            return np.asarray(img.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        msg = f"Expected an RGB image of shape (height, width, 3), got shape {array.shape}."
        raise ValueError(msg)
    return np.ascontiguousarray(array[..., :3]).astype(np.uint8)


class ImageClassifier(ConfigMixin, Game):
    """The image classification game: the probability of a class when only some regions are visible.

    The players are regions of the image:

    - for the vision transformers (``"vit_9_patches"``, ``"vit_16_patches"``, ``"vit_36_patches"``,
      ``"vit_144_patches"``), square groups of the model's patches, which are removed with a zero
      mask token;
    - for ``"resnet_18"`` or a custom classifier, SLIC superpixels, which are removed by filling
      them with the image's mean color.

    The explained class defaults to the class predicted on the full image.

    Attributes:
        image: The explained RGB image.
        class_index: The explained class.
        superpixels: The superpixel labels (``None`` for the vision transformers).
        model_commit: The Hugging Face commit of a vision transformer, ``None`` for other models.
    """

    def __init__(
        self,
        image: np.ndarray | str | Path,
        model: BuiltinModel | Callable[[np.ndarray], np.ndarray] = "vit_9_patches",
        *,
        n_superpixels: int = 14,
        class_index: int | None = None,
        batch_size: int = 16,
        device: str = "cpu",
        revision: str | None = None,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the image classification game.

        Args:
            image: An RGB image array of shape ``(height, width, 3)`` or the path of an image.
            model: A builtin model name or a classifier mapping a batch of images of shape
                ``(batch, height, width, 3)`` to class probabilities of shape
                ``(batch, n_classes)``. Defaults to ``"vit_9_patches"``.
            n_superpixels: The number of superpixels for ``"resnet_18"`` and custom classifiers.
                Defaults to ``14``.
            class_index: The explained class, or ``None`` for the class predicted on the image.
            batch_size: The number of masked images per forward pass. Defaults to ``16``.
            device: The torch device of the builtin models. Defaults to ``"cpu"``.
            revision: The Hugging Face revision (branch, tag, or commit) of the vision
                transformer. ``None`` loads the default branch; :attr:`model_commit` records what
                was loaded. ResNet-18 uses pinned torchvision weights and takes no revision.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.

        Raises:
            ValueError: If the model is unknown, or a revision is given for a model that is not a
                vision transformer.
        """
        self.image = _as_rgb_array(image)
        self.batch_size = batch_size
        self.superpixels: np.ndarray | None = None
        self.model_commit: str | None = None
        self._vit: ViTPatchModel | None = None

        if isinstance(model, str) and model in _VIT_MODELS:
            self._vit = ViTPatchModel(
                self.image,
                _VIT_MODELS[model],
                class_index=class_index,
                device=device,
                batch_size=batch_size,
                revision=revision,
            )
            n_players = self._vit.n_players
            self.class_index = self._vit.class_index
            self.model_commit = self._vit.model_commit
        else:
            if revision is not None:
                msg = "revision applies to the vision transformer models only."
                raise ValueError(msg)
            if model == "resnet_18":
                from ._resnet import ResNetClassifier

                classifier: Callable[[np.ndarray], np.ndarray] = ResNetClassifier(device=device)
            elif callable(model):
                classifier = model
            else:
                valid = [*_VIT_MODELS, "resnet_18"]
                msg = f"Unknown model {model!r}. Choose one of {valid} or pass a classifier."
                raise ValueError(msg)
            self._classifier = classifier
            self.superpixels = get_superpixels(self.image, n_superpixels)
            n_players = int(self.superpixels.max())
            self._fill = self.image.reshape(-1, 3).mean(axis=0)
            probabilities = np.asarray(self._classifier(self.image[None]))[0]
            self.class_index = int(np.argmax(probabilities)) if class_index is None else class_index

        empty_value = float(self._evaluate(np.zeros((1, n_players), dtype=bool))[0])
        super().__init__(
            n_players,
            normalize=normalize,
            normalization_value=empty_value,
            verbose=verbose,
        )

    def _masked_images(self, coalitions: np.ndarray) -> np.ndarray:
        images = np.repeat(self.image[None].astype(float), coalitions.shape[0], axis=0)
        for i, coalition in enumerate(coalitions):
            absent = ~coalition[self.superpixels - 1]  # type: ignore[index]
            images[i][absent] = self._fill
        return images.round().astype(np.uint8)

    def _evaluate(self, coalitions: np.ndarray) -> np.ndarray:
        if self._vit is not None:
            return self._vit(coalitions)
        values = []
        for start in range(0, coalitions.shape[0], self.batch_size):
            batch = self._masked_images(coalitions[start : start + self.batch_size])
            values.append(np.asarray(self._classifier(batch))[:, self.class_index])
        return np.concatenate(values).astype(float)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the probability of the explained class for each coalition of regions."""
        return self._evaluate(as_bool_coalitions(coalitions))

    @classmethod
    def from_config(
        cls,
        *,
        image: int | str = 0,
        model: BuiltinModel = "vit_9_patches",
        n_superpixels: int = 14,
        class_index: int | None = None,
        revision: str | None = None,
        device: str = "cpu",
        batch_size: int = 16,
        normalize: bool = True,
    ) -> Self:
        """Build the game for one of the ImageNet example images.

        The configuration records the Hugging Face commit of a vision transformer, so a new
        version of the model gets a new fingerprint and never reuses cached ground truth.

        Args:
            image: The index or file name of the example image (see
                :func:`shapiq_games.datasets.list_example_images`). Defaults to ``0``.
            model: The builtin model. Defaults to ``"vit_9_patches"``.
            n_superpixels: The number of superpixels for ``"resnet_18"``.
            class_index: The explained class, or ``None`` for the predicted class.
            revision: The Hugging Face revision of a vision transformer (``None`` for the default
                branch).
            device: The torch device (not part of the configuration). Defaults to ``"cpu"``.
            batch_size: The number of masked images per forward pass (not part of the
                configuration). Defaults to ``16``.
            normalize: Whether to center the game.

        Returns:
            The configured game.
        """
        game = cls(
            load_example_image(image),
            model,
            n_superpixels=n_superpixels,
            class_index=class_index,
            batch_size=batch_size,
            device=device,
            revision=revision,
            normalize=normalize,
        )
        return game._set_config(
            image=image,
            model=model,
            n_superpixels=n_superpixels,
            class_index=class_index,
            revision=revision,
            model_commit=game.model_commit,
            normalize=normalize,
        )
