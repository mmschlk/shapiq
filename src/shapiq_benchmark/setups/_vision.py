"""Setup of the image classification game on Imagenette."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from shapiq_benchmark.datasets import load_imagenette
from shapiq_games import ImageClassifier

from ._base import Setup, runtime_field

if TYPE_CHECKING:
    from shapiq_games.vision.image_classifier import BuiltinModel, Fill

__all__ = ["ImageClassifierSetup"]


@dataclass(frozen=True, kw_only=True)
class ImageClassifierSetup(Setup, name="image_classifier"):
    """An :class:`~shapiq_games.ImageClassifier` game for an Imagenette image.

    The images come from :func:`shapiq_benchmark.datasets.load_imagenette`, which downloads the
    archive on first use.

    Attributes:
        index: The position of the image in the split. Defaults to ``0``.
        split: The Imagenette split, ``"val"`` (default) or ``"train"``.
        size: The image size, ``"320px"`` (default) or ``"160px"``.
        model: The builtin model. Defaults to ``"vit_9_patches"``.
        n_superpixels: The number of superpixels for ``"resnet_18"``. Defaults to ``14``.
        fill: How ``"resnet_18"`` replaces removed superpixels (``None`` for the mean color).
        class_index: The explained ImageNet class: ``None`` for the predicted class, ``"label"``
            for the image's true class, or a class index.
        revision: The Hugging Face revision of a vision transformer (``None`` for the default
            branch).
        normalize: Whether to center the game. Defaults to ``True``.
        device: The torch device. A runtime field: it does not change the cache key.
        batch_size: The number of masked images per forward pass (a runtime field).

    Examples:
        >>> setup = ImageClassifierSetup(index=388, model="resnet_18", class_index="label")
        >>> game = setup.build()  # doctest: +SKIP
    """

    index: int = 0
    split: Literal["train", "val"] = "val"
    size: Literal["160px", "320px"] = "320px"
    model: BuiltinModel = "vit_9_patches"
    n_superpixels: int = 14
    fill: Fill | None = None
    class_index: int | Literal["label"] | None = None
    revision: str | None = None
    normalize: bool = True
    device: str = runtime_field("cpu")
    batch_size: int = runtime_field(16)

    def build(self) -> ImageClassifier:
        """Load the image and the model and build the game."""
        images = load_imagenette(split=self.split, size=self.size)
        label = int(images.labels[self.index])
        return ImageClassifier(
            images[self.index],
            self.model,
            n_superpixels=self.n_superpixels,
            fill=self.fill,
            class_index=label if self.class_index == "label" else self.class_index,
            batch_size=self.batch_size,
            device=self.device,
            revision=self.revision,
            normalize=self.normalize,
        )
