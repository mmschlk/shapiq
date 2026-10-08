"""Setups of the image games on Imagenette."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from shapiq_benchmark.datasets import (
    ImagenetteSize,
    ImagenetteSplit,
    load_imagenette,
)
from shapiq_games import ImageClassifier, ImageTextSimilarity
from shapiq_games.typing import ClipModel, Fill, ImageModel  # noqa: TC001  (field checks)

from ._base import Setup, runtime_field

__all__ = ["ImageClassifierSetup", "ImageTextSimilaritySetup"]


@dataclass(frozen=True, kw_only=True)
class ImageClassifierSetup(Setup, name="image_classifier"):
    """An :class:`~shapiq_games.ImageClassifier` game for an Imagenette image.

    The images come from :func:`shapiq_benchmark.datasets.load_imagenette`, which downloads the
    archive on first use.

    Attributes:
        index: The position of the image in the split. Defaults to ``0``.
        split: The Imagenette split, ``"val"`` (default) or ``"train"``.
        size: The image size, ``"320px"`` (default) or ``"160px"``.
        model: The builtin model: a vision transformer (``"vit_9_patches"``, default), DINOv2
            with token dropping (``"dinov2_16_patches"``, ``"dinov2_20_patches"``,
            ``"dinov2_25_patches"``), or ``"resnet_18"``.
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
    split: ImagenetteSplit = "val"
    size: ImagenetteSize = "320px"
    model: ImageModel = "vit_9_patches"
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


@dataclass(frozen=True, kw_only=True)
class ImageTextSimilaritySetup(Setup, name="image_text_similarity"):
    """An :class:`~shapiq_games.ImageTextSimilarity` game for an Imagenette image.

    Attributes:
        index: The position of the image in the split. Defaults to ``0``.
        split: The Imagenette split, ``"val"`` (default) or ``"train"``.
        size: The image size, ``"320px"`` (default) or ``"160px"``.
        model: ``"clip_vit_b16"`` (default) or ``"clip_vit_b32"``.
        grid: The ``(rows, columns)`` of the player grid. Defaults to ``(4, 4)``.
        text: The text to match: ``None`` (default) for CLIP's zero-shot ImageNet label of the
            image, ``"label"`` for the prompt of the image's true class, or any text.
        prompt_template: The prompt of a class. Defaults to ``"a photo of a {}."``.
        revision: The Hugging Face revision of the CLIP model (``None`` for the default branch).
        normalize: Whether to center the game. Defaults to ``True``.
        device: The torch device. A runtime field: it does not change the cache key.
        batch_size: The number of coalitions per forward pass (a runtime field).

    Examples:
        >>> setup = ImageTextSimilaritySetup(index=388, grid=(5, 5), text="label")
        >>> game = setup.build()  # doctest: +SKIP
    """

    index: int = 0
    split: ImagenetteSplit = "val"
    size: ImagenetteSize = "320px"
    model: ClipModel = "clip_vit_b16"
    grid: tuple[int, int] = (4, 4)
    text: str | None = None
    prompt_template: str = "a photo of a {}."
    revision: str | None = None
    normalize: bool = True
    device: str = runtime_field("cpu")
    batch_size: int = runtime_field(16)

    def build(self) -> ImageTextSimilarity:
        """Load the image and CLIP and build the game."""
        images = load_imagenette(split=self.split, size=self.size)
        text = self.text
        if text == "label":
            text = self.prompt_template.format(images.label_name(self.index))
        return ImageTextSimilarity(
            images[self.index],
            text,
            model=self.model,
            grid=self.grid,
            prompt_template=self.prompt_template,
            batch_size=self.batch_size,
            device=self.device,
            revision=self.revision,
            normalize=self.normalize,
        )
