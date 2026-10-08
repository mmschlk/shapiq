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
from shapiq_games.typing import (  # noqa: TC001  (resolved by the field checks)
    ClipModel,
    Fill,
    ImageModel,
    MaskStrategy,
)

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
            (``"dinov2_16_patches"``, ``"dinov2_20_patches"``, ``"dinov2_25_patches"``), or
            ``"resnet_18"``.
        n_superpixels: The number of superpixels for ``"resnet_18"``. Defaults to ``14``.
        fill: Remove players in image space with this fill (for ``"resnet_18"``, ``None`` means
            the mean color; a vision transformer given a fill removes its patches in image space;
            DINOv2 takes none).
        mask_strategy: How a vision transformer removes players in token space, ``"mask"``
            (``None``, the default) or ``"remove"``; DINOv2 only drops tokens (see
            :class:`~shapiq_games.ImageClassifier`).
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
    mask_strategy: MaskStrategy | None = None
    class_index: int | Literal["label"] | None = None
    revision: str | None = None
    normalize: bool = True
    device: str = runtime_field("cpu")
    batch_size: int = runtime_field(16)

    def __post_init__(self) -> None:
        """Check that DINOv2 is asked only to drop tokens, as in the paper."""
        super().__post_init__()
        if self.model.startswith("dinov2") and (
            self.fill is not None or self.mask_strategy == "mask"
        ):
            msg = "DINOv2 removes players only by dropping their tokens, as in the paper."
            raise ValueError(msg)

    def build(self) -> ImageClassifier:
        """Load the image and the model and build the game."""
        images = load_imagenette(split=self.split, size=self.size)
        label = int(images.labels[self.index])
        return ImageClassifier(
            images[self.index],
            self.model,
            n_superpixels=self.n_superpixels,
            fill=self.fill,
            mask_strategy=self.mask_strategy,
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
        mask_strategy: ``"remove"`` (``None``, the default) or ``"mask"`` in token space.
        fill: Remove regions in image space with this fill instead (``None`` for token space).
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
    mask_strategy: MaskStrategy | None = None
    fill: Fill | None = None
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
            mask_strategy=self.mask_strategy,
            fill=self.fill,
            batch_size=self.batch_size,
            device=self.device,
            revision=self.revision,
            normalize=self.normalize,
        )
