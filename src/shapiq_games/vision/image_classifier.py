"""Image classification games: the class probability with regions of the image removed."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from ._dinov2 import DINOV2_GRIDS, DinoV2TokenModel
from ._fill import fill_image, filled_images
from ._preprocess import as_rgb_array
from ._region_game import RegionGame, resolve_removal
from ._regions import check_regions
from ._superpixels import get_superpixels
from ._vit import VIT_PATCH_GRIDS, ViTTokenModel

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from shapiq.typing import CoalitionMatrix, GameValues
    from shapiq_games.typing import Fill, ImageModel, MaskStrategy

__all__ = ["ImageClassifier"]

_VIT_MODELS = {f"vit_{n}_patches": n for n in VIT_PATCH_GRIDS}
_DINOV2_MODELS = {f"dinov2_{n}_patches": n for n in DINOV2_GRIDS}


class ImageClassifier(RegionGame):
    """The image classification game: the probability of a class when only some regions are visible.

    The players are regions of the image, given by :attr:`regions` for every model:

    - for the vision transformers (``"vit_9_patches"``, ``"vit_16_patches"``, ``"vit_36_patches"``,
      ``"vit_144_patches"``), square groups of the model's 12 x 12 patches;
    - for DINOv2 (``"dinov2_16_patches"``, ``"dinov2_20_patches"``, ``"dinov2_25_patches"``: a
      4 x 4, 5 x 4, or 5 x 5 grid), rectangular groups of the model's 16 x 16 patch tokens;
    - for ``"resnet_18"`` or a custom classifier, SLIC superpixels or your own ``regions``.

    How absent players are removed:

    - **DINOv2** drops their patch tokens from the sequence (the present tokens keep their
      position embeddings), as in the benchmarking paper, and only so.
    - **The vision transformers** remove them in token space by default: their patch tokens are
      masked (``mask_strategy="mask"``, the default: the content is replaced by a zero mask token,
      the position embedding is kept) or dropped (``"remove"``). With a ``fill`` they remove them
      in image space instead.
    - **In image space** (``fill``; always for ResNet-18 and custom classifiers): their pixels are
      replaced by the image's mean color (the default), gray, black, a blurred copy, or any image
      of the same shape, and the model sees the filled image.

    ResNet-18 and DINOv2 see a ``224 x 224`` center crop, so the game explains that crop:
    :attr:`image` is what the model sees and every region is visible to it. The explained class
    defaults to the class predicted on the full image.

    To look at the game, :meth:`masked_image` shows what the model sees for a coalition,
    :meth:`attribution_map` spreads first-order values over the pixels for a heatmap, and
    :meth:`player_images` returns one image per player, e.g. for
    :func:`shapiq.plot.si_graph_plot`.

    Attributes:
        image: The explained RGB image (for ResNet-18 and DINOv2, its ``224 x 224`` crop).
        regions: The player of every pixel, of shape ``(height, width)``, numbered from ``0``.
        mask_strategy: How a transformer removes players in token space (``"mask"`` or
            ``"remove"``; always ``"remove"`` for DINOv2), or ``None`` in image space.
        class_index: The explained class.
        class_name: The name of the explained class, if the model provides class names.
        empty_value: The probability of the class with every region removed, before centering.
        original_model_output: The probability of the class on the full image.

    Examples:
        >>> image = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        >>> def classifier(images):  # (batch, height, width, 3) -> (batch, n_classes)
        ...     brightness = images.mean(axis=(1, 2, 3)) / 255
        ...     return np.stack([brightness, 1 - brightness], axis=1)
        >>> game = ImageClassifier(image, model=classifier, n_superpixels=8)
        >>> game.n_players
        8
        >>> from shapiq_games.vision import grid_regions
        >>> grid = ImageClassifier(image, classifier, regions=grid_regions(64, 64, 2, 2), fill="blur")
        >>> grid.n_players, grid.masked_image([1, 0, 0, 1]).shape
        (4, (64, 64, 3))

        The vision transformers mask or drop patch tokens, or remove pixels with a ``fill``; DINOv2
        drops tokens:

        >>> game = ImageClassifier(image, "vit_16_patches")  # doctest: +SKIP
        >>> game = ImageClassifier(image, "vit_16_patches", mask_strategy="remove")  # doctest: +SKIP
        >>> game = ImageClassifier(image, "vit_16_patches", fill="blur")  # doctest: +SKIP
        >>> game = ImageClassifier(image, "dinov2_20_patches")  # doctest: +SKIP
    """

    def __init__(
        self,
        image: np.ndarray | str | Path,
        model: ImageModel | Callable[[np.ndarray], np.ndarray] = "vit_9_patches",
        *,
        n_superpixels: int = 14,
        regions: np.ndarray | None = None,
        fill: Fill | np.ndarray | None = None,
        mask_strategy: MaskStrategy | None = None,
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
                Float images with values in ``[0, 1]`` (e.g. from ``matplotlib.pyplot.imread``)
                are scaled to ``0, ..., 255``.
            model: A builtin model name or a classifier mapping a batch of images of shape
                ``(batch, height, width, 3)`` to class probabilities of shape
                ``(batch, n_classes)``. Defaults to ``"vit_9_patches"``.
            n_superpixels: The number of SLIC superpixels for ``"resnet_18"`` and custom
                classifiers when no ``regions`` are given. Defaults to ``14``.
            regions: Your own players for ``"resnet_18"`` and custom classifiers: the player of
                every pixel, of shape ``(height, width)``, numbered ``0, ..., n - 1`` (see
                :func:`grid_regions`). For ResNet-18 they refer to its ``224 x 224`` crop.
            fill: Remove players in image space, replacing their pixels with ``"mean"`` (the
                image's mean color), ``"gray"``, ``"black"``, ``"blur"``, or an image of the same
                shape. ResNet-18 and custom classifiers always do (``"mean"`` by default); the
                vision transformers only when a fill is given; DINOv2 never.
            mask_strategy: How a vision transformer removes players in token space: ``"mask"``
                (``None``, the default) or ``"remove"`` (see above). DINOv2 only drops tokens
                (``None`` or ``"remove"``).
            class_index: The explained class, or ``None`` for the class predicted on the image.
            batch_size: The number of masked images per forward pass. The builtin models pad
                smaller batches to this size, so that a value does not depend on the batch.
                Defaults to ``16``.
            device: The torch device of the builtin models. Defaults to ``"cpu"``.
            revision: The Hugging Face revision (branch, tag, or commit) of the vision
                transformer or DINOv2. ``None`` loads the default branch. ResNet-18 uses pinned
                torchvision weights and takes no revision.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.

        Raises:
            ValueError: If the model is unknown; a revision is given for a model that is not a
                Hugging Face model (vision transformer or DINOv2); ``regions`` are given for a
                transformer (its players are its patch grid); ``mask_strategy`` is given for a
                model without tokens, or together with a ``fill``; DINOv2 is asked to mask or to
                fill; the regions, the fill or the mask strategy are invalid; or ``class_index``
                is out of range.
        """
        self.image = as_rgb_array(image)
        self.batch_size = batch_size
        self.mask_strategy: MaskStrategy | None = None
        # a model that removes the players itself, in token space, or the fill of image space
        self._token_model: ViTTokenModel | DinoV2TokenModel | None = None
        self._fill: np.ndarray | None = None

        if isinstance(model, str) and model in {**_VIT_MODELS, **_DINOV2_MODELS}:
            if regions is not None:
                msg = "regions apply to superpixel models; a transformer's players are its patches."
                raise ValueError(msg)
            is_vit = model in _VIT_MODELS
            strategy = resolve_removal(mask_strategy, fill, default="mask" if is_vit else "remove")
            if is_vit:
                vit = ViTTokenModel(
                    self.image,
                    _VIT_MODELS[model],
                    mask_strategy=strategy or "mask",
                    device=device,
                    batch_size=batch_size,
                    revision=revision,
                )
                if fill is not None:  # image space: the vision transformer classifies filled images
                    self._classifier, self._fill = vit.classify, fill_image(self.image, fill)
                token_model: ViTTokenModel | DinoV2TokenModel = vit
            elif strategy != "remove":
                msg = "DINOv2 removes players only by dropping their tokens, as in the paper."
                raise ValueError(msg)
            else:
                token_model = DinoV2TokenModel(
                    self.image,
                    _DINOV2_MODELS[model],
                    device=device,
                    batch_size=batch_size,
                    revision=revision,
                )
            self.image, self.regions = token_model.image, token_model.regions
            # the explained class comes from the token path, also when the image is filled
            probabilities = token_model(np.ones((1, token_model.n_players), dtype=bool))[0]
            categories: Sequence[str] | None = token_model.categories
            if strategy is not None:
                self._token_model, self.mask_strategy = token_model, strategy
        else:
            if revision is not None:
                msg = "revision applies to the Hugging Face models (vision transformers, DINOv2)."
                raise ValueError(msg)
            if mask_strategy is not None:
                msg = "mask_strategy applies to the transformers (vision transformers, DINOv2)."
                raise ValueError(msg)
            resolve_removal(None, fill, default=None)
            if model == "resnet_18":
                from ._resnet import ResNetClassifier

                resnet = ResNetClassifier(device=device, batch_size=batch_size)
                self.image = resnet.prepare(self.image)
                classifier: Callable[[np.ndarray], np.ndarray] = resnet
            elif callable(model):
                classifier = model
            else:
                valid = [*_VIT_MODELS, *_DINOV2_MODELS, "resnet_18"]
                msg = f"Unknown model {model!r}. Choose one of {valid} or pass a classifier."
                raise ValueError(msg)
            self._classifier = classifier
            if regions is None:
                self.regions = get_superpixels(self.image, n_superpixels) - 1
            else:
                self.regions = check_regions(regions, self.image)
            self._fill = fill_image(self.image, "mean" if fill is None else fill)
            probabilities = np.asarray(self._classifier(self.image[None]))[0]
            categories = getattr(classifier, "categories", None)

        if class_index is None:
            class_index = int(np.argmax(probabilities))
        elif not 0 <= class_index < len(probabilities):
            msg = f"class_index={class_index} is out of range for {len(probabilities)} classes."
            raise ValueError(msg)
        self.class_index = class_index
        self.class_name = None if categories is None else str(categories[class_index])
        super().__init__(normalize=normalize, verbose=verbose)

    def _evaluate(self, coalitions: CoalitionMatrix) -> GameValues:
        if self._token_model is not None:
            return self._token_model(coalitions)[:, self.class_index]
        values = []
        for start in range(0, coalitions.shape[0], self.batch_size):
            chunk = coalitions[start : start + self.batch_size]
            batch = filled_images(self.image, self.regions, cast("np.ndarray", self._fill), chunk)
            values.append(np.asarray(self._classifier(batch))[:, self.class_index])
        return np.concatenate(values).astype(float)
