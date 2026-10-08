"""Image classification games: the class probability with regions of the image removed."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions

from ._dinov2 import DINOV2_GRIDS, DinoV2TokenModel
from ._display import RegionPlots, gray_masked_image
from ._fill import fill_image, filled_images
from ._preprocess import as_rgb_array
from ._superpixels import get_superpixels
from ._vit import VIT_PATCH_GRIDS, ViTPatchModel

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from numpy.typing import ArrayLike

    from shapiq.typing import CoalitionMatrix, GameValues
    from shapiq_games.typing import Fill, ImageModel, MaskStrategy

__all__ = ["ImageClassifier", "grid_regions"]

_VIT_MODELS = {f"vit_{n}_patches": n for n in VIT_PATCH_GRIDS}
_DINOV2_MODELS = {f"dinov2_{n}_patches": n for n in DINOV2_GRIDS}


def grid_regions(height: int, width: int, rows: int, cols: int) -> np.ndarray:
    """Split an image into a ``rows x cols`` grid of regions.

    Args:
        height: The image height in pixels.
        width: The image width in pixels.
        rows: The number of grid rows.
        cols: The number of grid columns.

    Returns:
        The region of every pixel, of shape ``(height, width)``, numbered row by row from ``0``.

    Examples:
        >>> grid_regions(4, 6, rows=2, cols=3)
        array([[0, 0, 1, 1, 2, 2],
               [0, 0, 1, 1, 2, 2],
               [3, 3, 4, 4, 5, 5],
               [3, 3, 4, 4, 5, 5]])
    """
    row = np.minimum(np.arange(height) * rows // height, rows - 1)
    col = np.minimum(np.arange(width) * cols // width, cols - 1)
    return row[:, None] * cols + col[None, :]


def _check_regions(regions: np.ndarray, image: np.ndarray) -> np.ndarray:
    regions = np.asarray(regions)
    if regions.shape != image.shape[:2]:
        msg = f"regions must have the image's shape {image.shape[:2]}, got {regions.shape}."
        raise ValueError(msg)
    labels = np.unique(regions)
    if not np.issubdtype(regions.dtype, np.integer) or not np.array_equal(
        labels, np.arange(labels.shape[0])
    ):
        msg = "regions must label every pixel with a player 0, ..., n - 1, using every label."
        raise ValueError(msg)
    return regions.astype(int)


class ImageClassifier(RegionPlots, Game):
    """The image classification game: the probability of a class when only some regions are visible.

    The players are regions of the image, given by :attr:`regions` for every model:

    - for the vision transformers (``"vit_9_patches"``, ``"vit_16_patches"``, ``"vit_36_patches"``,
      ``"vit_144_patches"``), square groups of the model's 12 x 12 patches;
    - for DINOv2 (``"dinov2_16_patches"``, ``"dinov2_20_patches"``, ``"dinov2_25_patches"``: a
      4 x 4, 5 x 4, or 5 x 5 grid), rectangular groups of the model's 16 x 16 patch tokens;
    - for ``"resnet_18"`` or a custom classifier, SLIC superpixels or your own ``regions``.

    How absent players are removed:

    - **In token space** (the transformers, by default): their patch tokens are masked
      (``mask_strategy="mask"``: the content is replaced by a zero mask token, the position
      embedding is kept; the default of the vision transformers) or dropped from the sequence
      (``"remove"``: the present tokens keep their position embeddings; the default of DINOv2).
    - **In image space** (``fill``; always for ResNet-18 and custom classifiers): their pixels are
      replaced by the image's mean color (the default), gray, black, a blurred copy, or any image
      of the same shape, and the model sees the filled image. A transformer given a ``fill``
      removes its patch-grid players this way.

    The mask token is zeros for both transformers (their checkpoints carry no trained one).
    DINOv2's head averages all patch tokens, so masking is far out of distribution for it: one
    masked player can drop the explained probability to near zero, while dropping it hardly does.

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
            ``"remove"``), or ``None`` when players are removed in image space.
        class_index: The explained class.
        class_name: The name of the explained class, if the model provides class names.

    Examples:
        >>> image = np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)
        >>> def classifier(images):  # (batch, height, width, 3) -> (batch, n_classes)
        ...     brightness = images.mean(axis=(1, 2, 3)) / 255
        ...     return np.stack([brightness, 1 - brightness], axis=1)
        >>> game = ImageClassifier(image, model=classifier, n_superpixels=8)
        >>> game.n_players
        8
        >>> grid = ImageClassifier(image, classifier, regions=grid_regions(64, 64, 2, 2), fill="blur")
        >>> grid.n_players, grid.masked_image([1, 0, 0, 1]).shape
        (4, (64, 64, 3))

        The transformers remove players in token space, or in image space with a ``fill``:

        >>> game = ImageClassifier(image, "vit_16_patches")  # doctest: +SKIP
        >>> game = ImageClassifier(image, "vit_16_patches", mask_strategy="remove")  # doctest: +SKIP
        >>> game = ImageClassifier(image, "dinov2_20_patches", mask_strategy="mask")  # doctest: +SKIP
        >>> game = ImageClassifier(image, "dinov2_20_patches", fill="blur")  # doctest: +SKIP
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
                transformers only when a fill is given.
            mask_strategy: How a transformer removes players in token space: ``"mask"`` or
                ``"remove"`` (see above). ``None`` (default) takes the model's default:
                ``"mask"`` for the vision transformers, ``"remove"`` for DINOv2.
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
                model without tokens, or together with a ``fill``; or the regions or the fill are
                invalid.
        """
        self.image = as_rgb_array(image)
        self.batch_size = batch_size
        self.class_name: str | None = None
        self.mask_strategy: MaskStrategy | None = None
        # a model that removes the players itself, in token space
        self._patch_model: ViTPatchModel | DinoV2TokenModel | None = None

        if isinstance(model, str) and model in {**_VIT_MODELS, **_DINOV2_MODELS}:
            if regions is not None:
                msg = "regions apply to superpixel models; a transformer's players are its patches."
                raise ValueError(msg)
            if fill is not None and mask_strategy is not None:
                msg = "Choose mask_strategy (token space) or fill (image space), not both."
                raise ValueError(msg)
            strategy: MaskStrategy = mask_strategy or ("mask" if model in _VIT_MODELS else "remove")
            if model in _VIT_MODELS:
                patch_model: ViTPatchModel | DinoV2TokenModel = ViTPatchModel(
                    self.image,
                    _VIT_MODELS[model],
                    mask_strategy=strategy,
                    class_index=class_index,
                    device=device,
                    batch_size=batch_size,
                    revision=revision,
                )
                grid = VIT_PATCH_GRIDS[_VIT_MODELS[model]]
                height, width = self.image.shape[:2]
                self.regions = grid_regions(height, width, grid, grid)
            else:
                patch_model = DinoV2TokenModel(
                    self.image,
                    _DINOV2_MODELS[model],
                    mask_strategy=strategy,
                    class_index=class_index,
                    device=device,
                    batch_size=batch_size,
                    revision=revision,
                )
                self.image, self.regions = patch_model.image, patch_model.regions
            self.class_index = patch_model.class_index
            self.class_name = patch_model.class_name
            if fill is None:
                self._patch_model, self.mask_strategy = patch_model, strategy
            else:  # image space: the model classifies the filled images
                self._classifier = patch_model.classify
                self._baseline = fill_image(self.image, fill)
        else:
            if revision is not None:
                msg = "revision applies to the Hugging Face models (vision transformers, DINOv2)."
                raise ValueError(msg)
            if mask_strategy is not None:
                msg = "mask_strategy applies to the transformers (vision transformers, DINOv2)."
                raise ValueError(msg)
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
                self.regions = _check_regions(regions, self.image)
            self._baseline = fill_image(self.image, "mean" if fill is None else fill)
            probabilities = np.asarray(self._classifier(self.image[None]))[0]
            self.class_index = int(np.argmax(probabilities)) if class_index is None else class_index
            categories = getattr(classifier, "categories", None)
            if categories is not None:
                self.class_name = str(categories[self.class_index])

        n_players = int(self.regions.max()) + 1
        self._empty_value = float(self._evaluate(np.zeros((1, n_players), dtype=bool))[0])
        super().__init__(
            n_players,
            normalize=normalize,
            normalization_value=self._empty_value,
            verbose=verbose,
        )

    def _masked_images(self, coalitions: CoalitionMatrix) -> np.ndarray:
        return filled_images(self.image, self.regions, self._baseline, coalitions)

    def _evaluate(self, coalitions: CoalitionMatrix) -> GameValues:
        if self._patch_model is not None:
            return self._patch_model(coalitions)
        values = []
        for start in range(0, coalitions.shape[0], self.batch_size):
            batch = self._masked_images(coalitions[start : start + self.batch_size])
            values.append(np.asarray(self._classifier(batch))[:, self.class_index])
        return np.concatenate(values).astype(float)

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the probability of the explained class for each coalition of regions."""
        coalitions = as_bool_coalitions(coalitions)
        values = np.full(coalitions.shape[0], self._empty_value)
        present = coalitions.any(axis=1)  # the empty coalition is exactly the stored value
        if present.any():
            values[present] = self._evaluate(coalitions[present])
        return values

    def masked_image(self, coalition: ArrayLike) -> np.ndarray:
        """Return the image with the players outside ``coalition`` removed.

        In image space (a ``fill``, ResNet-18, custom classifiers) this is exactly the image the
        classifier sees. In token space the transformers remove patches inside the model, so
        their removed regions are shown in gray.

        Args:
            coalition: The players, as a boolean or 0/1 vector of length ``n_players``.

        Returns:
            The image as a ``uint8`` array of shape ``(height, width, 3)``.
        """
        coalition = as_bool_coalitions(coalition)
        if self._patch_model is None:
            return self._masked_images(coalition)[0]
        return gray_masked_image(self.image, self.regions, coalition[0])
