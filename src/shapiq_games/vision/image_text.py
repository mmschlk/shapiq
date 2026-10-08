"""Image-text games: how well the visible regions of an image match a text (CLIP)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions

from ._clip import ClipTokenModel, imagenet_class_names
from ._display import RegionPlots, gray_masked_image
from ._fill import fill_image, filled_images
from ._preprocess import as_rgb_array

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import ArrayLike

    from shapiq.typing import CoalitionMatrix, GameValues
    from shapiq_games.typing import ClipModel, Fill, MaskStrategy

__all__ = ["ImageTextSimilarity"]


class ImageTextSimilarity(RegionPlots, Game):
    """The image-text similarity game: how well the visible regions of an image match a text.

    CLIP embeds images and texts in one space. The players are rectangular regions of the image, a
    ``grid`` over CLIP's patch tokens (14 x 14 for ``"clip_vit_b16"``, 7 x 7 for
    ``"clip_vit_b32"``). The value of a coalition is the cosine similarity between the embedding of
    the image with only the coalition's regions and the embedding of the text. Absent regions are
    removed

    - in token space (by default): their tokens are dropped from the sequence
      (``mask_strategy="remove"``, the default; the present tokens keep their position embeddings)
      or masked (``"mask"``: CLIP has no mask token, so a masked token keeps only its position
      embedding), or
    - in image space (``fill``): their pixels are replaced, as in
      :class:`~shapiq_games.ImageClassifier`, and CLIP embeds the filled image.

    Without a text, the game explains CLIP's zero-shot ImageNet label of the full image: the
    prompt ``prompt_template.format(label)`` of the ImageNet class whose prompt is most similar to
    the image. CLIP sees a ``224 x 224`` center crop, so :attr:`image` is that crop.

    Similarities lie in a narrow band (about 0.2 to 0.35), so the game runs in float32; half
    precision would change the values by up to 1e-2.

    Attributes:
        image: The ``224 x 224`` crop CLIP sees.
        regions: The player of every pixel, of shape ``(224, 224)``, numbered row by row.
        text: The explained text.
        label: CLIP's zero-shot ImageNet label of the image if no text was given, else ``None``.
        original_model_output: The similarity of the full image and the text.
        mask_strategy: ``"mask"`` or ``"remove"`` in token space, ``None`` in image space.

    Examples:
        >>> image = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
        >>> game = ImageTextSimilarity(image, "a photo of a dog", grid=(5, 5))  # doctest: +SKIP
        >>> game.n_players  # doctest: +SKIP
        25
        >>> game = ImageTextSimilarity(image)  # the zero-shot label  # doctest: +SKIP
        >>> game.label, game.text  # doctest: +SKIP
        >>> game = ImageTextSimilarity(image, mask_strategy="mask")  # doctest: +SKIP
        >>> game = ImageTextSimilarity(image, fill="gray")  # image space  # doctest: +SKIP
    """

    def __init__(
        self,
        image: np.ndarray | str | Path,
        text: str | None = None,
        *,
        model: ClipModel = "clip_vit_b16",
        grid: tuple[int, int] = (4, 4),
        prompt_template: str = "a photo of a {}.",
        mask_strategy: MaskStrategy | None = None,
        fill: Fill | np.ndarray | None = None,
        batch_size: int = 16,
        device: str = "cpu",
        revision: str | None = None,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the image-text similarity game.

        Args:
            image: An RGB image array of shape ``(height, width, 3)`` or the path of an image.
            text: The text to match, or ``None`` (default) for CLIP's zero-shot ImageNet label of
                the image.
            model: ``"clip_vit_b16"`` (default) or ``"clip_vit_b32"``.
            grid: The ``(rows, columns)`` of the player grid. Defaults to ``(4, 4)``; the paper
                grids are ``(4, 4)``, ``(5, 4)``, and ``(5, 5)``.
            prompt_template: The prompt of an ImageNet class for the zero-shot label. Defaults to
                ``"a photo of a {}."``.
            mask_strategy: ``"remove"`` or ``"mask"`` (see above). ``None`` (default) means
                ``"remove"`` unless a ``fill`` is given.
            fill: Remove regions in image space instead, replacing their pixels with ``"mean"``,
                ``"gray"``, ``"black"``, ``"blur"``, or an image of the crop's shape.
            batch_size: The number of coalitions per forward pass; smaller batches are padded to
                it, so that a value does not depend on the batch. Defaults to ``16``.
            device: The torch device. Defaults to ``"cpu"``.
            revision: The Hugging Face revision (branch, tag, or commit) of the CLIP model.
                ``None`` loads the default branch.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.

        Raises:
            ValueError: If both ``mask_strategy`` and ``fill`` are given, or either is invalid.
        """
        if fill is not None and mask_strategy is not None:
            msg = "Choose mask_strategy (token space) or fill (image space), not both."
            raise ValueError(msg)
        self.mask_strategy: MaskStrategy | None = (
            None if fill is not None else mask_strategy or "remove"
        )
        self._clip = ClipTokenModel(
            as_rgb_array(image),
            tuple(grid),
            model=model,
            mask_strategy=self.mask_strategy or "remove",
            device=device,
            batch_size=batch_size,
            revision=revision,
        )
        self.image, self.regions = self._clip.image, self._clip.regions
        self._fill = None if fill is None else fill_image(self.image, fill)
        n_players = int(self.regions.max()) + 1
        full_image = self._image_embeddings(np.ones((1, n_players), dtype=bool))[0]
        self.label: str | None = None
        if text is None:
            labels = imagenet_class_names()
            prompts = [prompt_template.format(label) for label in labels]
            best = int(np.argmax(self._clip.text_embeddings(prompts) @ full_image))
            self.label, text = labels[best], prompts[best]
        self.text = text
        self._text_embedding = self._clip.text_embeddings([text])[0]
        self.original_model_output = float(full_image @ self._text_embedding)
        self._empty_value = float(self._similarity(np.zeros((1, n_players), dtype=bool))[0])
        super().__init__(
            n_players,
            normalize=normalize,
            normalization_value=self._empty_value,
            verbose=verbose,
        )

    def _image_embeddings(self, coalitions: CoalitionMatrix) -> np.ndarray:
        if self._fill is None:
            return self._clip.image_embeddings(coalitions)
        return self._clip.embed_images(
            filled_images(self.image, self.regions, self._fill, coalitions)
        )

    def _similarity(self, coalitions: CoalitionMatrix) -> GameValues:
        # a row-wise sum, not a matrix product, whose summation order depends on the batch
        return np.sum(self._image_embeddings(coalitions) * self._text_embedding, axis=1)

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Return the cosine similarity of the text and the image restricted to each coalition."""
        coalitions = as_bool_coalitions(coalitions)
        values = np.full(coalitions.shape[0], self._empty_value)
        present = coalitions.any(axis=1)  # the empty coalition is exactly the stored value
        if present.any():
            values[present] = self._similarity(coalitions[present])
        return values

    def masked_image(self, coalition: ArrayLike) -> np.ndarray:
        """Return the image with the players outside ``coalition`` removed.

        In image space this is the image CLIP embeds. In token space CLIP removes the tokens
        inside the model, so the removed regions are shown in gray.

        Args:
            coalition: The players, as a boolean or 0/1 vector of length ``n_players``.

        Returns:
            The image as a ``uint8`` array of shape ``(224, 224, 3)``.
        """
        coalitions = as_bool_coalitions(coalition)
        if self._fill is not None:
            return filled_images(self.image, self.regions, self._fill, coalitions)[0]
        return gray_masked_image(self.image, self.regions, coalitions[0])
