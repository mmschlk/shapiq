"""Image-text games: how well the visible regions of an image match a text (CLIP, SigLIP)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ._fill import fill_image
from ._image_text_model import ImageTextTokenModel, imagenet_class_names
from ._preprocess import as_rgb_array
from ._region_game import RegionGame, resolve_removal

if TYPE_CHECKING:
    from pathlib import Path

    from shapiq.typing import CoalitionMatrix, GameValues
    from shapiq_games.typing import Fill, ImageTextModel, MaskStrategy

__all__ = ["ImageTextSimilarity"]


class ImageTextSimilarity(RegionGame):
    """The image-text similarity game: how well the visible regions of an image match a text.

    CLIP, SigLIP, and SigLIP 2 embed images and texts in one space. The players are rectangular
    regions of the image, a ``grid`` over the model's patch tokens (14 x 14 for the ViT-B/16
    models, 7 x 7 for ``"clip_vit_b32"``). The value of a coalition is the cosine similarity
    between the embedding of the image with only the coalition's regions and the embedding of the
    text. SigLIP's logit is ``scale * cosine + bias``, so its Shapley values are ``scale`` times
    the game's. Absent regions are removed

    - in token space (by default): their tokens are dropped from the sequence
      (``mask_strategy="remove"``, the default; the present tokens keep their position embeddings)
      or masked (``"mask"``: the models have no mask token, so a masked token is a patch at the
      normalization mean, for CLIP only its position embedding, for SigLIP a gray patch), or
    - in image space (``fill``): their pixels are replaced, as in
      :class:`~shapiq_games.ImageClassifier`, and the model embeds the filled image.

    With every token dropped, CLIP keeps its class token; SigLIP has none and pools its tokens by
    attention, so the empty coalition is SigLIP's embedding of no token at all.

    Without a text, the game explains the model's zero-shot ImageNet label of the full image: the
    prompt ``prompt_template.format(label)`` of the ImageNet class whose prompt is most similar to
    the image. The models see a ``224 x 224`` image, CLIP a center crop and SigLIP the whole image
    resized, and :attr:`image` is what they see.

    Similarities lie in a narrow band (CLIP: about 0.2 to 0.35), so the game runs in float32; half
    precision would change the values by up to 1e-2.

    Attributes:
        image: The ``224 x 224`` image the model sees.
        regions: The player of every pixel, of shape ``(224, 224)``, numbered row by row.
        text: The explained text.
        label: The model's zero-shot ImageNet label of the image if no text was given, else ``None``.
        original_model_output: The similarity of the full image and the text.
        empty_value: The similarity with every region removed, before centering.
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
        >>> game = ImageTextSimilarity(image, "a dog", model="siglip2_vit_b16")  # doctest: +SKIP
    """

    def __init__(
        self,
        image: np.ndarray | str | Path,
        text: str | None = None,
        *,
        model: ImageTextModel = "clip_vit_b16",
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
            text: The text to match, or ``None`` (default) for the model's zero-shot ImageNet
                label of the image.
            model: CLIP (``"clip_vit_b16"``, default, or ``"clip_vit_b32"``), SigLIP
                (``"siglip_vit_b16"``), or SigLIP 2 (``"siglip2_vit_b16"``).
            grid: The ``(rows, columns)`` of the player grid. Defaults to ``(4, 4)``; the paper
                grids are ``(4, 4)``, ``(5, 4)``, and ``(5, 5)``.
            prompt_template: The prompt of an ImageNet class for the zero-shot label. Defaults to
                ``"a photo of a {}."``.
            mask_strategy: ``"remove"`` or ``"mask"`` (see above). ``None`` (default) means
                ``"remove"`` unless a ``fill`` is given.
            fill: Remove regions in image space instead, replacing their pixels with ``"mean"``,
                ``"gray"``, ``"black"``, ``"blur"``, or an image of :attr:`image`'s shape.
            batch_size: The number of coalitions per forward pass; smaller batches are padded to
                it, so that a value does not depend on the batch. Defaults to ``16``.
            device: The torch device. Defaults to ``"cpu"``.
            revision: The Hugging Face revision (branch, tag, or commit) of the model. ``None``
                loads the default branch.
            normalize: Whether to center the game such that the value of the empty coalition is
                zero. Defaults to ``True``.
            verbose: Whether to show a progress bar when evaluating the game.

        Raises:
            ValueError: If both ``mask_strategy`` and ``fill`` are given, or either is invalid.
        """
        self.mask_strategy = resolve_removal(mask_strategy, fill, default="remove")
        self._model = ImageTextTokenModel(
            as_rgb_array(image),
            tuple(grid),
            model=model,
            mask_strategy=self.mask_strategy or "remove",
            device=device,
            batch_size=batch_size,
            revision=revision,
        )
        self.image, self.regions = self._model.image, self._model.regions
        self._fill = None if fill is None else fill_image(self.image, fill)
        n_players = int(self.regions.max()) + 1
        full_image = self._image_embeddings(np.ones((1, n_players), dtype=bool))[0]
        self.label: str | None = None
        if text is None:
            labels = imagenet_class_names()
            prompts = [prompt_template.format(label) for label in labels]
            best = int(np.argmax(self._model.text_embeddings(prompts) @ full_image))
            self.label, text = labels[best], prompts[best]
        self.text = text
        self._text_embedding = self._model.text_embeddings([text])[0]
        super().__init__(normalize=normalize, verbose=verbose)

    def _image_embeddings(self, coalitions: CoalitionMatrix) -> np.ndarray:
        if self._fill is None:
            return self._model.image_embeddings(coalitions)
        return self._model.filled_embeddings(coalitions, self._fill)

    def _evaluate(self, coalitions: CoalitionMatrix) -> GameValues:
        # a row-wise sum, not a matrix product, whose summation order depends on the batch
        return np.sum(self._image_embeddings(coalitions) * self._text_embedding, axis=1)
