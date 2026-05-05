"""Image benchmark implementation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from shapiq.typing import IndexType, Model
from shapiq_games.benchmark.local_xai.benchmark_image import ImageClassifier

from .base import Benchmark
from .computers import ImageComputer

if TYPE_CHECKING:
    from shapiq import InteractionValues


_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def _collect_image_paths(root: Path) -> list[str]:
    """Collect image file paths from the given root path."""
    if root.is_file():
        return [str(root)]
    if not root.is_dir():
        msg = f"Image data path does not exist: {root}"
        raise ValueError(msg)
    image_paths = [
        str(path)
        for path in sorted(root.rglob("*"))
        if path.suffix.lower() in _IMAGE_EXTENSIONS
    ]
    if not image_paths:
        msg = f"No image files found under {root}."
        raise ValueError(msg)
    return image_paths


def _resolve_x_explain_path(
    data: str,
    x_explain: int | str | None,
) -> tuple[list[str] | None, str]:
    """Resolve the x_explain argument to an image path, and collect dataset image paths if needed."""
    if isinstance(x_explain, str):
        explain_path = Path(x_explain)
        if not explain_path.exists():
            msg = f"x_explain path does not exist: {x_explain}"
            raise ValueError(msg)
        return None, str(explain_path)

    data_path = Path(data)
    image_paths = _collect_image_paths(data_path)
    index = 0 if x_explain is None else x_explain
    if not isinstance(index, int):
        msg = "x_explain must be an int index or an image path string."
        raise ValueError(msg)
    if index < 0 or index >= len(image_paths):
        msg = f"x_explain index {index} is out of range for {len(image_paths)} images."
        raise ValueError(msg)
    return image_paths, image_paths[index]


class ImageBench(Benchmark[IndexType]):
    """Benchmark for image classification explanations."""

    def __init__(
        self,
        data: str,
        model: str | Model | Callable[[np.ndarray], np.ndarray],
        *,
        x_explain: int | str | None = 0,
        n_superpixel_resnet: int | None,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the benchmark with an image classifier game.

        Args:
            data: Image dataset directory or single image path.
            model: Image model identifier (e.g. "vit_16_patches", "resnet_18") or a
                callable that evaluates coalitions.
            x_explain: Index into the dataset or an image path string.
            n_superpixel_resnet: Number of superpixels for ResNet/SqueezeNet models.
            normalize: Whether to normalize the game values.
            verbose: Enable verbose output in the underlying model setup.
            random_state: Random state for dataset shuffling (unused for images).
        """
        if not isinstance(data, str):
            msg = "ImageBench expects a string path for data."
            raise ValueError(msg)
        if model== "vit_16_patches" and n_superpixel_resnet is None:
            msg = "n_superpixel_resnet must be set when using a ViT model."
            raise ValueError(msg)

        image_paths, image_path = _resolve_x_explain_path(data, x_explain) #TODO make less complicated
        self.dataset = image_paths
        self.x_train = image_paths if image_paths is not None else []
        self.model = model

        self._game = ImageClassifier(
            model_name=self.model,
            n_superpixel_resnet=n_superpixel_resnet,
            x_explain_path=image_path,
            normalize=normalize,
            verbose=verbose,
        )
        self._computer = ImageComputer(self._game)

    def exact_values(self, index: IndexType, order: int) -> InteractionValues:
        """Compute exact interaction values using the benchmark computer."""
        return self._computer.exact_values(index=index, order=order)

    @property
    def game(self) -> ImageClassifier:
        """Game instance used by this benchmark."""
        return self._game

    @property
    def computer(self) -> ImageComputer:
        """Ground truth computer used by this benchmark."""
        return self._computer
