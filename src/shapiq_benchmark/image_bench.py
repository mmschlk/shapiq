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
        n_superpixel_resnet: int = 14,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the benchmark with an image classifier game.

        Args:
            data: Path to an image file or a directory containing images.
            model: Model identifier (e.g. "resnet_18") or a fitted model object.
            x_explain: Index of the image to explain, or a path to an image file.
            n_superpixel_resnet: Number of superpixels for ResNet-based explanations.
            normalize: Whether to normalize interaction values.
            verbose: Whether to print verbose output during game initialization.
        """
        if not isinstance(data, str):
            msg = "ImageBench expects a string path for data."
            raise ValueError(msg)

        image_paths, image_path = _resolve_x_explain_path(
            data, x_explain
        )  # TODO make less complicated
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

    def exact_values(
        self, index: IndexType, order: int, budget: int | None = None
    ) -> InteractionValues:
        """Compute exact interaction values using the ImageBench computer.
        Args:
            index: The index for which to compute interaction values.
            order: The order of interactions to compute.
            budget: Optional budget for computation.
        Returns:
            InteractionValues: The computed interaction values.
        """
        return self._computer.exact_values(index=index, order=order, budget=budget)

    @property
    def game(self) -> ImageClassifier:
        """Game instance used by the Image Benchmark."""
        return self._game

    @property
    def computer(self) -> ImageComputer:
        """Ground truth computer used by the Image Benchmark."""
        return self._computer
