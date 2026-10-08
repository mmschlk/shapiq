"""The standard ImageNet preprocessing: resize the shorter side, then crop the center."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

__all__ = ["as_rgb_array", "center_crop", "displayed_image", "normalized_pixels"]


def as_rgb_array(image: np.ndarray | str | Path) -> np.ndarray:
    """Return an image (array or path) as an RGB ``uint8`` array; floats in ``[0, 1]`` are scaled."""
    if isinstance(image, str | Path):
        with Image.open(image) as img:
            return np.asarray(img.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        msg = f"Expected an RGB image of shape (height, width, 3), got shape {array.shape}."
        raise ValueError(msg)
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(np.nanmax(array)) <= 1.0:  # matplotlib and scikit-image floats
            array = array * 255.0
        array = np.rint(array)
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def center_crop(
    image: np.ndarray,
    resize: int,
    crop: int,
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> np.ndarray:
    """Resize the shorter side of an image to ``resize`` and crop its ``crop x crop`` center.

    Args:
        image: An RGB image of shape ``(height, width, 3)``.
        resize: The length of the shorter side after resizing.
        crop: The side length of the center crop.
        resample: The PIL resampling filter. Defaults to bilinear.

    Returns:
        The crop as an RGB ``uint8`` array of shape ``(crop, crop, 3)``.
    """
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    scale = resize / min(pil.size)
    width, height = round(pil.width * scale), round(pil.height * scale)
    pil = pil.resize((width, height), resample)
    left, top = (width - crop) // 2, (height - crop) // 2
    return np.asarray(pil.crop((left, top, left + crop, top + crop)))


def normalized_pixels(
    torch: Any,  # noqa: ANN401
    image: np.ndarray,
    mean: list[float],
    std: list[float],
    device: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Return an RGB ``uint8`` image as a normalized ``(1, 3, height, width)`` float tensor."""
    pixels = torch.as_tensor(np.array(image), device=device).permute(2, 0, 1)[None].float() / 255.0
    mean_ = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    std_ = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (pixels - mean_) / std_


def displayed_image(pixels: Any, mean: list[float], std: list[float]) -> np.ndarray:  # noqa: ANN401
    """Return a normalized ``(1, 3, height, width)`` tensor as the RGB ``uint8`` image it shows."""
    values = pixels[0].permute(1, 2, 0).float().cpu().numpy()
    rgb = (values * np.asarray(std) + np.asarray(mean)) * 255.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
