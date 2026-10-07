"""A torchvision ResNet-18 as an image classifier function (requires ``torch`` and ``torchvision``)."""

from __future__ import annotations

import numpy as np
from PIL import Image

from shapiq_games._optional import require

from ._batching import pad_batch

__all__ = ["ResNetClassifier"]


class ResNetClassifier:
    """ResNet-18 with the pinned ``IMAGENET1K_V1`` torchvision weights.

    The model sees a ``224 x 224`` center crop of the image resized to a shorter side of ``256``
    (its standard preprocessing). :meth:`prepare` produces that crop, so that the regions of an
    image game are drawn on exactly the pixels the model sees. Calling the classifier on a batch of
    prepared images of shape ``(batch, 224, 224, 3)`` returns class probabilities of shape
    ``(batch, 1000)``.

    Attributes:
        categories: The ImageNet class names.
    """

    def __init__(self, *, device: str = "cpu", batch_size: int = 16) -> None:
        """Load the model.

        Args:
            device: The torch device. Defaults to ``"cpu"``.
            batch_size: The number of images per forward pass; smaller batches are padded to it,
                so that an image's probabilities do not depend on the batch. Defaults to ``16``.
        """
        torch = require("torch", purpose="the ResNet image games")
        models = require("torchvision.models", purpose="the ResNet image games")
        self._torch = torch
        self._device = torch.device(device)
        self.batch_size = batch_size
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.categories: list[str] = list(weights.meta["categories"])
        self._model = models.resnet18(weights=weights).eval().to(self._device)
        transforms = weights.transforms()
        self._resize_size = int(transforms.resize_size[0])
        self._crop_size = int(transforms.crop_size[0])
        self._mean = torch.tensor(transforms.mean, device=self._device).view(1, 3, 1, 1)
        self._std = torch.tensor(transforms.std, device=self._device).view(1, 3, 1, 1)

    def prepare(self, image: np.ndarray) -> np.ndarray:
        """Return the ``224 x 224`` center crop the model sees, as an RGB ``uint8`` array."""
        pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
        scale = self._resize_size / min(pil.size)
        width, height = round(pil.width * scale), round(pil.height * scale)
        pil = pil.resize((width, height), Image.Resampling.BILINEAR)
        left, top = (width - self._crop_size) // 2, (height - self._crop_size) // 2
        pil = pil.crop((left, top, left + self._crop_size, top + self._crop_size))
        return np.asarray(pil)

    def __call__(self, images: np.ndarray) -> np.ndarray:
        """Return the class probabilities of a batch of prepared RGB images."""
        images = np.asarray(images)
        return np.concatenate(
            [
                self._probabilities(images[start : start + self.batch_size])
                for start in range(0, images.shape[0], self.batch_size)
            ]
        )

    def _probabilities(self, images: np.ndarray) -> np.ndarray:
        torch = self._torch
        # a copy: torch warns on read-only arrays (the prepared image) and rejects negative strides
        tensor = torch.as_tensor(np.array(pad_batch(images, self.batch_size)), device=self._device)
        tensor = tensor.permute(0, 3, 1, 2).float() / 255.0
        if tuple(tensor.shape[-2:]) != (self._crop_size, self._crop_size):
            tensor = torch.nn.functional.interpolate(
                tensor, size=(self._crop_size, self._crop_size), mode="bilinear", antialias=True
            )
        with torch.no_grad():
            logits = self._model((tensor - self._mean) / self._std)
        return torch.softmax(logits, dim=-1).cpu().numpy()[: images.shape[0]]
