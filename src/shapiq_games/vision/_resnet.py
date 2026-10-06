"""A torchvision ResNet-18 as an image classifier function (requires ``torch`` and ``torchvision``)."""

from __future__ import annotations

import numpy as np

from shapiq_games._optional import require

__all__ = ["ResNetClassifier"]


class ResNetClassifier:
    """ResNet-18 with the pinned ``IMAGENET1K_V1`` torchvision weights.

    Calling the classifier on a batch of RGB images of shape ``(batch, height, width, 3)``
    returns their class probabilities of shape ``(batch, 1000)``.
    """

    def __init__(self, *, device: str = "cpu") -> None:
        """Load the model.

        Args:
            device: The torch device. Defaults to ``"cpu"``.
        """
        torch = require("torch", purpose="the ResNet image games")
        models = require("torchvision.models", purpose="the ResNet image games")
        self._torch = torch
        self._device = torch.device(device)
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        self.categories: list[str] = list(weights.meta["categories"])
        self._model = models.resnet18(weights=weights).eval().to(self._device)
        self._preprocess = weights.transforms()

    def __call__(self, images: np.ndarray) -> np.ndarray:
        """Return the class probabilities of a batch of RGB images."""
        torch = self._torch
        tensor = torch.as_tensor(np.asarray(images), device=self._device)
        tensor = tensor.permute(0, 3, 1, 2).float() / 255.0
        with torch.no_grad():
            logits = self._model(self._preprocess(tensor))
        return torch.softmax(logits, dim=-1).cpu().numpy()
