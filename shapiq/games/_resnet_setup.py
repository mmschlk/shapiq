"""This file contains the ResNetModel class that is used as a benchmark game for `shapiq`.

Note to developers:
    This file should not be imported directly as it requires a lot of dependencies to be installed
    (e.g. `torch`, `torchvision`, `PIL`).
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL.Image import Image

__all__ = ["ResNetModel"]


class ResNetModel:

    """Sets up the ResNetModel model from torchvision."""

    def __init__(self, input_image: Image, verbose: bool = True) -> None:
        from torchvision.models import ResNet18_Weights, resnet18

        # setup model
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.eval()

        # setup image
        self.input_image = input_image

        preprocess = weights.transforms()
        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)

        # evaluate the model
        output = self.model(input_batch)
        class_id = int(torch.argmax(output[0]).item())
        original_score = float(output[0][class_id].item())
        class_label = weights.meta["categories"][class_id]

        if verbose:
            print(f"Predicted class: {class_label} with score: {original_score}")

    def __call__(self, coalition: np.ndarray) -> np.ndarray[float]:
        """Returns the class probability of the coalition.

        Args:
            coalition: The coalition of players (i.e. super-patches).

        Returns:
            The class probability of the coalition.
        """
        raise NotImplementedError("Not finished yet.")

    def model_call(self, input_image: torch.Tensor) -> torch.Tensor:
        """Calls the model with the input image.

        Args:
            input_image: The input image.

        Returns:
            The class probability
        """
        with torch.no_grad():
            output = self.model(input_image)
            output = F.softmax(output, dim=1)
            return output
