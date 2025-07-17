"""This file contains the ResNetModel class that is used as a benchmark game for `shapiq`.

Note to developers:
    This file should not be imported directly as it requires a lot of dependencies to be installed
    (e.g. `torch`, `torchvision`, `PIL`, and `skimage`).
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from skimage.segmentation import slic
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

__all__ = ["ResNetModel"]


class ResNetModel:
    """Sets up the ResNetModel model from torchvision as a callable function.

    Note:
        This class requires the `torch`, `torchvision`, `PIL`, and `skimage` packages to be
        installed.

    Args:
        input_image: The input image.
        n_superpixels: The number of superpixels to be searched for. Defaults to 14.
        verbose: Whether to print the predicted class and score. Defaults to True.
        batch_size: The batch size for the model evaluations. Defaults to 50.

    Attributes:
        model: The ResNet model.
        batch_size: The batch size for the model evaluations.
        class_score: The score of the original image.
        class_label: The class label of the original image.
        class_id: The class id of the original image.
        empty_value: The score of the background image.
        n_superpixels: The number of superpixels.
        superpixels: The superpixel mask found by SLICO.

    """

    def __init__(
        self,
        input_image: Image.Image,
        *,
        n_superpixels: int = 14,
        verbose: bool = True,
        batch_size: int = 50,
    ) -> None:
        # setup model
        weights = ResNet18_Weights.DEFAULT
        self.model = resnet18(weights=weights)
        self.model.eval()
        self.batch_size = batch_size

        # setup preprocess steps and transforms
        self._preprocess = weights.transforms()
        self._tensor_transform = transforms.ToTensor()

        # setup image and tensor
        self._image: Image.Image = copy.deepcopy(input_image)
        self._image_shape = np.asarray(self._image).shape
        self._image_tensor: torch.Tensor = self._tensor_transform(self._image)
        self._input_tensor: torch.Tensor = self._preprocess(self._image_tensor)

        # evaluate the model on the original image
        output = self.model_call(self._input_tensor.unsqueeze(0))
        class_id = int(torch.argmax(output[0]).item())
        self.class_score = float(output[0][class_id].item())
        self.class_label = weights.meta["categories"][class_id]
        self.class_id: int = int(class_id)

        if verbose:
            pass

        # get background tensor for gray image
        _background_image = np.zeros(self._image_shape, dtype=np.uint8)
        _background_image[:, :] = [127, 127, 127]
        self._background_image: Image.Image = Image.fromarray(_background_image)
        self._background_image_tensor: torch.Tensor = self._tensor_transform(self._background_image)
        self._background_input_tensor: torch.Tensor = self._preprocess(
            self._background_image_tensor,
        )

        # evaluate the model on the background
        output_background = self.model_call(self._background_input_tensor.unsqueeze(0))
        self.empty_value = float(output_background[0][class_id].item())

        # get superpixels
        self.n_superpixels, self.superpixels = self.get_superpixels(
            image=np.array(input_image),
            n_segments=n_superpixels,
        )

        # setup bool mask for all superpixels
        self._superpixel_masks = torch.zeros(
            (self.n_superpixels, self._image_tensor.shape[1], self._image_tensor.shape[2]),
            dtype=torch.bool,
        )
        for i in range(self.n_superpixels):
            mask = self.superpixels == i + 1
            self._superpixel_masks[i, :, :] = torch.tensor(mask, dtype=torch.bool)

    def __call__(self, coalitions: np.ndarray) -> np.ndarray[float]:
        """Returns the class probability of the coalition of superpixels.

        Superpixels not in the coalition are masked with a gray background.

        Args:
            coalitions: A 2d matrix of coalition of players (i.e. super-patches) in shape
            (n_coalitions, n_superpixels).

        Returns:
            The class probability of the coalition.

        """
        outputs = None
        for batch in range(0, len(coalitions), self.batch_size):
            output = self._call_batch(coalitions[batch : batch + self.batch_size])
            outputs = output if batch == 0 else np.concatenate((outputs, output), axis=0)
        return outputs

    def _call_batch(self, coalitions: np.ndarray) -> np.ndarray[float]:
        """Returns the class probability for a batch of coalitions.

        Args:
            coalitions: A 2d matrix of coalition of players (i.e. super-patches) in shape
            (n_coalitions, n_superpixels).

        Returns:
            The class probability of the coalition.

        """
        # create tensor dataset for all coalition in coalitions and apply the masks
        masked_images = torch.stack((self._image_tensor,) * len(coalitions))
        for i, coalition in enumerate(coalitions):
            for superpixel, is_present in enumerate(coalition, start=1):
                if not is_present:
                    masked_images[i, :, self._superpixel_masks[superpixel - 1]] = (
                        self._background_image_tensor[:, self._superpixel_masks[superpixel - 1]]
                    )
        # apply the model
        output = self.model_call(self._preprocess(masked_images))[..., self.class_id]
        return output.detach().numpy()

    def model_call(self, input_image: torch.Tensor) -> torch.Tensor:
        """Calls the model with the input image.

        Args:
            input_image: The input image.

        Returns:
            The class probability

        """
        with torch.no_grad():
            output = self.model(input_image)
            return F.softmax(output, dim=-1)

    @staticmethod
    def get_superpixels(image: np.ndarray, n_segments: int = 14) -> tuple[int, np.ndarray]:
        """Run SLIC and return the number of superpixels and the superpixel mask.

        Runs SLIC and retrying with randomized values if the number of superpixels does not match
        the desired number.

        Args:
            image: The image.
            n_segments: The number of segments. Defaults to 14.

        Returns:
            The number of superpixels and the superpixel mask.

        """
        # run slic for first time
        superpixels = slic(image, n_segments=n_segments, start_label=1, slic_zero=True)
        n_superpixels = len(np.unique(superpixels))

        # retry with increasing segments
        if n_superpixels < n_segments:
            iteration, n_segments_iter = 0, n_segments
            while iteration < 20 and n_superpixels < n_segments:
                n_segments_iter += 1
                superpixels = slic(image, n_segments=n_segments_iter, start_label=1, slic_zero=True)
                n_superpixels = len(np.unique(superpixels))
                iteration += 1

        # fallback to clipping the last superpixels
        if n_superpixels >= n_segments:
            # clip the superpixels to the desired number of segments
            superpixels = np.clip(superpixels, a_min=1, a_max=n_segments - 1)
            n_superpixels = n_segments

        return n_superpixels, superpixels
