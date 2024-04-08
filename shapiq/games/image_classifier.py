"""This module contains all benchmark games for image classification tasks."""

from typing import Optional

import numpy as np

from .base import Game


class ImageClassifierGame(Game):

    """An image classifier as a benchmark game.

    This benchmark game is based on image classifiers retrieved from huggingface's model hub or
    torchvision. The value function of the game is the class probability of the originally
    predicted class (the class with the highest probability on the original image) after removing
    not participating players, i.e., superpixels or patches, from the image.

    Two image classifier variants are available:
        - Vision Transformer (ViT) with 16 or 9 patches constituting the `n_players`. These model
            ids are `vit_16_patches` and `vit_9_patches`, respectively. For this model, the image is
            split into patches of equal size. Non-participating players are removed by setting the
            corresponding patch to the masking token.
        - ResNet-18, which is a convolutional neural network based on the ResNet architecture. For
            this model individual pixels are grouped together into superpixels. Non-participating
            players, i.e., superpixels, are removed by setting the corresponding superpixel to the
            mean color of the image (i.e. mean imputation).

    Note:
        Depending on the selected model, this game requires the `torch`, `torchvision`, and
            `transformers` packages to be installed.

    Args:
        path_to_values: The path to the precomputed values of the game. If provided, the game is
            initialized with the precomputed values. If not provided, the game is initialized with
            the given model.
        model: The model used for the game. This can be either a callable that takes an image in
            form of a numpy array or and returns the class probabilities, or a string that specifies
            the pre-trained model to be used. The default models are 'vit_16_patches',
            'vit_9_patches', and 'resnet_18'. Defaults to 'vit_16_patches'.
        x_explain: The image to be explained. If not provided, a random image from the benchmark set
            is selected. Defaults to None.
        normalize: Whether to normalize / center the game values. Defaults to True.
    """

    def __init__(
        self,
        path_to_values: Optional[str] = None,
        model: str = "vit_16_patches",
        x_explain: Optional[str] = None,
        normalize: bool = True,
        verbose: bool = True,
    ) -> None:
        # check path
        if path_to_values is not None:
            super().__init__(path_to_values=path_to_values)
            return

        if x_explain is None:
            raise ValueError("The image to be explained must be provided.")

        # validate inputs
        if model.lower() not in ["vit_16_patches", "vit_9_patches", "resnet_18"]:
            raise ValueError(
                f"Invalid model {model}. The model must be one of ['vit_16_patches', "
                f"'vit_9_patches', 'resnet_18']"
            )

        # read image with PIL
        from PIL import Image

        self.x_explain = Image.open(x_explain)

        # set model
        self.model_function = model
        if model == "vit_16_patches" or model == "vit_9_patches":
            from ._vit_setup import ViTModel

            n_players = 9
            if model == "vit_16_patches":
                n_players = 16
            vit_model = ViTModel(n_patches=n_players, input_image=self.x_explain, verbose=verbose)
            normalization_value = vit_model.empty_value
            self.model_function = vit_model
        else:
            raise NotImplementedError("Not finished yet.")
            from ._resnet_setup import ResNetModel

            resnet_model = ResNetModel(input_image=self.x_explain, verbose=verbose)
            n_players = resnet_model.n_superpixels
            normalization_value = resnet_model.empty_value
            self.model_function = resnet_model

        super().__init__(
            n_players=n_players, normalize=normalize, normalization_value=normalization_value
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """The value function of the game."""

        return self.model_function(coalitions)
