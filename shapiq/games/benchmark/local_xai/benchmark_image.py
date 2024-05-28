"""This module contains all benchmark games for image classification tasks."""

from typing import Optional
from warnings import warn

import numpy as np

from shapiq.games.base import Game


class ImageClassifier(Game):
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
        model_name: The model used for the game. This can be either a callable that takes an image in
            form of a numpy array or and returns the class probabilities, or a string that specifies
            the pre-trained model to be used. The default models are 'vit_16_patches',
            'vit_9_patches', and 'resnet_18'. Defaults to 'vit_16_patches'.
        n_superpixel_resnet: The approximate number of superpixels for the ResNet model to use.
            Defaults to 14. This is only used if the model is 'resnet_18'.
        x_explain_path: The image to be explained. If not provided, a random image from the benchmark set
            is selected. Defaults to None.
        normalize: Whether to normalize / center the game values. Defaults to True.

    Raises:
        ValueError: If an invalid model name is provided and the values are not precomputed.
        ValueError: If the image to be explained is not provided and the values are not precomputed.
        UserWarning: If the number of superpixels found is not equal to the provided number of
            superpixels.

    Examples:
        >>> from shapiq.games.benchmark.local_xai import ImageClassifier
        >>> game = ImageClassifier(x_explain_path='path/to/image.jpg')
        >>> game(game.grand_coalition)  # returns some value
        >>> game.n_players
        16
        >>> # precompute, save, and load values
        >>> game.precompute()
        >>> game.save_values('path/to/save.npz')
        >>> from shapiq.games import Game
        >>> loaded_game = Game(path_to_values='path/to/save.npz')

    """

    def __init__(
        self,
        model_name: str = "vit_16_patches",
        n_superpixel_resnet: int = 14,
        x_explain_path: Optional[str] = None,
        normalize: bool = True,
        verbose: bool = False,
        *args,
        **kwargs,
    ) -> None:

        if x_explain_path is None:
            raise ValueError("The image to be explained must be provided.")

        # validate inputs
        if model_name.lower() not in ["vit_16_patches", "vit_9_patches", "resnet_18"]:
            raise ValueError(
                f"Invalid model {model_name}. The model must be one of ['vit_16_patches', "
                f"'vit_9_patches', 'resnet_18']"
            )

        # read image with PIL
        from PIL import Image

        self.x_explain = Image.open(x_explain_path)

        # setup the models model
        self.model_function = model_name
        if model_name == "vit_16_patches" or model_name == "vit_9_patches":
            from shapiq.games.benchmark._setup._vit_setup import ViTModel

            n_players = 9
            if model_name == "vit_16_patches":
                n_players = 16
            vit_model = ViTModel(n_patches=n_players, input_image=self.x_explain, verbose=verbose)
            normalization_value = vit_model.empty_value
            self.model_function = vit_model
        else:
            from shapiq.games.benchmark._setup._resnet_setup import ResNetModel

            n_sp = n_superpixel_resnet
            resnet_model = ResNetModel(
                input_image=self.x_explain,
                verbose=verbose,
                batch_size=50,
                n_superpixels=n_sp,
            )
            n_players = resnet_model.n_superpixels
            # warn if not 14 superpixels
            warn(f"{n_players} superpixels found and not {n_sp}.") if n_players != n_sp else None
            normalization_value = resnet_model.empty_value
            self.model_function = resnet_model

        super().__init__(
            n_players=n_players,
            normalize=normalize,
            normalization_value=normalization_value,
            verbose=verbose,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """The value function of the game returning the class probability of the coalition with
            non-participating (removed players) masked on a superpixel or patches level.

        Args:
            coalitions: The coalitions of the game as a boolean array of shape (n_coalitions,
                n_players).

        Returns:
            The predicted class probability of the coalition given the image classifier model.
        """

        return self.model_function(coalitions)
