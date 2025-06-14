"""This module contains all benchmark games for image classification tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from warnings import warn

from shapiq.games.base import Game

if TYPE_CHECKING:
    import numpy as np


class ImageClassifier(Game):
    """An image classifier as a benchmark game.

    This benchmark game is based on image classifiers retrieved from huggingface's model hub or
    torchvision. The value function of the game is the class probability of the originally
    predicted class (the class with the highest probability on the original image) after removing
    not participating players, i.e., superpixels or patches, from the image.

    Several image classifier variants are available:
        - Vision Transformer (ViT) with various grid sizes constituting the `n_players`:
            - 144 patches: `vit_144_patches` (12x12 grid)
            - 36 patches: `vit_36_patches` (6x6 grid)
            - 16 patches: `vit_16_patches` (4x4 grid)
            - 9 patches: `vit_9_patches` (3x3 grid)
          For this model, the image is split into patches of equal size. Non-participating
          players are removed by setting the corresponding patch to the masking token.
        - ResNet-18, which is a convolutional neural network based on the ResNet architecture. For
            this model individual pixels are grouped together into superpixels. Non-participating
            players, i.e., superpixels, are removed by setting the corresponding superpixel to the
            mean color of the image (i.e. mean imputation).

    Note:
        Depending on the selected model, this game requires the `torch`, `torchvision`, and
            `transformers` packages to be installed.

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
        *,
        x_explain_path: str | None = None,
        normalize: bool = True,
        verbose: bool = False,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initializes the Image Classifier game.

        Args:
            model_name: The model used for the game. This can be either a callable that takes an
                image in form of a numpy array or and returns the class probabilities, or a string
                that specifies the pre-trained model to be used. The default models are
                ``'vit_144_patches'``, ``'vit_36_patches'``, ``'vit_16_patches'``,
                ``'vit_9_patches'``, and ``'resnet_18'``. Defaults to ``'vit_16_patches'``.

            n_superpixel_resnet: The approximate number of superpixels for the ResNet model to use.
                Defaults to ``14.`` This is only used if the model is 'resnet_18'.

            x_explain_path: The image to be explained. If not provided, a random image from the
                benchmark set is selected. Defaults to ``None``.

            normalize: Whether to normalize / center the game values. Defaults to ``True``.

            verbose: Whether to print the validation score of the model if trained. Defaults to
                ``True``.

            kwargs: Additional keyword arguments (not used).
        """
        if x_explain_path is None:
            msg = "The image to be explained must be provided."
            raise ValueError(msg)

        # validate inputs
        valid_models = [
            "vit_144_patches",
            "vit_36_patches",
            "vit_16_patches",
            "vit_9_patches",
            "resnet_18",
        ]
        if model_name.lower() not in valid_models:
            msg = f"Invalid model {model_name}. The model must be one of {valid_models}"
            raise ValueError(
                msg,
            )

        # read image with PIL
        from PIL import Image

        self.x_explain = Image.open(x_explain_path)

        # setup the models model
        self.model_function = model_name
        if "vit" in model_name:
            from shapiq.games.benchmark._setup._vit_setup import ViTModel

            # Extract the number of patches from the model name
            patch_sizes = {
                "vit_144_patches": 144,
                "vit_36_patches": 36,
                "vit_16_patches": 16,
                "vit_9_patches": 9,
            }
            n_players = patch_sizes[model_name]

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
            warn(
                f"{n_players} superpixels found and not {n_sp}.",
                stacklevel=2,
            ) if n_players != n_sp else None
            normalization_value = resnet_model.empty_value
            self.model_function = resnet_model

        super().__init__(
            n_players=n_players,
            normalize=normalize,
            normalization_value=normalization_value,
            verbose=verbose,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Returns the class probability of the coalition.

        The value function is the class probability of the coalition given the image classifier
        model. Non-participating players are removed from the image by setting the corresponding
        superpixel or patch to the mean color of the image.


        Args:
            coalitions: The coalitions of the game as a boolean array of shape (n_coalitions,
                n_players).

        Returns:
            The predicted class probability of the coalition given the image classifier model.

        """
        return self.model_function(coalitions)
