from __future__ import annotations

import numpy as np

from shapiq.imputer.base import Imputer

from .architecture import ModelArchitectureStrategy

from .utils import as_hwc_array, tensor_to_numpy, ImageLike


class ImageImputer(Imputer):
    """
    Imputer for images: creates masked versions of the input image based on player coalitions and returns model predictions.
    
    Requires pytorch to be installed for tensor operations. 
    Converts images to numpy arrays internally, and to tensors for model inference.    
    """
    def __init__(
        self,
        model_architecture: ModelArchitectureStrategy,
        image: ImageLike,
        normalize: bool = True,
        batch_size: int = 32,
    ):
                     
        self.image = as_hwc_array(image)
        self.architecture = model_architecture
        self.batch_size = batch_size

        self.architecture.prepare(self.image)
        self.n_features = self.architecture._player_strategy.n_players

        dummy_data = np.zeros((1, self.n_features))
        super().__init__(model=model_architecture.model, data=dummy_data)

        self.empty_prediction = self.calc_empty_prediction()
        if normalize:
            self.normalization_value = self.empty_prediction

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate the model for a batch of player coalitions.

        Converts ``coalitions`` to a boolean PyTorch tensor, splits it into
        mini-batches of at most :attr:`batch_size` rows, and issues one
        ``model`` forward call per mini-batch.

        Args:
            coalitions: Boolean array of shape ``(n_coalitions, n_players)``

        Returns:
            Float numpy array of shape ``(n_coalitions,)`` containing the
            scalar model output (logit or probability depending on
            :attr:`architecture`) for each coalition.
        
        """
        import torch
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
            
        n = len(coalitions)
        if n <= self.batch_size:
            coalitions_t = torch.from_numpy(coalitions).bool()
            return tensor_to_numpy(self.architecture.value_function(coalitions_t))
            
        chunks = [
            tensor_to_numpy(
                self.architecture.value_function(
                    torch.from_numpy(coalitions[start : start + self.batch_size]).bool()
                )
            )
            for start in range(0, n, self.batch_size)
        ]
        return np.concatenate(chunks, axis=0)

    def calc_empty_prediction(self) -> float:
        """Evaluate the model with all players absent to obtain the baseline prediction.

        Returns:
            The scalar model output when no players are present.
        """
        return float(self.value_function(np.zeros((1, self.n_features), dtype=bool))[0])

    @property
    def player_masks(self) -> np.ndarray:
        """Spatial masks per player as a ``(n_players, H, W)`` boolean numpy array.

        Returns:
            Boolean numpy array of shape ``(n_players, H, W)``.
        """
        return tensor_to_numpy(self.architecture.player_masks)