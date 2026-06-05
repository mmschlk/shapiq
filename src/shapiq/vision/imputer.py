import numpy as np

from shapiq.imputer.base import Imputer

from .architecture import ModelArchitectureStrategy, ResNetArchitecture, ViTArchitecture
from .players import PlayerStrategy
from .masking import PixelMaskingStrategy, LatentMaskingStrategy

from .utils import is_valid_image_shape


class ImageImputer(Imputer):
    """
    Imputer for images: creates masked versions of the input image based on player coalitions and returns model predictions.    
    """
    def __init__(
        self,
        model,
        image: np.ndarray,
        player_strategy: PlayerStrategy | None = None,
        masking_strategy: PixelMaskingStrategy | LatentMaskingStrategy | None = None,
        normalize: bool = True,
        model_architecture: ModelArchitectureStrategy | None = None,
        vit_processor=None,
    ):

        if not is_valid_image_shape(image):
            raise ValueError(
                f"Expected image with shape (H, W, C), got {image.shape}."
                "Convert your image to (H, W, C) format before passing it to the imputer."
            )
        
        self.image = image
        self.architecture = model_architecture or self._predict_model_architecture(model, masking_strategy, player_strategy, vit_processor)

        self.architecture.prepare(image)
        self.n_features = self.architecture._player_strategy.n_players

        dummy_data = np.zeros((1, self.n_features))
        super().__init__(model=model, data=dummy_data)

        self.empty_prediction = self.calc_empty_prediction()
        if normalize:
            self.normalization_value = self.empty_prediction

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """
        Calculates the value function for a batch of coalitions.
        
        Args:
            coalitions: (n_coalitions, n_players) boolean array
            
        Returns:
            (n_coalitions,) float array with model-Predictions
        
        """
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        return self.architecture.value_function(coalitions)

    def calc_empty_prediction(self) -> float:
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction of the model provided only missing features.

        """
        return float(self.architecture.value_function(np.zeros((1, self.n_features), dtype=bool))[0])

    @property
    def player_masks(self) -> np.ndarray | None:
        """Spatial masks per player, shape (n_players, H, W). None for latent-space architectures."""
        return getattr(self.architecture, "_player_masks", None)
    
    def _predict_model_architecture(self, model, masking_strategy=None, player_strategy=None, vit_processor=None) -> ModelArchitectureStrategy:
        """Auto-detects the model architecture and returns the appropriate ModelArchitectureStrategy."""
        
        import torchvision.models as models
        if isinstance(model, models.ResNet):
            return ResNetArchitecture(model, masking_strategy, player_strategy)
        
        from transformers import ViTForImageClassification
        if isinstance(model, ViTForImageClassification):
            if vit_processor is None:
                raise ValueError("Please provide a processor for ViT models.")
            return ViTArchitecture(model, vit_processor, masking_strategy, player_strategy)
        
        raise ValueError(f"Could not auto-detect architecture for model type '{type(model)}'.")