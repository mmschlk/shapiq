from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

class CNNMaskingStrategy(ABC):
    @abstractmethod
    def apply(self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        """
        Args:
            image:        (H, W, C) original image
            player_masks: (n_players, H, W) boolean masks per player
            coalition:    (n_coalitions, n_players) boolean array
        
        Returns:
            masked_images: (n_coalitions, H, W, C)
        """
        ...


class MeanColorMasking(CNNMaskingStrategy):
    """Imputes the masked pixels with the mean color of the entire image."""
    
    def apply(self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        n_coalitions = coalition.shape[0]
        H, W, _ = image.shape
        
        masked_images = np.stack([image] * n_coalitions, axis=0) # shape (n_coalitions, H, W, C)
        
        mask = np.zeros((n_coalitions, H, W), dtype=bool)
        for i, coal in enumerate(coalition):
            for j, is_present in enumerate(coal):
                if not is_present:
                    mask[i] |= player_masks[j]

        masked_images[mask] = image.mean(axis=(0, 1))
        return masked_images


class ZeroMasking(CNNMaskingStrategy):
    def __init__(self, value: float = 0.0):
        self.value = value
    
    def apply(self, image: np.ndarray, player_masks: np.ndarray, coalition: np.ndarray) -> np.ndarray:
        n_coalitions = coalition.shape[0]
        H, W, _ = image.shape
        
        masked_images = np.stack([image] * n_coalitions, axis=0) # shape (n_coalitions, H, W, C)
        
        mask = np.zeros((n_coalitions, H, W), dtype=bool)
        for i, coal in enumerate(coalition):
            for j, is_present in enumerate(coal):
                if not is_present:
                    mask[i] |= player_masks[j]

        masked_images[mask] = self.value
        return masked_images


class TransformerMaskingStrategy(ABC):
    """Defines how tokens are masked in latent/embedding space."""

    @abstractmethod
    def apply(
        self,
        coalitions: np.ndarray,
        token_masks: np.ndarray,
    ) -> torch.Tensor:               # (B, n_coalitions)
        ...
        
    def _to_token_mask(
        self,
        coalitions: np.ndarray,   # (n_coalitions, n_players)
        token_masks: np.ndarray, # (n_players, tokens_per_player)
    ) -> torch.Tensor:             # (n_coalitions, n_tokens)
        """Converts coalitions to token_mask.
        
        True  = token is masked (player absent)
        False = token is visible (player present)
        """
        import torch
        
        n_coalitions = coalitions.shape[0]
        n_tokens = int(token_masks.max()) + 1
        
        token_mask = torch.ones((n_coalitions, n_tokens), dtype=torch.bool)
        for i, coalition in enumerate(coalitions):
            for player, is_present in enumerate(coalition):
                if is_present:
                    token_mask[i, token_masks[player]] = False
        
        return token_mask


class BoolMaskedPosStrategy(TransformerMaskingStrategy):
    """Masks tokens via the token_mask argument in the model forward pass."""

    def apply(self, coalitions: np.ndarray, token_masks: np.ndarray) -> torch.Tensor:
        return self._to_token_mask(coalitions, token_masks)


class MaskTokenStrategy(TransformerMaskingStrategy):
    """Masks tokens by zeroing the mask_token embedding before the forward pass."""

    def __init__(self, model) -> None:
        self._model = model

    def apply(self, coalitions: np.ndarray, token_masks: np.ndarray) -> torch.Tensor:
        import torch
        
        token_mask = self._to_token_mask(coalitions, token_masks)
        self._model.vit.embeddings.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, self._model.config.hidden_size)
        )
        return token_mask
