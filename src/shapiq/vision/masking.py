"""Masking strategies for vision models.

Defines how to replace absent players in masked images before forwarding
through the model. Masking is applied in pixel space for CNNs and token
space for ViTs. Requires PyTorch to be installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from shapiq.typing import Model


class CNNMaskingStrategy(ABC):
    """Base class for pixel-space masking strategies used with CNN models.

    Implementations receive the original image as a ``(C, H, W)`` tensor and
    a coalition matrix, and return a batch of masked images ready for a
    single forward pass through the model.
    """

    @abstractmethod
    def apply(
        self, image: torch.Tensor, player_masks: torch.Tensor, coalitions: torch.Tensor
    ) -> torch.Tensor:
        """Apply masking to produce a batch of masked images.

        Args:
            image: Original image as a float32 ``(C, H, W)`` tensor.
            player_masks: Boolean tensor of shape ``(n_players, H, W)``
                mapping each player to its pixel region.
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``
                where ``True`` indicates a player is present (unmasked).

        Returns:
            Float32 tensor of shape ``(n_coalitions, C, H, W)`` with absent
            players replaced by the imputation value.
        """
        ...

    def _build_pixel_mask(
        self,
        player_masks: torch.Tensor,
        coalitions: torch.Tensor,
    ) -> torch.Tensor:
        """Build a combined pixel absence mask for all coalitions.

        Args:
            player_masks: Boolean tensor of shape ``(n_players, H, W)``.
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``.

        Returns:
            Boolean tensor of shape ``(n_coalitions, H, W)`` where ``True``
            means the pixel belongs to an absent player and should be imputed.
        """
        absent_players = ~coalitions  # (n_coalitions, n_players)

        n_players, H, W = player_masks.shape
        masks_flat = player_masks.view(n_players, -1).float()  # (n_players, H*W)

        # Union pixel masks of all absent players per coalition
        pixel_mask = (absent_players.float() @ masks_flat).bool()  # (n_coalitions, H*W)
        return pixel_mask.view(-1, H, W)  # (n_coalitions, H, W)


class MeanColorMasking(CNNMaskingStrategy):
    """Imputes absent player regions with the per-channel mean color of the original image.

    The mean is computed per channel across all spatial positions of the
    original image and broadcast into the masked regions.
    """

    def apply(
        self, image: torch.Tensor, player_masks: torch.Tensor, coalitions: torch.Tensor
    ) -> torch.Tensor:
        """Apply mean color masking to absent player regions."""
        import torch

        pixel_mask = self._build_pixel_mask(player_masks, coalitions)  # (n_coalitions, H, W)
        mean_color = image.mean(dim=(1, 2))  # (C,)

        return torch.where(
            pixel_mask.unsqueeze(1),  # (n_coalitions, 1, H, W)
            mean_color[None, :, None, None],  # (1, C, 1, 1)
            image.unsqueeze(0),  # (1, C, H, W)
        )


class ZeroMasking(CNNMaskingStrategy):
    """Imputes absent player regions with a constant scalar value.

    Args:
        value: The fill value used for masked pixels. Defaults to ``0.0``.
    """

    def __init__(self, value: float = 0.0) -> None:
        """Initialize the zero masking strategy with a specified fill value."""
        self.value = value

    def apply(
        self, image: torch.Tensor, player_masks: torch.Tensor, coalitions: torch.Tensor
    ) -> torch.Tensor:
        """Apply zero (or constant) masking to absent player regions."""
        import torch

        pixel_mask = self._build_pixel_mask(player_masks, coalitions)  # (n_coalitions, H, W)

        return torch.where(
            pixel_mask.unsqueeze(1),  # (n_coalitions, 1, H, W)
            torch.tensor(self.value, dtype=image.dtype, device=image.device),
            image.unsqueeze(0),  # (1, C, H, W)
        )


class TransformerMaskingStrategy(ABC):
    """Base class for token-space masking strategies used with ViT models.

    Implementations convert a coalition matrix into a ``bool_masked_pos``
    tensor suitable for passing directly to a ViT forward call.
    """

    @abstractmethod
    def apply(
        self,
        coalitions: torch.Tensor,
        token_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Convert coalitions to a token-level boolean mask.

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``
                where ``True`` indicates a player is present.
            token_masks: Integer tensor of shape
                ``(n_players, tokens_per_player)`` mapping each player to its
                flat token indices.

        Returns:
            Boolean tensor of shape ``(n_coalitions, n_tokens)`` where
            ``True`` means the token is masked (player absent) and ``False``
            means the token is visible (player present).
        """
        ...

    def _to_token_mask(
        self,
        coalitions: torch.Tensor,
        token_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Convert a coalition tensor to a flat token-level boolean mask.

        Tokens belonging to absent players are set to ``True`` (masked);
        tokens belonging to present players are set to ``False`` (visible).

        Args:
            coalitions: Boolean tensor of shape ``(n_coalitions, n_players)``
                where ``True`` indicates a player is present.
            token_masks: Integer tensor of shape
                ``(n_players, tokens_per_player)`` containing the flat token
                indices for each player.

        Returns:
            Boolean tensor of shape ``(n_coalitions, n_tokens)`` on the same
            device as ``coalitions``.
        """
        import torch

        n_players = token_masks.shape[0]
        n_tokens = int(token_masks.max()) + 1

        # (n_players, n_tokens): one-hot encoding of which tokens belong to which player
        player_to_token = torch.zeros(
            (n_players, n_tokens), dtype=torch.bool, device=coalitions.device
        )
        player_to_token.scatter_(dim=1, index=token_masks, value=True)  # (n_players, n_tokens)

        # A token is visible (False) if at least one present player owns it
        visible = coalitions.float() @ player_to_token.float()  # (n_coalitions, n_tokens)
        return ~visible.bool()


class BoolMaskedPosStrategy(TransformerMaskingStrategy):
    """Masks tokens by passing ``bool_masked_pos`` directly to the model forward call.

    This strategy requires the model to support the ``bool_masked_pos``
    argument (e.g. :class:`~transformers.ViTForMaskedImageModeling`).
    """

    def apply(self, coalitions: torch.Tensor, token_masks: torch.Tensor) -> torch.Tensor:
        """Apply boolean masking by converting coalitions to a ``bool_masked_pos`` tensor."""
        return self._to_token_mask(coalitions, token_masks)


class MaskTokenStrategy(TransformerMaskingStrategy):
    """Masks tokens by zeroing the mask_token embedding before the forward pass."""

    def __init__(self, model: Model) -> None:
        """Initialise with the ViT model whose mask token will be zeroed.

        Args:
            model: A ViT model with a ``vit.embeddings.mask_token``
                parameter.
        """
        self._model = model

    def apply(self, coalitions: torch.Tensor, token_masks: torch.Tensor) -> torch.Tensor:
        """Apply masking by setting the model's mask_token embedding to zero."""
        import torch

        self._model.vit.embeddings.mask_token = torch.nn.Parameter(
            torch.zeros(1, 1, self._model.config.hidden_size)
        )
        return self._to_token_mask(coalitions, token_masks)
