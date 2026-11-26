"""FIxLIP: cross-modal explanations of vision-language models such as CLIP and SigLIP.

Needs torch and Hugging Face ``transformers`` models.
"""

from dataclasses import dataclass
from typing import Any, Self, override

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jaxtyping import Bool, Float

from shapiq.masker.base import require_eager
from shapiq.players import FinitePlayers
from shapiq.typing import typechecked

from .base import XAIGame

SIGLIP_TEXT_LENGTH = 64  # SigLIP was trained on texts padded to 64 tokens


def _embedding(outputs: Any) -> torch.Tensor:  # noqa: ANN401 - a tensor or a model output
    """Return the unit-norm embedding of ``get_*_features``: a tensor, or ``pooler_output``."""
    embedding = outputs if torch.is_tensor(outputs) else outputs.pooler_output
    return torch.nn.functional.normalize(embedding, dim=-1)


@typechecked
@dataclass(frozen=True, slots=True, weakref_slot=True)
class FIxLIPGame[P](XAIGame[P, "FIxLIPGame[P]"]):
    """The logit of an image and a text whose absent patches and tokens are removed.

    FIxLIP (Baniecki et al., 2025) explains the interactions between an image and a text in CLIP
    and SigLIP. The players are the image's patches, row by row, then the text's tokens. Absent
    patches are zero in the normalized pixels, the normalization mean; absent tokens are removed
    from the attention mask. The value is the model's logit of the image and the text, its
    ``logits_per_image``: ``scale * cosine`` for CLIP, plus a bias for SigLIP.

    ``pixel_values`` (``(1, 3, height, width)``) and ``input_ids`` (``(1, length)``) are the
    processor's outputs, ``first_token`` the position of the first explained token. Build the
    game from an image and a text with ``FIxLIPGame.from_processor``. Coalitions are evaluated
    in batches of ``batch_size`` on the model's device.
    """

    pixel_values: torch.Tensor
    input_ids: torch.Tensor
    first_token: int
    batch_size: int = 16

    @override
    def _infer_value_spec(self) -> jax.ShapeDtypeStruct:
        """Return the dtype used for the logits in this representation."""
        return jax.ShapeDtypeStruct((), jnp.float32)

    @classmethod
    def from_processor(
        cls,
        model: Any,  # noqa: ANN401 - a transformers CLIP or SigLIP model
        processor: Any,  # noqa: ANN401 - its processor
        image: Any,  # noqa: ANN401 - any image the processor takes
        text: str,
        *,
        batch_size: int = 16,
    ) -> Self:
        """Return the game of ``image`` and ``text``, processed for ``model``.

        The explained tokens are those between CLIP's start and end tokens, and SigLIP's tokens
        before its end token.
        """
        if model.config.model_type == "siglip":
            inputs = processor(
                images=image,
                text=text,
                padding="max_length",
                max_length=SIGLIP_TEXT_LENGTH,
                return_tensors="pt",
            )
            first = 0
            n_tokens = int((inputs["input_ids"] != processor.tokenizer.pad_token_id).sum()) - 1
        else:
            inputs = processor(images=image, text=text, padding=True, return_tensors="pt")
            first, n_tokens = 1, inputs["input_ids"].shape[1] - 2
        pixel_values = inputs["pixel_values"]
        patch = model.config.vision_config.patch_size
        n_patches = (pixel_values.shape[-2] // patch) * (pixel_values.shape[-1] // patch)
        players = FinitePlayers(n_patches + n_tokens)
        return cls(model, players, pixel_values, inputs["input_ids"], first, batch_size)

    @property
    def n_patches(self) -> int:
        """Return the number of image players."""
        model: Any = self.model
        patch = model.config.vision_config.patch_size
        return (self.pixel_values.shape[-2] // patch) * (self.pixel_values.shape[-1] // patch)

    @override
    def _eval_bool(
        self, coalition: Bool[jax.Array, "#*batch {self.n_players}"]
    ) -> Float[jax.Array, "*batch"]:
        """Embed every coalition's image and text in batches and return their logits."""
        require_eager("FIxLIP", coalition)  # torch models
        flat = np.array(coalition).reshape(-1, self.n_players)  # writable, for torch
        logits = [
            self._logits(flat[start : start + self.batch_size])
            for start in range(0, len(flat), self.batch_size)
        ]
        values = np.concatenate(logits) if logits else np.zeros(0, np.float32)
        return jnp.asarray(values).reshape(coalition.shape[:-1])

    def _logits(self, coalitions: Bool[np.ndarray, "n n_players"]) -> Float[np.ndarray, " n"]:
        """Return the logits of a batch of coalitions."""
        model: Any = self.model
        device = next(model.parameters()).device
        n, n_patches = len(coalitions), self.n_patches
        patch = model.config.vision_config.patch_size
        rows = self.pixel_values.shape[-2] // patch
        present = torch.as_tensor(coalitions, device=device)
        # Repeat every patch's membership over its pixels.
        image_mask = present[:, :n_patches].reshape(n, 1, rows, -1)
        image_mask = image_mask.repeat_interleave(patch, -2).repeat_interleave(patch, -1)
        attention = torch.ones((n, self.input_ids.shape[-1]), dtype=torch.long, device=device)
        attention[:, self.first_token : self.first_token + self.n_players - n_patches] = present[
            :, n_patches:
        ]
        with torch.no_grad():
            image = _embedding(
                model.get_image_features(pixel_values=self.pixel_values.to(device) * image_mask)
            )
            text = _embedding(
                model.get_text_features(
                    input_ids=self.input_ids.to(device).repeat(n, 1), attention_mask=attention
                )
            )
            logits = model.logit_scale.exp() * torch.sum(image * text, dim=-1)
            if getattr(model, "logit_bias", None) is not None:
                logits = logits + model.logit_bias
        return logits.reshape(n).float().numpy(force=True)
