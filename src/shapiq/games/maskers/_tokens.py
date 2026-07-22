"""Token masking for sequence models of any array backend."""

from __future__ import annotations

from dataclasses import dataclass

from array_api_compat import array_namespace, device

from shapiq.games.maskers._base import BackendArray
from shapiq.games.maskers._baseline import BaselineMasker


@dataclass(frozen=True)
class TokenMasker[InputT: BackendArray](BaselineMasker[InputT]):
    """Masker replacing absent tokens with a mask token id.

    Players are token positions — the trailing axis of the token ids;
    leading axes become the explanation target shape, so a batch of
    sequences is explained at once. Present positions keep the explained
    sequence's ids, absent positions receive the mask token (a tokenizer's
    mask or pad id), so the model scores partially masked sequences.

    A special kind of ``BaselineMasker``: the baseline is one token id
    broadcast over every position (or a per-position id array), and the
    masked ids keep the inputs' integer dtype, backend, and device.
    Grouping subword tokens into word players mirrors how
    ``SuperpixelMasker`` groups pixels and is a planned sibling.

    Example:
        >>> masker = TokenMasker(inputs=token_ids, baseline=tokenizer.mask_token_id)
        >>> predictor = ModelMaskedPredictor(masker=masker, model=score_tokens)
    """

    baseline: InputT | int  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Broadcast a scalar mask token, then validate like a baseline."""
        try:
            array_namespace(self.baseline)
        except TypeError:
            if self.inputs.ndim < 1:
                msg = "inputs must carry at least the trailing token axis"
                raise ValueError(msg) from None
            xp = array_namespace(self.inputs)
            filled = xp.full(
                (int(self.inputs.shape[-1]),),
                self.baseline,
                dtype=self.inputs.dtype,
                device=device(self.inputs),
            )
            object.__setattr__(self, "baseline", filled)
        super().__post_init__()
