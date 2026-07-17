"""Import error handling for shapiq.vision."""

from __future__ import annotations

_vision_msg = (
    "The vision package requires the optional dependencies torch "
    "and scikit-image but they are not installed. "
    "Install them with: pip install shapiq[vision]"
)
_vision_import_error = ImportError(_vision_msg)


_model_protocol_msg = (
    "The provided model is not compatible with the required callable model "
    "interface. It must at least be callable with a single positional "
    "argument `input` and return a tensor or an object exposing a `.logits` "
    "attribute."
)


def _vision_protocol_error(obj: object, expected: str | None = None) -> TypeError:
    """Return a TypeError for an incompatible vision model."""
    prefix = f"Expected compatibility with {expected}. " if expected else ""
    return TypeError(f"{prefix}{_model_protocol_msg} Got: {type(obj)!r}")
