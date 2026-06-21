"""Import error handling for shapiq.vision."""

from __future__ import annotations

_vision_msg = (
    "The vision package requires the optional dependencies torch "
    "and scikit-image but they are not installed. "
    "Install them with: pip install shapiq[vision]"
)
_vision_import_error = ImportError(_vision_msg)
