"""Import error handling for ShaplEIG."""

from __future__ import annotations

_shapleig_msg = (
    "ShaplEIG requires the optional dependencies torch, gpytorch, "
    "botorch, and linear-operator but they are not installed. "
    "Install them with: pip install 'shapiq[shapleig]'"
)
_shapleig_import_error = ImportError(_shapleig_msg)
