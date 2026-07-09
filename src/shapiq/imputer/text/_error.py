"""Import error handling for the text module."""

from __future__ import annotations

_text_msg = (
    "The text explanation module requires the optional dependencies "
    "torch, transformers, and nltk, but they are not installed.\n"
    "Install them with:\n\n"
    "    pip install 'shapiq[text]'"
)

_text_import_error = ImportError(_text_msg)