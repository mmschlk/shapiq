"""The shapiq-games library."""

import importlib.util
import warnings

_missing = [
    module
    for module in ["transformers", "torch", "tabpfn"]
    if importlib.util.find_spec(module) is None
]

if _missing:
    msg = (
        "The 'shapiq_games' package requires optional dependencies that are more than the normal"
        " shapiq package. Install and import the packages as you go or preferably via,\n\n "
        "    uv add --group all_ml"
    )
    warnings.warn(msg, ImportWarning, stacklevel=2)
