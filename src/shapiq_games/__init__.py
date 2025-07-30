"""The shapiq-games library."""

import importlib.util

_missing = [
    module
    for module in ["transformers", "torch", "tabpfn"]
    if importlib.util.find_spec(module) is None
]

if _missing:
    msg = (
        "The 'shapiq_games' package requires optional dependencies. "
        "Please install shapiq with the 'games' extra:\n"
        "    uv pip install 'shapiq[games]'\n"
        "or\n"
        "    pip install 'shapiq[games]'\n"
    )
    raise ImportError(msg)
