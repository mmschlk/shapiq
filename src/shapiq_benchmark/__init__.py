"""Benchmark helpers for shapiq.

The benchmark model backends (``optuna``, ``tabpfn``, ``lightgbm``, ``xgboost``)
are optional and imported lazily. Install them with
``pip install 'shapiq[benchmark]'``. The package still imports without them; the
extras are only needed to build the corresponding models or to run the Optuna
hyperparameter optimization.
"""

import importlib.util
import warnings

_missing = [
    module
    for module in ["optuna", "tabpfn", "lightgbm", "xgboost"]
    if importlib.util.find_spec(module) is None
]

if _missing:
    msg = (
        "The 'shapiq_benchmark' package uses optional model backends that are not part of the"
        f" core shapiq install and are currently missing: {', '.join(_missing)}. Install them"
        " via,\n\n    pip install 'shapiq[benchmark]'\n"
    )
    warnings.warn(msg, ImportWarning, stacklevel=2)
