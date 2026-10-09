"""TabPFN models by version (requires ``tabpfn>=6.0``) and their ``inf`` missing values."""

from __future__ import annotations

import importlib
import importlib.metadata
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq.utils.modules import safe_isinstance
from shapiq_games._optional import require

if TYPE_CHECKING:
    from shapiq_games.typing import Task

__all__ = [
    "DEFAULT_TABPFN_VERSION",
    "build_tabpfn",
    "check_inf_baseline",
    "require_inf_passthrough",
]

DEFAULT_TABPFN_VERSION = "v2"
"""The TabPFN version used when none is chosen: v2, whose checkpoints download without a license."""

_PASSTHROUGH_INF_VERSION = (8, 1)  # the first tabpfn release with inference_config PASSTHROUGH_INF


def build_tabpfn(
    task: Task,
    version: str = DEFAULT_TABPFN_VERSION,
    **params: Any,
) -> Any:  # noqa: ANN401
    """Build an unfitted TabPFN model of a version, with the version's default settings.

    Args:
        task: ``"classification"`` or ``"regression"``.
        version: A TabPFN version the installed ``tabpfn`` knows, e.g. ``"v2"`` (default),
            ``"v2.5"``, ``"v3"``, or ``"v3.5"``. Downloading the checkpoints of the versions
            after v2 needs a Prior Labs license token (see the TabPFN documentation).
        **params: Settings overriding the version's defaults, e.g. ``n_estimators`` or
            ``model_path`` (a checkpoint of your own).

    Returns:
        The unfitted ``TabPFNClassifier`` or ``TabPFNRegressor``.

    Raises:
        ValueError: If the installed ``tabpfn`` does not know ``version``.
    """
    tabpfn = require("tabpfn", purpose=f"TabPFN {version}")
    try:
        model_version = importlib.import_module("tabpfn.constants").ModelVersion
    except (ImportError, AttributeError):  # tabpfn before 6.0
        model_version = None
    known = [str(member.value) for member in model_version] if model_version is not None else []
    if model_version is None or version not in known:
        installed = _installed_version()
        hint = f"it knows {', '.join(known)}" if known else "choosing a version needs tabpfn>=6.0"
        msg = (
            f"tabpfn {installed} has no TabPFN {version!r}: {hint}. Upgrade tabpfn for newer ones."
        )
        raise ValueError(msg)
    cls = tabpfn.TabPFNClassifier if task == "classification" else tabpfn.TabPFNRegressor
    return cls.create_default_for_version(model_version(version), **params)


def require_inf_passthrough() -> None:
    """Raise unless the installed TabPFN reads ``inf`` as a missing value (``tabpfn>=8.1``).

    Raises:
        ImportError: If ``tabpfn`` is not installed.
        ValueError: If the installed ``tabpfn`` is older than 8.1.
    """
    require("tabpfn", purpose="TabPFN with inf as a missing value")
    installed = _installed_version()
    version = tuple(int(part) for part in re.findall(r"\d+", installed)[:2])
    if version < _PASSTHROUGH_INF_VERSION:
        msg = (
            "TabPFN reads inf as a missing value only from tabpfn 8.1 on (built with "
            f"inference_config={{'PASSTHROUGH_INF': True}}); installed: tabpfn {installed}."
        )
        raise ValueError(msg)


def check_inf_baseline(model: Any, baseline: np.ndarray) -> None:  # noqa: ANN401
    """Reject an ``inf`` baseline for a TabPFN model that does not read it as a missing value.

    Without ``PASSTHROUGH_INF``, tabpfn 8.1 and later reject ``inf``, and older releases transform it
    in their preprocessing, so the game would silently explain something else. Other models and
    finite baselines pass.

    Raises:
        ValueError: If ``baseline`` has an ``inf`` and ``model`` is a TabPFN model that would not
            read it as missing.
    """
    if not np.isinf(baseline).any() or not safe_isinstance(
        model, ["tabpfn.TabPFNClassifier", "tabpfn.TabPFNRegressor"]
    ):
        return
    require_inf_passthrough()
    config = getattr(model, "inference_config", None) or {}
    if isinstance(config, dict):
        passthrough = config.get("PASSTHROUGH_INF")
    else:
        passthrough = getattr(config, "PASSTHROUGH_INF", None)
    if not passthrough:
        msg = "Build the TabPFN model with inference_config={'PASSTHROUGH_INF': True}."
        raise ValueError(msg)


def _installed_version() -> str:
    # through the module attribute, so that tests can fake the installed version
    return importlib.metadata.version("tabpfn")
