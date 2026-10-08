"""TabPFN models by version (requires ``tabpfn>=6.0``)."""

from __future__ import annotations

import importlib
import importlib.metadata
from typing import Any, Literal

from shapiq_games._optional import require

__all__ = ["DEFAULT_TABPFN_VERSION", "build_tabpfn"]

DEFAULT_TABPFN_VERSION = "v2"
"""The TabPFN version used when none is chosen: v2, whose checkpoints download without a license."""


def build_tabpfn(
    task: Literal["classification", "regression"],
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
    if version not in known:
        installed = importlib.metadata.version("tabpfn")
        hint = f"it knows {', '.join(known)}" if known else "choosing a version needs tabpfn>=6.0"
        msg = (
            f"tabpfn {installed} has no TabPFN {version!r}: {hint}. Upgrade tabpfn for newer ones."
        )
        raise ValueError(msg)
    cls = tabpfn.TabPFNClassifier if task == "classification" else tabpfn.TabPFNRegressor
    return cls.create_default_for_version(model_version(version), **params)
