"""Setups of the causal games on the synthetic study of Curth and van der Schaar (2021)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from shapiq_benchmark.datasets import load_curthvds_synthetic
from shapiq_benchmark.models import build_model
from shapiq_games import GlobalConfoundingXAI, LocalConfoundingXAI
from shapiq_games.causal import tabpfn_regressor

from ._base import Setup

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np

    from shapiq_games.causal import Mode

__all__ = ["GlobalConfoundingSetup", "LocalConfoundingSetup"]


@dataclass(frozen=True, kw_only=True)
class _ConfoundingSetup(Setup):
    """Shared fields: the synthetic study, the aggregation mode, and the regressor."""

    n: int = 500
    d: int = 4
    setting: Literal["i", "ii"] = "ii"
    mode: Mode = "signed"
    regressor: str = "tabpfn"
    random_state: int = 42

    def _data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, Callable[[], Any]]:
        """Return covariates, treatment, outcome, and a factory of seeded regressors."""
        frame = load_curthvds_synthetic(
            n=self.n, d=self.d, random_state=self.random_state, setting=self.setting
        )
        covariates = [column for column in frame.columns if column not in {"Treatment", "Outcome"}]
        name, random_state = self.regressor, self.random_state

        def regressor() -> Any:  # noqa: ANN401
            if name == "tabpfn":  # configured as in the ConfoundingSHAP paper
                return tabpfn_regressor(random_state=random_state)
            return build_model(name, "regression", random_state=random_state)

        return (
            frame[covariates].to_numpy(dtype=float),
            frame["Treatment"].to_numpy(dtype=float),
            frame["Outcome"].to_numpy(dtype=float),
            regressor,
        )


@dataclass(frozen=True, kw_only=True)
class GlobalConfoundingSetup(_ConfoundingSetup, name="global_confounding"):
    """A :class:`~shapiq_games.GlobalConfoundingXAI` game on the synthetic study.

    The reference effects come from an S-learner on all covariates.

    Attributes:
        n: The number of samples. Defaults to ``500``.
        d: The number of covariates (players), at least ``4``. Defaults to ``4``.
        setting: ``"i"`` (homogeneous) or ``"ii"`` (heterogeneous effect, default).
        mode: ``"signed"`` (default), ``"abs"``, or ``"sq"``.
        regressor: ``"tabpfn"`` (default, configured as in the ConfoundingSHAP paper) or a model
            name of :mod:`shapiq_benchmark.models` (e.g. ``"linear"``).
        random_state: The seed of the data and the regressors.

    Examples:
        >>> GlobalConfoundingSetup(n=200, regressor="linear").build().n_players
        4
    """

    def build(self) -> GlobalConfoundingXAI:
        """Generate the data and build the game."""
        x, treatment, outcome, regressor = self._data()
        return GlobalConfoundingXAI(x, treatment, outcome, mode=self.mode, regressor=regressor)


@dataclass(frozen=True, kw_only=True)
class LocalConfoundingSetup(_ConfoundingSetup, name="local_confounding"):
    """A :class:`~shapiq_games.LocalConfoundingXAI` game for one unit of the synthetic study.

    Attributes:
        unit: The index of the explained unit. Defaults to ``0``.
        n: The number of samples. Defaults to ``500``.
        d: The number of covariates (players), at least ``4``. Defaults to ``4``.
        setting: ``"i"`` (homogeneous) or ``"ii"`` (heterogeneous effect, default).
        mode: ``"signed"`` (default), ``"abs"``, or ``"sq"``.
        regressor: ``"tabpfn"`` (default) or a model name of :mod:`shapiq_benchmark.models`.
        random_state: The seed of the data and the regressors.

    Examples:
        >>> LocalConfoundingSetup(n=200, unit=2, regressor="linear").build().n_players
        4
    """

    unit: int = 0

    def build(self) -> LocalConfoundingXAI:
        """Generate the data and build the game."""
        x, treatment, outcome, regressor = self._data()
        return LocalConfoundingXAI(
            x, treatment, outcome, unit=self.unit, mode=self.mode, regressor=regressor
        )
