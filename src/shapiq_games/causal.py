r"""Causal games: which covariates explain the confounding bias of a treatment effect estimate.

Implements the ConfoundingSHAP value functions (Brockschmidt et al., 2026,
https://arxiv.org/abs/2605.10533). For a coalition :math:`S`, an S-learner fitted on
:math:`(X_S, A) \to Y` gives the treatment effect estimate when adjusting only for :math:`S`; its
deviation from the reference effect estimate :math:`\hat\tau` (adjusting for all covariates) is
the confounding bias left by omitting the other covariates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np

from shapiq.game import Game
from shapiq_games._base import ConfigMixin, as_bool_coalitions
from shapiq_games.datasets import load_curthvds_synthetic

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["GlobalConfoundingXAI", "LocalConfoundingXAI", "tabpfn_regressor"]

type Mode = Literal["signed", "abs", "sq"]
type RegressorFactory = Callable[[], Any]


def tabpfn_regressor(device: str = "cpu", n_estimators: int = 1, random_state: int = 42) -> Any:  # noqa: ANN401
    """Return a TabPFN regressor configured as in the ConfoundingSHAP paper (requires ``tabpfn``)."""
    from shapiq_games._optional import require

    tabpfn = require("tabpfn", purpose="the default regressor of the causal games")
    return tabpfn.TabPFNRegressor(
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
        inference_config={"REGRESSION_Y_PREPROCESS_TRANSFORMS": (None,)},
    )


def _predictor(model: Any) -> Callable[[np.ndarray], np.ndarray]:  # noqa: ANN401
    return lambda x: np.asarray(model.predict(x), dtype=float).reshape(-1)


def _fit_s_learner(
    x_s: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    regressor: RegressorFactory,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Fit an S-learner on ``(X_S, A) -> Y`` and return the predictors for ``A = 1`` and ``A = 0``."""
    model = regressor()
    model.fit(np.column_stack([x_s, treatment]), outcome)
    predict = _predictor(model)

    def predict_treated(x: np.ndarray) -> np.ndarray:
        return predict(np.column_stack([x, np.ones(x.shape[0])]))

    def predict_control(x: np.ndarray) -> np.ndarray:
        return predict(np.column_stack([x, np.zeros(x.shape[0])]))

    return predict_treated, predict_control


def _fit_projection(
    x_s: np.ndarray, tau_hat: np.ndarray, regressor: RegressorFactory
) -> Callable[[np.ndarray], np.ndarray]:
    """Project the reference effect onto the coalition's covariates (a constant without any)."""
    if x_s.shape[1] == 0:
        mean = float(np.mean(tau_hat))
        return lambda x: np.full(x.shape[0], mean)
    model = regressor()
    model.fit(x_s, tau_hat)
    return _predictor(model)


def _aggregate(bias: np.ndarray, mode: Mode) -> float:
    if mode == "signed":
        return float(np.mean(-bias))
    if mode == "abs":
        return float(np.mean(np.abs(bias)))
    if mode == "sq":
        return float(np.mean(bias**2))
    msg = f"mode must be 'signed', 'abs', or 'sq', got {mode!r}."
    raise ValueError(msg)


class _ConfoundingGame(ConfigMixin, Game):
    """Shared state of the confounding games: data, reference effect, regressor, and a cache."""

    def __init__(
        self,
        x: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        tau_hat: np.ndarray | None,
        *,
        mode: Mode,
        regressor: RegressorFactory | None,
    ) -> None:
        if mode not in ("signed", "abs", "sq"):
            msg = f"mode must be 'signed', 'abs', or 'sq', got {mode!r}."
            raise ValueError(msg)
        self.X = np.asarray(x, dtype=float)
        self.A = np.asarray(treatment, dtype=float).reshape(-1)
        self.Y = np.asarray(outcome, dtype=float).reshape(-1)
        self.mode: Mode = mode
        self.regressor: RegressorFactory = regressor if regressor is not None else tabpfn_regressor
        if tau_hat is None:  # the reference: an S-learner on all covariates
            treated, control = _fit_s_learner(self.X, self.A, self.Y, self.regressor)
            tau_hat = treated(self.X) - control(self.X)
        self.tau_hat = np.asarray(tau_hat, dtype=float).reshape(-1)
        naive_effect = float(np.mean(self.Y[self.A == 1]) - np.mean(self.Y[self.A == 0]))
        self.empty_value = _aggregate(np.array([naive_effect - float(np.mean(self.tau_hat))]), mode)
        self._cache: dict[tuple[int, ...], float] = {(): self.empty_value}
        super().__init__(self.X.shape[1], normalize=False, normalization_value=self.empty_value)

    def _coalition_value(self, players: tuple[int, ...]) -> float:
        raise NotImplementedError

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Return the (cached) confounding value of each coalition."""
        coalitions = as_bool_coalitions(coalitions)
        values = np.empty(coalitions.shape[0])
        for i, coalition in enumerate(coalitions):
            key = tuple(int(j) for j in np.flatnonzero(coalition))
            if key not in self._cache:
                self._cache[key] = self._coalition_value(key)
            values[i] = self._cache[key]
        return values


class GlobalConfoundingXAI(_ConfoundingGame):
    r"""The global confounding game: the bias of the average treatment effect adjusting for S.

    With ``mode="signed"``, the value is :math:`\bar\tau - \overline{\hat\mu_1(X_S) - \hat\mu_0(X_S)}`;
    with ``"abs"`` or ``"sq"`` it is the mean absolute or squared conditional bias against the
    projection of :math:`\hat\tau` onto :math:`X_S`. The empty coalition is the naive difference
    in means. Every new coalition fits two to four regressors, so values are cached.

    Examples:
        >>> from sklearn.linear_model import LinearRegression
        >>> rng = np.random.default_rng(0)
        >>> X = rng.normal(size=(200, 4))
        >>> treatment = (rng.random(200) < 0.5).astype(float)
        >>> outcome = X[:, 0] + treatment * (1.0 + X[:, 1])
        >>> game = GlobalConfoundingXAI(X, treatment, outcome, regressor=LinearRegression)
        >>> game.n_players
        4
    """

    def __init__(
        self,
        x: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        tau_hat: np.ndarray | None = None,
        *,
        mode: Mode = "signed",
        regressor: RegressorFactory | None = None,
    ) -> None:
        """Initialize the global confounding game.

        Args:
            x: The covariates of shape ``(n_samples, n_features)`` (the players).
            treatment: The binary treatment of shape ``(n_samples,)``.
            outcome: The outcome of shape ``(n_samples,)``.
            tau_hat: The reference conditional treatment effects of shape ``(n_samples,)``.
                If ``None``, an S-learner with ``regressor`` on all covariates provides them.
            mode: ``"signed"``, ``"abs"``, or ``"sq"``. Defaults to ``"signed"``.
            regressor: A function returning a fresh, seeded regressor. Defaults to TabPFN as in
                the paper (requires ``tabpfn``).
        """
        super().__init__(x, treatment, outcome, tau_hat, mode=mode, regressor=regressor)

    def _coalition_value(self, players: tuple[int, ...]) -> float:
        x_s = self.X[:, list(players)]
        treated, control = _fit_s_learner(x_s, self.A, self.Y, self.regressor)
        effect = treated(x_s) - control(x_s)
        if self.mode == "signed":
            return float(np.mean(self.tau_hat) - np.mean(effect))
        projection = _fit_projection(x_s, self.tau_hat, self.regressor)
        return _aggregate(effect - projection(x_s), self.mode)

    @classmethod
    def from_config(
        cls,
        *,
        n: int = 500,
        d: int = 4,
        setting: Literal["i", "ii"] = "ii",
        mode: Mode = "signed",
        random_state: int = 42,
        regressor: RegressorFactory | None = None,
    ) -> Self:
        r"""Build the game on the synthetic study of Curth and van der Schaar (2021).

        The reference effects :math:`\hat\tau` come from an S-learner on all covariates.

        Args:
            n: The number of samples. Defaults to ``500``.
            d: The number of covariates (players), at least ``4``. Defaults to ``4``.
            setting: ``"i"`` (homogeneous) or ``"ii"`` (heterogeneous effect). Defaults to ``"ii"``.
            mode: ``"signed"``, ``"abs"``, or ``"sq"``.
            random_state: The seed of the data and the default regressor.
            regressor: A function returning a fresh regressor; defaults to a seeded TabPFN.

        Returns:
            The configured game.
        """
        custom_regressor = regressor is not None
        x, treatment, outcome, regressor = _curthvds_setup(n, d, setting, random_state, regressor)
        game = cls(x, treatment, outcome, mode=mode, regressor=regressor)
        if custom_regressor:  # a custom regressor cannot be fingerprinted
            return game
        return game._set_config(
            dataset="curthvds_synthetic",
            n=n,
            d=d,
            setting=setting,
            mode=mode,
            random_state=random_state,
        )


class LocalConfoundingXAI(_ConfoundingGame):
    """The local confounding game: the bias of the conditional effect at one unit adjusting for S.

    Examples:
        >>> from sklearn.linear_model import LinearRegression
        >>> rng = np.random.default_rng(0)
        >>> X = rng.normal(size=(200, 4))
        >>> treatment = (rng.random(200) < 0.5).astype(float)
        >>> outcome = X[:, 0] + treatment * (1.0 + X[:, 1])
        >>> game = LocalConfoundingXAI(X, treatment, outcome, unit=3, regressor=LinearRegression)
        >>> game.n_players
        4
    """

    def __init__(
        self,
        x: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        tau_hat: np.ndarray | None = None,
        unit: int | np.ndarray = 0,
        *,
        mode: Mode = "signed",
        regressor: RegressorFactory | None = None,
    ) -> None:
        """Initialize the local confounding game.

        Args:
            x: The covariates of shape ``(n_samples, n_features)`` (the players).
            treatment: The binary treatment of shape ``(n_samples,)``.
            outcome: The outcome of shape ``(n_samples,)``.
            tau_hat: The reference conditional treatment effects of shape ``(n_samples,)``.
                If ``None``, an S-learner with ``regressor`` on all covariates provides them.
            unit: The explained unit as an index into ``x`` or its covariates. Defaults to ``0``.
            mode: ``"signed"``, ``"abs"``, or ``"sq"``. Defaults to ``"signed"``.
            regressor: A function returning a fresh, seeded regressor. Defaults to TabPFN.
        """
        covariates = np.asarray(x, dtype=float)
        if isinstance(unit, int | np.integer):
            self.unit = covariates[int(unit)].copy()
        else:
            self.unit = np.asarray(unit, dtype=float).reshape(-1)
        super().__init__(x, treatment, outcome, tau_hat, mode=mode, regressor=regressor)

    def _coalition_value(self, players: tuple[int, ...]) -> float:
        x_s = self.X[:, list(players)]
        unit_s = self.unit[list(players)].reshape(1, -1)
        treated, control = _fit_s_learner(x_s, self.A, self.Y, self.regressor)
        projection = _fit_projection(x_s, self.tau_hat, self.regressor)
        bias = treated(unit_s)[0] - control(unit_s)[0] - projection(unit_s)[0]
        return _aggregate(np.array([bias]), self.mode)

    @classmethod
    def from_config(
        cls,
        *,
        unit: int = 0,
        n: int = 500,
        d: int = 4,
        setting: Literal["i", "ii"] = "ii",
        mode: Mode = "signed",
        random_state: int = 42,
        regressor: RegressorFactory | None = None,
    ) -> Self:
        """Build the game for one unit of the synthetic study of Curth and van der Schaar (2021).

        Args:
            unit: The index of the explained unit. Defaults to ``0``.
            n: The number of samples. Defaults to ``500``.
            d: The number of covariates (players), at least ``4``. Defaults to ``4``.
            setting: ``"i"`` (homogeneous) or ``"ii"`` (heterogeneous effect). Defaults to ``"ii"``.
            mode: ``"signed"``, ``"abs"``, or ``"sq"``.
            random_state: The seed of the data and the default regressor.
            regressor: A function returning a fresh regressor; defaults to a seeded TabPFN.

        Returns:
            The configured game.
        """
        custom_regressor = regressor is not None
        x, treatment, outcome, regressor = _curthvds_setup(n, d, setting, random_state, regressor)
        game = cls(x, treatment, outcome, unit=unit, mode=mode, regressor=regressor)
        if custom_regressor:  # a custom regressor cannot be fingerprinted
            return game
        return game._set_config(
            dataset="curthvds_synthetic",
            unit=unit,
            n=n,
            d=d,
            setting=setting,
            mode=mode,
            random_state=random_state,
        )


def _curthvds_setup(
    n: int,
    d: int,
    setting: str,
    random_state: int,
    regressor: RegressorFactory | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, RegressorFactory]:
    frame = load_curthvds_synthetic(n=n, d=d, random_state=random_state, setting=setting)
    covariates = [column for column in frame.columns if column not in {"Treatment", "Outcome"}]
    if regressor is None:

        def regressor() -> Any:  # noqa: ANN401
            return tabpfn_regressor(random_state=random_state)

    return (
        frame[covariates].to_numpy(dtype=float),
        frame["Treatment"].to_numpy(dtype=float),
        frame["Outcome"].to_numpy(dtype=float),
        regressor,
    )
