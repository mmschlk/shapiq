r"""Causal games: which covariates explain the confounding bias of a treatment effect estimate.

Implements the ConfoundingSHAP value functions (Brockschmidt et al., 2026,
https://arxiv.org/abs/2605.10533). For a coalition :math:`S`, an S-learner fitted on
:math:`(X_S, A) \to Y` gives the treatment effect estimate when adjusting only for :math:`S`; its
deviation from the reference effect estimate :math:`\hat\tau` (adjusting for all covariates) is
the confounding bias left by omitting the other covariates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from shapiq.game import Game
from shapiq_games._base import as_bool_coalitions
from shapiq_games._tabpfn import DEFAULT_TABPFN_VERSION, build_tabpfn

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.typing import CoalitionMatrix, FloatVector, GameValues
    from shapiq_games.typing import ConfoundingMode, PredictFunction

__all__ = ["GlobalConfoundingXAI", "LocalConfoundingXAI", "tabpfn_regressor"]

type RegressorFactory = Callable[[], Any]


def tabpfn_regressor(
    device: str = "cpu",
    n_estimators: int = 1,
    random_state: int = 42,
    *,
    version: str = DEFAULT_TABPFN_VERSION,
    **params: Any,
) -> Any:  # noqa: ANN401
    """Return a TabPFN regressor configured as in the ConfoundingSHAP paper (requires ``tabpfn``).

    Args:
        device: The torch device. Defaults to ``"cpu"``.
        n_estimators: The number of ensemble members. Defaults to ``1``.
        random_state: The seed. Defaults to ``42``.
        version: The TabPFN version, e.g. ``"v2"`` (default, downloads without a license),
            ``"v2.5"`` (the paper's), ``"v3"``, or ``"v3.5"``. The versions after v2 need a
            Prior Labs license token to download (see the TabPFN documentation).
        **params: Further TabPFN settings, e.g. ``model_path`` for a checkpoint of your own.

    Returns:
        The unfitted regressor.

    Raises:
        ValueError: If the installed ``tabpfn`` does not know ``version``.

    Examples:
        >>> from functools import partial
        >>> regressor = partial(tabpfn_regressor, version="v3")  # doctest: +SKIP
        >>> GlobalConfoundingXAI(X, treatment, outcome, regressor=regressor)  # doctest: +SKIP
    """
    config = {"REGRESSION_Y_PREPROCESS_TRANSFORMS": (None,), **params.pop("inference_config", {})}
    return build_tabpfn(
        "regression",
        version,
        device=device,
        n_estimators=n_estimators,
        random_state=random_state,
        inference_config=config,
        **params,
    )


def _predictor(model: Any) -> PredictFunction:  # noqa: ANN401
    return lambda x: np.asarray(model.predict(x), dtype=float).reshape(-1)


def _fit_s_learner(
    x_s: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    regressor: RegressorFactory,
) -> tuple[PredictFunction, PredictFunction]:
    """Fit an S-learner on ``(X_S, A) -> Y`` and return the predictors for ``A = 1`` and ``A = 0``."""
    model = regressor()
    model.fit(np.column_stack([x_s, treatment]), outcome)
    predict = _predictor(model)

    def predict_treated(x: np.ndarray) -> FloatVector:
        return predict(np.column_stack([x, np.ones(x.shape[0])]))

    def predict_control(x: np.ndarray) -> FloatVector:
        return predict(np.column_stack([x, np.zeros(x.shape[0])]))

    return predict_treated, predict_control


def _fit_projection(
    x_s: np.ndarray, tau_hat: np.ndarray, regressor: RegressorFactory
) -> PredictFunction:
    """Project the reference effect onto the coalition's covariates (a constant without any)."""
    if x_s.shape[1] == 0:
        mean = float(np.mean(tau_hat))
        return lambda x: np.full(x.shape[0], mean)
    model = regressor()
    model.fit(x_s, tau_hat)
    return _predictor(model)


def _aggregate(bias: np.ndarray, mode: ConfoundingMode) -> float:
    if mode == "signed":
        return float(np.mean(-bias))
    if mode == "abs":
        return float(np.mean(np.abs(bias)))
    if mode == "sq":
        return float(np.mean(bias**2))
    msg = f"mode must be 'signed', 'abs', or 'sq', got {mode!r}."
    raise ValueError(msg)


class _ConfoundingGame(Game):
    """Shared state of the confounding games: data, reference effect, regressor, and a cache."""

    def __init__(
        self,
        x: np.ndarray,
        treatment: np.ndarray,
        outcome: np.ndarray,
        tau_hat: np.ndarray | None,
        *,
        mode: ConfoundingMode,
        regressor: RegressorFactory | None,
    ) -> None:
        if mode not in ("signed", "abs", "sq"):
            msg = f"mode must be 'signed', 'abs', or 'sq', got {mode!r}."
            raise ValueError(msg)
        self.X = np.asarray(x, dtype=float)
        self.A = np.asarray(treatment, dtype=float).reshape(-1)
        self.Y = np.asarray(outcome, dtype=float).reshape(-1)
        self.mode: ConfoundingMode = mode
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

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
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
        mode: ConfoundingMode = "signed",
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
            regressor: A function returning a fresh, seeded regressor. Defaults to
                :func:`tabpfn_regressor`: TabPFN v2 configured as in the paper (requires
                ``tabpfn``); ``partial(tabpfn_regressor, version="v2.5")`` is the paper's model.
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
        mode: ConfoundingMode = "signed",
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
            regressor: A function returning a fresh, seeded regressor. Defaults to
                :func:`tabpfn_regressor` (TabPFN v2 configured as in the paper).
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
