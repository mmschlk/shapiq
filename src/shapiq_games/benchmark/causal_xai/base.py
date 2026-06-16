"""Confounding XAI games for confounding-bias attribution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from shapiq.game import Game

if TYPE_CHECKING:
    from collections.abc import Callable

    from tabpfn import TabPFNRegressor

_TABPFN_INFERENCE_CONFIG = {"REGRESSION_Y_PREPROCESS_TRANSFORMS": (None,)}


def _make_tabpfn(device: str, n_estimators: int = 1) -> TabPFNRegressor:
    from tabpfn import TabPFNRegressor

    return TabPFNRegressor(
        device=device,
        n_estimators=n_estimators,
        n_jobs=1,
        inference_config=_TABPFN_INFERENCE_CONFIG,
    )


def _constant_predictor(y: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    mean = float(np.mean(y))
    return lambda X: np.full(X.shape[0], mean, dtype=float)


def _fit_s_learner(
    X_S: np.ndarray,
    A: np.ndarray,
    Y: np.ndarray,
    *,
    device: str,
    n_estimators: int,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Fit m_S(x_S, a) = E[Y | X_S, A=a]; return (predict_m1, predict_m0)."""
    XA = np.concatenate([X_S, A.reshape(-1, 1)], axis=1)
    model = _make_tabpfn(device, n_estimators)
    model.fit(XA, Y)

    def predict_m1(X_q: np.ndarray) -> np.ndarray:
        return np.asarray(
            model.predict(np.concatenate([X_q, np.ones((len(X_q), 1))], axis=1)),
            dtype=float,
        ).reshape(-1)

    def predict_m0(X_q: np.ndarray) -> np.ndarray:
        return np.asarray(
            model.predict(np.concatenate([X_q, np.zeros((len(X_q), 1))], axis=1)),
            dtype=float,
        ).reshape(-1)

    return predict_m1, predict_m0


def _fit_tau_projection(
    X_S: np.ndarray,
    tau_hat: np.ndarray,
    *,
    device: str,
    n_estimators: int,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit g_S(x_S) = E[tau_hat | X_S]; return a predictor."""
    if X_S.shape[1] == 0:
        return _constant_predictor(tau_hat)
    model = _make_tabpfn(device, n_estimators)
    model.fit(X_S, tau_hat)
    return lambda X_q: np.asarray(model.predict(X_q), dtype=float).reshape(-1)


def _aggregate(bias: np.ndarray, mode: str) -> float:
    """Map per-sample bias to a coalition value."""
    if mode == "signed":
        return float(np.mean(-bias))
    if mode == "abs":
        return float(np.mean(np.abs(bias)))
    if mode == "sq":
        return float(np.mean(bias**2))
    msg = f"mode must be 'signed', 'abs', or 'sq'; got {mode!r}."
    raise ValueError(msg)


def _empty_coalition_value(Y: np.ndarray, A: np.ndarray, tau_hat: np.ndarray, mode: str) -> float:
    """Coalition value for S={}: constant predictions for all observations."""
    empty_bias = (float(np.mean(Y[A == 1])) - float(np.mean(Y[A == 0]))) - float(np.mean(tau_hat))
    return _aggregate(np.array([empty_bias]), mode)


class GlobalConfoundingXAI(Game):
    """Global confounding attribution game according to Brockschmidt et al. (2026) :cite:t:`brockschmidt2026`.

    The coalition value v(S) measures the confounding bias attributable to
    feature subset S, averaged over all observations:

        v(S) = mean_i[ f(b_S(x_i)) ]

    where b_S(x_i) = (m_S(x_i, 1) - m_S(x_i, 0)) - g_S(x_i).

    For ``mode='signed'`` the per-sample mean factors out by the law of iterated
    expectations (E[g_S(X_S)] = E[tau_hat]), so no tau projection is fitted:
    v(S) = mean(tau_hat) - mean(m_S(1) - m_S(0)).
    """

    def __init__(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        tau_hat: np.ndarray,
        *,
        mode: Literal["signed", "abs", "sq"] = "signed",
        device: str = "cpu",
        n_estimators: int = 1,
    ) -> None:
        """Initialize the GlobalConfoundingXAI game.

        Args:
            X: Covariate matrix of shape ``(n_samples, n_features)``.
            A: Binary treatment vector of shape ``(n_samples,)``.
            Y: Outcome vector of shape ``(n_samples,)``.
            tau_hat: Pre-estimated CATE vector of shape ``(n_samples,)``.
            mode: Aggregation mode for the bias. One of ``'signed'``, ``'abs'``, ``'sq'``.
                Defaults to ``'signed'``.
            device: Device for TabPFN. Defaults to ``'cpu'``.
            n_estimators: Number of TabPFN estimators. Defaults to ``1``.
        """
        self.X = np.asarray(X, dtype=float)
        self.A = np.asarray(A).reshape(-1)
        self.Y = np.asarray(Y).reshape(-1)
        self.tau_hat = np.asarray(tau_hat).reshape(-1)
        self.mode = mode
        self.device = device
        self.n_estimators = n_estimators
        self._cache: dict[tuple[int, ...], float] = {}

        self.empty_value = _empty_coalition_value(self.Y, self.A, self.tau_hat, mode)
        self._cache[()] = self.empty_value

        super().__init__(
            n_players=self.X.shape[1],
            normalize=False,
            normalization_value=self.empty_value,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Compute global coalition values.

        Args:
            coalitions: Binary matrix of shape ``(n_coalitions, n_features)``.

        Returns:
            Coalition values of shape ``(n_coalitions,)``.
        """
        worth = np.empty(len(coalitions), dtype=float)
        for i, coalition in enumerate(coalitions):
            key = tuple(int(j) for j in np.flatnonzero(coalition))
            if key in self._cache:
                worth[i] = self._cache[key]
                continue

            X_S = self.X[:, list(key)]
            predict_m1, predict_m0 = _fit_s_learner(
                X_S, self.A, self.Y, device=self.device, n_estimators=self.n_estimators
            )

            if self.mode == "signed":
                # E[g_S(X_S)] = E[tau_hat] by iterated expectations; skip tau projection.
                value = float(np.mean(self.tau_hat) - np.mean(predict_m1(X_S) - predict_m0(X_S)))
            else:
                predict_tau_S = _fit_tau_projection(
                    X_S, self.tau_hat, device=self.device, n_estimators=self.n_estimators
                )
                bias = (predict_m1(X_S) - predict_m0(X_S)) - predict_tau_S(X_S)
                value = _aggregate(bias, self.mode)

            self._cache[key] = value
            worth[i] = value
        return worth


class LocalConfoundingXAI(Game):
    """Local confounding attribution game for a single observation according to Brockschmidt et al. (2026) :cite:t:`brockschmidt2026`.

    The coalition value v_i(S) measures the confounding bias attributable to
    feature subset S for observation x_i:

        v_i(S) = f(b_S(x_i))

    where b_S(x_i) = (m_S(x_i, 1) - m_S(x_i, 0)) - g_S(x_i).

    Models are fitted on the full dataset but evaluated at x_i only.
    """

    def __init__(
        self,
        X: np.ndarray,
        A: np.ndarray,
        Y: np.ndarray,
        tau_hat: np.ndarray,
        x_i: np.ndarray | int,
        *,
        mode: Literal["signed", "abs", "sq"] = "signed",
        device: str = "cpu",
        n_estimators: int = 1,
    ) -> None:
        """Initialize the LocalConfoundingXAI game.

        Args:
            X: Covariate matrix of shape ``(n_samples, n_features)``.
            A: Binary treatment vector of shape ``(n_samples,)``.
            Y: Outcome vector of shape ``(n_samples,)``.
            tau_hat: Pre-estimated CATE vector of shape ``(n_samples,)``.
            x_i: The observation to explain. Integer index into X, or a 1-d array of
                shape ``(n_features,)``.
            mode: Aggregation mode. One of ``'signed'``, ``'abs'``, ``'sq'``.
            device: Device for TabPFN.
            n_estimators: Number of TabPFN estimators.
        """
        self.X = np.asarray(X, dtype=float)
        self.A = np.asarray(A).reshape(-1)
        self.Y = np.asarray(Y).reshape(-1)
        self.tau_hat = np.asarray(tau_hat).reshape(-1)
        self.mode = mode
        self.device = device
        self.n_estimators = n_estimators
        self._cache: dict[tuple[int, ...], float] = {}

        if isinstance(x_i, int | np.integer):
            self.x_i = self.X[int(x_i)].copy()
        else:
            self.x_i = np.asarray(x_i, dtype=float).reshape(-1)

        self.empty_value = _empty_coalition_value(self.Y, self.A, self.tau_hat, mode)
        self._cache[()] = self.empty_value

        super().__init__(
            n_players=self.X.shape[1],
            normalize=False,
            normalization_value=self.empty_value,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Compute local coalition values for observation x_i.

        Args:
            coalitions: Binary matrix of shape ``(n_coalitions, n_features)``.

        Returns:
            Coalition values of shape ``(n_coalitions,)``.
        """
        worth = np.empty(len(coalitions), dtype=float)
        for i, coalition in enumerate(coalitions):
            key = tuple(int(j) for j in np.flatnonzero(coalition))
            if key in self._cache:
                worth[i] = self._cache[key]
                continue

            X_S = self.X[:, list(key)]
            x_i_S = self.x_i[list(key)].reshape(1, -1)

            predict_m1, predict_m0 = _fit_s_learner(
                X_S, self.A, self.Y, device=self.device, n_estimators=self.n_estimators
            )
            predict_tau_S = _fit_tau_projection(
                X_S, self.tau_hat, device=self.device, n_estimators=self.n_estimators
            )

            bias_i = float((predict_m1(x_i_S)[0] - predict_m0(x_i_S)[0]) - predict_tau_S(x_i_S)[0])
            value = _aggregate(np.array([bias_i]), self.mode)

            self._cache[key] = value
            worth[i] = value
        return worth
