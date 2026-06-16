"""Benchmark games for the confounding XAI coalition game."""

from __future__ import annotations

from .base import GlobalConfoundingXAI, _fit_s_learner
from shapiq_games.datasets import load_curthvds_synthetic


class CurthVDS(GlobalConfoundingXAI):
    """Curth-VDS synthetic confounding attribution benchmark game.

    Loads a synthetic observational dataset from Curth and van der Schaar
    (2021) :cite:t:`curth2021` and estimates tau_hat via a T-Learner (TabPFN)
    on the full data before initializing the global game.
    """

    def __init__(
        self,
        *,
        n: int = 500,
        d: int = 4,
        seed: int = 42,
        setting: str = "ii",
        mode: str = "signed",
        device: str = "cpu",
        n_estimators: int = 1,
    ) -> None:
        """Initialize the CurthVDS benchmark game.

        Args:
            n: Number of observations. Defaults to ``500``.
            d: Number of features. Defaults to ``4``.
            seed: Random seed for data generation. Defaults to ``42``.
            setting: DGP setting. ``'ii'`` has heterogeneous treatment effects;
                ``'i'`` has no treatment effect. Defaults to ``'ii'``.
            mode: Aggregation mode. One of ``'signed'``, ``'abs'``, ``'sq'``.
            device: Device for TabPFN. Defaults to ``'cpu'``.
            n_estimators: Number of TabPFN estimators. Defaults to ``1``.
        """
        df = load_curthvds_synthetic(n=n, d=d, seed=seed, setting=setting)
        feature_cols = [c for c in df.columns if c not in {"Treatment", "Outcome"}]
        X = df[feature_cols].to_numpy(dtype=float)
        A = df["Treatment"].to_numpy(dtype=float)
        Y = df["Outcome"].to_numpy(dtype=float)

        predict_m1, predict_m0 = _fit_s_learner(X, A, Y, device=device, n_estimators=n_estimators)
        tau_hat = predict_m1(X) - predict_m0(X)

        super().__init__(
            X=X,
            A=A,
            Y=Y,
            tau_hat=tau_hat,
            mode=mode,
            device=device,
            n_estimators=n_estimators,
        )
