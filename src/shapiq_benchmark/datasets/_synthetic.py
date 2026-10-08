"""Seeded generators of synthetic datasets.

All generators are deterministic given ``random_state`` (default ``42``), so a synthetic dataset
is identified by its name and parameters.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from ._registry import DatasetSpec, register_dataset

__all__ = ["CausalSetting", "load_curthvds_synthetic"]

type CausalSetting = Literal["i", "ii"]
"""The setting of :func:`load_curthvds_synthetic`: ``"i"`` without effect heterogeneity, ``"ii"`` with."""

_DEFAULT_SEED = 42


def _with_irrelevant(
    data: dict[str, np.ndarray],
    rng: np.random.Generator,
    n_irrelevant: int,
    draw: Literal["uniform01", "uniform11", "normal", "binary"],
    n_samples: int,
) -> pd.DataFrame:
    for i in range(1, n_irrelevant + 1):
        if draw == "uniform01":
            data[f"irr{i}"] = rng.uniform(0, 1, n_samples)
        elif draw == "uniform11":
            data[f"irr{i}"] = rng.uniform(-1, 1, n_samples)
        elif draw == "normal":
            data[f"irr{i}"] = rng.normal(0, 1, n_samples)
        else:  # binary
            data[f"irr{i}"] = rng.integers(0, 2, n_samples).astype(float)
    return pd.DataFrame(data)


def load_condind(
    n_samples: int = 1000,
    n_irrelevant: int = 3,
    random_state: int | None = _DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Conditional independence: ``y = 1[x1 + x2 > 1]`` with uniform features."""
    rng = np.random.default_rng(random_state)
    x1 = rng.uniform(0, 1, n_samples)
    x2 = rng.uniform(0, 1, n_samples)
    y = (x1 + x2 > 1).astype(int)
    x = _with_irrelevant({"x1": x1, "x2": x2}, rng, n_irrelevant, "uniform01", n_samples)
    return x, pd.Series(y, name="y")


def load_xor(
    n_samples: int = 1000,
    n_irrelevant: int = 2,
    noise: float = 0.05,
    random_state: int | None = _DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """XOR of two binary features with label noise."""
    rng = np.random.default_rng(random_state)
    x1 = rng.integers(0, 2, n_samples)
    x2 = rng.integers(0, 2, n_samples)
    y = (x1 != x2).astype(int)
    if noise > 0:
        flip = rng.random(n_samples) < noise
        y[flip] = 1 - y[flip]
    data = {"x1": x1.astype(float), "x2": x2.astype(float)}
    x = _with_irrelevant(data, rng, n_irrelevant, "binary", n_samples)
    return x, pd.Series(y, name="y")


def load_group(
    n_samples: int = 1000,
    n_irrelevant: int = 2,
    random_state: int | None = _DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Four Gaussian clusters with XOR-like labels."""
    rng = np.random.default_rng(random_state)
    centers = np.array([[-2, -2], [2, 2], [-2, 2], [2, -2]], dtype=float)
    cluster_labels = [0, 1, 0, 1]
    per_cluster = n_samples // 4
    counts = [per_cluster] * 4
    counts[-1] += n_samples - sum(counts)
    points, labels = [], []
    for k, (count, center) in enumerate(zip(counts, centers, strict=True)):
        points.append(rng.normal(loc=center, scale=0.8, size=(count, 2)))
        labels.extend([cluster_labels[k]] * count)
    xy, y = np.vstack(points), np.array(labels)
    perm = rng.permutation(n_samples)
    xy, y = xy[perm], y[perm]
    x = _with_irrelevant({"x1": xy[:, 0], "x2": xy[:, 1]}, rng, n_irrelevant, "normal", n_samples)
    return x, pd.Series(y, name="y")


def load_cross(
    n_samples: int = 1000,
    a: float = 0.3,
    n_irrelevant: int = 2,
    random_state: int | None = _DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Cross: ``y = 1[|x1| < a] XOR 1[|x2| < a]``."""
    rng = np.random.default_rng(random_state)
    x1 = rng.uniform(-1, 1, n_samples)
    x2 = rng.uniform(-1, 1, n_samples)
    y = ((np.abs(x1) < a) ^ (np.abs(x2) < a)).astype(int)
    x = _with_irrelevant({"x1": x1, "x2": x2}, rng, n_irrelevant, "uniform11", n_samples)
    return x, pd.Series(y, name="y")


def load_chess(
    n_samples: int = 1000,
    m: int = 8,
    n_irrelevant: int = 2,
    random_state: int | None = _DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Chessboard pattern on an ``m x m`` grid."""
    rng = np.random.default_rng(random_state)
    x1 = rng.uniform(0, 1, n_samples)
    x2 = rng.uniform(0, 1, n_samples)
    row = (x1 * m).astype(int).clip(0, m - 1)
    col = (x2 * m).astype(int).clip(0, m - 1)
    y = ((row + col) % 2).astype(int)
    x = _with_irrelevant({"x1": x1, "x2": x2}, rng, n_irrelevant, "uniform01", n_samples)
    return x, pd.Series(y, name="y")


def load_sphere(
    n_samples: int = 1000,
    radius: float = 0.7,
    n_irrelevant: int = 2,
    random_state: int | None = _DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Points inside a circle of the given radius."""
    rng = np.random.default_rng(random_state)
    x1 = rng.uniform(-1, 1, n_samples)
    x2 = rng.uniform(-1, 1, n_samples)
    y = (x1**2 + x2**2 <= radius**2).astype(int)
    x = _with_irrelevant({"x1": x1, "x2": x2}, rng, n_irrelevant, "uniform11", n_samples)
    return x, pd.Series(y, name="y")


def load_disjunct(
    n_samples: int = 1000,
    thresholds: tuple[float, ...] = (0.8, 0.8, 0.8),
    n_irrelevant: int = 2,
    random_state: int | None = _DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Disjunction: ``y = 1`` if any relevant feature exceeds its threshold."""
    rng = np.random.default_rng(random_state)
    relevant = {f"x{i + 1}": rng.uniform(0, 1, n_samples) for i in range(len(thresholds))}
    condition = np.zeros(n_samples, dtype=bool)
    for i, threshold in enumerate(thresholds):
        condition |= relevant[f"x{i + 1}"] > threshold
    x = _with_irrelevant(dict(relevant), rng, n_irrelevant, "uniform01", n_samples)
    return x, pd.Series(condition.astype(int), name="y")


def load_random(
    n_samples: int = 1000,
    n_features: int = 5,
    p: float = 0.5,
    random_state: int | None = _DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """Uniform features and labels drawn independently of them (no signal)."""
    rng = np.random.default_rng(random_state)
    data = {f"x{i + 1}": rng.uniform(0, 1, n_samples) for i in range(n_features)}
    return pd.DataFrame(data), pd.Series(rng.binomial(1, p, n_samples), name="y")


def load_independentlinear60(
    n_samples: int = 1000,
    random_state: int | None = _DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """60 independent features, linear target on every third of the first 30 features."""
    rng = np.random.default_rng(random_state)
    n_features = 60
    beta = np.zeros(n_features)
    beta[0:30:3] = 1
    x = rng.standard_normal((n_samples, n_features))
    x -= x.mean(0)
    y = x @ beta + rng.standard_normal(n_samples) * 0.01
    return pd.DataFrame(x), pd.Series(y, name="target")


def load_corrgroups60(
    n_samples: int = 1000,
    random_state: int | None = _DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.Series]:
    """60 features in strongly correlated groups of three, linear target."""
    rng = np.random.default_rng(random_state)
    n_features = 60
    beta = np.zeros(n_features)
    beta[0:30:3] = 1
    correlation = np.eye(n_features)
    for i in range(0, 30, 3):
        correlation[i, i + 1] = correlation[i + 1, i] = 0.99
        correlation[i, i + 2] = correlation[i + 2, i] = 0.99
        correlation[i + 1, i + 2] = correlation[i + 2, i + 1] = 0.99
    x = rng.standard_normal((n_samples, n_features))
    x -= x.mean(0)
    sigma = x.T @ x / x.shape[0]
    whitening = np.linalg.cholesky(np.linalg.inv(sigma)).T
    x = (x @ whitening.T) @ np.linalg.cholesky(correlation).T
    y = x @ beta + rng.standard_normal(n_samples) * 0.01
    return pd.DataFrame(x), pd.Series(y, name="target")


def load_curthvds_synthetic(
    n: int = 500,
    d: int = 4,
    random_state: int = _DEFAULT_SEED,
    setting: CausalSetting = "ii",
) -> pd.DataFrame:
    """Synthetic observational study with known causal roles (Curth and van der Schaar, 2021).

    The covariates are split into instruments, confounders, effect modifiers and outcome-only
    covariates. The returned frame contains the covariates plus the binary ``Treatment`` and the
    ``Outcome`` columns.

    Args:
        n: The number of samples.
        d: The number of covariates, at least ``4``.
        random_state: The seed of the generator.
        setting: ``"i"`` (no treatment effect heterogeneity) or ``"ii"`` (heterogeneous effect).

    Returns:
        The study as a data frame.
    """
    min_covariates = 4
    if d < min_covariates:
        msg = "d must be at least 4."
        raise ValueError(msg)
    if setting not in {"i", "ii"}:
        msg = "setting must be 'i' or 'ii'."
        raise ValueError(msg)
    rng = np.random.default_rng(random_state)
    n_c = n_tau = n_z = max(1, d // 4)
    n_o = max(0, d - n_c - n_tau - n_z)
    x_c = rng.normal(0.0, 1.0, (n, n_c))
    x_tau = rng.normal(0.0, 1.0, (n, n_tau))
    x_z = rng.normal(0.0, 1.0, (n, n_z))
    x_o = rng.normal(0.0, 1.0, (n, n_o)) if n_o > 0 else None
    x_co = np.concatenate([x_c, x_o], axis=1) if x_o is not None else x_c
    mu0 = np.sum(x_co**2, axis=1)
    mu1 = mu0 + (np.sum(x_tau**2, axis=1) if setting == "ii" else 0.0)
    m = np.mean(x_c**2, axis=1)
    propensity = 1.0 / (1.0 + np.exp(-3.0 * (m - float(np.quantile(m, 0.5)))))
    treatment = rng.binomial(1, propensity, n).astype(int)
    outcome = mu0 * (1 - treatment) + mu1 * treatment + rng.normal(0.0, 1.0, n)

    def _columns(values: np.ndarray, base: str) -> dict[str, np.ndarray]:
        if values.shape[1] == 1:
            return {base: values[:, 0]}
        return {f"{base}{j + 1}": values[:, j] for j in range(values.shape[1])}

    data: dict[str, np.ndarray] = {}
    data.update(_columns(x_z, "Instrument"))
    data.update(_columns(x_c, "Confounder"))
    data.update(_columns(x_tau, "EffectModifier"))
    if x_o is not None:
        data.update(_columns(x_o, "OutcomeOnly"))
    data["Treatment"] = treatment
    data["Outcome"] = outcome
    return pd.DataFrame(data)


_SYNTHETIC = [
    ("chess", "classification", load_chess),
    ("condind", "classification", load_condind),
    ("corrgroups60", "regression", load_corrgroups60),
    ("cross", "classification", load_cross),
    ("disjunct", "classification", load_disjunct),
    ("group", "classification", load_group),
    ("independentlinear60", "regression", load_independentlinear60),
    ("random", "classification", load_random),
    ("sphere", "classification", load_sphere),
    ("xor", "classification", load_xor),
]

for _name, _task, _loader in _SYNTHETIC:
    register_dataset(
        DatasetSpec(
            name=_name,
            task=_task,  # type: ignore[arg-type]
            loader=_loader,
            source="seeded generator",
            kind="synthetic",
        )
    )
