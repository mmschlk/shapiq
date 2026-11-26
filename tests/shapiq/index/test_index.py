"""The Shapley kernel reproduces the Shapley value, and only the value exposes it."""

from dataclasses import FrozenInstanceError, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ndlike.jax_functions import jax_broadcast_to

from shapiq.game.discretederivative import subset_patterns
from shapiq.game.exact import ExactComputer
from shapiq.game.table import TableGame
from shapiq.index import (
    AlphaBetaIndex,
    BanzhafIndex,
    LeastSquaresIndex,
    ShapleyInteractionIndex,
    ShapleyValueIndex,
    WeightedBanzhafIndex,
)
from shapiq.players import FinitePlayers

N_PLAYERS = 5
TABLE = jax.random.normal(jax.random.key(0), (2**N_PLAYERS,))


def _least_squares_fit(index: LeastSquaresIndex, table: np.ndarray) -> np.ndarray:
    """Fit v(S) ≈ b + sum of S's player values under the kernel; inf entries are constraints."""
    patterns = subset_patterns(N_PLAYERS)
    kernel = np.asarray(index.least_squares_weight(patterns.sum(-1), N_PLAYERS), dtype=float)
    coalitions = np.asarray(patterns)
    design = np.hstack([np.ones((len(coalitions), 1)), coalitions])
    fitted, exact = np.isfinite(kernel), np.isinf(kernel)
    weighted = design[fitted].T * kernel[fitted]
    constraints = design[exact]
    # KKT system of the equality-constrained weighted least squares.
    n_rows = len(constraints)
    lhs = np.block(
        [[weighted @ design[fitted], constraints.T], [constraints, np.zeros((n_rows, n_rows))]]
    )
    rhs = np.concatenate([weighted @ table[fitted], table[exact]])
    return np.linalg.solve(lhs, rhs)[1 : N_PLAYERS + 1]


def test_shapley_kernel_reproduces_the_shapley_value():
    index = ShapleyValueIndex()
    exact = ExactComputer.from_game(TableGame(TABLE, FinitePlayers(N_PLAYERS)))(index).values
    fit = _least_squares_fit(index, np.asarray(TABLE, dtype=float))

    assert isinstance(index, LeastSquaresIndex)
    assert np.allclose(fit, exact, atol=1e-5)


def test_only_the_shapley_value_exposes_the_kernel():
    # SII shares the Shapley weights, but it is not the kernel's least-squares fit.
    assert not isinstance(ShapleyInteractionIndex(max_order=2), LeastSquaresIndex)


@pytest.mark.parametrize(
    ("index", "constants"),
    [
        (BanzhafIndex(max_order=2), {"p": 0.5}),
        (ShapleyInteractionIndex(max_order=2), {"alpha": 1.0, "beta": 1.0}),
        (ShapleyValueIndex(), {"alpha": 1.0, "beta": 1.0}),
    ],
)
def test_fixed_indices_preserve_constants_when_batched(index, constants):
    assert index.shape == ()
    assert set(index.protected_values()) == {"_batch"}
    kwargs = {} if isinstance(index, ShapleyValueIndex) else {"max_order": index.max_order}
    assert type(index)(**kwargs)._batch is index._batch
    for name, value in constants.items():
        with pytest.raises(TypeError, match="unexpected keyword"):
            type(index)(**kwargs, **{name: value + 1})
        with pytest.raises(FrozenInstanceError):
            setattr(index, name, value + 1)
        with pytest.raises((TypeError, ValueError), match="init=False"):
            replace(index, **{name: value + 1})

    for rebuilt, shape in [
        (index.with_protected_values(index.protected_values()), ()),
        (jax.jit(lambda x: x)(index), ()),
        (index[..., None, None], (1, 1)),
        (index.reshape(1, 1), (1, 1)),
        (jax_broadcast_to(index, (2, 3)), (2, 3)),
        (jax_broadcast_to(index, (2, 3))[1], (3,)),
        (jax_broadcast_to(index, (2, 3)).reshape(6), (6,)),
    ]:
        assert type(rebuilt) is type(index)
        assert rebuilt.shape == shape
        assert rebuilt._batch.shape == (*shape, 0)
        assert rebuilt._batch.dtype == jnp.uint8
        assert rebuilt._batch.nbytes == 0
        assert rebuilt.max_order == index.max_order
        for name, value in constants.items():
            assert getattr(rebuilt, name).shape == ()
            assert jnp.all(getattr(rebuilt, name) == value)

        weights = jax.jit(
            lambda idx: idx.discrete_derivative_weight(jnp.array(2), jnp.array(1), N_PLAYERS)
        )(rebuilt)
        assert weights.shape == shape
        assert jnp.allclose(
            weights, index.discrete_derivative_weight(jnp.array(2), jnp.array(1), N_PLAYERS)
        )

        if isinstance(rebuilt, LeastSquaresIndex):
            kernel = rebuilt.least_squares_weight(jnp.array(2), N_PLAYERS)
            assert kernel.shape == shape
            assert jnp.allclose(kernel, index.least_squares_weight(jnp.array(2), N_PLAYERS))

    sizes = jnp.arange(3)
    assert jnp.allclose(
        jax.jit(lambda idx: idx.discrete_derivative_weight(sizes, jnp.array(1), N_PLAYERS))(index),
        index.discrete_derivative_weight(sizes, jnp.array(1), N_PLAYERS),
    )


def test_alpha_beta_parameters_stay_compact():
    index = AlphaBetaIndex(
        max_order=2, alpha=jnp.array([[1.0], [4.0]]), beta=jnp.array([1.0, 2.0, 3.0])
    )
    assert index.shape == (2, 3)
    assert index.alpha.shape == (2, 1)
    assert index.beta.shape == (3,)

    weights = jax.jit(
        lambda idx: idx.discrete_derivative_weight(jnp.array(2), jnp.array(1), N_PLAYERS)
    )(index)
    assert weights.shape == (2, 3)
    for i in range(2):
        for j in range(3):
            scalar = AlphaBetaIndex(max_order=2, alpha=index.alpha[i, 0], beta=index.beta[j])
            assert jnp.allclose(
                weights[i, j],
                scalar.discrete_derivative_weight(jnp.array(2), jnp.array(1), N_PLAYERS),
            )


def test_index_parameters_preserve_weak_scalars_and_explicit_array_dtypes():
    weighted = WeightedBanzhafIndex(p=1, max_order=1)
    alpha_beta = AlphaBetaIndex(alpha=2, beta=1.0, max_order=1)
    assert weighted.p.weak_type and alpha_beta.alpha.weak_type and alpha_beta.beta.weak_type

    weighted = WeightedBanzhafIndex(p=np.array([0.25, 0.5], dtype=np.float16), max_order=1)
    alpha_beta = AlphaBetaIndex(alpha=np.float16(2), beta=jnp.bfloat16(1), max_order=1)
    assert weighted.p.dtype == jnp.float16 and not weighted.p.weak_type
    assert alpha_beta.alpha.dtype == jnp.float16 and not alpha_beta.alpha.weak_type
    assert alpha_beta.beta.dtype == jnp.bfloat16 and not alpha_beta.beta.weak_type

    large = AlphaBetaIndex(alpha=jnp.float16(50000), beta=jnp.float16(50000), max_order=1)
    weights = large.discrete_derivative_weight(jnp.array(1), jnp.array(1), 3)
    assert weights.dtype == jnp.asarray(0.0).dtype
    assert jnp.isclose(weights, 0.25, atol=0.03)
