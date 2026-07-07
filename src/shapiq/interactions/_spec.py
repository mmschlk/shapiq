"""The argmin presentation of interaction indices.

An argmin specification describes an index as the solution operator of a
symmetric weighted least squares fit: a row kernel over coalition sizes, a
cardinality basis whose design value depends only on the interaction size
and its intersection with the coalition, and optional exact-interpolation
constraints at the empty and grand coalition. Such a fit is linear in the
game and permutation-equivariant, so it compiles to a coalition functional.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import cache
from math import comb

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class ArgminSpecification:
    """A symmetric weighted least squares fit defining an interaction index.

    The fitted model assigns each interaction size ``s`` up to the order a
    coefficient per interaction, with the design value on a coalition ``T``
    given by ``basis_weights[s][|S & T|]``; row ``0`` is the constant column
    of the order-0 coefficient. Rows are weighted by ``row_weights`` per
    coalition size, and the interpolation flags pin the model to the empty
    or grand coalition exactly (their row weight is then conventionally
    zero). The index's attributions are the fitted coefficients.
    """

    row_weights: Array
    basis_weights: Array
    interpolate_empty: bool
    interpolate_grand: bool


def membership_basis(order: int) -> Array:
    """Return the membership basis: an interaction's column is one on supersets.

    Row 0 is the constant column; row ``s`` has design value one exactly when
    the whole interaction is inside the coalition.
    """
    return jnp.eye(order + 1)


def bernoulli_basis(order: int) -> Array:
    """Return the Bernoulli basis of kADD-SHAP per interaction and intersection size.

    Row 0 is the constant column; row ``s`` weights a coalition by a Bernoulli
    sum over the intersection size, so order-1 coefficients of a Shapley-kernel
    fit remain Shapley values at every order.
    """
    bernoulli = bernoulli_numbers(order)
    table = [[0.0] * (order + 1) for _ in range(order + 1)]
    table[0][0] = 1.0
    for size in range(1, order + 1):
        for intersection in range(1, size + 1):
            table[size][intersection] = sum(
                comb(intersection, top) * bernoulli[size - top]
                for top in range(1, intersection + 1)
            )
    return jnp.asarray(table)


@cache
def bernoulli_numbers(order: int) -> tuple[float, ...]:
    """Return the Bernoulli numbers up to ``order`` with the B(1) = -1/2 convention."""
    numbers = [Fraction(1)]
    for m in range(1, order + 1):
        acc = sum((Fraction(comb(m + 1, j)) * numbers[j] for j in range(m)), Fraction(0))
        numbers.append(-acc / (m + 1))
    return tuple(float(number) for number in numbers)
