"""Exact solves for the Vandermonde systems of the polynomial TreeSHAP machinery.

``TreeSHAPIQ`` and ``LinearTreeSHAP`` repeatedly need ``inv(vander(D[:i]).T) @ rhs``
for prefixes ``D[:i]`` of a fixed interpolation grid. These Vandermonde systems are
severely ill-conditioned in double precision — the conditioning is non-monotonic in
the prefix length and peaks at interior prefixes (``i ~ n/2``), so explicit
``float64`` inversion drifts at the ~1e-7 level from interpolation degree ~20,
returns wrong values from ~32 on the default grids, and raises ``LinAlgError``
for very deep trees (~60+).

The conditioning is purely a floating-point artifact: the interpolation nodes are
distinct, so the systems are exactly solvable over the rationals. This module
therefore solves them **exactly** and returns the correctly rounded ``float64``
result, at every depth:

- **O(n^2) Björck-Pereyra recursion** instead of an O(n^3) inverse. The dual
  Vandermonde recursion (Golub & Van Loan, Algorithm 4.6.2) factors the inverse
  into bidiagonal steps, which keeps the exact-arithmetic operand sizes small.
- **Scaled-integer fixed-point arithmetic** (standard-library ``int``): every node
  and intermediate value is represented as ``round(v * 2**bits)``. With ``bits``
  large enough the rounded result equals the exact rational solution rounded to
  ``float64``; integer arithmetic keeps a depth-100 grid's full prefix workload in
  the sub-second range, where naive ``fractions.Fraction`` elimination needs minutes.
- **A convergence certificate instead of an error analysis**: each system is solved
  at increasing precision (128, 256, ... bits) until two consecutive precision
  levels produce bitwise-identical ``float64`` outputs. Agreement of two scaled
  truncation grids that fine means the value has converged to the rounding of the
  exact solution, so the returned coefficients carry no conditioning error at all.

Every path either returns the exact result or refuses loudly: degenerate inputs
(coincident, non-finite, or extreme custom nodes) raise ``ValueError``, and
N-matrix rows whose magnitude exceeds what the downstream float64 pipeline can
evaluate raise :class:`~shapiq.utils.errors.RepresentationLimitError`.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from shapiq.utils.errors import RepresentationLimitError

if TYPE_CHECKING:
    from collections.abc import Callable

# Precision ladder for the convergence certificate. 128 bits converges for every
# Chebyshev-grid prefix up to at least depth 100; the 256-bit rung confirms it
# whenever the cheap float64 cross-check cannot (ill-conditioned prefixes), and
# the higher rungs exist for adversarial custom grids (LinearTreeSHAP(base_func=...)).
_PRECISION_BITS = (128, 256, 512, 1024, 2048, 4096)

# Largest N-matrix entry the downstream float64 pipeline can absorb. The
# coefficients themselves are exact, but the explainers consume them in float64
# inner products whose summands scale with the largest entry while the result is
# O(prediction). Measured end to end on chain trees of depth 20-40, the relative
# cancellation error tracks ``max|N| * 1e-13`` within one order of magnitude, so
# at 3e10 the expected loss is in the 0.3%-3% range. Beyond that the explainers
# refuse instead of returning values whose error may exceed ~1% of the
# prediction. For the default Chebyshev grids this corresponds to an
# interpolation degree of ~29 for LinearTreeSHAP's right-hand sides and ~25
# for TreeSHAPIQ's (its identity N matrix, built for every index, saturates
# first). The bound is empirical, not a worst-case proof.
_REPRESENTATION_LIMIT = 3.0e10


def _check_row_magnitude(row: np.ndarray) -> None:
    """Refuse N-matrix rows whose magnitude exceeds the float64 pipeline's limit.

    The Vandermonde solves themselves are exact at any depth, but the
    monomial-basis N-matrix entries grow exponentially with the interpolation
    degree, and the downstream float64 evaluation cancels them against each
    other. Beyond ``_REPRESENTATION_LIMIT`` the empirically calibrated loss may
    exceed ~1% of the result, so this raises instead of letting the explainer
    return silently wrong values.

    Raises:
        RepresentationLimitError: If the row magnitude exceeds the limit.
    """
    peak = float(np.max(np.abs(row)))
    if peak > _REPRESENTATION_LIMIT:
        msg = (
            "The interpolation degree of this tree is too large for the float64 "
            f"TreeSHAP pipeline: the exact interpolation coefficients reach {peak:.1e}, "
            "and evaluating them in double precision would lose more than ~1% of the "
            "result to cancellation. Reduce the interpolation degree — depending on "
            "the explainer and index this is the tree depth, min(tree depth, features "
            "in the tree), or the number of features in the tree; the default grids "
            "support up to ~29 with LinearTreeSHAP and ~25 with TreeSHAPIQ. This is a "
            "structural limit of the monomial-basis representation, not of the solver."
        )
        raise RepresentationLimitError(msg)


def build_n_matrix(
    grid: np.ndarray,
    rhs_for_row: Callable[[int], np.ndarray],
) -> np.ndarray:
    """Build an interpolation N matrix row by row from exact Vandermonde solves.

    Row ``i`` is the solution of ``vander(grid[:i]).T @ x = rhs_for_row(i)``.
    Every row is checked against the float64 representation limit as soon as it
    is computed, so an over-deep tree is refused after the first offending row
    instead of after the full (expensive) construction.

    Args:
        grid: The full interpolation grid (its length is the maximum degree).
        rhs_for_row: Callback returning the length-``i`` right-hand side of row ``i``.

    Returns:
        The ``(len(grid) + 1, len(grid))`` N matrix (row 0 stays zero).

    Raises:
        RepresentationLimitError: If any row exceeds the representation limit.
        ValueError: For degenerate grids (coincident, non-finite, or extreme
            nodes), propagated from :func:`solve_vandermonde`.
    """
    depth = len(grid)
    n_matrix = np.zeros((depth + 1, depth))
    for i in range(1, depth + 1):
        n_matrix[i, :i] = solve_vandermonde(grid[:i], rhs_for_row(i))
        _check_row_magnitude(n_matrix[i, :i])
    return n_matrix


def _to_fixed_point(value: float, bits: int) -> int:
    """Return ``value * 2**bits`` as an integer (truncated toward minus infinity).

    The truncation direction is irrelevant for the convergence certificate: the
    per-step error stays below ``2**-bits`` in either case, and the certificate
    only accepts results that are stable under a 128-bit precision increase.
    """
    fraction = Fraction(value)
    return (fraction.numerator << bits) // fraction.denominator


def _bjorck_pereyra_fixed_point(points: np.ndarray, rhs: np.ndarray, bits: int) -> np.ndarray:
    """Solve ``vander(points).T @ x = rhs`` in scaled-integer arithmetic.

    Implements the dual Björck-Pereyra recursion on values scaled by ``2**bits``.
    ``np.vander`` uses decreasing powers, so the right-hand side is reversed to
    obtain the increasing-power moment problem the recursion expects.
    """
    n = len(points)
    nodes = [_to_fixed_point(p, bits) for p in points.tolist()]
    x = [_to_fixed_point(v, bits) for v in rhs.tolist()[::-1]]
    for k in range(n - 1):
        node_k = nodes[k]
        for i in range(n - 1, k, -1):
            x[i] = x[i] - (node_k * x[i - 1] >> bits)
    for k in range(n - 2, -1, -1):
        for i in range(k + 1, n):
            x[i] = (x[i] << bits) // (nodes[i] - nodes[i - k - 1])
        for i in range(k + 1, n):
            x[i - 1] = x[i - 1] - x[i]
    scale = 1 << bits
    return np.array([float(Fraction(value, scale)) for value in x])


def solve_vandermonde(points: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``vander(points).T @ x = rhs`` exactly, rounded to ``float64``.

    The system is solved in scaled-integer arithmetic and certified by agreement
    with an independent computation: first against a plain float64 LAPACK solve
    (which corroborates the well-conditioned common case at negligible cost), and
    otherwise against the next rung of the precision ladder, until two
    computations agree bitwise. An agreeing result equals the exact rational
    solution rounded to ``float64`` up to an overwhelmingly improbable
    coincidence of correlated rounding (both computations would have to land in
    the same wrong rounding bin), regardless of the conditioning of the system.
    Works for any grid of distinct interpolation nodes.

    Solves are memoized on the byte representation of ``(points, rhs)``: the
    explainers issue identical solves for every tree of equal depth in an
    ensemble, which therefore costs one tree's worth of work.

    Args:
        points: Interpolation nodes (a prefix of the interpolation grid). The nodes
            must be pairwise distinct.
        rhs: Right-hand side vector of the same length.

    Returns:
        The solution vector ``x`` (the ``float64`` rounding of the exact solution).

    Raises:
        ValueError: If the inputs are degenerate — coincident or non-finite
            nodes, a non-finite right-hand side, or custom nodes so close
            together (or so large) that the exact solution leaves the float64
            range or does not stabilize within the precision ladder. The
            default Chebyshev grids never trigger any of these.
    """
    points = np.asarray(points, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if points.ndim != 1 or points.shape != rhs.shape:
        msg = "The interpolation nodes and right-hand side must be 1-D arrays of equal length."
        raise ValueError(msg)
    if not (np.isfinite(points).all() and np.isfinite(rhs).all()):
        msg = "The interpolation nodes and right-hand side must be finite."
        raise ValueError(msg)
    if len(points) == 1:
        return rhs.copy()
    solution = _solve_vandermonde_cached(points.tobytes(), rhs.tobytes())
    return solution.copy()  # the cached array must stay immutable


_DEGENERATE_NODES_MSG = (
    "The exact solution leaves the float64 range or does not stabilize within "
    f"{_PRECISION_BITS[-1]} fractional bits; the interpolation nodes are too close "
    "together or too large in magnitude. Use better-separated nodes of moderate "
    "magnitude."
)


def clear_solver_cache() -> None:
    """Drop all memoized Vandermonde solutions (for cold-timing tests or memory-sensitive hosts)."""
    _solve_vandermonde_cached.cache_clear()


@lru_cache(maxsize=4096)
def _solve_vandermonde_cached(points_bytes: bytes, rhs_bytes: bytes) -> np.ndarray:
    n = len(points_bytes) // 8  # float64 buffers; lengths are equal by construction
    points = np.frombuffer(points_bytes, dtype=float, count=n)
    rhs = np.frombuffer(rhs_bytes, dtype=float, count=n)

    previous = None
    for bits in _PRECISION_BITS:
        try:
            current = _bjorck_pereyra_fixed_point(points, rhs, bits)
        except ZeroDivisionError:
            # Two nodes collapsed onto the same scaled integer at this precision.
            # Genuinely distinct nodes separate at a finer rung (and stay separated
            # at every rung above it), so climb instead of misreporting them as
            # coincident; truly coincident nodes collapse at every rung and fall
            # through to the error below.
            previous = None
            continue
        except OverflowError:
            # float(Fraction(...)) overflows when the exact solution leaves the
            # float64 range (nearly-coincident or very large custom nodes).
            raise ValueError(_DEGENERATE_NODES_MSG) from None
        if previous is None:
            # Cheap independent certificate: a float64 LAPACK solve corroborates
            # the well-conditioned common case at a fraction of the cost of a
            # second fixed-point solve. It must solve the same decreasing-power
            # system the fixed-point recursion solves by reversing the rhs.
            try:
                float_solution = np.linalg.solve(np.vander(points).T, rhs)
                if np.array_equal(current, float_solution):
                    return current
            except np.linalg.LinAlgError:  # float64 considers the system singular
                pass
        elif np.array_equal(previous, current):
            return current
        previous = current
    if previous is None:
        msg = "The interpolation nodes must be pairwise distinct."
        raise ValueError(msg)
    # Reachable only for custom grids with nearly-coincident nodes, where no
    # float64 rounding of the (astronomically large) exact solution stabilizes.
    raise ValueError(_DEGENERATE_NODES_MSG)
