"""Numerically guarded solves for the Vandermonde systems of TreeSHAP-IQ.

The polynomial TreeSHAP machinery repeatedly needs ``inv(vander(points).T) @ rhs``,
where ``points`` is a prefix ``D[:i]`` of a fixed interpolation grid ``D`` whose size
equals the interpolation degree (``min(tree depth, n_features_in_tree)`` for
TreeSHAPIQ, the full tree depth for LinearTreeSHAP). These Vandermonde systems are
severely ill-conditioned, and the conditioning is **non-monotonic in the prefix
length**: the first ``i`` points of an ``n``-point grid typically cluster near one
endpoint, so the condition number peaks around ``i ~ n/2``. For the default
Chebyshev grids, measured peak prefix condition numbers are ``4.1e11`` at grid size
26, ``1.3e12`` at 27 and ``4.2e13`` at 30 — precision degrades for grids larger
than ~26 (at *interior* prefixes, not only at the full size), and the systems become
singular to machine precision for very deep trees (sizes around 55-60), which
previously surfaced as an unexplained ``LinAlgError``.

This module centralises the solve:

- **Grid certification instead of assumptions.** For each distinct grid, the peak
  condition number over all prefixes is measured once and cached
  (:func:`grid_is_certified`); call sites hoist this decision out of their
  construction loops, so certified grids take a plain ``np.linalg.solve`` with no
  per-solve diagnostic work. The certification is computed on the actual grid, so
  custom interpolation bases (``LinearTreeSHAP(base_func=...)``) are handled
  correctly rather than assumed to behave like Chebyshev nodes.
- **Checked path with one SVD.** Uncertified grids are solved by SVD-based least
  squares; the singular values of that same decomposition give the exact condition
  number of the matrix actually solved, and rank-deficient systems degrade into a
  least-squares solution instead of crashing.
- **One warning per matrix, not per prefix.** Call sites construct whole N-matrices
  in a loop over prefixes; per-prefix warnings would flood the user (a depth-60
  construction touches dozens of degenerate prefixes). The loop passes a
  :class:`VandermondeDiagnostics` collector and emits a single summary warning.

Design note: rank-deficient systems deliberately **warn and return** the
least-squares fallback rather than raising. Before this module, the same depths
either crashed (``LinAlgError``, degree ~60) or — worse — **silently returned
wrong values** (rank-deficient prefixes appear from grid size ~32; completeness-axiom errors up to
orders of magnitude beyond the prediction itself). Warning loudly while degrading is strictly
more informative than either previous behaviour, and pipelines that want hard
failures can promote the warning with the ``warnings`` module.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

# Above this condition number, double precision has lost ~12 of its ~16 significant
# digits; results may no longer be exact. (Empirical anchor for the default
# Chebyshev grids: the peak prefix condition number stays below this threshold up
# to grid size 26 and exceeds it from size 27 on; see module docstring. The
# certification below measures each actual grid, so the constant is a warning
# threshold, not a structural assumption.)
_COND_WARN_THRESHOLD = 1e12

# Certification is only attempted for grids up to this size; larger grids go
# straight to the checked path (their peak prefix conditioning exceeds the
# threshold for every node family used in practice, and bounding the cache work
# keeps certification O(small)).
_CERTIFY_MAX_SIZE = 32


@lru_cache(maxsize=128)
def _max_prefix_condition(grid_key: tuple[float, ...]) -> float:
    """Measured peak condition number over all prefixes of the given grid."""
    grid = np.asarray(grid_key)
    worst = 1.0
    for i in range(2, len(grid) + 1):
        try:
            worst = max(worst, float(np.linalg.cond(np.vander(grid[:i]).T)))
        except np.linalg.LinAlgError:  # pragma: no cover - cond rarely raises
            return float("inf")
    return worst


class VandermondeDiagnostics:
    """Collects conditioning diagnostics across the solves of one N-matrix.

    Use one instance per matrix-construction loop and call :meth:`emit` once after
    the loop, so the user sees a single summary warning instead of one warning per
    degenerate prefix.
    """

    def __init__(self) -> None:
        self.worst_condition: float = 0.0
        self.rank_deficient_sizes: list[int] = []
        self.worst_residual: float = 0.0

    def emit(self, stacklevel: int = 4) -> None:
        """Emit at most one summary warning for the collected diagnostics.

        Args:
            stacklevel: Forwarded to :func:`warnings.warn`; pass the depth of the
                user's call site relative to ``emit`` (the construction loops sit
                at different depths in LinearTreeSHAP vs TreeSHAPIQ).
        """
        if self.rank_deficient_sizes:
            warnings.warn(
                f"Interpolation systems of size {self.rank_deficient_sizes} are "
                "singular to machine precision (very deep tree). Least-squares "
                "fallbacks were used (worst relative residual "
                f"{self.worst_residual:.1e}) - the resulting explanation values are "
                "NOT reliable. Limit the tree depth to obtain exact values.",
                RuntimeWarning,
                stacklevel=stacklevel,
            )
        elif self.worst_condition > _COND_WARN_THRESHOLD:
            warnings.warn(
                "The interpolation systems for this tree reach a condition number "
                f"of {self.worst_condition:.1e}; TreeSHAP-IQ values may lose "
                "precision. Consider limiting the tree depth (the interpolation "
                "degree is min(tree depth, number of features in the tree)).",
                RuntimeWarning,
                stacklevel=stacklevel,
            )


def grid_is_certified(grid: np.ndarray) -> bool:
    """Whether every prefix system of ``grid`` is measured well-conditioned.

    Note that "certified" bounds, but does not eliminate, precision loss: the
    threshold admits condition numbers up to 1e12 (a size-26 Chebyshev grid peaks
    at ~4e11). Certification costs one SVD per prefix on first touch of a grid
    (cached afterwards, and shared across the N-matrix builders), roughly doubling
    the linear-algebra work of the first construction for that grid size.

    Call once per interpolation grid (the measurement is cached per grid) and pass
    the result to :func:`solve_vandermonde` as ``certified=`` so the per-prefix
    solves carry no diagnostic work at all for ordinary trees.
    """
    grid = np.asarray(grid)
    if len(grid) > _CERTIFY_MAX_SIZE:
        return False
    return _max_prefix_condition(tuple(grid.tolist())) <= _COND_WARN_THRESHOLD


def solve_vandermonde(
    points: np.ndarray,
    rhs: np.ndarray,
    *,
    certified: bool = False,
    diagnostics: VandermondeDiagnostics | None = None,
) -> np.ndarray:
    """Solve ``vander(points).T @ x = rhs`` without forming an explicit inverse.

    Args:
        points: Interpolation nodes (a prefix of the interpolation grid).
        rhs: Right-hand side vector of the same length.
        certified: Result of :func:`grid_is_certified` for the *full* interpolation
            grid that ``points`` is a prefix of (conditioning peaks at interior
            prefixes, so safety is a property of the whole grid). When ``True``, a
            plain solve is used with no diagnostic work. Defaults to ``False``
            (checked SVD-based path).
        diagnostics: Optional collector. If given, conditioning findings are
            recorded on it (call :meth:`VandermondeDiagnostics.emit` after the
            construction loop) instead of warning per call.

    Returns:
        The solution vector ``x``.

    Warns:
        RuntimeWarning: Only when ``diagnostics`` is ``None``: if the solved
            system's condition number exceeds ``1e12`` (possible precision loss),
            or if it is rank-deficient and a least-squares fallback is returned
            whose values are not reliable.
    """
    V = np.vander(points).T
    size = len(points)
    if certified:
        # Certified grid: every prefix system measured well-conditioned.
        return np.linalg.solve(V, rhs)

    # Checked path: one SVD yields the solution, the exact condition number of this
    # very matrix, and rank-deficiency detection.
    solution, _, rank, singular_values = np.linalg.lstsq(V, rhs, rcond=None)
    if rank < size:
        residual = float(np.linalg.norm(V @ solution - rhs))
        scale = float(np.linalg.norm(rhs)) or 1.0
        if diagnostics is not None:
            diagnostics.rank_deficient_sizes.append(size)
            diagnostics.worst_residual = max(diagnostics.worst_residual, residual / scale)
        else:
            warnings.warn(
                f"The interpolation system of size {size} is singular to machine "
                "precision (very deep tree). A least-squares fallback is returned "
                f"with relative residual {residual / scale:.1e} - the resulting "
                "explanation values are NOT reliable. Limit the tree depth to "
                "obtain exact values.",
                RuntimeWarning,
                stacklevel=2,
            )
        return solution
    cond = float(singular_values[0] / singular_values[-1])
    if diagnostics is not None:
        diagnostics.worst_condition = max(diagnostics.worst_condition, cond)
    elif cond > _COND_WARN_THRESHOLD:
        warnings.warn(
            f"The interpolation system of size {size} has condition number "
            f"{cond:.1e}; TreeSHAP-IQ values for trees this deep may lose "
            "precision. Consider limiting the tree depth (the interpolation "
            "degree is min(tree depth, number of features in the tree)).",
            RuntimeWarning,
            stacklevel=2,
        )
    return solution
