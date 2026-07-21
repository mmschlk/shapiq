"""LeverageSHAP regression approximator (Algorithm 1 of Musco and Witter, 2025)."""

from __future__ import annotations

import math
import random as _py_random
import sys
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from shapiq.approximator.regression.base import solve_regression
from shapiq.interaction_values import InteractionValues

from .base import Regression

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.game import Game
    from shapiq.typing import FloatVector

ValidRegressionLeverageSHAPIndices = Literal["SV"]


class LeverageSHAP(Regression[ValidRegressionLeverageSHAPIndices]):
    r"""Leverage SHAP approximator for Shapley values.

    Leverage SHAP, introduced by Musco and Witter (2025) :cite:t:`Musco.2025`, is a
    lightweight modification of KernelSHAP that comes with provable accuracy
    guarantees. Like KernelSHAP, it recovers the Shapley values as the solution of a
    weighted least-squares problem over sampled coalitions; unlike KernelSHAP, it
    samples coalitions proportional to their statistical *leverage scores* rather than
    the heuristic Shapley kernel weights. The key result of the paper is that these
    leverage scores have a simple closed form -- the score of a coalition depends only
    on its size, ``l_z = 1/C(n, ||z||)`` (Lemma 3.2) -- which makes leverage-score
    sampling tractable despite the exponentially many coalitions.

    This class is a faithful implementation of Algorithm 1 of the paper. Given a target
    budget ``m`` of game evaluations, it:

    1. Solves for an oversampling parameter ``c`` by binary search so that the expected
       number of sampled coalitions matches the budget,
       ``m - 2 = sum_{s=1}^{n-1} min(C(n, s), 2c)`` (Equation 12). Two evaluations are
       reserved for the empty and grand coalitions.
    2. Draws coalition pairs ``(z, z̄)`` by Bernoulli sampling without replacement
       (Algorithm 2). For each size ``s`` the number of pairs is drawn as
       ``m_s ~ Binomial(C(n, s), min(1, 2c / C(n, s)))``. Sizes whose entire layer fits
       within the ``2c`` budget are taken exhaustively; since ``C(n, s)`` is symmetric
       and peaks in the middle, that covers the smallest and largest sizes, leaving only
       the middle sizes to be subsampled.
    3. Reweights each sampled row by ``w(||z||) / min(1, 2c * l_z)``, where
       ``w(s) = (s-1)! (n-s-1)! / n!`` is the Shapley kernel weight -- the standard
       importance-sampling correction that keeps the estimate unbiased.
    4. Projects out the efficiency constraint to obtain an unconstrained regression
       (Lemma 3.1), solves it by weighted least squares, and adds the efficiency offset
       back in.

    Paired sampling (always including a coalition's complement) and sampling without
    replacement are built into the algorithm; both are variance-reduction tricks that
    the optimized KernelSHAP in the SHAP library also uses.

    Note:
        The number of game evaluations is random -- it concentrates tightly around
        ``budget`` but may over- or undershoot slightly. The paper also describes a fully
        deterministic variant that fixes ``m_s`` to the Binomial's expectation; we use the
        random Binomial form to match the paper's reported experiments.

    Example:
        >>> from shapiq.approximator import LeverageSHAP
        >>> from shapiq_games.synthetic import DummyGame
        >>> n = 5
        >>> game = DummyGame(n=n, interaction=(1, 2))
        >>> approximator = LeverageSHAP(n=n, random_state=42)
        >>> sv_estimates = approximator.approximate(budget=100, game=game)
        >>> print(sv_estimates.values)
        [0.  0.2 0.7 0.7 0.2 0.2]

    See Also:
        - :class:`~shapiq.approximator.regression.kernelshap.KernelSHAP`: The original
          KernelSHAP approximator that Leverage SHAP refines.
    """

    valid_indices: tuple[ValidRegressionLeverageSHAPIndices, ...] = ("SV",)

    def __init__(
        self,
        n: int,
        *,
        pairing_trick: bool = True,
        sampling_weights: np.ndarray | None = None,
        random_state: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the LeverageSHAP approximator.

        Args:
            n: The number of players.

            pairing_trick: Inert; kept only for interface compatibility with the other
                regression approximators. Algorithm 1 always samples pairs ``(z, z̄)``
                together, so toggling this flag has no effect on the output.

            sampling_weights: Inert; kept only for interface compatibility. LeverageSHAP
                uses its own leverage-score-based sampling scheme and ignores this argument.

            random_state: The random state of the estimator. Defaults to ``None``.

            **kwargs: Additional keyword arguments (not used, only for compatibility).
        """
        super().__init__(
            n,
            max_order=1,
            index="SV",
            random_state=random_state,
            pairing_trick=pairing_trick,
            sampling_weights=sampling_weights,
        )

    def approximate(
        self,
        budget: int,
        game: Game | Callable[[np.ndarray], np.ndarray],
        *args: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> InteractionValues:
        """Approximate the Shapley values via leverage-score-guided sampling.

        Args:
            budget: Target number of game evaluations (Algorithm 1 input ``m``).
            game: The game to approximate.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            The estimated Shapley values as an :class:`~shapiq.InteractionValues` object.
            Its ``estimation_budget`` reports the number of coalitions actually evaluated,
            which concentrates around ``budget`` but may over- or undershoot it (see the
            class docstring).

        Raises:
            ValueError: If ``budget`` is less than ``2`` (the empty and grand coalitions
                must both be evaluated), or if the game returns non-finite (NaN/Inf) values.
        """
        Z, weights = self._sample(budget)
        game_values: FloatVector = game(Z)
        # Number of coalitions actually evaluated. Because BernoulliSample draws a random
        # (Binomial) number of pairs, this concentrates around ``budget`` but can over- or
        # undershoot it; report the realized count so downstream cost accounting is honest.
        n_evaluations = int(Z.shape[0])
        if not np.all(np.isfinite(game_values)):
            msg = "Game returned NaN or Inf values. LeverageSHAP requires finite game values."
            raise ValueError(msg)
        v0 = float(game_values[np.sum(Z, axis=1) == 0][0])

        n = self.n
        coalition_sizes = Z.sum(axis=1)
        v_grand = float(game_values[coalition_sizes == n][0])
        efficiency_shift = (v_grand - v0) / n

        interior = (coalition_sizes > 0) & (coalition_sizes < n)
        Z_int = Z[interior].astype(float)
        v_int = game_values[interior]
        s_int = coalition_sizes[interior]
        w_is = weights[interior]

        if len(Z_int) == 0:
            sv = np.concatenate([[v0], np.full(n, efficiency_shift)])
        else:
            # Keep all LeverageSHAP-specific math here: the shared solver only handles
            # the weighted least-squares step.
            A = Z_int - (s_int / n)[:, np.newaxis]
            b = (v_int - v0) - efficiency_shift * s_int
            phi_perp = solve_regression(
                X=A,
                y=b,
                kernel_weights=w_is,
                use_svd=True,
            )
            sv = np.concatenate([[v0], phi_perp + efficiency_shift])
        return InteractionValues(
            values=sv,
            index=self.approximation_index,
            interaction_lookup=self.interaction_lookup,
            baseline_value=v0,
            min_order=self.min_order,
            max_order=self.max_order,
            n_players=self.n,
            estimated=not budget >= 2**self.n,
            estimation_budget=n_evaluations,
            target_index=self.index,
        )

    def _sample(self, budget: int) -> tuple[np.ndarray, np.ndarray]:
        r"""Algorithm 1, lines 1-7: BernoulliSample plus IS reweighting.

        This method implements the custom Bernoulli sampling logic required by
        LeverageSHAP, bypassing the generic ``CoalitionSampler``. This is necessary
        to strictly enforce the $2c$ threshold boundaries (Equation 12). Because the
        leverage score $l_z = 1/\binom{n}{s}$ is largest for the few extreme-size
        coalitions, those layers fit within the $2c$ budget and are evaluated
        exhaustively (both the smallest coalitions and their large-cardinality
        complements, since $\binom{n}{s} = \binom{n}{n-s}$ is small at both extremes),
        while leverage sampling otherwise spreads samples uniformly across sizes.

        Args:
            budget: Target number of evaluations ``m``.

        Returns:
            Z: Boolean coalition matrix of shape ``(n_coalitions, n)`` containing
                the empty coalition, the grand coalition, and the BernoulliSample
                pairs.
            weights: Per-coalition IS weights ``w(s) / min(1, 2c·l_z)`` with
                arbitrary positive scale (only relative weights matter for lstsq).
                Empty/grand coalitions get weight 0 (they enter via the efficiency
                shift, not the regression).
        """
        if budget < 2:
            msg = "Budget must be at least 2 to evaluate baseline and grand coalition."
            raise ValueError(msg)

        n = self.n
        m = min(budget, 2**n)  # cap budget at full enumeration (2^n)

        z_empty = np.zeros(n, dtype=bool)
        z_grand = np.ones(n, dtype=bool)

        c = self._find_c(n, m)  # oversampling parameter from Eq. 12
        Z_pairs, sizes = self._bernoulli_sample(n, c)

        # IS weights (Algorithm 1 line 7)
        if Z_pairs.shape[0] > 0:
            weights_pairs = np.empty(Z_pairs.shape[0], dtype=float)
            two_c = 2.0 * c
            fact_n = math.factorial(n)  # big-int; reused across sizes

            for i, s in enumerate(sizes):
                full_count = math.comb(n, s)
                # A size is included deterministically iff its whole layer fits the 2c
                # budget. Compare in big-int space (full_count is an exact int) so the
                # threshold never forms float(C(n, s)), which overflows for large n.
                if full_count <= two_c:
                    # p = 1: the IS weight is the raw Shapley kernel weight
                    # w(s) = (s-1)!(n-s-1)!/n!, evaluated as a big-int ratio.
                    weights_pairs[i] = (math.factorial(s - 1) * math.factorial(n - s - 1)) / fact_n
                else:
                    # p = 2c/C(n,s): the IS weight w(s)/p collapses analytically to
                    # w(s)*C(n,s)/(2c) = 1/(s*(n-s)*2c), so the binomial cancels and
                    # cannot overflow. (C(n,s) == C(n,n-s) makes this symmetric across
                    # a (z, z̄) pair, matching the paired-sampling design.)
                    weights_pairs[i] = 1.0 / (s * (n - s) * two_c)

            Z = np.vstack([z_empty[None, :], z_grand[None, :], Z_pairs])
        else:
            weights_pairs = np.empty(0, dtype=float)
            Z = np.vstack([z_empty[None, :], z_grand[None, :]])

        # Empty/grand get weight 0: they enter via the efficiency shift, not the regression.
        weights = np.concatenate([[0.0, 0.0], weights_pairs])
        return Z, weights

    @staticmethod
    def _find_c(n: int, m: int) -> float:
        """Algorithm 1, line 2: binary search for ``c`` solving Eq. 12.

        ``m - 2 = sum_{s=1}^{n-1} min(C(n,s), 2c)``.
        """
        MAX_BISECT_ITER = 200

        if n < 2:
            return 0.0
        target = m - 2  # budget minus empty + grand
        if target <= 0:
            return 0.0  # nothing left to sample beyond empty + grand

        binoms = [math.comb(n, s) for s in range(1, n)]  # kept as int to avoid float overflow

        def total(c_: float) -> float:
            two_c = 2.0 * c_
            return float(sum(min(b, two_c) for b in binoms))

        # Grow the upper bound by doubling rather than float(max_binom), which overflows for large n.
        hi = 1.0
        while total(hi) < target:
            hi *= 2.0
        lo = 0.0
        for _ in range(MAX_BISECT_ITER):
            mid = 0.5 * (lo + hi)
            if total(mid) >= target:
                hi = mid
            else:
                lo = mid
            if hi - lo < 1e-12 * max(1.0, hi):
                break
        return 0.5 * (lo + hi)

    def _bernoulli_sample(self, n: int, c: float) -> tuple[np.ndarray, np.ndarray]:
        """Algorithm 2 (BernoulliSample) of Musco and Witter (2025) :cite:t:`Musco.2025`.

        For each size ``s in {1, ..., floor(n/2)}`` draws ``m_s ~ Binomial`` pairs
        ``(z, z̄)`` without replacement. The middle size (when ``n`` is even and
        ``s = n/2``) is partitioned by fixing ``z_n = 1`` so each unordered pair
        is sampled at most once.

        Returns:
            Z_pairs: Boolean coalition matrix with both ``z`` and ``z̄`` appended
                consecutively for each pair.
            sizes: Cardinality of each row of ``Z_pairs``.
        """
        if n < 2 or c <= 0.0:
            return np.zeros((0, n), dtype=bool), np.zeros(0, dtype=int)

        z_list: list[np.ndarray] = []
        sizes_list: list[int] = []

        # Convert numpy seed to a Python random seed so randrange supports
        # arbitrary-precision integers (needed for large n where C(n, n/2) overflows int64).
        py_seed = int(self._rng.integers(0, 2**32))
        py_rng = _py_random.Random(py_seed)  # noqa: S311 - reproducible, non-crypto sampling

        two_c = 2.0 * c
        for s in range(1, n // 2 + 1):  # sizes 1..⌊n/2⌋ (the rest are covered via complement z̄)
            is_middle = (n % 2 == 0) and (s == n // 2)  # pair would self-complement here
            full_count = math.comb(n, s)  # C(n, s); exact big-int, never cast to float

            # Number of distinct unordered pairs at this size.
            pool_size = math.comb(n - 1, s - 1) if is_middle else full_count

            if full_count <= two_c:
                # Whole layer fits the 2c budget (inclusion probability 1): include all
                # pairs. The int-vs-float comparison is exact in Python, so it stays
                # overflow-safe even when C(n, s) is astronomically large.
                m_s = pool_size
            elif pool_size > 2**31 - 1:
                # Pool exceeds a conservative int32 cap (2**31 - 1) → fall back to
                # Poisson with the analytic mean ``pool_size * 2c / C(n, s)``, which
                # equals ``2c`` (non-middle) or ``2c*s/n`` (middle). This cap is
                # self-imposed (not numpy's own Binomial ``n`` limit, which is C long /
                # int64): computing the mean this way avoids forming ``float(C(n, s))``
                # (which overflows for n ≳ 1030). In this regime prob → 0 with the mean
                # fixed, so Poisson matches Binomial.
                poisson_mean = two_c * s / n if is_middle else two_c
                m_s = min(int(self._rng.poisson(poisson_mean)), pool_size)
            else:
                # pool_size fits in an int32 (numpy Binomial's ``n``-argument limit).
                # full_count equals pool_size for non-middle sizes and twice pool_size
                # for the middle size, i.e. at most ~2^32, which fits exactly in a
                # double, so prob = 2c/C(n,s) is a safe float. Binomial(pool_size, prob)
                # matches the paper's pseudocode (sample Binomial(C(n,s), prob) then
                # halve for the middle size).
                prob = two_c / full_count
                m_s = int(self._rng.binomial(pool_size, prob))

            if m_s == 0:
                continue

            indices = self._sample_without_replacement(pool_size, m_s, py_rng)

            for idx in indices:
                if is_middle:
                    # Sample over n-1 items with size s-1, then fix z_n = 1 so each
                    # unordered pair is produced exactly once.
                    z_partial = self._combo(n - 1, s - 1, idx)
                    z = np.zeros(n, dtype=bool)
                    z[: n - 1] = z_partial
                    z[n - 1] = True
                else:
                    z = self._combo(n, s, idx)
                z_bar = ~z  # complement (paired sampling)
                z_list.append(z)
                z_list.append(z_bar)
                sizes_list.append(int(z.sum()))
                sizes_list.append(int(z_bar.sum()))

        if z_list:
            return np.array(z_list), np.array(sizes_list, dtype=int)
        return np.zeros((0, n), dtype=bool), np.zeros(0, dtype=int)

    @staticmethod
    def _sample_without_replacement(total: int, k: int, py_rng: _py_random.Random) -> list[int]:
        """Sample ``k`` distinct integers from ``[0, total)`` without replacement.

        ``total`` may be an arbitrary-precision Python int (for large ``n``).
        """
        if k >= total:
            return list(range(total))  # asking for everything → return all indices

        # Fallback for astronomically large binomial pools (e.g. n=101) where
        # range(total) exceeds C ssize_t (sys.maxsize) and crashes random.sample.
        # Since k is tiny compared to total, rejection collisions are practically impossible.
        if total > sys.maxsize:
            seen: set[int] = set()
            while len(seen) < k:
                seen.add(py_rng.randrange(total))
            return list(seen)

        # random.sample supports range objects efficiently without materializing them and is
        # robust even when k is a large fraction of total.
        return py_rng.sample(range(total), k)

    @staticmethod
    def _combo(n: int, s: int, i: int) -> np.ndarray:
        """Algorithm 3: ``i``-th lexicographic combination of size ``s`` from ``n`` items.

        Returns a boolean vector of length ``n`` with exactly ``s`` True entries.
        ``i`` is 0-indexed.
        """
        z = np.zeros(n, dtype=bool)
        if s == 0:
            return z
        k = s  # remaining slots to fill
        j = 0  # current position
        while k > 0 and j < n:
            # Combinations that include position j (choose the other k-1 from the n-j-1
            # positions after it). If i falls in that block, j is in the combination;
            # otherwise skip past the block and move on.
            count = math.comb(n - j - 1, k - 1)
            if i < count:
                z[j] = True
                k -= 1
            else:
                i -= count
            j += 1
        return z
