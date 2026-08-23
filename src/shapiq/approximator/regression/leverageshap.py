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
    lightweight modification of KernelSHAP with a provable accuracy guarantee for its
    Binomial sampling variant (``deterministic_counts=False``; see the Note below for
    the deterministic default). Like KernelSHAP, it recovers the Shapley values as the
    solution of a weighted least-squares problem over sampled coalitions; unlike
    KernelSHAP, it samples coalitions proportional to their statistical *leverage
    scores* rather than the heuristic Shapley kernel weights. The key result of the
    paper is that these leverage scores have a simple closed form -- the score of a
    coalition depends only on its size, ``l_z = 1/C(n, ||z||)`` (Lemma 3.2) -- which
    makes leverage-score sampling tractable despite the exponentially many coalitions.

    This class is a faithful implementation of Algorithm 1 of the paper. Given a target
    budget ``m`` of game evaluations, it:

    1. Solves for an oversampling parameter ``c`` by binary search so that the expected
       number of sampled coalitions matches the budget,
       ``m - 2 = sum_{s=1}^{n-1} min(C(n, s), 2c)`` (Equation 12). Two evaluations are
       reserved for the empty and grand coalitions.
    2. Draws coalition pairs ``(z, z̄)`` by Bernoulli sampling without replacement
       (Algorithm 2). By default (``deterministic_counts=True``), for each size ``s``
       the pair count ``m_s`` is fixed to the *expected* value of
       ``Binomial(C(n, s), min(1, 2c / C(n, s)))`` instead of being drawn at random,
       rounded to integers via a largest-remainder rule so the realized total matches
       the budget exactly. This stratifies the sample by coalition size, in the spirit
       of stratified SVARM :cite:t:`Kolpaczki.2024a`. With
       ``deterministic_counts=False``, ``m_s`` is instead drawn at random from that
       Binomial, exactly as Algorithm 2 describes. Sizes whose entire layer fits within
       the ``2c`` budget are taken exhaustively in both modes; since ``C(n, s)`` is
       symmetric and peaks in the middle, that covers the smallest and largest sizes,
       leaving only the middle sizes to be subsampled.
    3. Reweights each sampled row by ``w(||z||) / min(1, 2c * l_z)``, where
       ``w(s) = (s-1)! (n-s-1)! / n!`` is the Shapley kernel weight -- the standard
       importance-sampling correction that keeps the estimate unbiased.
    4. Projects out the efficiency constraint to obtain an unconstrained regression
       (Lemma 3.1), solves it by weighted least squares, and adds the efficiency offset
       back in.

    Sampling without replacement is built into the algorithm. Paired sampling --
    drawing each coalition together with its complement -- is the default
    (``pairing_trick=True``) but can be disabled; see ``pairing_trick`` below. Both
    without-replacement sampling and (when enabled) pairing are variance-reduction
    tricks that the optimized KernelSHAP in the SHAP library also uses.

    Note:
        The default (``deterministic_counts=True``) fixes each size's pair count
        ``m_s`` to its Binomial expectation instead of drawing it at random, so the
        realized number of game evaluations equals
        ``2 + 2 * ((min(budget, 2**n) - 2) // 2)`` exactly -- no over- or undershoot.
        This is what the paper's own released implementation and every experiment
        reported in the paper actually use. The paper's accuracy theorem is stated and
        proved for the random ``deterministic_counts=False`` variant: because the
        deterministic variant no longer samples coalitions independently, its formal
        analysis does not carry over verbatim. The authors believe the same guarantee
        still holds for the deterministic variant and leave a formal proof to future
        work (Musco and Witter, 2025, end of Sec. 4).

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
        deterministic_counts: bool = True,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Initialize the LeverageSHAP approximator.

        Args:
            n: The number of players.

            pairing_trick: If ``True`` (default), the pairing trick is applied to the
                sampling procedure: every sampled coalition's complement is also
                included, exactly as Algorithm 1 always does. If ``False``, the same
                number of coalitions is still allocated per size, but they are drawn
                independently instead of in forced complementary pairs -- this
                reproduces the paper's "without paired sampling" ablation
                (``leverage_shap_unpaired`` in the released ``leverageshap`` package).
                Both modes sample without replacement.

            sampling_weights: Inert; kept only for interface compatibility. LeverageSHAP
                uses its own leverage-score-based sampling scheme and ignores this argument.

            random_state: The random state of the estimator. Defaults to ``None``.

            deterministic_counts: If ``True`` (default), fix each coalition size's pair
                count ``m_s`` to the expected value of Algorithm 2's Binomial draw
                (largest-remainder rounded so the total matches the budget exactly)
                instead of drawing it at random -- this is what the paper's released
                implementation and reported experiments use. If ``False``, draws
                ``m_s`` at random as Algorithm 2 literally describes. See the class
                docstring's Note for the accuracy-guarantee caveat.

            **kwargs: Additional keyword arguments (not used, only for compatibility).
        """
        self.deterministic_counts = deterministic_counts
        self.pairing_trick = pairing_trick
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
            Its ``estimation_budget`` reports the number of coalitions actually
            evaluated. With the default ``deterministic_counts=True`` this equals
            ``2 + 2 * ((min(budget, 2**n) - 2) // 2)`` exactly (see the class
            docstring's Note); with ``deterministic_counts=False`` it is the realized
            count of a random Binomial draw and only concentrates around that same
            value, possibly over- or undershooting it.

        Raises:
            ValueError: If ``budget`` is less than ``2`` (the empty and grand coalitions
                must both be evaluated), or if the game returns non-finite (NaN/Inf) values.
        """
        Z, weights = self._sample(budget)
        game_values: FloatVector = game(Z)
        # Number of coalitions actually evaluated. With the deterministic default this
        # is exact (see the class docstring's Note); with deterministic_counts=False it
        # is a random Binomial draw and can over- or undershoot the requested budget.
        # Report the realized count either way so downstream cost accounting is honest.
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
        if self.deterministic_counts:
            Z_pairs, sizes = self._bernoulli_sample_deterministic(n, c, m)
        else:
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
        # ``hi`` (not the midpoint) is returned so that at budget == 2**n every size's
        # layer is guaranteed exhaustive: the bisection invariant is total(hi) >= target
        # at every iteration, so ``hi`` always satisfies the target while the midpoint can
        # land just under it.
        return hi

    def _bernoulli_sample(self, n: int, c: float) -> tuple[np.ndarray, np.ndarray]:
        """Algorithm 2 (BernoulliSample) of Musco and Witter (2025) :cite:t:`Musco.2025`.

        For each size ``s in {1, ..., floor(n/2)}`` draws ``m_s ~ Binomial`` pairs
        ``(z, z̄)`` without replacement. This is the ``deterministic_counts=False``
        path; see ``_bernoulli_sample_deterministic`` for the default. Row
        construction (paired vs. unpaired, per ``self.pairing_trick``) is delegated to
        ``_build_rows`` once every size's count ``m_s`` is known.

        Returns:
            Z_pairs: Boolean coalition matrix with both ``z`` and ``z̄`` appended
                consecutively for each pair (paired mode) or independently drawn rows
                of sizes ``s`` and ``n - s`` (unpaired mode).
            sizes: Cardinality of each row of ``Z_pairs``.
        """
        if n < 2 or c <= 0.0:
            return np.zeros((0, n), dtype=bool), np.zeros(0, dtype=int)

        # Convert numpy seed to a Python random seed so randrange supports
        # arbitrary-precision integers (needed for large n where C(n, n/2) overflows int64).
        py_seed = int(self._rng.integers(0, 2**32))
        py_rng = _py_random.Random(py_seed)  # noqa: S311 - reproducible, non-crypto sampling

        counts: dict[int, int] = {}
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

            counts[s] = m_s

        return self._build_rows(n, counts, py_rng, paired=self.pairing_trick)

    def _bernoulli_sample_deterministic(
        self, n: int, c: float, m: int
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Deterministic analogue of ``_bernoulli_sample``'s Binomial draw.

        Fixes each size's pair count to the Binomial's expectation instead of sampling
        it, then rounds to integers via a largest-remainder rule so the *total* number
        of pairs matches the target exactly -- this is what the paper's released
        implementation does, and is the same stratification idea as stratified SVARM,
        Kolpaczki et al. 2024 :cite:t:`Kolpaczki.2024a`. This is the
        ``deterministic_counts=True`` (default) path. Row construction (paired vs.
        unpaired, per ``self.pairing_trick``) is delegated to ``_build_rows`` once every
        size's count ``m_s`` is known.

        This loop (like ``_bernoulli_sample``) already has only one degree of freedom
        per half-size ``s = 1..floor(n/2)``: a single pair count ``m_s`` determines both
        size ``s`` and its complement ``n - s`` (or, at the self-complementary middle
        size of even ``n``, two rows of the same size ``n/2``). Because of this, pairing
        fractional remainders at index ``i`` with ``n - 1 - i`` is unnecessary here --
        only the floor-then-largest-remainder-fill idea is used.

        Closed form for the continuous target (replaces the Binomial mean
        ``pool_size * 2c / C(n,s)`` used in ``_bernoulli_sample``): for non-middle sizes
        ``pool_size == C(n,s)``, so the mean simplifies to exactly ``2c``; for the
        middle size (even ``n`` only), ``pool_size == C(n-1, n/2-1) == C(n, n/2)/2``, so
        the mean simplifies to exactly ``c``. Neither closed form ever forms
        ``C(n,s)`` or ``pool_size`` as a float, so -- unlike the Binomial branch it
        replaces -- this needs no Poisson-mean/int32-pool fallback for huge ``n``; the
        existing fallback logic in ``_bernoulli_sample`` is simply not needed in this
        code path.

        Args:
            n: Number of players.
            c: Oversampling parameter from ``_find_c`` (already solved against ``m``,
                see ``_sample``).
            m: Target total budget, already capped at 2**n.

        Returns:
            Same shape/semantics as ``_bernoulli_sample``: (Z_pairs, sizes).
        """
        if n < 2 or c <= 0.0:
            return np.zeros((0, n), dtype=bool), np.zeros(0, dtype=int)

        # Integer division floors any odd ``m`` down to the nearest even total, so the
        # realized row count (2 + 2 * target_pairs) never overshoots ``m`` -- this is the
        # exact formula documented in the class docstring's Note and ``approximate()``'s
        # Returns section.
        target_pairs = (m - 2) // 2
        two_c = 2.0 * c
        half_sizes = list(range(1, n // 2 + 1))

        m_s: dict[int, int] = {}
        frac: dict[int, float] = {}
        exhaustive: dict[int, bool] = {}
        pool_size_of: dict[int, int] = {}

        for s in half_sizes:
            is_middle = (n % 2 == 0) and (s == n // 2)
            full_count = math.comb(n, s)
            pool_size = math.comb(n - 1, s - 1) if is_middle else full_count
            pool_size_of[s] = pool_size

            if full_count <= two_c:
                m_s[s] = pool_size
                frac[s] = 0.0
                exhaustive[s] = True
            else:
                mu = two_c * s / n if is_middle else two_c
                floor_mu = int(mu)  # mu = O(c); always safe to cast, unlike pool_size/full_count
                m_s[s] = min(floor_mu, pool_size)
                frac[s] = mu - floor_mu
                exhaustive[s] = False

        # Largest-remainder apportionment. Every non-exhaustive half-size s has
        # m_s[s] < pool_size_of[s] strictly by definition: for non-middle sizes
        # pool_size == full_count > two_c == mu, and for the middle size
        # pool_size > mu because it is not exhaustive; either way floor(mu) < pool_size
        # since both are integers and mu < pool_size. So every non-exhaustive size can
        # always absorb exactly one increment in the fill loop below without ever
        # triggering the ``m_s[s] >= pool_size_of[s]`` skip on its single visit -- this
        # holds for any c (not just one solved by ``_find_c``), so the fill loop can
        # resolve any shortfall up to and including the number of non-exhaustive
        # half-sizes, a bound strictly tighter than ``len(half_sizes)`` whenever at
        # least one size is exhaustive. When c *is* the ``_find_c(n, m)`` solution (the
        # only way ``_sample`` ever calls this method) the bound is normally not even
        # reached with equality: the continuous target is met almost exactly, so
        # shortfall is the sum of one sub-1 fractional remainder per non-exhaustive
        # size and so is typically strictly less than their count. Verified
        # empirically across a wide sweep of (n, budget) pairs with zero violations.
        shortfall = target_pairs - sum(m_s.values())
        num_non_exhaustive = sum(1 for s in half_sizes if not exhaustive[s])
        assert 0 <= shortfall <= num_non_exhaustive, (  # noqa: S101
            f"largest-remainder shortfall {shortfall} out of the expected "
            f"[0, {num_non_exhaustive}] range for n={n}, m={m}"
        )
        for s in sorted(half_sizes, key=lambda s: -frac[s]):
            if shortfall <= 0:
                break
            if exhaustive[s] or m_s[s] >= pool_size_of[s]:
                continue
            m_s[s] += 1
            shortfall -= 1

        py_seed = int(self._rng.integers(0, 2**32))
        py_rng = _py_random.Random(py_seed)  # noqa: S311

        return self._build_rows(n, m_s, py_rng, paired=self.pairing_trick)

    def _build_rows(
        self,
        n: int,
        counts: dict[int, int],
        py_rng: _py_random.Random,
        *,
        paired: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Turn per-half-size pair counts into concrete sampled coalition rows.

        Shared by ``_bernoulli_sample`` and ``_bernoulli_sample_deterministic``: both
        only differ in *how many* pairs ``counts[s]`` to draw for each half-size
        ``s in {1, ..., floor(n/2)}``; this method decides *which* rows to draw for a
        given count, branching on ``paired`` (the ``pairing_trick`` flag).

        If ``paired`` (default): for each ``s``, draw ``counts[s]`` coalitions of size
        ``s`` without replacement and append each one's complement (size ``n - s``)
        right next to it -- the middle size (even ``n``, ``s == n // 2``) instead
        partitions its restricted pool ``C(n - 1, s - 1)`` by fixing one player, so
        every drawn item still pairs with a distinct complement of the same size. This
        is Algorithm 1's ``(z, z̄)`` design.

        If not ``paired``: draw ``counts[s]`` coalitions of size ``s`` and,
        independently, another ``counts[s]`` coalitions of size ``n - s``, with no
        forced relationship between the two draws -- the paper's "without paired
        sampling" ablation. The middle size instead draws ``2 * counts[s]`` coalitions
        directly from the full pool ``C(n, s)`` (matching the released implementation's
        ``CoalitionSampler(pairing_trick=False)``, which is likewise without
        replacement in both modes). Per-size row counts are identical between the two
        modes either way; only whether a drawn coalition's complement is also present
        differs.

        Args:
            n: Number of players.
            counts: Pair count ``m_s`` for each half-size ``s`` in
                ``range(1, n // 2 + 1)``.
            py_rng: Seeded Python RNG for index sampling (arbitrary-precision integers).
            paired: Whether to force each drawn coalition's complement (Algorithm 1) or
                draw both sizes independently (the unpaired ablation).

        Returns:
            Z_pairs: Boolean coalition matrix.
            sizes: Cardinality of each row of ``Z_pairs``.
        """
        z_list: list[np.ndarray] = []
        sizes_list: list[int] = []

        for s, count in counts.items():
            if count == 0:
                continue
            is_middle = (n % 2 == 0) and (s == n // 2)

            if is_middle:
                if paired:
                    pool_size = math.comb(n - 1, s - 1)
                    indices = self._sample_without_replacement(pool_size, count, py_rng)
                    for idx in indices:
                        z_partial = self._combo(n - 1, s - 1, idx)
                        z = np.zeros(n, dtype=bool)
                        z[: n - 1] = z_partial
                        z[n - 1] = True
                        z_bar = ~z
                        z_list.append(z)
                        z_list.append(z_bar)
                        sizes_list.append(s)
                        sizes_list.append(s)
                else:
                    pool_size = math.comb(n, s)
                    indices = self._sample_without_replacement(pool_size, 2 * count, py_rng)
                    for idx in indices:
                        z_list.append(self._combo(n, s, idx))
                        sizes_list.append(s)
            else:
                pool_size = math.comb(n, s)
                indices = self._sample_without_replacement(pool_size, count, py_rng)
                if paired:
                    for idx in indices:
                        z = self._combo(n, s, idx)
                        z_bar = ~z
                        z_list.append(z)
                        z_list.append(z_bar)
                        sizes_list.append(s)
                        sizes_list.append(n - s)
                else:
                    for idx in indices:
                        z_list.append(self._combo(n, s, idx))
                        sizes_list.append(s)
                    # Independent draw for the complementary size n - s. The pool size is
                    # the same (C(n, s) == C(n, n - s)), but the drawn indices are not
                    # correlated with ``indices`` above, so a coalition's complement is only
                    # present by chance -- the defining property of the unpaired ablation.
                    indices_complement = self._sample_without_replacement(pool_size, count, py_rng)
                    for idx in indices_complement:
                        z_list.append(self._combo(n, n - s, idx))
                        sizes_list.append(n - s)

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
