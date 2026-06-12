"""ShaplEIG: Bayesian experimental design for Shapley value estimation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from tqdm import tqdm

from shapiq.approximator.base import Approximator
from shapiq.approximator.sampling import CoalitionSampler
from shapiq.interaction_values import InteractionValues

try:
    import torch

    from . import _shapley_math as sm
    from ._surrogate import HammingGP

except ImportError as err:
    from ._error import _shapleig_import_error

    raise _shapleig_import_error from err

if TYPE_CHECKING:
    from collections.abc import Callable

    from shapiq.game import Game
    from shapiq.typing import CoalitionMatrix, GameValues

ValidShaplEIGIndices = Literal["SV"]


class ShaplEIG(Approximator[ValidShaplEIGIndices]):
    """Bayesian experimental design (BED) approximator for Shapley values.

    ShaplEIG fits a Gaussian process surrogate with a weighted Hamming product
    kernel on the queried coalition values and sequentially selects the next
    coalition to evaluate by maximizing the closed-form expected information
    gain (EIG) about the Shapley values. The Shapley structure is handled via
    elementary-symmetric-polynomial (ESP) identities of the kernel's
    generating polynomials, so the full ``2^n`` coalition space is never
    enumerated and each iteration costs only polynomial time in ``n``.
    (``n``, shapiq's number of players, is denoted ``p`` in the paper.)

    The returned :class:`~shapiq.interaction_values.InteractionValues` contain
    the posterior-mean Shapley value estimates (order 1, index ``"SV"``).

    Requires the optional ``shapleig`` extra
    (``pip install shapiq[shapleig]``) for torch / gpytorch / botorch.

    Example:
        >>> from shapiq_games.synthetic import DummyGame
        >>> from shapiq.approximator import ShaplEIG
        >>> game = DummyGame(n=7, interaction=(1, 2))
        >>> approximator = ShaplEIG(n=7, random_state=42, show_progress=False)
        >>> values = approximator.approximate(budget=50, game=game)
        >>> values.index, values.max_order
        ('SV', 1)

    References:
        Rundel, D., et al. (2026). ShaplEIG: Bayesian Experimental Design for
        Shapley Value Estimation.
    """

    valid_indices: tuple[ValidShaplEIGIndices, ...] = ("SV",)

    def __init__(
        self,
        n: int,
        *,
        initial_design_size: int | None = None,
        max_candidates: int | None = 1024,
        refit: str = "every_iteration",
        warmstart: bool = False,
        show_progress: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initialize the ShaplEIG approximator.

        Args:
            n: Number of players.
            initial_design_size: Number of coalitions in the initial design
                (drawn with the coalition sampler before the BED loop starts).
                Defaults to ``n + 1``.
            max_candidates: Maximum size of the sampled candidate set the EIG
                is optimized over; games with fewer coalitions outside the
                initial design fall back to the exhaustive candidate set (all
                not-yet-evaluated coalitions). ``None`` forces the exhaustive
                candidate set, which is feasible only for ``n <= 16``.
                Defaults to ``1024``, as in the reference paper.
            refit: When to refit the GP hyperparameters: ``"every_iteration"``
                (default) or the staged schedule ``"init_64_factor_4"`` (every
                iteration for the first 64, then every 8th for 128, every
                16th for 256, every 32nd afterwards) for large budgets. The
                final surrogate is always refit.
            warmstart: If ``True``, hyperparameter refits start from the
                previous iteration's fitted hyperparameters instead of fresh
                initial values.
            show_progress: Whether to display a tqdm progress bar over the
                BED iterations.
            random_state: Random state seeding both the coalition sampling
                and the GP hyperparameter fitting. Defaults to ``None``.
        """
        # Optional dependencies of the `shapleig` extra: import in the
        # constructor (never at package import time) so that a missing extra
        # fails immediately with an actionable message.

        # ShaplEIG selects coalitions adaptively (EIG argmax), so the base
        # class's sampling configuration must not suggest otherwise. Sampling
        # enters only through the initial design and the candidate set, whose
        # internal samplers follow the protocol of the reference paper
        # (uniform coalition-size weights + pairing trick, `_coalition_sampler`).
        super().__init__(
            n=n,
            max_order=1,
            index="SV",
            min_order=0,
            sampling_weights=None,
            pairing_trick=False,
            random_state=random_state,
        )
        self.initial_design_size = (
            int(initial_design_size) if initial_design_size is not None else n + 1
        )
        self.max_candidates = int(max_candidates) if max_candidates is not None else None
        self.refit = refit
        self.warmstart = warmstart
        self.show_progress = show_progress

    def approximate(
        self,
        budget: int,
        game: Game | Callable[[CoalitionMatrix], GameValues],
        **_: dict,
    ) -> InteractionValues:
        """Run the BED loop and return posterior-mean Shapley value estimates.

        Args:
            budget: Total number of game evaluations (initial design plus one
                evaluation per BED iteration).
            game: The game to approximate, as a callable mapping a binary
                coalition matrix of shape ``(m, n)`` to ``m`` values.

        Returns:
            The estimated Shapley values as ``InteractionValues``.
        """
        # The ESP math and the GP fitting require double precision throughout
        # (float32 intermediates change fitted hyperparameters and thereby
        # selections). Enforce float64 locally and restore the user's default.
        interactions,_ = self.approximate_with_variance(budget=budget, game=game)
        return interactions

    def approximate_with_variance(
        self,
        *,
        budget: int,
        game: Game | Callable[[CoalitionMatrix], GameValues],
    ) -> tuple[InteractionValues, np.ndarray]:
        """BED loop body (runs under enforced float64, see `approximate`)."""
        if budget <= self.initial_design_size:
            msg = (
                f"Budget ({budget}) must exceed the initial design size "
                f"({self.initial_design_size})."
            )
            raise ValueError(msg)
        iterations = budget - self.initial_design_size

        # Seed the GP hyperparameter fitting; the coalition sampling is seeded
        # through `self._random_state` when the design samplers are built
        # (kept current by `set_random_state`).
        if self._random_state is not None:
            torch.manual_seed(self._random_state)

        archive_X, candidate_set = self._generate_design(iterations)
        archive_Y = self._evaluate(game, archive_X)
        baseline_value = self._baseline_value(game, archive_X, archive_Y)

        gp = self._surrogate(archive_X, archive_Y)
        cache = sm.ShaplEIGCache()

        iterator = range(iterations)
        if self.show_progress:
            iterator = tqdm(iterator, desc="ShaplEIG", unit="it")

        best_idx: int | None = None
        refit = True
        for iteration_idx in iterator:
            # --- surrogate (rebuild on refit — cold-started unless
            # `warmstart`; hyperparameters frozen otherwise) ---
            refit = self._refit_scheduled(iteration_idx)
            if iteration_idx > 0:
                if refit and not self.warmstart:
                    gp = self._surrogate(archive_X, archive_Y)
                else:
                    gp.update_data(archive_X, archive_Y)
            if refit:
                gp.fit()

            # --- EIG over the candidates; select and query ---
            utilities = self._eig(
                gp,
                cache,
                candidate_set,
                no_refit_step=(iteration_idx > 0 and not refit),
                prev_selected_idx=best_idx,
            )
            best_idx = int(torch.argmax(utilities))
            new_x = candidate_set[best_idx : best_idx + 1]
            new_y = self._evaluate(game, new_x)

            archive_X = torch.cat([archive_X, new_x], dim=0)
            archive_Y = torch.cat([archive_Y, new_y], dim=0)
            candidate_set = torch.cat([candidate_set[:best_idx], candidate_set[best_idx + 1 :]])

        # --- final surrogate (always refit) and posterior-mean SVs ---
        if self.warmstart:
            gp.update_data(archive_X, archive_Y)
        else:
            gp = self._surrogate(archive_X, archive_Y)
        gp.fit()
        gp_tensors = gp.tensors()
        K_chol = sm.psd_chol(sm.noisy_train_kernel(gp_tensors))
        A_KZX = sm.a_kzw(gp_tensors.train_X, gp_tensors.lengthscales, gp_tensors.outputscale)
        shapley_values = sm.affine_posterior_mean(A_KZX, gp_tensors, K_chol)

        # NOTE (future work, coordinated with the maintainers): the surrogate
        # posterior also provides the marginal SV variances,
        AEA = sm.aea(sm.akzza(ls, s), A_KZX, K_chol) # TODO: compute the correct AEA
        sv_variances = AEA.diagonal() * gp_tensors.emp_std**2 
        # Expose them via an `approximate_with_variance` once
        # `InteractionValues` supports uncertainty information.

        values = np.zeros(self.n + 1)
        values[0] = baseline_value
        values[1:] = shapley_values.numpy()
        interaction_lookup: dict[tuple[int, ...], int] = {(): 0}
        interaction_lookup |= {(i,): i + 1 for i in range(self.n)}
        return InteractionValues(
            values=values,
            index="SV",
            max_order=1,
            min_order=0,
            n_players=self.n,
            interaction_lookup=interaction_lookup,
            # Always flagged as estimated: even when the budget covers all
            # 2^n coalitions, the GP posterior mean is subject to numerical
            # inaccuracies (the estimator is consistent, not exact).
            estimated=True,
            estimation_budget=budget,
            baseline_value=float(baseline_value),
            target_index=self.index,
        ), sv_variances.numpy()

    # ------------------------------------------------------------------
    # building blocks
    # ------------------------------------------------------------------

    def _generate_design(self, iterations: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw the initial design and the candidate set (binary, float64).

        Both draws come from fresh, identically seeded coalition samplers
        (:meth:`_coalition_sampler`):

        - the initial design sampler draws exactly ``initial_design_size``
          coalitions — its composition therefore does not depend on the
          candidate budget;
        - the candidate sampler draws the combined budget, keeping the first
          ``candidate_size`` coalitions not already in the initial design.
        """
        init_size = self.initial_design_size
        if self.max_candidates is None:
            if self.n > 16:
                msg = (
                    "An exhaustive candidate set is infeasible for more than 16 "
                    f"players (n={self.n}); set `max_candidates` to optimize the "
                    "EIG over a sampled candidate subset instead."
                )
                raise ValueError(msg)
            candidate_size = 2**self.n - init_size
        else:
            # `max_candidates` is an upper bound: small games fall back to the
            # exhaustive candidate set (all coalitions outside the initial
            # design).
            candidate_size = min(self.max_candidates, 2**self.n - init_size)
        if candidate_size < iterations:
            msg = (
                f"Budget requires {iterations} BED iterations but only "
                f"{candidate_size} candidate coalitions are available; reduce "
                "the budget (or increase `max_candidates` if the candidate set "
                "is not already exhaustive)."
            )
            raise ValueError(msg)

        initial_sampler = self._coalition_sampler()
        initial_sampler.sample(init_size)
        initial_design = torch.tensor(initial_sampler.coalitions_matrix)

        candidate_sampler = self._coalition_sampler()
        candidate_sampler.sample(init_size + candidate_size)
        candidates = torch.tensor(candidate_sampler.coalitions_matrix)
        in_initial_design = (
            (candidates[:, None, :] == initial_design[None, :, :]).all(dim=2).any(dim=1)
        )
        candidates = candidates[~in_initial_design][:candidate_size, :]

        return initial_design.to(torch.float64), candidates.to(torch.float64)

    def _coalition_sampler(self) -> CoalitionSampler:
        """Fresh seeded sampler with the reference design protocol.

        Uniform coalition-size weights and the pairing trick, as in the
        reference paper. A fresh sampler per draw keeps repeated
        ``approximate`` calls on the same instance deterministic.
        """
        return CoalitionSampler(
            n_players=self.n,
            sampling_weights=np.ones(self.n + 1),
            pairing_trick=True,
            random_state=self._random_state,
        )

    def _surrogate(self, train_X: torch.Tensor, train_Y: torch.Tensor) -> HammingGP:
        """Fresh (cold-start) surrogate on the current archive.

        Uses the validated reference settings throughout (standardized fixed
        noise ``1e-6``, minimum lengthscale ``1e-6``, 5 fitting attempts).
        """
        return HammingGP(train_X, train_Y)

    def _refit_scheduled(self, iteration_idx: int) -> bool:
        """Whether the GP hyperparameters are refit at this iteration."""
        if self.refit == "every_iteration":
            return True
        if self.refit == "init_64_factor_4":
            if iteration_idx < 64:
                return True
            if iteration_idx < 64 + 128:
                return iteration_idx % 8 == 0
            if iteration_idx < 64 + 128 + 256:
                return iteration_idx % 16 == 0
            return iteration_idx % 32 == 0
        msg = f"Unknown refit schedule {self.refit!r}."
        raise ValueError(msg)

    def _eig(
        self,
        gp: HammingGP,
        cache: sm.ShaplEIGCache,
        candidates: torch.Tensor,
        *,
        no_refit_step: bool,
        prev_selected_idx: int | None,
    ) -> torch.Tensor:
        """EIG utilities of the candidates under the current surrogate.

        In a no-refit step the hyperparameters are unchanged: the cached
        A·K(Z,Z)·Aᵀ is reused, the column of the last queried coalition is
        appended to A·K(Z,X), and the selected candidate's column is dropped
        from A·K(Z,W).
        """
        gp_tensors = gp.tensors()
        K_chol = sm.psd_chol(sm.noisy_train_kernel(gp_tensors))

        if no_refit_step:
            if prev_selected_idx is None:
                msg = "No-refit steps require the previously selected index."
                raise RuntimeError(msg)
            new_col = sm.a_kzw(
                gp_tensors.train_X[-1:],
                gp_tensors.lengthscales,
                gp_tensors.outputscale,
            )[:, 0]
            cache.append_train_column(new_col)
            cache.drop_candidate_column(prev_selected_idx)
            A_KZX, A_KZW, AKA = cache.A_KZX, cache.A_KZW, cache.AKA
            if A_KZX is None or A_KZW is None or AKA is None:
                msg = "No-refit step requires a populated cache."
                raise RuntimeError(msg)
        else:
            A_KZX = sm.a_kzw(gp_tensors.train_X, gp_tensors.lengthscales, gp_tensors.outputscale)
            A_KZW = sm.a_kzw(candidates, gp_tensors.lengthscales, gp_tensors.outputscale)
            AKA = sm.akzza(gp_tensors.lengthscales, gp_tensors.outputscale)
            cache.A_KZX, cache.A_KZW, cache.AKA = A_KZX, A_KZW, AKA

        K_XW = sm.hamming_kernel(
            gp_tensors.train_X,
            candidates,
            gp_tensors.lengthscales,
            gp_tensors.outputscale,
        )
        B = sm.a_sigma_w(A_KZW, A_KZX, K_chol, K_XW)
        AEA = sm.aea(AKA, A_KZX, K_chol)
        noisy_var_diag = gp.posterior_variance_diag(candidates, observation_noise=True)
        return sm.shapleig_utilities(AEA, B, noisy_var_diag, gp_tensors.emp_std)

    @staticmethod
    def _evaluate(
        game: Game | Callable[[CoalitionMatrix], GameValues],
        coalitions: torch.Tensor,
    ) -> torch.Tensor:
        """Query the game on binary coalitions; values as a ``(m, 1)`` tensor."""
        values = game(coalitions.numpy().astype(bool))
        return torch.as_tensor(values, dtype=torch.float64).reshape(-1, 1)

    def _baseline_value(
        self,
        game: Game | Callable[[CoalitionMatrix], GameValues],
        archive_X: torch.Tensor,
        archive_Y: torch.Tensor,
    ) -> float:
        """Empty-coalition value (the game's baseline value).

        The coalition sampler's border trick always places the empty coalition
        in the initial design, so its value is read from the archive; querying
        the game directly is a safety net only.
        """
        empty_rows = (archive_X.sum(dim=1) == 0).nonzero()
        if len(empty_rows) > 0:
            return float(archive_Y[int(empty_rows[0]), 0])
        return float(self._evaluate(game, torch.zeros((1, self.n)))[0, 0])
