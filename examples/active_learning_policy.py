"""Playground for a third-party active-learning policy on the carry contract.

Run with:
    uv run python examples/active_learning_policy.py

Nothing here subclasses a shipped policy. ``ToyBED`` is built entirely on
the public currency — evidence in, ``Estimate(surrogate, Provenance)``
out — and honors the one rule that makes budgets composable: everything
not in the evidence is recomputed from it. In return it gets, for free,
what the shipped policies get: split-invariant budgets, exact resume from
any carry, banked remainders when its candidate pool runs dry, and a
variance capability riding the provenance.

The model is a conjugate Bayesian linear fit on the Moebius basis;
proposals pick the candidate coalition that most shrinks the posterior
trace of the Shapley values (a toy Bayesian experimental design loop).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from shapiq import (
    BasisGame,
    CallableGame,
    DenseCoalitionArray,
    Estimate,
    MoebiusBasis,
    NoEvidence,
    Provenance,
    SampledEvidence,
    shapley_values,
    to_basis,
)
from shapiq.games import interaction_terms

if TYPE_CHECKING:
    from jax import Array

    from shapiq.sampling import Evidence

N_PLAYERS = 8


@dataclass(frozen=True)
class ToyBED:
    """A frozen third-party policy: verbs on the outside, evidence inside."""

    game: CallableGame
    order: int = 2
    sigma: float = 0.05
    tau: float = 2.0
    n_candidates: int = 100
    candidate_key: int = 99

    def estimate(self, budget: int) -> Estimate:
        fresh = self.at_evidence(NoEvidence(), 0)
        return self.refine(fresh, budget)

    def refine(self, carry: Estimate, budget: int) -> Estimate:
        n = self.game.n_players
        evidence, bank = carry.evidence, carry.bank + budget
        if not isinstance(evidence, SampledEvidence) and bank >= 2:
            seeds = jnp.stack([jnp.zeros(n, dtype=bool), jnp.ones(n, dtype=bool)])
            evidence = SampledEvidence(
                coalitions=DenseCoalitionArray(seeds),
                values=jnp.asarray(self.game(DenseCoalitionArray(seeds))),
            )
            bank -= 2
        while bank > 0 and isinstance(evidence, SampledEvidence):
            candidates = self._candidates(evidence)
            if candidates.shape[0] == 0:
                break  # candidate pool exhausted: the remainder stays banked
            pick = candidates[self._propose(evidence, candidates)][None, :]
            values = jnp.asarray(self.game(DenseCoalitionArray(jnp.asarray(pick))))
            evidence = evidence.append(DenseCoalitionArray(jnp.asarray(pick)), values)
            bank -= 1
        return self.at_evidence(evidence, bank)

    def at_evidence(self, evidence: Evidence, bank: int) -> Estimate:
        n = self.game.n_players
        if not isinstance(evidence, SampledEvidence):
            return Estimate(
                BasisGame(MoebiusBasis(), {}, n),
                Provenance(evidence=evidence, bank=bank, shortfall="no evidence yet"),
            )
        terms, mean, cov, sv_map = self._posterior(evidence)
        sv_cov = sv_map @ cov @ sv_map.T
        variance = {frozenset([player]): float(sv_cov[player, player]) for player in range(n)}
        # a third-party policy hands its own coefficients straight to the carry
        surrogate = BasisGame(
            MoebiusBasis(), None, n, terms=terms, values=np.asarray(mean, dtype=np.float64),
        )
        return Estimate(surrogate, Provenance(evidence=evidence, bank=bank, variance=variance))

    def _posterior(
        self,
        evidence: SampledEvidence,
    ) -> tuple[tuple[frozenset[int], ...], np.ndarray, np.ndarray, np.ndarray]:
        n = self.game.n_players
        terms = interaction_terms(n, self.order)
        masks = np.asarray(evidence.coalitions.to_dense(), dtype=bool)
        design = np.asarray(MoebiusBasis().atoms(masks, terms, xp=np))
        precision = design.T @ design / self.sigma**2 + np.eye(len(terms)) / self.tau**2
        cov = np.linalg.inv(precision)
        mean = cov @ design.T @ np.asarray(evidence.values, dtype=np.float64) / self.sigma**2
        sv_map = np.zeros((n, len(terms)))
        for column, term in enumerate(terms):
            for player in term:
                sv_map[player, column] = 1.0 / len(term)
        return terms, mean, cov, sv_map

    def _candidates(self, evidence: SampledEvidence) -> np.ndarray:
        n = self.game.n_players
        rows, seen, local = [], set(evidence.key_index()), set()
        for unit in range(self.n_candidates):
            rng = np.random.default_rng((self.candidate_key, unit))
            size = int(rng.integers(1, n))
            row = np.zeros(n, dtype=bool)
            row[rng.choice(n, size=size, replace=False)] = True
            key = np.packbits(row).tobytes()
            if key not in seen and key not in local:
                local.add(key)
                rows.append(row)
        return np.array(rows, dtype=bool) if rows else np.empty((0, n), dtype=bool)

    def _propose(self, evidence: SampledEvidence, candidates: np.ndarray) -> int:
        terms, _, cov, sv_map = self._posterior(evidence)
        design = np.asarray(MoebiusBasis().atoms(candidates, terms, xp=np))
        projected = design @ (sv_map @ cov).T
        denominator = self.sigma**2 + np.einsum("md,de,me->m", design, cov, design)
        return int(np.argmax((projected**2).sum(axis=1) / denominator))


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    weights = jnp.asarray(rng.normal(size=N_PLAYERS), dtype=jnp.float32)
    pairs = {(0, 1): 1.7, (1, 2): -1.1, (4, 5): 2.3, (6, 7): -0.6}

    def game_value(coalitions) -> Array:
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        total = masks @ weights
        for (left, right), strength in pairs.items():
            total = total + strength * masks[..., left] * masks[..., right]
        return total

    game = CallableGame(fn=game_value, n_players=N_PLAYERS)
    policy = ToyBED(game)

    # split invariance: the proposal is memoryless given the carry
    whole = policy.estimate(40)
    split = policy.refine(policy.estimate(15), 25)
    print(f"evidence identical, whole vs 15+25 split: {whole.evidence == split.evidence}")

    # the carry IS a moebius game: exact read-outs apply directly
    truth = shapley_values(to_basis(game, MoebiusBasis()))
    posterior_sv = shapley_values(whole)
    print(f"max |posterior SV - exact SV| at budget 40: "
          f"{np.abs(posterior_sv - truth).max():.3f}")

    # uncertainty is a capability riding the provenance
    early = policy.estimate(15)
    late = policy.refine(early, 25)
    early_std = np.mean([np.sqrt(v) for v in early.variance.values()])
    late_std = np.mean([np.sqrt(v) for v in late.variance.values()])
    print(f"mean posterior std, budget 15 -> 40: {early_std:.4f} -> {late_std:.4f}")

    # a dry candidate pool banks the remainder instead of dropping it
    small_pool = ToyBED(game, n_candidates=25)
    exhausted = small_pool.estimate(500)
    print(f"candidate pool of 25: spent {exhausted.evidence.n_samples} evaluations, "
          f"banked {exhausted.bank}")
