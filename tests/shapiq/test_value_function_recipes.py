"""Acceptance recipes for the value-function rebuild (issue 14).

Three compositions that must fall out of library primitives: the
gradient bridge (Owen's theorem live), the ProxySHAP recipe (fit a
surrogate, correct on the residual, pay zero extra evaluations), and an
active-learning policy built entirely on the public carry contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest

from shapiq import (
    SV,
    CallableGame,
    Estimate,
    Regression,
    banzhaf_values,
    fidelity,
    fit_game,
    integrated_gradients,
    multilinear_diagonal_gradient,
    shapley_values,
    to_basis,
    uniform_measure,
)
from shapiq.coalitions import DenseCoalitionArray
from shapiq.games import MoebiusBasis, interaction_terms
from shapiq.sampling import EmptyState, SamplingState

N_PLAYERS = 8


def structured_game(n_players: int = N_PLAYERS) -> CallableGame:
    """Order-2 structure plus a redundancy block and order-4 synergy."""
    rng = np.random.default_rng(7)
    weights = jnp.asarray(rng.normal(size=n_players), dtype=jnp.float32)
    pairs = jnp.asarray(rng.normal(size=(n_players, n_players)) * 0.5, dtype=jnp.float32)

    def fn(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        redundancy = 1.5 * masks[..., [2, 4, 6]].max(axis=-1)
        synergy = 0.8 * masks[..., [1, 3, 5, 7]].prod(axis=-1)
        quadratic = masks @ weights + 0.5 * jnp.einsum("...i,ij,...j->...", masks, pairs, masks)
        return quadratic + redundancy + synergy

    return CallableGame(fn=fn, n_players=n_players)


# ---------------------------------------------------------------- gradient bridge


def test_owen_integrated_gradients_on_the_multilinear_extension_are_shapley():
    exact = to_basis(structured_game(), MoebiusBasis())
    owen = integrated_gradients(
        lambda t: multilinear_diagonal_gradient(exact, t),
        N_PLAYERS,
        steps=256,
    )
    assert np.allclose(owen, shapley_values(exact), atol=1e-6)


def test_center_gradient_of_the_multilinear_extension_is_banzhaf():
    exact = to_basis(structured_game(), MoebiusBasis())
    center = multilinear_diagonal_gradient(exact, 0.5)
    assert np.allclose(center, banzhaf_values(exact), atol=1e-9)


def test_a_different_extension_of_the_same_game_attributes_differently():
    # add a bump vanishing at every vertex: same game, new extension
    exact = to_basis(structured_game(), MoebiusBasis())
    bump = 1.2

    def bumped(t: float) -> np.ndarray:
        gradient = multilinear_diagonal_gradient(exact, t)
        gradient[0] += bump * (1 - 2 * t) * t  # d/dz0 of C * z0 (1 - z0) z1
        gradient[1] += bump * t * (1 - t)
        return gradient

    attribution = integrated_gradients(bumped, N_PLAYERS, steps=512)
    truth = shapley_values(exact)
    # completeness still holds, the attribution moved by exactly -C/6, +C/6
    assert np.allclose(attribution.sum(), truth.sum(), atol=1e-4)
    assert attribution[0] - truth[0] == pytest.approx(-bump / 6, abs=1e-4)
    assert attribution[1] - truth[1] == pytest.approx(+bump / 6, abs=1e-4)


# ---------------------------------------------------------------- proxy recipe


def test_proxyshap_recipe_from_primitives_costs_zero_extra_evaluations():
    n = 10  # 1024 coalitions; the evidence must NOT identify the game
    calls = []
    base = structured_game(n)

    def recording(coalitions):
        calls.append(int(coalitions.to_dense().shape[-2]))
        return base(coalitions)

    game = CallableGame(fn=recording, n_players=n)
    exact_sv = shapley_values(to_basis(base, MoebiusBasis()))

    policy = Regression(game, SV(), random_state=0, deduplicate=True)
    direct = policy.estimate(160)
    evaluations_after_direct = sum(calls)

    # -- the recipe: fit, subtract, re-solve on rebased evidence ---------
    evidence = direct.evidence
    masks = jnp.asarray(evidence.coalitions.to_dense())
    proxy = fit_game(np.asarray(masks), np.asarray(evidence.values), n, order=2)
    residual_values = jnp.asarray(evidence.values) - jnp.asarray(
        proxy._host_values(np.asarray(masks, dtype=bool)),
    )
    rebased = SamplingState(
        coalitions=evidence.coalitions,
        values=residual_values,
        target_shape=(),
    )
    correction = Regression(base - proxy, SV(), random_state=0).at_evidence(rebased, bank=0)
    combined = shapley_values(proxy) + np.array(
        [float(correction[(player,)]) for player in range(n)],
    )
    # ---------------------------------------------------------------------

    assert sum(calls) == evaluations_after_direct  # the correction was free
    direct_error = np.abs(np.array([float(direct[(p,)]) for p in range(n)]) - exact_sv).max()
    combined_error = np.abs(combined - exact_sv).max()
    assert combined_error < direct_error  # the proxy soaked up real structure
    surrogate = proxy + correction.as_game()
    assert fidelity(base, surrogate, uniform_measure(n)) > 0.5


# ---------------------------------------------------------------- active learning


@dataclass(frozen=True)
class ToyBED:
    """A third-party active policy built on the public carry contract.

    Conjugate Bayesian linear model on the moebius basis; proposals
    maximize the trace reduction of the Shapley-value posterior. All
    state is derived from the carried evidence, so split invariance and
    exact replay hold by construction.
    """

    game: CallableGame
    order: int = 2
    sigma: float = 0.05
    tau: float = 2.0
    n_candidates: int = 100
    candidate_key: int = 99

    def estimate(self, budget: int) -> Estimate:
        fresh = Estimate(evidence=EmptyState(), bank=0, n_players=self.game.n_players, view=None)
        return self.refine(fresh, budget)

    def refine(self, carry: Estimate, budget: int) -> Estimate:
        n = self.game.n_players
        evidence, bank = carry.evidence, carry.bank + budget
        if not isinstance(evidence, SamplingState) and bank >= 2:
            seeds = jnp.stack([jnp.zeros(n, dtype=bool), jnp.ones(n, dtype=bool)])
            evidence = SamplingState(
                coalitions=DenseCoalitionArray(seeds),
                values=jnp.asarray(self.game(DenseCoalitionArray(seeds))),
            )
            bank -= 2
        while bank > 0 and isinstance(evidence, SamplingState):
            candidates = self._candidates(evidence)
            if candidates.shape[0] == 0:
                break  # support exhausted: the remainder stays banked
            pick = candidates[self._propose(evidence, candidates)][None, :]
            values = jnp.asarray(self.game(DenseCoalitionArray(jnp.asarray(pick))))
            evidence = evidence.append(DenseCoalitionArray(jnp.asarray(pick)), values)
            bank -= 1
        return self.at_evidence(evidence, bank)

    def at_evidence(self, evidence, bank: int) -> Estimate:
        if not isinstance(evidence, SamplingState):
            return Estimate(evidence=evidence, bank=bank, n_players=self.game.n_players, view=None)
        _, mean, cov, sv_map = self._posterior(evidence)
        sv_cov = sv_map @ cov @ sv_map.T
        variance = {
            frozenset([player]): float(sv_cov[player, player])
            for player in range(self.game.n_players)
        }
        estimate = Estimate(
            evidence=evidence,
            bank=bank,
            n_players=self.game.n_players,
            view=None,
            variance=variance,
        )
        object.__setattr__(estimate, "_posterior_mean", mean)
        return estimate

    def _posterior(self, evidence: SamplingState):
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

    def _candidates(self, evidence: SamplingState) -> np.ndarray:
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

    def _propose(self, evidence: SamplingState, candidates: np.ndarray) -> int:
        terms, _, cov, sv_map = self._posterior(evidence)
        design = np.asarray(MoebiusBasis().atoms(candidates, terms, xp=np))
        projected = design @ (sv_map @ cov).T
        denominator = self.sigma**2 + np.einsum("md,de,me->m", design, cov, design)
        return int(np.argmax((projected**2).sum(axis=1) / denominator))


def bed_game() -> CallableGame:
    rng = np.random.default_rng(11)
    weights = jnp.asarray(rng.normal(size=N_PLAYERS), dtype=jnp.float32)
    pairs = {(0, 1): 1.7, (1, 2): -1.1, (4, 5): 2.3, (6, 7): -0.6}

    def fn(coalitions):
        masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)
        total = masks @ weights
        for (left, right), strength in pairs.items():
            total = total + strength * masks[..., left] * masks[..., right]
        return total

    return CallableGame(fn=fn, n_players=N_PLAYERS)


def sv_posterior(estimate: Estimate) -> np.ndarray:
    mean = estimate._posterior_mean
    n = estimate.n_players
    values = np.zeros(n)
    for coefficient, term in zip(mean, interaction_terms(n, 2), strict=True):
        for player in term:
            values[player] += coefficient / len(term)
    return values


def test_active_policies_on_the_carry_contract_are_split_invariant():
    policy = ToyBED(bed_game())
    whole = policy.estimate(40)
    split = policy.refine(policy.estimate(15), 25)
    assert whole.evidence == split.evidence
    assert np.allclose(sv_posterior(whole), sv_posterior(split), atol=0, rtol=0)
    # rollback and replay: derive-from-evidence makes resume exact
    exact_truth = shapley_values(to_basis(bed_game(), MoebiusBasis()))
    assert np.abs(sv_posterior(whole) - exact_truth).max() < 0.2
    # uncertainty is a capability: posterior variance shrinks with budget
    early = policy.estimate(15)
    late = policy.refine(early, 25)
    early_std = np.mean([np.sqrt(v) for v in early.variance.values()])
    late_std = np.mean([np.sqrt(v) for v in late.variance.values()])
    assert late_std < early_std


def test_active_policies_bank_when_the_candidate_pool_runs_dry():
    policy = ToyBED(bed_game(), n_candidates=25)
    estimate = policy.estimate(500)
    assert estimate.bank > 0  # banked, not dropped
    assert estimate.bank + (estimate.evidence.n_samples - 2) == 500 - 2
    assert estimate.evidence.n_samples <= 2 + 25
