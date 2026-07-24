# 15. Explanations are games

Date: 2026-07-24

## Status

Accepted. Supersedes the output-side vocabulary of earlier ADRs: explanation
arrays as a species (their storage and lookup contracts) are replaced by the
game currency below. The sampling core of ADR 0013/0014 (stateless samplers,
banked budgets, evidence-owned identity) is unchanged and slots underneath.

## Context

Every explanation the library produces is, mathematically, a value function:
Shapley values are the coefficients of the best additive game under the
Shapley measure (Charnes), FBII of the best k-additive game under the uniform
measure (Hammer–Holzman), order n recovers the game exactly (Moebius). v1's
ProxySHAP had already built the consequences by hand — game subtraction,
exact read-outs of fitted surrogates, evidence shared across estimators by
seed discipline — because the library had no primitives for them. A separate
explanation-array species forced every such composition through conversions
and made "explain the explanation", residual correction, and fidelity
measurement foreign operations.

## Decision

One currency: **explainers map games to games.**

- A **Game** is a value function; the boundary also accepts plain dense
  masks. Games carry algebra (sum, difference, scaling).
- A **Basis** is a value object owning its atoms (Moebius/AND,
  Co-Moebius/OR, Fourier/XOR); extension is implementing the protocol,
  never registering a string. A **BasisGame** is a game known through
  coefficients on a declared basis — two planes on one object: calling
  evaluates the surrogate, indexing reads a coefficient. Sparse storage is
  fewer terms, not a separate kind. The empty interaction is an ordinary
  slot whose meaning each index family declares (baseline, fitted
  intercept, or absent).
- An **Estimate** is a BasisGame with provenance: evidence, bank, index,
  producer fingerprint, optional variance. Estimates are inert; process
  verbs (estimate/refine/at_evidence) live on frozen policies, and a policy
  refuses to continue another policy's carry. Everything not in the
  evidence is recomputed from it or checkpointed with it — that rule is
  what makes budget splits, rollback, and replay exact, stalls included.
- Exact and tree explainers return the same currency: an exact estimate
  carries the full sweep as evidence (spending visible), a tree estimate
  carries none.
- The center is the pair **(game, measure)**: projection indices are
  subspace-plus-measure, towers compose only under a shared measure,
  sampling laws target measures, fidelity is distance under one. The
  gradient bridge generalizes the pair to (game, extension): the extension
  is part of a gradient method's identity (Owen's theorem is the
  multilinear special case).

## Consequences

Composition is native: residual games, explanation addition, fit-then-read
with free corrections (ProxySHAP as a recipe), third-party policies handing
their own coefficients to the carry, and fidelity as a library verb. The
explanation-array machinery (dense and sparse species, orientation, batched
array lookups) is deleted; coefficient reads return host float64 (which game
a coefficient vector describes is semantic exactness). Costs accepted: the
two call planes must stay sharply distinguished on one object, and index
families owe an explicit empty-slot/surrogate-semantics declaration. Design
record with evidence: `docs/plans/issue-14-value-functions-api.md`.
