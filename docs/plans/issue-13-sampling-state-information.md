# Issue 13 — Sampling-state information density

Status: investigated (2026-07-21), direction proposed, not started. Scratchpad experiments:
`exp1_state_info.py`, `exp2_counts_and_law.py` (run against a frozen copy of `a3d0f529`).

## Problem

`SamplingState` records coalitions, values, target shape, and history cut points — nothing
else. Estimator families on the roadmap (SHAP-IQ, SVARM-IQ, stratified sampling, unbiased
KernelSHAP) are all Horvitz-Thompson-shaped: they need to know *how often* each coalition
was drawn and *with what probability*. v1's `CoalitionSampler` carried exactly this —
`coalitions_counter`, `coalitions_probability` (size prob × in-size prob, with log-space
variants for large `n`), `is_coalition_size_sampled` (the border trick's exact-vs-sampled
flag) — and both v1 estimator families consumed it through one derived quantity,
`sampling_adjustment_weights = counter / (probability × N)`.

## What the investigation established

By the fold-in determinism design, `(state, sampler)` jointly contain **all** of this
information already — nothing needs to be *stored*. What is missing is addressability.
Three channels, three current hiding places:

1. **Multiplicity** — the stream keeps repeats: a deduplicated run with budget 120 stored
   248 rows (values bit-identical across duplicates), so counts are derivable by grouping.
   Measured: 6.7 ms at 20k samples × 50 players, 27 ms at 50k × 200 — cheap enough for a
   derived, cached accessor. Today the dedup machinery computes this grouping and keeps it
   as `_dedup_keys` on the *approximator* (with the branch-copy token dance), not on the
   state where other consumers could reach it.
2. **The sampling law** — analytic sampler knowledge (`SizeKernelSampler._sizes` +
   `_size_probabilities`, `ProductKernelSampler.p`), but private, and **hidden by
   composition**: `PairedSampler` does not forward it, and pairing *transforms* it
   (marginal law becomes `q̃(S) = (q(S) + q(S̄))/2`; the Shapley kernel is size-symmetric
   so there `q̃ = q`). Any public law channel must compose through wrappers, not forward.
3. **Stratum/exactness structure** — "rows 0..n_seed−1 are exact" is positional
   convention plus `sampler.n_seed_samples`. v1's border trick (enumerate cheap sizes
   exactly, sample the rest) generalizes this seed block and is a real variance win v2
   currently cannot express.

## Measured evidence

- **Unique+count-weighted WLS ≡ full-stream OLS**: max coefficient difference 3.1e-15,
  with 1998 → 592 rows (n=10, budget 2000). The Regression solve could always run on the
  unique view — never worse, smaller at small-to-mid `n` where repeats are common.
- **HT reweighting to a different measure** (the v1 adjustment-weights pattern, the
  skeleton of every SHAP-IQ-style port): with counts + the sampler's law, estimating a
  uniform-measure mean from Shapley-kernel samples converges (err 0.100 → 0.011 → 0.005 at
  budgets 500/5k/20k); the naive stream mean stays biased at 0.12. Writing this today
  requires `sampler.sampler._sizes` — two layers of privacy.

## Design proposal (detailed 2026-07-21, not built)

Store nothing new; make the existing information addressable.

### `SamplingState.unique(n_samples=None) -> UniqueView`

`UniqueView` = (coalitions: CoalitionArray in first-occurrence order, first_indices,
counts, inverse — host int arrays, per the host-side-index-math law). Decisions:

- **Storage mirrors the chunk design**: per-chunk packed-bit key lists computed once and
  shared structurally on `append` (like the value chunks); the merged first-index dict and
  counts built lazily per state and cached like `_materialized()`. No cross-object
  mutation, no validity token — immutability replaces the `_dedup_keys` token dance.
- **Prefix parameter** because consumers slice pending rows off the tail
  (`usable = n_samples - pending`); unique over the full stream then filtering would count
  pending repeats. Prefix grouping is a host slice of the flat key list.
- **First-occurrence order** makes the view budget-split invariant (stream bit-identity
  transfers) — a pinnable property.
- **Shared-target precondition** as for dedup, same teaching error.
- Consumers: dedup (`_known_coalitions`/`_dedup_keys` on `EvidenceApproximator` deleted,
  dedup reads the state's index); Regression unique-solve (√count row scaling — proven
  ≡ to 3e-15, 3.4x smaller at n=10); stall detection; effective sample size
  `(Σc)²/Σc²` for convergence diagnostics.

### `LawfulSampler` capability protocol

`log_probability(coalitions) -> Array`, a `@runtime_checkable` protocol in `sampling`.
Contract semantics:

- **The marginal law of one sampled stream position, after wrapper transformations** —
  the quantity HT estimators sum over. Wrappers transform, never forward:
  `PairedSampler.log_probability(c) = logaddexp(inner(c), inner(~c)) − log 2`.
- **Log-space only, one method** (v1's ~1022-player lesson); outside support → `−inf`,
  not an error. Seed rows sit outside the law (deterministic; sliced off by position).
- **Absence is meaningful**: kernel/product samplers implement
  (`size_prob[|S|]/C(n,|S|)`; `p^|S|(1−p)^(n−|S|)`); permutation-walk samplers do not
  (unit-correlated marginals; walk estimators need no law). Estimators capability-check —
  register where the variance lives, capability-check the other axis.
- v1's two-factor split (size × in-size) deferred to a `StratifiedLaw` sibling when
  SVARM-IQ brings its stratum needs — protocol lands with its first consumer.

### Exactness strata (design note only)

"Rows 0..n_seed−1 are exact" stays positional convention + `sampler.n_seed_samples`.
When the border trick is ported (exhaustive cheap sizes, v1's `is_coalition_size_sampled`),
rows partition into exact/sampled strata and the law contract must condition on the
sampled stratum; the sampler's schedule answers membership arithmetically. No stored flags.

### Sequencing

1. Unique view: small, independent; land **before or with the history rework** (both
   remove parallel structures around `_state.py`; one neighborhood, one pass).
2. Law protocol: **with the sampled k-SII port as first consumer** (v1's sampled k-SII is
   the SHAP-IQ estimator family; counts + law are its entire input beyond values — the HT
   experiment here is the skeleton). Per the protocol-lands-with-consumer rule.
3. Orthogonal to value-layout canonicalization (coalitions, not values).

Open naming: `unique()` vs `multiplicities`; `LawfulSampler` vs `SamplingLaw`; "sampling
law" as a CONTEXT.md glossary entry (leaning yes — the contract leans on the term).

Explicitly rejected: eagerly storing counts or probabilities in the state (a parallel
structure with lockstep invariants — the disease the history rework is curing; the law is
analytic, and rollback/history slicing would otherwise have to maintain stored copies).

### Acceptance criteria

- `unique()` counts match an `np.unique` oracle; view is budget-split invariant; prefix
  variant excludes pending repeats; dedup suite green with the approximator-side key
  machinery deleted; explanation estimates bit-identical before/after.
- `LawfulSampler`: law sums to 1 over support (exact enumeration, small n); empirical
  frequencies converge to the law; `PairedSampler` symmetrization pinned; `−inf` outside
  support.
- ADR when it lands: "sampling information is addressable, not stored" (state owns
  identity, sampler owns law, schedule owns strata).

## Consumers this unlocks

KernelSHAP unique-solve (smaller WLS, identical estimate); SHAP-IQ / SVARM-IQ /
stratified / unbiased-KernelSHAP ports (counts + law are their entire input beyond
values); convergence diagnostics (`std_error`/CI backlog item — wants counts and law
weights); dedup stall detection (reads the same key index).
