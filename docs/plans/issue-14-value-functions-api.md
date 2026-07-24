# Issue 14 — Value functions all the way down (the "world A" rebuild)

Status: design converged in discussion 2026-07-24; validated by a scratchpad prototype
(`worlda.py` + `demo.py`, session-ephemeral — every number in the Evidence appendix comes
from it; say the word to vendor the scripts into the repo). Rebuild not started. This
document is the working record: thesis, decisions, evidence, landmines, and the arc plan.

## Thesis

A clean-slate re-centering of the API on one closure property: **explainers map value
functions to value functions.** A model behind a masker is a game v; an explanation is
again a game — a lossy, interpretable compression of v whose parameters are the
attribution/interaction values. The Shapley value is the best additive game under the
Shapley measure (Charnes); FBII is the best k-additive game under the uniform measure
(Hammer–Holzman); order n recovers v exactly (Möbius). Input type = output type buys
composition (residual chains, explain-the-explanation), comparison (two methods = two
games, their distance is computable), and fidelity as a first-class verb.

The true center is the **pair (game, measure)**. The same object appeared three times in
the v2 arc under different names — the regression kernel, the sampling law, the
projection inner product — and this design names it once. Corollaries proven in the
prototype: the projection tower composes only under a shared measure (R2), and the
gradient bridge adds a third instance: *the extension is part of the gradient method*
(R6), like the measure is part of the index and the chunk policy is part of the game's
numeric identity.

## Decisions

- **D1 — Explanations ARE games** (option A; option B "explanations carry a `.game`"
  rejected: one concept, one owner). "Is a game" means *satisfies the Game protocol* —
  structural typing, composition inside is invisible.
- **D2 — Two planes on one object**: `phi[T]` reads a coefficient, `phi(masks)`
  evaluates the surrogate. They differ by the intercept and lower orders; both planes
  need sharp names and teaching errors. Settled early because every user will conflate
  them once.
- **D3 — Three tiers, hard-typed**: extensional (eval-only, costly, opaque) →
  intensional (readable finite parametrization) → credal (posterior over intensional).
  The subtype axis is "can I read your coefficients", never "are you cheap". Plots,
  aggregations, exact solvers type on intensional. If this split goes muddy the design
  collapses into capability soup.
- **D4 — Intensional = (basis, coefficients, storage)**. Bases are declared: Möbius
  (AND/synergy), co-Möbius (OR/redundancy — the dual game's Möbius), Fourier–Walsh
  (XOR/parity — orthonormal under uniform; Banzhaf values are its degree-1
  coefficients), interaction transforms (Shapley, with Bernoulli-number inverses; the
  kADD solve already lives in this basis). Sparse vs dense storage is orthogonal to
  basis. Sparsity is basis-relative: OR(4) is 1 coefficient in co-Möbius and a 15-term
  alternating smear in Möbius (R1) — basis choice is a modeling statement (synergy vs
  redundancy vs parity). Structured parametrizations (tree leaves, GP posteriors) are
  intensional without being linear bases; conversion verbs exist where cheap (the tree
  cext's (−1)^|W| read-out IS a to-Möbius conversion).
- **D5 — Uncertainty is a capability**, not a subspecies: `variance` present when the
  estimator provides it (GP posterior, MC variance), absent otherwise.
- **D6 — Math on the value, process on the policy** (the optax shape). The estimate is
  inert: a game plus a provenance *record*. All process verbs live on the frozen
  approximator policy: `policy.estimate(v, budget, key)`, `policy.refine(est, budget)`,
  `policy.rollback(est, steps)`, `policy.from_evidence(...)`. Fluent
  `est.refine(...)` rejected — it needed an empty-estimate scaffold just to exist
  (the smell of verbs on the wrong noun) and hid the policy inside the result.
- **D7 — Approximators become fully stateless** (finishing what ADR 0014 started for
  samplers): frozen config bundles owning sampler + family expansion + solve as
  *identity, not history*. All evolving state rides in the returned estimate. The
  functional shape of estimation is literal: `refine : (estimate, budget) → estimate` —
  a fold with the explanation as carry.
- **D8 — Provenance rule**: provenance = evidence + checkpointed non-derivables;
  everything else is recomputed. (The quiet-counter replay bug was this rule violated;
  ShaplEIG's warmstart hyperparameters are the same class of state at scale.) Split
  invariance generalizes from unit-index arithmetic to: **the proposal is memoryless
  given the carry** — which is what makes active learning a citizen, not an exception.
- **D9 — Keys enter once, ride in provenance** (jaxy in idiom, not in `lax.scan`):
  explicit key at entry, fold-in derivation per unit, recorded so `refine` continues
  the stream without re-threading randomness. The outer loop is host Python and never
  jits — dedup is dict work, games call arbitrary frameworks; don't chase it.
- **D10 — Chunking is a game transformer**: `ChunkedGame(v, batch_size)`, the same
  composition move as `PairedSampler` one tier up. The canonical chunk shape is what
  makes bit-identity true in float32 (XLA values are batch-shape-dependent — the
  review's dedup-split finding), so the evaluation policy is honestly part of the
  game's numeric identity. Exact sweeps and sampling share the wrapper;
  `ChunkedMaskedPredictor` slims to torch policy beneath it.
- **D11 — Estimator taxonomy: three roles.** *Estimate* (Monte-Carlo coefficient
  estimators: walks, kernels, SHAP-IQ), *fit-then-read* (learn an intensional
  surrogate, read it exactly: proxy models, SPEX, trees, GPs), *steer* (policies that
  read the fitted surrogate to propose the next evaluation: BED/EIG). ProxySHAP =
  fit-then-read + residual-corrected estimate; ShaplEIG = fit-then-read + steer.
- **D12 — The gradient bridge is an adapter, not a bolt-on.** `Extension` = a
  differentiable function on the cube agreeing with v at the vertices. Two
  constructors: the model's own extension (autodiff — the deep-learning door) and the
  multilinear extension (analytic for intensional games, sampled for extensional).
  Gradient explainers are path policies over extensions producing AdditiveGames — same
  closure type, budget = gradient evaluations, second evidence species (path points +
  gradients) under the unchanged provenance rule. Owen's theorem makes IG-on-MLE
  exactly SV (R6); the IG-vs-SV gap becomes a computable extension-artifact diagnostic
  no other library offers.

## Ownership map

| noun | owns | pointedly does not own |
| --- | --- | --- |
| extensional game | evaluation + boundary metadata; transformers: chunked, residual (v − v̂), scaled | any memory of being evaluated |
| masker | model + data → game | (unchanged) |
| sampler | key + draw law, `draws(unit_indices)` pure; active variants read the carry: `propose(key, estimate, candidates)` | state, budgets, walks |
| index | identity, subspace, **measure**, surrogate semantics (derivative family declares its aggregation) | estimation machinery |
| approximator policy | frozen: sampler, expansion, solve, evaluation policy + ALL process verbs | state of any kind |
| estimate (the carry) | surrogate coefficients + provenance record: evidence, source ref, key, bank, ±variance; math verbs: `()`, `[]`, project, to_basis, fidelity, arithmetic, detach | the loop (process re-enters the policy) |
| evidence | coalitions/values (or path points/gradients), cuts, identity (`unique`/`key_index`) — the sufficient statistic for exact resume | interpretation (a *sample of* a game, not a game) |

The one heavy fact in the record: the source-game reference (enables two-argument
`refine`; a live model rides along). `detach()` sheds it. Named trade, accepted.

## What survives from the current line, unchanged in substance

Stateless samplers with fold-in draws and laws (ADR 0014), banked whole-unit budgets
and split invariance, `SamplingState` with chunked append / cuts / state-owned identity
(`packed_keys`/`unique`/`key_index` — the candidate-set maintenance of active learning
is the same set-difference), canonical value layout (vmap-friendly vectorization over
targets/instances via `share_samples`), maskers-are-math, indices as instances (ADR
0012), the family-registry pattern, teaching errors, the tree stack and its cexts, the
review round's lessons (Tier-1 fixes are absorbed by D8/D10 — do not fix twice on the
old line if the rebuild proceeds). What dissolves: `Explainer` as a stateful noun,
`ExplanationArray` as a species separate from games, ProxySHAP/RegressionMSR as
monolith classes (they become recipes), v1's shared-seed discipline (explicit
`evidence=` passing).

## The v1 census (nothing contradicts the ontology)

permutation SV/SII/STII → estimate-role walk families (landed). KernelSHAP / FSII /
FBII / kADD → projection estimators, literally `project(order, measure)`.
SHAP-IQ / UnbiasedKernelSHAP / SVARM-IQ → Horvitz-Thompson consumers of counts × law
(issue-13 built their inputs). Owen / stratified → sampling-law policies (need
per-stratum addressability = exactness-strata row). ProxySHAP / RegressionMSR →
four-line recipe (R3). SPEX / ProxySPEX → fit-then-read with sparse Fourier surrogates
(forces D4). OddSHAP → paired sampling + odd-Fourier solve (pairing kills even
parities — why pairing helps SV). ShaplEIG → the active-learning archetype (forces D5,
validates D8; its candidate cap becomes a banked remainder, its variance return becomes
the capability, its four loose loop variables become one carry). v1's stateful
`CoalitionSampler` → dead, already superseded.

## Landmines (open, named, not yet settled)

1. The two call planes (D2) need their final spelling and teaching errors.
2. The extensional/intensional split must be enforced by types from day one.
3. Index metadata regrows (surrogate semantics + measure declarations) — keep it to
   load-bearing members; the derivative family's game semantics require an explicit
   aggregation choice (k-SII/STII), which the API states rather than hides.
4. No orthonormal basis exists for the Shapley measure (degenerate kernel at ∅/N —
   constrained least squares, not coefficient read-off); the uniform/Banzhaf column is
   the luxurious one. Reflect the asymmetry, don't paper over it.
5. The practitioner path must stay one attribute deep ("where are my SHAP values")
   regardless of the ontology underneath.
6. pytraverse (probly) is the designated vehicle for two seams when they arrive —
   pytree/structured maskers and model-graph surgery for `ModelExtension` or
   layer-rule methods — plumbing beside the core, same dispatch family as flextype.
7. Vocabulary for the glossary (grill before arc 1): Estimate? IntensionalGame?
   Surrogate? Extension? Measure? — CONTEXT.md changes only when code lands.

## Evidence appendix (prototype, 2026-07-24; exact ground truth throughout)

- **R1 basis zoo** (n=8, R=4 players): support sizes — AND: 1/15/16, OR: 15/1/16,
  XOR: 15/15/2 across Möbius/co-Möbius/Fourier; OR's Möbius smear printed sign by sign;
  XOR's Fourier support is 2 because the 0/1 encoding parks an affine shift in ∅.
  Banzhaf↔Fourier: max |BV_i − 2·f̂({i})| = 1.5e-15.
- **R2 projection tower** (generic game, order-2 projection loses 88.9% energy):
  `P₁∘P₂ = P₁` to 7e-16 (uniform) / 8e-14 (soft-Shapley) under a shared measure; mixed
  measures diverge by 1.65. First tower attempt was vacuous (quadratic game inside the
  subspace) — projection tests need energy above the target order.
- **R3 ProxySHAP from primitives**: four lines (`fit_game`; `v − proxy`;
  `from_evidence(E.minus(proxy), …)`; add read-outs); median worst-player SV error
  0.226 → 0.153 at budget 160, correction costing zero extra game calls.
- **R4 toy active learner** (conjugate Bayesian linear on Möbius ≤ 2, EIG on the SV
  functional): split invariance and rollback replay pass at atol=0 rtol=0 purely by
  derivability (no carried counters); EIG vs random at budget 20: SV error 0.046 vs
  0.82; posterior std 0.070 → 0.019; candidate cap banks 416 of 500 instead of
  warn-and-drop.
- **R5 order dial**: OR(4) fidelity under uniform: 0.00 / 0.27 / 0.67 / 0.93 / 1.00
  for orders 0–4 (why an additive story captures 27% of a redundancy game).
- **R6 gradient bridge**: IG on the multilinear extension = SV to 4e-16 (Owen);
  BII({1,2}) by definition = MLE mixed partial at center = −3.694650; a bump extension
  vanishing at every vertex keeps completeness and shifts IG by exactly ∓C/6.

## Rebuild plan (arcs; each self-contained, test-first)

0. **Grill the vocabulary** against CONTEXT.md; settle D2's spellings; pick the branch
   (`shapiqv2-value-functions` off `shapiqv2-index-rethink`; old line frozen as
   reference, no double-fixing).
1. **The math tier**: Game protocol + arithmetic transformers, ParametricGame with the
   three bases + two planes, measures, exact verbs (`project`, `to_basis`, `fidelity`,
   `sv_from_moebius`-style read-outs). Acceptance: R1/R2/R5 as pinned tests, plus
   parity against the existing `ExactExplainer` on shared indices.
2. **The process tier**: Evidence (port `SamplingState` — it already fits), frozen
   policies with the verb mixin, the banked loop + dedup + chunked transformer
   underneath (D10 fixes the float32 finding structurally). Acceptance: current
   estimate pins ported + R4's replay/split checks; the review's Tier-1 findings
   verified moot by construction.
3. **Families as recipes**: permutation walks and kernel regressions on the new spine;
   ProxySHAP recipe (R3) and the SHAP-IQ port (counts × law) as the validating
   consumers; toy BED promoted to a real policy.
4. **The gradient bridge**: Extension adapter + path policies (R6 as tests); pytraverse
   enters only if/when structured maskers or model surgery arrive.
5. Facade, plots, on-ramps — after the core proves out, as before.
