# Issue 9 — The sampling rework: sources, one loop, one seam

Status: **largely landed** (2026-07-22, `a498e5f5`, ADR 0014) — slices 1-5 and 7 in one
pass; slice 6 (bounded-batch evaluation shared with the exact sweep,
`ChunkedMaskedPredictor` slimming) remains open. Landed beyond the design: history
checkpoints carry the bank as `(n_samples, bank)` pairs, so rollback restores the exact
resume point instead of forfeiting the remainder. Known cost: the deduplication charge
scan is a host loop per row (permutation dedup 31 vs 27 ms on the benchmark; deduplicated
regression dropped 62.5 → 9.5 ms from the batched scan). Originally: This supersedes the
original "game-call seam" scope: after the maintainer waived ADR 0004's exact-spending
constraint ("we are prototyping exactly for this"), the seam merged with the follow-through
of the sampler-vehicle arc (ADR 0013) into one rework. Design converged in discussion
2026-07-22; a scratchpad prototype (ephemeral — its results are recorded in Evidence
below) validated the load-bearing claims against the installed library.

## Problem (restated after convergence)

Approximator logic leaked into samplers across three refactors, and the plumbing kept
growing to manage other plumbing: walk layouts ride inside the sampler (as plans) because
the sampler meters budget; exact spending forces the pending subsystem (partial units
evaluated but masked, resume machinery); pending forces sampler evolution
(`_units_started`, `_pending_pos`, `mutable`); evolution forces the parallel sampler
history with its lockstep check. Samplers take a `state` argument they never read
(`noqa: ARG002 — schedule samplers are not adaptive` admits it). `Approximator.sample` and
`EvidenceApproximator._sample_deduplicated` are two loops kept behaviorally identical by
tests alone. And `ExactExplainer` materializes all `2**n` coalitions at once because
bounded-batch evaluation has no home. The guiding sentence for the fix, from the
maintainer: **an approximator has a sampler and houses the approximator logic.**

## Design: three pillars, one owner each

### 1. Samplers are stateless draw values

```
Sampler                      # ABC: n_players, key, shape policy; draws(indices)
├── PermutationSampler       # draws = player positions; antithetic = reversal
└── CoalitionSampler         # ABC: draws = masks; antithetic = complement (defined once)
    ├── SizeKernelSampler    # size-weighted distribution; declares its law
    │   └── ShapleyKernelSampler
    └── ProductKernelSampler # iid membership flips (p); declares its law
        └── BanzhafKernelSampler

Paired(sampler)              # composition beside the tree: attaches to any sampler
                             # declaring an antithesis; [draw, antithesis] per unit;
                             # symmetrizes the law when the inner sampler declares one

AntitheticDraws              # capability protocol, ONE method: antithetic(draws)
LawfulSampler                # capability protocol: log_probability(coalitions)
```

Draws derive from `fold_in(key, unit_index)` — order-free, so samplers never evolve: no
`_evolve`, no `mutable` flag, no sampler history. Level one of the tree is organized by
draw type (fixes what antithesis means), level two by distribution (fixes the law). Shape
policy (`share_samples`, target axes) stays sampler-owned. Concrete names keep the
`KernelSampler` suffix (decided: it earns its keep — sampling proportional to a kernel is
why the unweighted regression solve is correct). Pairing stays composition, one level
down, because *what antithetic means* is sampler-kind knowledge and must not become a
loop branch; the public `paired: bool | None` parameter survives unchanged as the way
users ask for it.

### 2. The loop lives on Approximator (EvidenceApproximator folds in)

Every shipped approximator is evidence-based, so the two-layer
`Approximator`/`EvidenceApproximator` split collapses into one `Approximator` base that
owns the single sample loop:

```
sample(budget):
    available = bank + budget            # first call: evaluate + charge the prelude
    units     = available // unit_rows   # whole units only
    bank'     = available % unit_rows
    draws  = sampler.draws(range(units_done, units_done + units))
    masks  = family_render(draws)        # identity for coalition samplers
    values = _call_game(masks)           # pillar 3
    state.append(masks, values)
```

The family contract stays a registry, not a class: `permutation_family` provides
(render, unit length, prelude, explain); `regression_family` provides (sampler builder,
kernel, solve) as today. The approximator stores what it needs as plain attributes set in
`__init__` — nothing travels, so nothing needs a bundle name (`WalkPlan` and the
never-born `EstimationPlan` both go).

**Banked budget semantics** (supersedes ADR 0004's exact spending; budgets stay
denominated in game evaluations — the honest unit survives):

- Whole units only; the remainder is banked as one integer and spent first on the next
  call. Split invariance holds by arithmetic: `floor((bank + b)/L)` accumulates the same
  units for any split of a cumulative budget.
- Explain-visible evidence is **identical to today at every cumulative budget** (today
  masks pending rows, so usable walks are `floor((spent − seeds)/L)` in both worlds) —
  the estimate pins carry over unchanged. What changes: stored streams no longer contain
  partial-unit rows, and the game is called strictly less (pending rows were evaluations
  that informed nothing until later).
- Deduplication charges novel coalitions only; whole units are drawn until the charge
  meets the available budget, and the final unit may overshoot into the bank (negative =
  borrowed, repaid by the next call; invariant `|bank| < unit_rows`). A stall after K
  quiet units warns and banks the remainder.

### 3. The game-call seam (the original issue 9, kept modest)

`_call_game` is a small evaluation-policy seam: the plain policy forwards to the game;
the deduplicating policy consults the state's key index (`packed_keys`/`unique` — the
state owns coalition identity, ADR 0013), evaluates novel rows, stitches duplicates from
stored values, and reports the novel charge; a later bounded-batch policy chunks any
coalition block through any game. The seam is shared by the sample loop **and**
`ExactExplainer`'s full sweep — which is what finally lets `ChunkedMaskedPredictor` slim
to pure torch execution policy (no-grad, device), with chunk arithmetic living here once
for every backend.

## Inventory

**Dies**: `UnitScheduleSampler` and `KernelSampler` as layers; the pending-samples
subsystem (machinery + concept + glossary entry); sampler evolution and `mutable`;
the sampler-history tuple and its lockstep check (closing the F3 rework's sampler limb);
`WalkPlan`/`ChainPlan` as public names; `EvidenceApproximator` as a layer; the unused
`state` parameter on sampling.

**Slims**: `PairedSampler` ~90 → ~15 lines (draw doubling + law symmetrization);
`AntitheticDraws` four methods → one; `EmptyState` (the state-side history rework — always-on
cut points, per-cut bank — folds into this arc, retiring `track_history` and
`HistoryError`).

**Stays**: the state with its unique view and coalition identity; the family registries;
the law capability (re-seated per source; `Paired`'s graft moves down a level with it);
the user grammar — `Regression(game, SV()).sample(500).explain()` does not move; the
estimate pins.

## Evidence (prototype, 2026-07-22)

Five validations against the installed library at `ef41ad95`: (V1) SV chain stream
bit-identical minus pending, 2 game calls saved at budget 40; (V2) `pump(15)+pump(25) ==
pump(40)` including the bank; (V3) STII stream bit-identical with the prelude and the
library's own `TaylorPlan.render` reused unchanged as the expansion — plans survive as
family expansions; (V4) paired+deduplicated kernel stream bit-identical to
`Regression(SV, dedup)`, `spent == distinct evaluated (30 == 30)`, dedup split
invariance; (V5) overshoot borrow: budget 5 → spent 6, bank −1; +10 → spent 16, bank −1.

## Slices (each self-contained)

1. Draw sources + `Paired` + the two capabilities: `sampling/` rewrite; draw streams
   pinned bit-identical to current sampled streams.
2. The loop on `Approximator` (fold `EvidenceApproximator`) + banked budgets; SV and the
   kernel path migrate; estimate pins hold; `bank`/`spent` become visible fields.
3. SII/STII migrate (plans become family expansions); `WalkPlan`/`ChainPlan` deleted.
4. Dedup as a `_call_game` policy consuming the state's key index; borrow + stall
   semantics; benchmark re-baselined (must not be slower).
5. State-side history: always-on cut points with per-cut bank; `EmptyState` slims;
   `track_history` and `HistoryError` die.
6. Bounded-batch policy + `ExactExplainer` through the seam; `ChunkedMaskedPredictor`
   slims to torch execution policy.
7. Docs: superseding ADR (0004/0003 amendments, 0013 continuity note), glossary
   (Pending Samples out, Bank in; Sampler entry sharpened), backlog truing.

Follow-up arc (not this issue): the sampled k-SII / SHAP-IQ port as the design's
validating consumer — it reads multiplicities from `unique()` and probabilities from the
law, and proves the whole shape.

## Acceptance criteria

- Estimates bit-identical to today at every cumulative budget (existing pins are the net).
- Draw streams bit-identical to current sampled streams (sources replicate the fold-in
  derivations exactly).
- `spent` equals game evaluations performed; `|bank| <` unit rows; split invariance
  including the bank, pinned.
- Dedup: `spent ==` distinct coalitions evaluated; stall warns; benchmark equal or
  faster.
- One sample loop, one game-call seam, zero public-grammar change.
