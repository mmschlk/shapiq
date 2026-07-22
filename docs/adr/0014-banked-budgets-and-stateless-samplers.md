# 14. Banked budgets and stateless samplers

Date: 2026-07-22

## Status

Accepted. Supersedes ADR 0004's exact-spending and pending-sample semantics and
ADR 0003's sampler-evolution and sampler-history machinery; ADR 0013's walk plans
and sampling law survive with their meaning intact, re-seated by this decision
(plans become family-owned walk layouts, the law lives on draw sources).

## Context

Exact budget spending forced a chain of machinery that existed to manage other
machinery: partial units were evaluated but masked (pending samples), pending
forced samplers to evolve resume state, evolution forced a parallel sampler
history with a lockstep check, and the walk layout had to ride inside the
sampler because the sampler metered the budget. Pending rows were game
evaluations that informed no estimate until a later call completed them — cost
without evidence. The maintainer waived exact spending during the issue-9 design
("we are prototyping exactly for this"), with the guiding sentence: an
approximator has a sampler and houses the approximator logic.

## Decision

- **Samplers are stateless draw values.** A sampler is `(key, shape policy)`
  plus `draws(unit_indices)`; draws derive from `fold_in(key, unit_index)` and
  are order-free, so samplers never evolve and are never tracked. The hierarchy
  is organized by draw type, then distribution: `Sampler` →
  `PermutationSampler` (positions; antithesis = reversal) and
  `CoalitionSampler` (masks; antithesis = complement, defined once) →
  `SizeKernelSampler`/`ProductKernelSampler` (each declaring its law).
  `PairedSampler` composes on any sampler declaring an antithesis and emits
  `[draw, antithesis]` per unit.
- **One loop on `Approximator`** (`EvidenceApproximator` folded in). Budgets
  stay denominated in game evaluations — the honest unit — but spend in whole
  units: the seed block (empty and grand coalition plus the family prelude)
  once, then sampled units, with the remainder **banked as one integer** and
  spent first on the next call. Split invariance holds by arithmetic; nothing
  is evaluated that cannot inform an estimate; explain-visible evidence is
  identical to the exact-spending world at every cumulative budget, which is
  why the estimate pins survived the rework unchanged.
- **History is identity and always on.** One `(n_samples, bank)` checkpoint per
  sample call rides in the state's cut points, so `rollback()` restores the
  exact resume point — rolling back and resampling the same budgets replays
  bit-identically — and `history()` needs no flag. `track_history`,
  `HistoryError`, and the pending vocabulary are gone.
- **Deduplication is a policy inside the loop.** Novel rows are charged, whole
  units are drawn until the charge meets the available budget, and the final
  unit may overshoot into the bank (negative = borrowed, repaid next call;
  `|bank|` stays under one unit). A stall after quiet units warns and banks the
  remainder. `spent` always equals game evaluations performed.

## Consequences

- Deleted: pending samples (machinery, concept, glossary), sampler evolution
  and `mutable`, sampler history and its lockstep check, `UnitScheduleSampler`
  and `KernelSampler` as layers, `WalkPlan`/`ChainPlan` as public names,
  `EvidenceApproximator`, `HistoryError`, the unused `state` parameter on
  sampling.
- `bank` and `spent` are visible approximator fields; `sample(b)` with
  `b < min_budget` may evaluate nothing and bank everything — boundary
  validations fire on the first evaluating call.
- The batched deduplication charge scan made deduplicated regression ~6.5x
  faster on the benchmark; the game-call seam (`_call_game`) is the future home
  of bounded-batch evaluation shared with the exact sweep (issue-9 slice 6,
  not yet built).
