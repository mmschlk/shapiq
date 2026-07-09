# Issue 8 — A game-call seam owning deduplication

Status: **not started**

## Problem

Game evaluation is scattered and deduplication lives at the wrong layer. `Approximator.sample`
calls `self.game(coalitions)` directly (`src/shapiq/explainers/_approximator.py`), while
`EvidenceApproximator._sample_deduplicated` reimplements the whole sample-evaluate-append cycle
around its duplicate bookkeeping (`src/shapiq/explainers/_evidence.py`). Deduplication is a
property of *how the game is called* — evaluate each distinct coalition at most once — not of how
budgets are scheduled, so the sampling loop and the dedup loop should not be two separate code
paths that must be kept behaviorally identical by tests alone.

## Direction

Introduce one internal seam through which every approximator evaluation flows — a
`_call_game(coalitions)` on the approximator (or a small evaluation-policy object it owns):

- The plain policy forwards to `self.game`.
- The deduplicating policy consults stored evidence, evaluates only novel coalitions, stitches
  duplicates from stored values, and reports how much budget was actually consumed — so
  `sample()` keeps one loop for scheduling and budget accounting regardless of policy.
- The stall warning and exact-budget semantics (ADR 0004) stay observable behavior of `sample()`;
  only their implementation moves.

## Constraints

- Sampled streams and deduplicated estimates are contractual and tested bit-identically; this is
  a pure restructuring with zero stream changes.
- The functional contract stands: `sample()` returns a new approximator; policies must not hide
  mutable state that history/rollback cannot see.
- Coordinate with issue 4's incremental dedup key store (`_known_coalitions` currently rebuilds
  per call): the key store naturally becomes state owned by the deduplicating policy, so sequence
  the two issues rather than interleaving them.

## Acceptance criteria

- One evaluation path: `EvidenceApproximator` no longer duplicates the sample loop.
- All existing sampling, deduplication, history, and stall-warning tests pass unchanged.
- No public API change.
