# Issue 4 — Sampling-path performance

Status: **not started** · Order: **fourth (last)**

## Problem

Sampling overhead was measured at roughly 0.7 ms per evaluation on the kernel sampler — irrelevant
next to a large torch forward pass, dominant next to synthetic games and small models. The
overhead is Python-level per-unit work on the `sample()` path:

- **One JAX dispatch per sampled unit.** `UnitScheduleSampler._sample` loops
  `while remaining > 0` at `src/shapiq/sampling/_schedule.py:104`, materializing one unit per
  iteration via `_unit_masks(units)` — one `jax.random.fold_in` plus one dispatch each. The
  kernel sampler's quantum is 1–2, so this is roughly one dispatch per evaluation.
- **Per-walk mask construction.** The permutation samplers rebuild masks per walk
  (`_player_positions` fold_in at `src/shapiq/sampling/_permutation.py:90`; `_walk_masks` loops
  bounded by order/pattern count).
- **Deduplication key handling.** `_coalition_keys` converts rows one `.tobytes()` at a time
  (`src/shapiq/explainers/_evidence.py:241-244`), and `_known_coalitions` rebuilds the full key
  dict from the entire state on every `sample()` call (`_evidence.py:151-159`) — quadratic
  across many small calls.

Explicitly **not** hot: the `explain()` loops in `_permutation.py` iterate over interaction sizes
and patterns (bounded by order), each iteration a single vectorized operation over all walks and
targets.

## Hard constraint

Sampled streams are contractual. Split invariance and deduplication identity are tested
**bit-identically**, and `fold_in(key, unit_index)` was chosen precisely so unit generation is
order-free: batched generation over a range of unit indices can and must reproduce the sequential
stream exactly. Any change that alters a sampled stream is out of scope for this issue.

## Approach (strictly in order)

1. **Measure.** Benchmark script covering sampler-only and end-to-end runs; workloads are issue
   2's torch example plus synthetic games, varied over `n_players`, budget, and quantum. Record
   the baseline table in this file before touching code.
2. **Batch unit generation.** Vectorize `_sampled_unit_masks` over an array of unit indices
   (batched `fold_in` / `vmap`), and restructure `_sample` to emit many units per dispatch while
   preserving resume and pending-sample semantics.
3. **Incremental deduplication keys.** Stop rebuilding `_known_coalitions` per call — persist
   keys in a way compatible with functional state (cache keyed on state identity, or keys carried
   forward by the state); batch the per-row `.tobytes()` conversion.
4. **Micro-optimizations** only after 1–3, and only where the benchmark says so.

## Work breakdown

- [ ] Benchmark script + baseline table recorded here (per sampler family; sampler-only vs
  end-to-end; a "sampling overhead per evaluation" headline number).
- [ ] Batched unit generation in `UnitScheduleSampler` and subclass `_sampled_unit_masks`
  (kernel first — smallest quantum, worst dispatch ratio — then permutation walks).
- [ ] Incremental dedup key store; batched key construction.
- [ ] After table: set and check a concrete speedup target (expectation: ≥ 10× on kernel-sampler
  unit generation).

## Acceptance criteria

- Before/after numbers recorded in this file.
- All stream-identity tests pass bit-identically — no tolerance loosening anywhere.
- No public API change (sampler classes, quanta, and budget semantics untouched).

## Dependencies

After issue 3 (both modify `src/shapiq/explainers/_evidence.py`). The benchmark's realistic
workload comes from issue 2's example; synthetic-game benchmarks can start earlier if needed.
