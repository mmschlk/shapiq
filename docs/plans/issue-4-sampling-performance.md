# Issue 4 — Sampling-path performance

Status: **landed 2026-07-09** (batched unit generation + incremental dedup keys) · Order: **fourth (last)**

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

## Design notes from the 2026-07-08 review

Captured from the code-review TODOs so the ideas survive their removal from the source:

- **Batched unit ranges as the sampler API.** Rework `_sampled_unit_masks(unit_index)` toward a
  range form (`start_unit`, `end_unit`, or an index array) so many units are generated per JAX
  dispatch; `fold_in(key, unit_index)` keeps the stream order-free, so a `vmap` over the index
  range must be bit-identical to the sequential loop.
- **Pairing as a compositional layer — landed (2026-07-09).** `PairedSampler(sampler)` wraps
  any `UnitScheduleSampler`: the wrapper owns the schedule (budgets, pending, seeds) and uses
  the wrapped sampler purely as a unit renderer, so samplers carry zero pairing logic and new
  samplers get pairing for free. The public `AntitheticDraws` hook (`unit_draw` /
  `render_draw` / `antithetic_draw`) lets a sampler declare what pairing means (permutation
  walks pair as the reversed permutation); hookless samplers pair by row complement. A mask-level
  wrapper was rejected: complementing rendered walk rows corrupts permutation layouts, while
  the draw-level antithesis (complement coalition, reversed permutation) renders bona-fide
  units. Kernel streams stayed bit-identical; `PermutationSampling(..., paired=True)` gained
  antithetic permutation walks as a new opt-in. Batching hook: `_unit_draw(unit_index)` is the
  surface to vectorize over unit-index ranges.
- **Growing evidence buffers.** `_append_coalitions` (`src/shapiq/sampling/_state.py`)
  concatenates full dense arrays per `sample()` call — quadratic over many small calls. The
  sampling state wants an amortized structure: a JAX-backed growing buffer (capacity doubling or
  a user-provided pre-budget that pre-allocates for the expected total budget) with cheap appends,
  while preserving the functional state contract and history semantics.
- **`_sample(state, budget)` signature smell.** Schedule samplers ignore `state` (it exists for
  adaptive samplers). When the sampler API is reworked for batching, revisit the protocol so the
  non-adaptive case does not carry an unused parameter — for example a narrower
  `ScheduleSampler` protocol or passing only what adaptivity needs.

## Baseline (2026-07-09, `benchmark/sampling.py`)

jax 0.10.1 on cpu (Apple Silicon), `n_players=14`, budget 4096, median of 3 after one warmup.
The synthetic quadratic game costs 0.1 us/eval, so end-to-end numbers are sampling overhead.

| workload | total ms | us/eval |
|---|---|---|
| sampler-only: ShapleyKernelSampler | 1363.0 | 332.8 |
| sampler-only: BanzhafKernelSampler | 466.9 | 114.0 |
| sampler-only: PairedSampler(ShapleyKernel) | 758.3 | 185.1 |
| sampler-only: PermutationSIISampler(order=2) | 63.8 | 15.6 |
| sampler-only: PairedSampler(PermutationSII(order=2)) | 56.0 | 13.7 |
| sampler-only: PermutationSTIISampler(order=2) | 28.5 | 7.0 |
| game-only: quadratic on 4096 masks | 0.3 | 0.1 |
| end-to-end: Regression(SV) paired | 758.1 | 185.1 |
| end-to-end: Regression(SV) paired dedup | 2067.8 | 504.8 |
| end-to-end: PermutationSampling(SII order=2) | 59.3 | 14.5 |
| end-to-end: PermutationSampling(SII order=2) dedup | 133.3 | 32.5 |
| end-to-end: Regression(SV) dedup, 64 split calls | 2125.7 | 519.0 |

Readings: kernel samplers pay one JAX dispatch chain per coalition (the diagnosed problem);
permutation walks amortize dispatches over 25-208 coalitions per render and are 20-50x
cheaper per coalition already. Deduplication nearly triples Regression end-to-end (key rebuild
plus per-row Python). Splitting one dedup budget into 64 calls adds only ~3% — the
`_append_coalitions` quadratic is **not** a measurable cost at realistic scales, so growing
evidence buffers stay deferred (design note below kept for the record).

## After (2026-07-09, same machine, same workloads)

| workload | total ms | us/eval | speedup |
|---|---|---|---|
| sampler-only: ShapleyKernelSampler | 5.7 | 1.4 | 238x |
| sampler-only: BanzhafKernelSampler | 1.4 | 0.3 | 334x |
| sampler-only: PairedSampler(ShapleyKernel) | 3.5 | 0.9 | 217x |
| sampler-only: PermutationSIISampler(order=2) | 1.5 | 0.4 | 43x |
| sampler-only: PairedSampler(PermutationSII(order=2)) | 2.2 | 0.5 | 25x |
| sampler-only: PermutationSTIISampler(order=2) | 5.0 | 1.2 | 6x |
| end-to-end: Regression(SV) paired | 3.7 | 0.9 | 205x |
| end-to-end: Regression(SV) paired dedup | 77.4 | 18.9 | 27x |
| end-to-end: PermutationSampling(SII order=2) | 1.8 | 0.4 | 33x |
| end-to-end: PermutationSampling(SII order=2) dedup | 30.6 | 7.5 | 4x |
| end-to-end: Regression(SV) dedup, 64 split calls | 1372.9 | 335.2 | 1.5x |

The >=10x kernel unit-generation target is met with margin; sampling overhead on the
kernel path dropped from ~333 to ~1.4 us per evaluation and now sits within ~10x of the
raw game evaluation instead of ~3000x. The remaining outlier is dedup with many tiny
budget splits at high coalition-space coverage (25% here): with the key rebuild gone, the
cost is genuine novelty hunting — many small sampler rounds per call, each fetching exactly
the remaining novel count. That fetch policy is contractual: fetching more would append
extra free-duplicate evidence and make the stored state depend on how the budget was
split. If this ever matters, the lever is cheaper per-round fixed costs, not bigger rounds.

## What landed

- `UnitScheduleSampler._sampled_unit_batch(unit_indices)` — batched unit rendering,
  default sequential stack so custom samplers stay correct; `_sample` emits all full units
  of a call in one batched dispatch (seed and pending units keep the scalar path).
  `_unit_keys` batches `fold_in` over unit indices; bit-identity to the sequential stream
  is pinned by `tests/shapiq/test_batched_sampling.py` across kernels, walks, pairing,
  target shapes, custom samplers, splits, and resumes.
- Kernel samplers render via `_unit_from_key` and vmap it over unit keys; permutation
  samplers gain `unit_draws` (vmap over `unit_draw`) and their `render_draw` broadcasts
  over leading batch axes; `PairedSampler` batches through the `AntitheticDraws` hook
  (which gained `unit_draws`) or complements the wrapped batch.
- Dedup keys are carried forward on the approximator as a first-index map with a
  `(n_samples, len)` validity token: the tip of a sampling chain extends the map in place,
  branches (two samples from one approximator) detect the foreign extension and copy
  their own entries, rollback/history miss the token and rebuild. Key construction is one
  `packbits` + one `tobytes` per round. Branch consistency is pinned in
  `test_deduplicated_sampling.py`.
- `benchmark/sampling.py` reproduces both tables.

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

- [x] Benchmark script + baseline table recorded here (per sampler family; sampler-only vs
  end-to-end; a "sampling overhead per evaluation" headline number).
- [x] Batched unit generation in `UnitScheduleSampler` and subclass `_sampled_unit_masks`
  (kernel first — smallest quantum, worst dispatch ratio — then permutation walks).
- [x] Incremental dedup key store; batched key construction.
- [x] After table: set and check a concrete speedup target (expectation: ≥ 10× on kernel-sampler
  unit generation) — landed at 238×.

## Acceptance criteria

- Before/after numbers recorded in this file.
- All stream-identity tests pass bit-identically — no tolerance loosening anywhere.
- No public API change (sampler classes, quanta, and budget semantics untouched).

## Dependencies

After issue 3 (both modify `src/shapiq/explainers/_evidence.py`). The benchmark's realistic
workload comes from issue 2's example; synthetic-game benchmarks can start earlier if needed.
