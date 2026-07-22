# Consolidated backlog

One table for everything planned, postponed, or discussed and not yet landed. Shorthand is
fine here — per-issue files and ADRs stay authoritative for detail. Sources: the issue files,
the five-agent review rounds (issue-12), the outside API review (`shapiqv2apireview.md`,
findings cited as F1–F6 / Part III / IV / VI), and session decisions. Last trued: 2026-07-21.

## A — Core arcs (ossify-sensitive; order matters)

| Item | What | Source | Status / when |
| --- | --- | --- | --- |
| Amortized lifecycle ADR | fit-once/explain-many doesn't fit `Explainer(game).explain()`; sibling ABC vs `explain(target)`; facade currency = (model, masker, index) | VI.3 | **deferred** (user 2026-07-21) — still before facade; also decides `__call__` fate (F5e) |
| History rework | keep `history()`/`rollback()` as anytime-estimator identity; always-on cut points (+ per-cut bank); `track_history`, `HistoryError`, `EmptyState` edge-case law die — the sampler-history limb dies with stateless samplers | F3-revised | **folds into the issue-9 arc** (slice 5) |
| Sampling-state addressability (remainder) | landed 2026-07-21 (ADR 0013): `unique()`/`packed_keys()` on the state, dedup as a consumer, `LawfulSampler` on kernel/product samplers with pairing symmetrization. Remaining: wire Regression's solve to the unique view (√count weighting — proven equivalent, smaller solves), exactness-strata design when the border trick ports, law's first estimator consumer = sampled k-SII port | issue-13 | remainder open |
| Sampling rework (merged issue 9) | stateless draw-source hierarchy (`Sampler` → Permutation/Coalition → kernel concretes; `Paired` slims to draw doubling + law symmetrization), ONE sample loop on `Approximator` (`EvidenceApproximator` folds in) with banked whole-unit budgets (ADR 0004 exact spending waived — pending subsystem dies), `_call_game` seam (dedup policy; later bounded-batch shared with Exact, `ChunkedMaskedPredictor` slims); estimates stay bit-identical, prototype-validated (V1–V5) | issue-9 (rewritten 2026-07-22) | **designed — next arc**; SHAP-IQ port validates after |

## B — API surface & ergonomics (small–medium)

| Item | What | Source | Status / when |
| --- | --- | --- | --- |
| Bulk accessors | `explanation.values(order=k)` + aligned `interactions(order=k)`; document the `[]`/`()` planes side by side (keeping both — decided) | F6 | open — small, anytime |
| CallableGame dense default | user callables receive dense bool jax arrays (today: raw `CoalitionArray`; every example's first line is `to_dense()`); ABC stays | F5d | open — small |
| `share_samples` → bool | axis-tuple design to the drawer until the VLM/multi-target story arrives | F5a | open — small |
| Drop `approximate()` | real alias duplication; `__call__` waits on the lifecycle ADR | F5e | open |
| `ensure_bool` trim | keep teaching sites, drop flag-policing; much of it dies with the history rework anyway | F5c | open — low, batch |
| Reprs | samplers, ExactExplainer, TreeExplainer, InterventionalTreeGame are grade-F defaults; TreeModel dumps full arrays (SamplingState got its repr in #564) | issue-12 | open |
| `np.asarray(explanation)` | silent 0-d object array today → real dense exporter | issue-12 | open |
| Naming | `Regression` → `KernelRegression`?, `sampling_quantum` → `unit_size`? | Part III | open — decide at facade/namespace time |

## C — On-ramps & breadth (the beta milestone)

| Item | What | Source | Status / when |
| --- | --- | --- | --- |
| Facade | `shapiq.explain(model, x, ...)` routing trees → TreeExplainer, small games → Exact, else sampled; converters already form the model-routing layer | Part IV | open — after lifecycle ADR |
| Namespace diet | 80 top-level names → ~18; index-zoo long tail → `shapiq.indices` | Part IV | open — with facade |
| Background/marginal masker | background-dataset baselines (v1 marginal imputer story) | issue-12 numpy on-ramp | open |
| README quickstart | | issue-12 | open |
| Plots | biggest v1 regression | issue-12 | open — rank |
| Sampled k-SII | v1 regression | issue-12 | open — rank |
| Aggregation utilities | SII → k-SII etc. | issue-12 | open — rank |
| PathDependentTreeGame | v1 regression | issue-12 | open — rank |
| Feature names / pandas | everything is positional today | IV.3 | open |
| Convergence diagnostics | `std_error`/CI hook on approximators; evidence state has what's needed | IV.3 | open |
| Flax example file | decided: "jax world" is an example, not a module — file never written | issue-12 (landed §) | open — small |

## D — Kernel & performance knobs

| Item | What | Source | Status / when |
| --- | --- | --- | --- |
| Kernel order > 4 | packed 4×16-bit key ABI caps interaction order; needs key-ABI rewrite | issue-11 | open |
| OpenMP in the kernel | needs GIL release first | issue-11 | open |
| E/R extraction in C | move expectation/reachability extraction into the cext | issue-11 | open |
| Log-space omega | omega weights overflow past ~1022 players | review rounds | open |

## E — Contracts, docs, tests

| Item | What | Source | Status / when |
| --- | --- | --- | --- |
| Glossary/docs | `baseline` homonym (reference point vs `v(empty)`), TreeExplainer entry, remaining ADR 0009/0005 drift | issue-12 | open |
| ValueT vector-space sentence | Game docstring: values must form a vector space over the reals; nonlinearity belongs in the link function | VI.2 | open — one line |
| FIxLIP orientation design note | bipartite/cross-modal structure is masker/game metadata, not an orientation flag; sketch the requirement before reintroducing anything | F5b-revised | open — when FIxLIP nears |
| PR #564 test pins | `interaction_rank` vs `iter_interactions` property test; chunked-state append→materialize equivalence + misalignment error; `expand_ellipsis` cases | #564 review | offered, open |
| CHANGELOG.md | missing entirely | review rounds | open |

## Closed — considered and rejected (do not re-open without new evidence)

- **Singleton index rewrite** (commit `0dfffe84` on `shapiqv2-first-approx`; the name-literal
  `Index[Literal[...]]` plan): a probed alternative, not pending work. Settled by `4a6736af`
  (weighted Banzhaf family: a continuous `p` can only live on an instance) and declared dead
  2026-07-21 — recorded as ADR 0012. Indices stay instantiated value objects per
  ADR 0005/0008/0009/0012, with extensional equality as shipped.
