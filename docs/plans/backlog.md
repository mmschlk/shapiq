# Consolidated backlog

One table for everything planned, postponed, or discussed and not yet landed. Shorthand is
fine here — per-issue files and ADRs stay authoritative for detail. Sources: the issue files,
the five-agent review rounds (issue-12), the outside API review (`shapiqv2apireview.md`,
findings cited as F1–F6 / Part III / IV / VI), and session decisions. Last trued: 2026-07-21.

## A — Core arcs (ossify-sensitive; order matters)

| Item | What | Source | Status / when |
| --- | --- | --- | --- |
| Amortized lifecycle ADR | fit-once/explain-many doesn't fit `Explainer(game).explain()`; sibling ABC vs `explain(target)`; facade currency = (model, masker, index) | VI.3 | open — grill session, **before facade**; also decides `__call__` fate (F5e) |
| Value-layout canonicalization | one internal layout (value axes leading), convert once at game boundary + explanation construction; kills the 23 `to_leading`/`to_trailing` sites | Part III, VI.2 | open — before new estimator families |
| History rework | keep `history()`/`rollback()` as anytime-estimator identity; rebuild as always-on cut points + sampler replay (arithmetic fast-forward — samplers are pure functions of emitted count); drop `track_history`, parallel sampler tuple, lockstep check, `EmptyState` edge-case law, `HistoryError` | F3-revised | direction accepted — after `_state.py` settles post-#564 |
| Game-call seam | issue 9: one evaluation path per approximator, dedup becomes an evaluation policy; backend-agnostic chunking generalization lives here too | issue-9 | not started |

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
| Review-file housekeeping | `shapiqv2apireview.md` untracked in repo root → `docs/plans/` or delete | this round | open — trivial |

## Closed — considered and rejected (do not re-open without new evidence)

- **Singleton index rewrite** (commit `0dfffe84` on `shapiqv2-first-approx`; the name-literal
  `Index[Literal[...]]` plan): a probed alternative, not pending work. Settled by `4a6736af`
  (weighted Banzhaf family: a continuous `p` can only live on an instance) and declared dead
  2026-07-21 — recorded as ADR 0012. Indices stay instantiated value objects per
  ADR 0005/0008/0009/0012, with extensional equality as shipped.
