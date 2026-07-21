# shapiq v2 API Review — Is It Overengineered?

*One-pass review of the v2 prototype (`src/shapiq`, examples, tests). ADRs and
CONTEXT.md deliberately not consulted, per request. Date: 2026-07-11.*

*Method: full read of all ~7,900 source lines across `coalitions`, `games`,
`sampling`, `interactions`, `explanations`, `explainers`, `trees`, and
`games/torch`, plus the examples and the ergonomics/identity test files. Claims
about "consumed nowhere" are grep-verified against `src/`.*

---

## Verdict

The core is **not** overengineered. The two load-bearing bets — interaction indices
as typed value objects with mathematical capability protocols, and immutable
evidence-carrying approximators — map onto the actual structure of the mathematics,
not onto speculative software patterns, and both demonstrably pay for themselves in
the code that consumes them.

However, there is a ring of ceremony *around* that core which **is** overengineered,
concentrated in six identifiable places: unused index metadata, the extensional
equality system, the history/rollback machinery, a pair of competing extension
mechanisms patched together with subclass guards, several strains of speculative
generality, and defensive runtime type-policing that a statically-typed API already
covers. Together these account for roughly 20–25% of the code and — more importantly
— about half of the *conceptual* surface a contributor or power user must absorb.

At the same time, the library is **under**-engineered exactly where "best XAI
library" will be decided: there is no high-level entry point, and the ~70-name flat
top-level namespace presents internal machinery as if it were the user API.

A useful frame: v2 today is a superb **kernel** with no **hood**. The kernel should
be defended and even simplified; the hood is the missing investment.

---

## Part I — What's earning its complexity (keep, and defend)

### 1. Index objects with capability protocols

`interactions/_indices.py` defines indices as frozen dataclasses (`SII(order=2)`,
`WeightedFBII(p=0.3, order=2)`) and lets estimators discover what they can do with
an index through three *structural* capabilities:

| Capability | Protocol method | Mathematical meaning |
|---|---|---|
| `CardinalInteractionIndex` | `derivative_weights(n, s)` | cardinality-weighted discrete derivatives |
| `GeneralizedValueIndex` | `marginal_weights(n, s)` | cardinality-weighted bloc marginals |
| `RegressionIndex` | `regression_kernel(n)` | kernel-weighted least-squares fit |

This is the right decomposition because it matches how the literature actually
organizes these indices. The evidence that it pays off is in the consumers:

- Each of the 21 shipped indices is ~40 declarative lines; adding an index means
  declaring weights, not writing an estimator.
- `ExactExplainer` (`explainers/_exact.py`) supports the entire zoo through three
  math-shaped code paths (`_weighted_derivatives`, `_weighted_marginals`, the
  regression solvers) plus three dedicated solvers — instead of a 21-armed
  `if index == "SII"` ladder, which is exactly the v1 failure mode this design
  escapes.
- `TreeExplainer` gets SV, BV, SII, BII, the weighted-Banzhaf family, CHII, STII,
  and both Moebius transforms *for free* from the cardinal capability — one
  closed-form algorithm, many indices.

This is the crown jewel of the design. Nothing below should be read as an argument
against it.

### 2. The immutable approximator loop

```python
approximator = PermutationSampling(game, SII(order=2), random_state=0)
approximator = approximator.sample(500)      # returns a NEW approximator
explanation  = approximator.explain()
```

Under the hood this buys several properties that are genuinely rare and
differentiating for an approximation library:

- **Budget-split invariance.** Units derive randomness from
  `fold_in(random_state, unit_index)` (`sampling/_schedule.py:171-177`), so
  `.sample(100)` and `.sample(7).sample(13).sample(80)` produce bit-identical
  states. Your own example asserts this (`examples/permutation_sampling.py:66-68`).
  This makes iterative refinement ("sample until the CI is tight") trivially
  correct, which is the actual workflow of approximation users.
- **Exact budget accounting with pending units.** A permutation walk cut short by
  the budget stays pending and resumes on the next call; `explain()` masks
  incomplete units. Budgets mean "game evaluations", which is the only honest unit.
- **Deduplication as a policy, not an estimator change.** `deduplicate=True`
  (`explainers/_evidence.py:110-164`) reuses stored values for repeated coalitions,
  charges the budget only for novel evaluations, and provably returns the same
  estimate over the same units — plus a `SamplingStallWarning` when the game's
  2^n coalitions are exhausted. This is thoughtful, correct, and hard to retrofit.
- **`min_budget`** (`_evidence.py:45-56`) saves users the seed-block arithmetic and
  its docstring is honest about being a floor, not a guarantee.

### 3. The game/masker layering

```
Masker → MaskedPredictor → MaskedGame → Explainer
```

Each layer has one job: maskers turn coalitions into model-native inputs in the
model's own backend; masked predictors own call policy (the torch one adds no-grad,
chunking, and device-following); `MaskedGame` owns the value-space declaration and
the prediction→value conversion; explainers only ever see `Game`. Two details are
particularly well done:

- **One backend seam.** `coalition_masks_like` (`games/_masker.py:56-76`) is the
  *only* place JAX-side coalitions cross into another backend (DLPack with a host
  fallback), and `require_shared_backend` gives a translated error instead of a
  deep backend stack trace. Everything else is `array_namespace(x)` local compute.
- **`ChunkedMaskedPredictor`** (`games/torch/_chunked.py`) is the best-engineered
  file in the repo: chunk sizing derived from flat instance count, one chunk alive
  at a time, device resolved per prediction so `model.to("cuda")` after
  construction just works. This is the kind of operational care users feel.

### 4. Teaching errors and the tests that protect them

`reject_common_index_mistakes` catches `index="SII"` (string) and `index=SII`
(class) with errors that show the working idiom; `InsufficientSamplesError` tells
the user the *exact* budget shortfall and reminds them that `sample()` returns a new
object. `test_api_ergonomics.py` pins these messages down — testing error-message
quality is rare and it is exactly what a library with "best in class" ambitions
should do. Keep this, but see Part II.5 for where the same instinct overshoots.

---

## Part II — Overengineering findings

### Finding 1: Dead metadata on every index

**What it is.** The `InteractionIndex` protocol demands seven members
(`explainers/_base.py:37-45` lists them for the error hint):

```
name, order, order_semantics, min_interaction_size,
includes_empty_interaction, preserves_value, generalizes
```

**Evidence.** Grep over `src/`:

- `order_semantics` — declared 21×, consumed **nowhere** in `src/` (only asserted
  in `tests/shapiq/test_index_identity.py`).
- `preserves_value` — declared 21×, consumed **nowhere** in `src/`.
- `includes_empty_interaction` — derived property, consumed **nowhere** in `src/`.
- `generalizes` — consumed only by the equality collapse (`_indices.py:148`).
- `min_interaction_size` — genuinely consumed (`_exact.py:166`,
  `explanations/_base.py:92`, `check_represented_window`). **Keep.**
- `name`, `order` — genuinely consumed everywhere. **Keep.**

**Why it's a problem.** Every index author — including external contributors adding
their own index, which the capability design explicitly invites — must correctly
answer four questions no code asks. The answers *are* mathematically meaningful
(coverage-vs-identity order semantics is a real and subtle distinction; the
numerical test that order-1 restrictions equal the generalized value is good
science), but metadata that only tests read is documentation wearing a protocol's
costume. It inflates the protocol, the docstrings, the mental model, and the
"missing protocol members" error message — for zero runtime behavior.

**What to do.** Keep `name`, `order`, `min_interaction_size` in the protocol. Move
`order_semantics`, `preserves_value`, `generalizes` into docstrings (and keep the
numerical property tests — they can test the *math* without the *protocol*). If a
future estimator or docs generator needs them, reintroduce them with a consumer in
the same commit. Protocol members should be pulled in by consumers, not pushed in by
taxonomy.

### Finding 2: `ExtensionalEquality` — an equality relation the library's own dispatch ignores

**What it is.** `_indices.py:135-167`: indices compare "as attribution rules on
nonempty interactions". Order-1 instances collapse onto the value they generalize,
and uniform weightings collapse onto their unweighted siblings:

- Nine types at order 1 all equal `SV()`: SII, CHII, STII, KSII, FSII, KADDSHAP,
  SGV, CHGV, JointSV (pinned by `test_index_identity.py:80`).
- `WeightedBV(p=0.5) == BV()`, `WeightedBII(p=0.5, order=k) == BII(order=k)`,
  `WeightedFBII(p=0.5, order=k) == FBII(order=k)` — each via a hand-written
  `_identity` override on the class (`_indices.py:273-277`, `366-372`, `551-557`).

**Why it's a problem.**

1. **Equal objects, unequal behavior.** Dispatch throughout the library is keyed on
   *exact type* (see Finding 4). So:
   - `FSII(order=1) == SV()`, yet `PermutationSampling(game, SV())` works while
     `PermutationSampling(game, FSII(order=1))` raises (only SV/SII/STII have
     permutation families).
   - `SII(order=1) == SV()`, yet `Regression(game, SV())` works while
     `Regression(game, SII(order=1))` raises (only SV/FSII/FBII/WeightedFBII/
     KADDSHAP have regression families).
   An equivalence relation that construction, dispatch, and support matrices all
   ignore is a loaded gun. The docstring itself has to carve out the exemption
   ("Estimator dispatch is keyed on index types and is unaffected") — when equality
   needs a disclaimer about where it doesn't apply, it's the wrong equality.
2. **Silent dict/set collapse.** `{SV(): res_a, SII(order=1): res_b}` has one entry.
   Any user caching results per index — the most natural benchmarking pattern, and
   this team benchmarks for a living — silently merges nine distinct configurations.
3. **Maintenance surface.** Each collapse rule is a per-class `_identity` override
   with an `# noqa: SLF001 - sibling rule` escape hatch; adding a weighted variant
   means remembering to hand-wire its collapse. The cleverness is not localized.

**What to do.** Use plain frozen-dataclass equality (type + params). Document the
mathematical identities — "SII at order 1 *is* the Shapley value" — in docstrings
and keep the numerical tests that verify them. If "same attribution rule" is needed
operationally later (e.g., result-cache keys), expose it as an explicit method
(`index.rule_key()`), not as `__eq__`/`__hash__` where it ambushes dicts.

### Finding 3: History/rollback — the largest removable block

**What it is.** A full provenance subsystem threaded through the core:

- `track_history` flags on `EmptyState`, `SamplingState`, and every approximator
  constructor.
- `SamplingState.rollback(steps)` / `.history(reverse=, include_self=)` with a
  compact `_history_n_samples` cut-point tuple and `_slice_to` reconstruction
  (`sampling/_state.py:117-230`).
- `EmptyState` with its *own* overridden `rollback`/`history` semantics ("an empty
  state with history enabled lists only itself", rollback(0) allowed, rollback(1)
  raises) — ~70 lines of docstring-heavy edge-case law (`_state.py:45-114`).
- A *parallel* sampler-history tuple on `Approximator` that must stay in lockstep
  with the state history (`explainers/_approximator.py:29, 51, 114-124`), including
  a runtime consistency check — `"state and sampler history lengths differ"`
  (`_approximator.py:96-98`). When a design needs a runtime check that its two
  histories haven't drifted apart, the invariant lives in the wrong place.
- A dedicated `HistoryError`, and constructor-time interlocks
  ("history cannot be enabled with mutable state or sampler").

**Why it's a problem.** The approximator is already immutable — that's the whole
architecture. Immutability means history is a *user-side one-liner*:

```python
snapshots = [approx := approx.sample(40) for _ in range(5)]   # history
approx = snapshots[-2]                                        # rollback
```

This is strictly more powerful than the built-in (arbitrary branching, no
`track_history=True` opt-in ceremony, no masked feature behind a flag) and costs the
library nothing. The built-in re-implements it with: two flags, two error types'
worth of edge cases, a compact encoding, a parallel bookkeeping structure, a
lockstep invariant, and ~200 lines + tests. The one thing the built-in adds —
memory compactness via shared underlying arrays and cut points — also falls out of
the user-side version, because JAX arrays are immutable and the snapshots share
buffers anyway.

This looks like the team's benchmarking needs (convergence curves are your papers'
bread and butter) leaking into the core API.

**What to do.** Delete `track_history`/`rollback`/`history`/`HistoryError` from the
core. Put a ten-line `record_convergence(approximator, budgets)` helper in the
benchmark layer if the loop is common. Document the walrus idiom prominently — it's
a selling point of the immutable design, not a workaround. Expected effect: the
`Approximator` base shrinks to ~40 lines, `sampling/_state.py` roughly halves, and
`EmptyState` becomes trivial.

### Finding 4: Two competing extension models, reconciled by subclass police

**What it is.** The library has two extension mechanisms:

- **Structural capabilities** (Part I.1): `isinstance(index, CardinalInteractionIndex)`
  — anyone's index works anywhere the capability suffices (`ExactExplainer`,
  `TreeExplainer`).
- **Nominal families**: `singledispatch` on the *exact* index type pairs a sampler
  with its estimator (`permutation_family`, `_permutation.py:212`;
  `regression_family`, `_regression.py`) or an algorithm with a game type
  (`tree_explanation`, `_tree.py:95`).

Because `singledispatch` resolves along the MRO, a *subclass* of `SII` would
silently inherit SII's walk estimator even if its semantics differ. The codebase
defends against this with **four** hand-written guard blocks that detect
subclass-of-supported and raise a teaching error:

- `ExactExplainer.__init__` — loop over `(KSII, FBII, WeightedFBII, KADDSHAP)`
  (`_exact.py:92-99`)
- `Regression.__init__` (`_regression.py:135-143`)
- `PermutationSampling.__init__` (`_permutation.py:120-127`)
- `TreeExplainer.__init__` — same pattern for game types (`_tree.py:59-67`)

Plus dedicated tests (`test_permutation_family_dispatch.py`,
`test_entry_point_dispatch.py`) pinning the guards.

**Why it's a problem.** The guards are correct *given the design* — the failure they
prevent (subclass silently gets the parent's kernel-matched sampler) is real. But
needing four copies of "stop my own dispatch mechanism from firing" is the design
telling you the mechanism is wrong. `singledispatch`'s MRO walking is exactly the
feature being fought; using it and then policing it is complexity spent on both
sides of the same fence. Contributors, meanwhile, must learn *three* dispatch
systems (capabilities, singledispatch families, flexdispatch backend seams) and
which one applies where.

**What to do.** Replace `singledispatch` families with a plain dict registry keyed
by exact type — `_FAMILIES: dict[type, PermutationFamily]` with an explicit
`register(index_type)` function. Exact-type lookup has no MRO fallback, so all four
guards and their tests disappear; a subclass is simply *unregistered* and gets the
ordinary "no family registered for X" teaching error. Alternatively, hang the family
off the index itself (`index.permutation_family()` as an optional capability), which
unifies the two models completely — but the dict registry is the smaller, safer
change and preserves the "families register atomically" property the docstrings
rightly prize (`_permutation.py:200-206`).

### Finding 5: Speculative generality in the small

Five independent strains; each is small, together they're a pattern.

**(a) `share_samples: bool | int | tuple[int, ...]`** (`sampling/_base.py:85-119`).
Per-axis sample sharing across explanation-target axes, with axis normalization,
duplicate detection, negative-axis handling, and a derived `shared_target_shape`.
The `bool` is load-bearing (deduplication requires shared samples; the error at
`_evidence.py:72-77` says so). The axis-tuple variant — "share coalitions across
axis 1 of the target batch but not axis 0" — has no visible story, no example, and
no test that motivates it. Cut to `bool`; reintroduce axes when a real use case
(e.g., per-class sharing in a multiclass batch) arrives.

**(b) `InteractionOrientation`** — `"undirected" | "directed"` threads through six
files (`_types.py`, `_validation.py`, `_iteration.py`, both explanation classes,
`explainers/_base.py`), and validation ends with `"{index_name} currently supports
only undirected interactions"` (`_validation.py:57-59`). A parameter whose only
accepted value is its default is a comment, not a parameter. Delete it; re-thread
when a directed index actually ships. (This one is pure carrying cost: every
explanation constructor, `normalize_interaction` call, and repr drags it along.)

**(c) Runtime type-policing.** 13 `ensure_bool` call sites, plus the
bool-rejecting `validate_int`/`validate_n_players`/`normalize_shape` pattern —
~74 defensive-validation call sites and ~153 `raise` statements in ~7,900 lines
(one per 52 lines). The high-traffic teaching errors are worth every line
(index-as-string, budget-as-float, rebind reminder). But
`ensure_bool("reverse", reverse)` and `ensure_bool("include_self", include_self)`
inside `history()` police keyword flags that a type checker — which this project
explicitly targets — already covers, and that no user has ever gotten wrong in a
damaging way (`True`/`1` confusion on a `reverse` flag is harmless). There's also a
taxonomy wobble: `_validate_history_steps` raises `HistoryError` (an `IndexError`
subclass) for what is a `TypeError` everywhere else (`_state.py:233-239`).
Rule of thumb to adopt: **validate values that silently corrupt math** (shapes,
weight-vector lengths — `_require_weight_length` in `_exact.py:202` is a great
example, since JAX would silently mis-index); **don't validate types that a type
checker sees and that fail loudly anyway.**

**(d) `CoalitionArray`** — an ABC (`coalitions/_base.py`) + a `_DenseStorage`
protocol + `__bool__` ambiguity guard + `empty/zeros/ones` constructors, with
exactly **one** implementation (`DenseCoalitionArray`). Meanwhile, at the single
boundary users touch most — writing a game — the first line of every callable in
your own examples and tests is the escape hatch:

```python
def game_value(coalitions):
    masks = jnp.asarray(coalitions.to_dense(), dtype=jnp.float32)   # everyone does this
```

An abstraction that 100% of call sites immediately unwrap is friction, not
protection. The sparse-coalitions future is plausible (explanations already ship
dense+sparse, so the pattern is proven there) — but coalitions haven't needed it
yet, and the ABC can be introduced *when* the second implementation lands without
breaking anyone if the dense array is what users see today. Minimum fix if the ABC
stays: make `CallableGame` hand plain dense bool arrays to user callables by
default (it already has a `coalition_converter` hook — flip the default), so the
90% path never sees the wrapper.

**(e) Duplicate entry points.** `Explainer.__call__` aliases `explain()`;
`Approximator.approximate(budget)` aliases `sample(budget).explain()`. Two names
for everything doubles docs and halves grep-ability. Pick the explicit one; drop
the aliases while nothing depends on them.

### Finding 6: Explanation ergonomics — two indexing planes and no bulk accessor

**What it is.** On `ExplanationArray`:

- `explanation[0]` → slices *target* axes (batch of explained instances);
  raises `IndexError("cannot index a scalar explanation")` on scalar targets.
- `explanation((0, 1))` → looks up the *interaction* (0,1) via `__call__`.
- Bulk access → reach into `attributions_by_order[1]`, a dict of blocks whose
  column order is implicit (lexicographic combinations); your own image example
  does `explanation.attributions_by_order[1].reshape(GRID)`
  (`examples/image_superpixels.py:62`).

**Why it's a problem.** `[]` and `()` doing categorically different things on the
same object is a permanent source of confusion — square brackets are the universal
"index me" affordance, and the scalar-target `IndexError` will be many users' first
contact with the distinction. Meanwhile the operation users perform most — "give me
all order-1 attributions as an array" — has no blessed accessor, only a raw dict
with an ordering contract you have to know. The per-interaction `__call__` also does
host-side position lookup via `list(iter_interactions(...)).index(...)`
(`explanations/_dense.py:181-189`) — a linear scan per query that will sting when
someone loops over all pairs of a 50-player game (your example loops exactly like
this).

**What to do.**

1. Add a bulk accessor: `explanation.values(order=1)` returning the block, plus
   `explanation.interactions(order=k)` returning the aligned member tuples/array.
   Most users never need per-interaction lookup once this exists.
2. Keep `()`-lookup (it reads well) but back it with a cached rank computation —
   the combinatorial-rank code already exists in `_permutation.py:537-545`.
3. Consider whether `[]`-on-targets earns its place at all, or whether
   `explanation.at(i)` / `.for_target(i)` communicates better. If `[]` stays,
   document the two planes side-by-side in one place.

---

## Part III — Smaller observations

**The shape-algebra tax.** Games declare `target_shape` (explained instances) and
`value_shape` (vector-valued games), and the value contract is
`(*broadcast(target_shape, coalition_leading), n_samples, *value_shape)`
(`games/_base.py:36-59`). Both needs are real (batched explanation, multiclass
outputs). But every estimator pays a `to_leading`/`to_trailing` toll to move value
axes front and back (`explainers/_valueaxes.py`, called 14× across the explainers),
and states store values in a *third* layout (sample axis at the target position,
value axes trailing — `_evidence.py:235-241`). Three layouts for one tensor is a
standing invitation for silent misalignment; the module docstring itself explains
the bug class it's defending against. Consider canonicalizing one internal layout
(value axes leading, since that's what accumulation wants) at the game boundary,
converting once on entry and once at explanation construction.

**Error taxonomy.** `HistoryError(IndexError)` is surprising (and doubles as a
TypeError — see Finding 5c). `UnsupportedGameError` exists in `errors.py` but the
unsupported-game paths in `TreeExplainer` raise bare `TypeError` instead — either
use the domain error or drop it. Small, but a "best in class" library's exception
table gets read.

**Naming.** `Regression` as a top-level class name reads like a model, not an
explainer (`shapiq.Regression(game, FSII(order=2))`). `KernelRegression` or
`RegressionApproximator` would self-describe. Likewise `EvidenceApproximator` is an
internal concept exported at top level (via `shapiq.explainers`) — it's plumbing.
`sampling_quantum` is precise but cute; `unit_size` would need no glossary entry.

**Vocabulary tax (the meta-observation).** To read this codebase you must acquire:
*units, quanta, seed blocks, pending samples, evidence, walks, chains, families,
capabilities, extensional equality, coverage vs. identity semantics, orientation,
targets, value axes, anchors, antithetic draws*. Each term is individually
well-chosen and consistently used — the writing quality is genuinely high — but the
count is a proxy for concept count, and concept count is what overengineering
actually costs. Every cut in Part II removes vocabulary with it: dropping history
removes rollback semantics; dropping extensional equality removes a whole equality
theory; dropping orientation removes a word that currently means nothing.

**Dependencies.** `numpy + array-api-compat + jax + flextype` is coherent with the
stated strategy (JAX internal, array-API at the seams, flextype for lazy backends).
Two honest costs to keep in view: JAX transitively pulls `scipy`/`ml_dtypes`/
`opt_einsum`, so "very little dependencies" reads as "few *direct* dependencies";
and JAX's float32 default means exact solvers quietly run float32 unless users
enable x64 — `ExactExplainer`'s docstring discloses this, which is good; the
eventual facade should probably surface it too. The flexdispatch usage itself is
exemplary restraint: exactly two seams (`to_values`, `to_tree_model`), both
genuinely needing lazy backend registration.

**What I did *not* find.** No premature performance abstraction (the JAX code is
direct and vectorized — `_weighted_derivatives` is a clean einsum, the STII
estimator's position-recovery trick at `_permutation.py:470-474` is documented
where it's subtle); no config-object disease; no inheritance depth beyond 2; no
premature plugin system. The overengineering here is *conceptual*, not structural —
which is the fixable kind.

---

## Part IV — Under-engineered where "best library" is decided

### The missing facade

The core story today is four-object assembly plus an index object:

```python
masker    = SuperpixelMasker(inputs=image, baseline=0.0, labels=labels)
predictor = ChunkedMaskedPredictor(masker=masker, model=cnn, batch_size=64)
game      = MaskedGame(masked_predictor=predictor, link_function=probs)
explainer = ExactExplainer(game, SV())
values    = explainer.explain()
```

The competition is `shap.Explainer(model, background)(X)`. Every layer above is
*right* — but it's the kernel, and users should meet it only when they outgrow:

```python
explanation = shapiq.explain(model, x, baseline=background, index=SII(order=2))
```

The dispatch infrastructure for this facade **already exists in the repo**:
`to_tree_model` flexdispatches on sklearn/XGBoost/LightGBM/CatBoost types without
importing them (`trees/_conversion.py:59-77`), and `to_values` does the same for
torch. A `shapiq.explain` / `shapiq.Explainer` that routes tree models to
`TreeExplainer`, small games to `ExactExplainer`, and everything else to a sensible
approximator with a budget parameter is the same pattern one level up. This is the
single highest-leverage piece of work available, and it converts every finding
above from "user-facing wart" to "expert-layer detail".

### The namespace diet

`shapiq/__init__.py` exports ~70 names. `PermutationSTIISampler`, `EmptyState`,
`UnitScheduleSampler`, `validate_interaction_metadata`, and `ExtensionalEquality`
sit at the same shelf height as `SV`. Flat namespaces are read as "this is the
API"; today the API says the user must understand sampler internals.

Proposed top level (~18 names):

```
explain (facade), Explainer, ExactExplainer, TreeExplainer,
PermutationSampling, Regression (renamed),
SV, BV, SII, BII, STII, KSII, FSII, FBII, KADDSHAP, Moebius,   # + weighted/GV in shapiq.indices
CallableGame, MaskedGame,
errors (module)
```

Everything else stays importable from its subpackage (`shapiq.sampling`,
`shapiq.games`, `shapiq.interactions`) — discoverable by the people extending the
library, invisible to the people using it. The 21-index zoo itself is *not* bloat
(it's the library's research identity, and Part I.1 makes each index cheap), but the
long tail (SGV/BGV/CHGV/IGV/EGV/JointSV/CoMoebius/Weighted*) belongs in
`shapiq.indices`, not the front page.

### Ergonomics still to build (expected, given prototype stage)

Named after what the examples strain against: feature names / pandas-aware maskers
(everything is positional today), the bulk accessor of Finding 6, convergence
diagnostics on approximators (a `std_error` or CI hook — the evidence state has
everything needed), and eventually plotting. None of this is criticism of the
prototype scope; it's where the freed complexity budget should go.

---

## Part V — Prioritized recommendations

| # | Action | Effort | Payoff |
|---|---|---|---|
| 1 | Delete history/rollback subsystem; document the walrus-snapshot idiom | M | −~200 lines, −2 flags, −1 error type, Approximator base → ~40 lines |
| 2 | Replace singledispatch families with exact-type dict registries | S | deletes all 4 subclass guards + their tests; one dispatch story |
| 3 | Drop `order_semantics`, `preserves_value`, `generalizes` from the protocol (keep the math in docstrings + numeric tests) | S | protocol 7→4 members; honest contract |
| 4 | Replace `ExtensionalEquality` with plain dataclass equality | S | removes the equal-but-rejected trap and per-class `_identity` overrides |
| 5 | Delete `InteractionOrientation`; cut `share_samples` to `bool`; trim `ensure_bool`-style policing to the teaching-error sites | S | thinner signatures everywhere |
| 6 | `CallableGame` hands dense bool arrays to user callables by default; defer the `CoalitionArray` ABC until a sparse impl exists | S | removes the universal `to_dense()` first line |
| 7 | Add `explanation.values(order=k)` + cached interaction ranks; reconsider `[]`-on-targets | M | fixes the most-trodden ergonomic path |
| 8 | Build `shapiq.explain` facade on the existing flexdispatch pattern | M–L | the actual "best library" move |
| 9 | Namespace diet: ~70 → ~18 top-level names | S | the API reads as the API |
| 10 | Rename `Regression`; canonicalize one internal value layout | M | fewer silent-misalignment opportunities |

*Items 1–6 subtract roughly 20–25% of current code and half the conceptual surface;
items 7–9 are where the freed budget earns compound interest.*

---

## Bottom line

Keep the bones — indices-with-capabilities, immutable sampling with fold-in
reproducibility, the game/masker layering, the two flexdispatch seams, the teaching
errors. Cut the ceremony — history, dead metadata, extensional equality,
orientation, axis-sharing, type-policing, subclass guards. Then spend what you freed
on the facade and the bulk-access ergonomics, because a rival-less XAI library is
decided at `shapiq.explain(...)`, not at `UnitScheduleSampler._sampled_unit_batch`.
The encouraging part: every problem found here is a *subtraction* problem, and the
core you'd be subtracting down to is the strongest XAI kernel design I've seen.

---

# Part VI — Addendum: re-assessment under the full roadmap

*Added after the maintainers clarified the v2 scope: essentially every form of
Shapley-value-based explanation — including amortized explainers (FastSHAP-style
surrogate networks, TabPFN-based approaches), FIxLIP-style interactions for
vision-language encoders with (semi-)directed cross-modal interactions, and games
whose values are not necessarily scalar reals. The original review (Parts I–V) is
left unchanged as a record; this part states what survives, what flips, and what
the roadmap newly raises.*

## VI.1 Findings that get *stronger* under the roadmap

**Exact-type registries instead of singledispatch + guards (Finding 4) — much
stronger.** "Every form of Shapley explanation" means many more estimator families
(SVARM-style stratification, amortized training objectives per index, FIxLIP
solvers…). Under the current pattern, *each new entry point ships its own
subclass-guard block* — the four existing guards become eight, then twelve.

Spelled out, since this point matters most as the estimator count grows: the
families are `functools.singledispatch` functions, and singledispatch resolves
along the **MRO** — `class MyIndex(SII)` with different semantics would silently
receive SII's walk sampler and estimator, producing wrong numbers with no error.
The docstrings correctly forbid this ("MRO resolution never hands an index a
shipped estimator silently"), and the defense is a hand-written
`type(index) not in registered and isinstance(index, registered)` guard repeated
at four entry points. The mechanism (MRO fallback) and the policy (no MRO
fallback) are fighting each other. A plain mapping dissolves the conflict:

```python
_FAMILIES: dict[type, PermutationFamily] = {}

def permutation_family(index) -> PermutationFamily:
    family = _FAMILIES.get(type(index))          # no inheritance fallback, by construction
    if family is None:
        raise _unsupported(index)                # one place, not four
    return family
```

A subclass is simply *not found* — all four guards and their tests become
structurally unnecessary. Atomic registration, third-party extension, and the
supported-types teaching error all survive (`_FAMILIES.keys()` replaces
`.registry`); the nicer "you subclassed SII; register your own family" hint moves
into the single lookup (on a miss, `isinstance(index, tuple(_FAMILIES))` appends
it). The trade is only singledispatch's decorator ergonomics, recovered by a
two-line `register` helper.

**ExtensionalEquality (Finding 2) — stronger, with one addition.** More indices ×
more estimators = more cells in the support matrix where equal-comparing objects
behave differently; the trap surface grows quadratically with the roadmap. The
addition: amortized explainers make an *explicit* "same attribution rule" key
genuinely useful — a surrogate network trained for SV should serve
`SII(order=1)` queries. That is exactly the `index.rule_key()` method suggested in
Finding 2 — an explicit, opt-in identity for caching and model-serving lookup —
and exactly not `__eq__`/`__hash__`, where the collapse ambushes every dict in
user code.

**Explanation ergonomics (Finding 6) — stronger.** Vision-language games mean
hundreds of players (patches + tokens); the current per-interaction lookup's linear
scan over `combinations(n, k)` becomes unusable there, so the cached
combinatorial-rank fix stops being a nicety. And with vector-valued attributions,
reaching into `attributions_by_order[k]` and knowing the implicit column order by
heart gets worse — the bulk accessor becomes the primary API, not sugar.

**Facade + namespace diet (Part IV) — existential.** Every new method family
multiplies top-level names under the current export-everything pattern. Breadth is
only a feature if a curated entry layer routes it; otherwise breadth *is* the
overwhelm. The facade also becomes the place where "which estimator for this
model/index/budget" knowledge lives — with amortization in the mix, that routing
table is a real asset worth designing deliberately.

**Type-policing trim (5c), alias removal (5e) — unchanged.**

## VI.2 Findings I revise

**History/rollback (Finding 3) — revised from "delete the feature" to "keep the
feature, delete the mechanism's redundancy" (after maintainer feedback).** Two
corrections. First, a retraction: an earlier version of this addendum argued
history burdens the amortized branch — wrong; history lives on
`Approximator`/`SamplingState`, and amortized explainers would sit beside
`Explainer` as a sibling, never touching it. Second, the maintainers' framing is
accepted: an approximator *is* an anytime, iterative estimator, and `history()`
materializes that identity — convergence trajectories are what this library is
about, not a benchmarking side-quest.

Accepting the feature sharpens the critique of its mechanism, though:

1. **The evidence is already fully stored.** `SamplingState` keeps every coalition
   and value; history needs only the cut points, and `_history_n_samples` is
   exactly that tuple of ints. This part is already minimal.
2. **The parallel sampler-history tuple is redundant — by the design's own
   guarantee.** Every shipped sampler is a `UnitScheduleSampler` whose state
   (`_units_started`, `_pending_pos`) is a pure function of the emitted sample
   count, because of the fold-in/budget-split-invariance property. The sampler at
   any historical cut point is therefore reconstructible by replay
   (`sampler_at(n_samples)`). The parallel tuple on `Approximator`, its lockstep
   bookkeeping, and the runtime `"state and sampler history lengths differ"`
   check carry information the engineered determinism already guarantees. They
   would only be needed for adaptive samplers (none exist — the
   `noqa: ARG002 - schedule samplers are not adaptive` comment says so) or
   mutable ones (`mutable` returns `False` on everything shipped).
3. **If history is identity, it shouldn't be opt-in.** Cut points cost a tuple of
   ints per `sample()` call. Always-on history removes the `track_history` flag
   from three constructors, the "history is not enabled" error paths, the
   mutable-state interlocks, and `EmptyState`'s ~70 lines of rollback/history
   edge-case law (fresh approximator → `history()` returns `[self]`).

Revised recommendation: keep `history()` and `rollback()` as the public face of
"approximators are anytime estimators"; implement them as always-on cut points
plus sampler replay. Same API, minus one flag, one error type's worth of paths,
the parallel tuple, and the lockstep check. If adaptive samplers later land, that
commit reintroduces stored sampler history for exactly the samplers that need it.
(Part V, item 1 should be read with this revision.)

**`InteractionOrientation` (Finding 5b) — revised from "delete" to "design it
against FIxLIP now, or keep only a verified seam."** With cross-modal, semi-directed
interactions on the roadmap, the placeholder has a real story — the original
"speculative generality" charge is withdrawn. But a caution replaces it: a bare
`"undirected" | "directed"` string may be the wrong *shape* for the actual feature.
FIxLIP-style cross-modal interactions are less "ordered tuples of players" and more
"players partitioned into modality groups, with interactions *across* groups" —
which likely wants player-group metadata on the game/masker (which players are
patches, which are tokens) rather than an orientation flag on every explanation.
Today the directed branch of `normalize_interaction` just preserves tuple order;
nothing expresses bipartite structure. Recommendation: sketch the FIxLIP
requirement concretely (even as a failing test or design note) and check the
placeholder can express it. A placeholder that cannot express its planned feature
is still dead weight — worse, it occupies the name the real design will want.

**`CoalitionArray` ABC (Finding 5d) — revised from "defer the ABC" to "keep the
ABC; still fix the boundary default."** With hundreds of players (VLM), structured
walks, and amortized training batches, a second coalition representation (sparse,
lazy, or group-structured) is plausible and near — the abstraction is now
justified insurance rather than speculation. Unchanged, however: user-written
callables should receive plain dense boolean arrays by default
(`CallableGame`'s converter default), because the 90% path escaping the wrapper on
line one is a boundary-design fact independent of how many implementations exist.

**`share_samples` axis-tuples (Finding 5a) — softened.** Cross-modal and
multi-target explanation grids (e.g. an image×text target lattice where coalitions
should be shared along one modality axis) may be the story this feature was waiting
for. Revised recommendation: still ship `bool` only *today*, but keep the axis
design in a drawer and reintroduce it together with the concrete VLM use case and
an example that exercises it — features should land with their stories.

**Dead metadata (Finding 1) — reframed rather than reversed.** The rule matures
from "delete what nothing consumes" to "**a protocol member lands in the same
commit as its first consumer**." On this roadmap, two of the three may well find
consumers: `generalizes` is the natural backbone of amortized serving ("this
network answers any index whose order-1 restriction is SV") and of `rule_key()`;
`order_semantics` governs whether explanations are reusable across orders
(coverage indices: an order-3 explanation answers order-2 queries — valuable for
caching and progressive UIs). If those consumers are genuinely planned, keep the
members and name the consumer in the docstring; `preserves_value` still has no
candidate consumer I can see and belongs in prose.

**Shape/value algebra (Part III) — reframed from "tax" to "core, so canonicalize
now."** Non-scalar game values were treated as a legitimate-but-costly need in the
original review; the roadmap makes them first-class. That *raises* the priority of
the recommendation rather than changing it: with many more estimators coming, every
one of them pays the `to_leading`/`to_trailing` toll and risks the silent
misalignment the module docstring warns about. Pick one internal canonical layout
(value axes leading), convert once at the game boundary and once at explanation
construction, and new estimators never touch axis moves. The generic `Game[ValueT]`
+ boundary validation design is *validated* by the roadmap — this was the right
bet. One contract to write down explicitly: estimators are linear in game values,
so `ValueT` must form a vector space over the reals (addition, scalar
multiplication, and the centering `v − v(∅)` must be meaningful). Embeddings and
class-probability vectors qualify; anything nonlinear must happen in the link
function, before values exist. Stating this one sentence in the `Game` docstring
prevents an entire class of misuse.

## VI.3 New issue the roadmap raises (not in the original review)

**The `Explainer(game, index)` lifecycle does not fit amortized explainers.** The
current base binds *one game* at construction and `explain()` takes no arguments —
perfect for exact, closed-form, and sampling explainers, where a game (one instance
to explain) is the unit of work. Amortized explainers invert this: they are
*trained once* across a distribution of instances (each instance inducing a game
via the masker) and then *explain many* new instances cheaply. Their natural shape
is fit-once/explain-many:

```python
amortized = AmortizedExplainer(masker_factory_or_model, SV(), ...)
amortized.fit(X_train)          # or train(...)
explanation = amortized.explain(x_new)     # explain() takes an argument!
```

Two design consequences worth settling *before* the `Explainer` ABC ossifies:

1. **The masker layer, not the game, is the shared currency across the whole
   roadmap.** Games are per-instance; maskers (and masked predictors) describe the
   model+masking policy that amortized training needs. This is good news — the
   layering from Part I.3 already isolates the right reusable piece — but it means
   the facade should be organized around (model, masker, index), with games as an
   internal per-instance construction.
2. **Decide whether `explain()` takes a target.** Either the base grows an optional
   argument now, or amortized explainers get a sibling ABC and the facade unifies
   them. Both are fine; discovering the mismatch after the ecosystem depends on
   `explain()`'s nullary signature is not. This is the one place where I'd
   *pre-invest* in design (the thing this review mostly argues against), because
   the cost of retrofitting a base-class signature after release is ecosystem-wide.

## VI.4 Revised bottom line

The roadmap does not rehabilitate the ceremony — it makes the case for cutting it
*stronger*, because every concept in the shared core is multiplied by every new
method family. What it does change: keep `history()`/`rollback()` as the anytime-
estimator identity but rebuild them on always-on cut points plus sampler replay,
keep the `CoalitionArray` ABC, design the directed/bipartite interaction feature
properly instead of deleting the placeholder, keep the metadata members that have
named, planned consumers, canonicalize the value layout now, and settle the
amortized-explainer lifecycle before the `Explainer` ABC hardens. The strategic picture sharpens into a rule of
thumb: **breadth belongs in the registry of indices, games, and families — the
extensible edges the architecture already got right — while the shared core
(explainer bases, state, explanation containers) must get *smaller* as the roadmap
gets bigger, because it is multiplied by everything.**
