# Green-room brief: a Shapley-interactions library, from zero

You are building a Python library for Shapley-value and Shapley-interaction
explanations of machine-learning models, from scratch. You get this brief,
the published literature, and nothing else.

**Rules of the room.** Do not consult the shapiq codebase (any version),
its documentation, its issue tracker, or publications presenting the
shapiq package itself — those leak the incumbent API this exercise exists
to challenge. Everything else in the published literature is fair game;
the reading list at the end names the works we like, not the works you
are limited to. Where this brief is silent, decide and write the decision
down. Your interpretation of the silence is a deliverable.

## The domain, in the language you may assume

A model behind a feature-removal scheme induces a **cooperative game**: a
value function `v` assigning a real value (or a vector of them) to every
**coalition** of players. Explanations attribute `v` to players and to
player **interactions** up to some **order**, under one of many published
**interaction indices** (Shapley values, Shapley/Banzhaf interaction
indices, Shapley–Taylor, faithful/kernel variants, k-additive fits, the
Möbius transform). Exact computation enumerates all `2^n` coalitions;
beyond ~15 players everything is estimation under a spend denominated in
**game evaluations**. One game evaluation is one query of `v` for one
(coalition, instance) pair; a vector-valued result counts once, and a
batched call that answers 512 pairs costs 512. This unit is fixed by the
brief; whether you expose a "budget" abstraction above it, and what that
abstraction looks like, is your design. The **Möbius transform** and
**least-squares / kernel-weighted** views of these indices are textbook.
Explanations can be judged by **fidelity**: how well the (sparser)
attribution object accounts for the model's behavior.

## Pillars (non-negotiable; the brief's one big bet is yours to elaborate)

**Pillars are bets, not axioms.** Non-negotiable means each pillar must be
pushed seriously to its limit — not that it must win. If a pillar
demonstrably fails a scenario — worse user code, a broken guarantee, a
performance cost you can measure and cannot pay — a documented rejection
is a *stronger* deliverable than a forced fit. A valid rejection names the
concrete scenario where the pillar breaks, shows the evidence, and ships a
worked alternative that passes the scenarios. An unargued rejection, or
one without a working alternative, is a failure. Record every rejection or
near-rejection in `DECISIONS.md`.

1. **Everything is a game.** The value function is the central abstraction.
   A model behind a feature-removal scheme is a game — and the library's
   *outputs* must live in the same currency: an explanation is itself a
   game, a sparser and readable one. Explainers map games to games. You
   work out the consequences, and they are the heart of this exercise:
   What does it mean to *evaluate* an explanation? How does a user read
   its attributions without confusing reading with evaluating? Where do
   the byproducts of estimation — spend, uncertainty, resumability —
   live, if the output is a game? What happens when you add two
   explanations, or subtract one from its model? If this pillar never
   fights you, you have not pushed it far enough; where it fought you
   belongs in `DECISIONS.md`. And we will check that game-ness *does
   work*, not just names things: at least one mechanism in your library
   must exist that is only possible, or dramatically simpler, because
   explanations are games — game arithmetic behind S4's residual, S5's
   correction as a combination of games, something we have not thought
   of — and `DESIGN.md` must point at it. Inventing such a mechanism is
   encouraged; a library where the pillar is vocabulary rather than
   load-bearing structure has not met the pillar.

2. **Optax-shaped process.** Take the discipline of the JAX ecosystem
   (optax, equinox, flax) as your model: configuration is immutable
   values, computation is pure transformation, and any state that evolves
   between calls is a first-class value the *user* holds and passes —
   never hidden attributes mutating on a long-lived object.

3. **JAX-first numerics, open boundary.** The numerical core computes in
   JAX, batched — no per-coalition Python loops on hot paths. The model
   boundary stays backend-open (a torch model receives torch tensors;
   think array-api at the removal boundary, not conversion ceremonies).

4. **Value axes and batching are a contract, not an accident.** Games may
   return vector values (class probabilities, margins) and be explained
   for many instances at once. Declare an explicit layout contract for
   how target axes, sample axes, and value axes compose — at the public
   boundary and internally — before writing estimators, and enforce it
   at the game boundary.

## Toolbox (provided, not prescribed)

For the backend boundary of pillar 3 we can offer `flextype` — lazy,
type-driven dispatch that lets you register behavior for backend types
(a torch tensor, a jax array) without importing any backend until a
value of its type actually arrives. It is installable from PyPI
(`pip install flextype`) — install it and see what it can do. Its
dispatch keys are types written as strings (e.g. `"torch.Tensor"`), so
registering behavior for a backend never imports it; this is exactly the
laziness a lazy implementation needs. We use it, we like it, and we
heavily encourage it *where it fits*: converting model-native
predictions to values, moving arrays across the removal boundary,
keeping optional backends optional. You are free to replace it with
anything that achieves the same laziness. One boundary on the
encouragement: how *your library* is extended by third parties (S6) is
yours to design from scratch — do not let a dispatch tool you were
handed decide that architecture for you.

## Design values (the taste you are being hired for)

- Identity is typed. A misspelled method or index name must be impossible
  to pass, not silently wrong. No string-keyed dispatch on the public API.
- Errors teach. Every rejection at the boundary states what to do instead.
- Configuration does not mutate. Whatever objects hold your settings, a
  user reading them twice reads the same thing. State that evolves must be
  explicit, inspectable, and live somewhere a user can point at.
- Two copies of one fact means one of them is lying. Prefer deriving to
  storing; prefer deleting a concept to documenting its confusion.
- Functional over stateful where it does not cost clarity; the numerical
  core should feel at home next to JAX-ecosystem libraries.
- Small constructors. If an `__init__` needs a dozen parameters, the type
  is wrong, not the docstring.
- Composition over inheritance. A tower of subclasses is a design smell;
  so is a registry you must touch to extend the library.
- Laziness is a feature. Defer expensive materialization until something
  is actually read; cache only what is value-equivalent to recomputing.
  Data should cross backends and accumulate without copying wherever the
  backends allow it (dlpack-style interchange, views, shared buffers) —
  and where correctness genuinely demands ownership, copy once,
  deliberately, and say so. An accidental copy is a bug you have not
  measured yet.

## Acceptance scenarios (the spec is here)

Your design is judged by whether these read naturally and run correctly in
it. Write each as a runnable script.

**S1 — On-ramps.** A trained torch tabular classifier, an image model over
superpixels, a text model over tokens, and a scikit-learn random forest are
each explained: Shapley values and order-2 interactions, exact for small
`n`, estimated for larger `n`. Feature removal is by baseline imputation
only — a removed player's features are replaced by a fixed reference value
— for every model, including the torch ones; what infrastructure this
requires is yours to decide and design. Agreement is tested, not asserted:
an estimator handed all `2^n` game evaluations must reproduce the exact
result to a numerical tolerance you state and justify, and below
exhaustion its error against the exact result must shrink across three
increasing spends on a fixed seed. The tabular classifier is also
explained for both class log-probabilities at once (vector values) and
for a batch of ten instances in one pass — the pillar-4 contract,
exercised. Peak memory for the image model is bounded regardless of how
many coalitions an estimator requests at once.

**S2 — Anytime, split-invariant, replayable.** Spending 4000 game
evaluations and then 4000 more yields *bit-identical* results to spending
8000 once — including when the spend is an awkward fraction of whatever
internal step size you choose. Bit-identity is required on one platform
and software stack, not across platforms; but the accumulation scheme must
be invariant to how a spend is split, by construction, and how you
achieved that is a `DECISIONS.md` entry. Any intermediate result can be
revisited later and continued, by the same process or a fresh one, with
identical outcomes. The same seed always reproduces the same evaluations.
An explanation can be audited after the fact: what was spent, on what,
under which settings.

**S3 — Never pay twice.** Against an expensive model, a mode in which no
(coalition, instance) pair is ever evaluated twice: repeats are free, the
spend counts only true game evaluations, and when the space is exhausted
(or novelty dries up) the unspent remainder is explicit — queryable by
the caller, visible in the audit, and returned rather than silently lost.
S2's guarantees still hold in this mode.

**S4 — Quality and comparison.** Quantify in one call how faithful an
explanation is to its model. Compare two different methods' explanations
of the same model on equal footing — same game, same spend in game
evaluations, and (where the methods permit) the same evaluation trace;
state which footing your comparison uses. Quantify what an explanation
*missed* — and explain the missed part itself.

**S5 — Correction at zero cost.** Given a cheap approximate model of `v`
(from domain knowledge or a fitted surrogate), produce an explanation that
uses the approximation for what it knows and the already-spent evaluations
of S2 for what it does not — with **zero** additional game evaluations.
Demonstrate empirically a case where this beats the direct estimate, and
for the case where it cannot help, give both a constructed example and a
written argument (a proof sketch is acceptable) for *why* no
zero-additional-cost correction can beat the direct estimate there.

**S6 — Outsiders extend it.** A third party implements (a) a new
interaction index and (b) a new estimation strategy — an active-learning
loop that picks its next coalition by a posterior criterion and reports
per-attribution uncertainty — *without modifying your library*. Both
enjoy S2's guarantees. If your design needs the third party to subclass
more than one thing or register strings, revisit it.

**S7 — Fast where it counts.** On a cheap 12-player game, sampling-based
estimation sustains at least the order of one million game evaluations
per second end to end — measured warm (JIT compilation excluded), with
repeats allowed (a 12-player game has only 4096 coalitions; S3's dedup
mode is off here), on hardware you record alongside the number.
Evaluating a fitted attribution object as a function (if your design has
such a thing) is vectorized, not a Python loop over terms.

**S8 — Structure beats sampling.** Some model families admit exact
explanation from the model's *structure*, with zero game evaluations:
for tree ensembles, exact Shapley values and interactions follow from
one pass over the trees (the TreeSHAP lineage). Your library must offer
this shortcut for S1's random forest, and the shortcut must not fork the
library: its output is the same kind of object as every other path —
composable in S4's comparisons, honest under S2's audit, which must read
a spend of exactly zero — and it must agree with the black-box exact
route on the same forest to your stated tolerance. Design so a second
structure family (linear models, say) could ride the same seam without a
redesign, and point at that seam in `DESIGN.md`. If structure-exploiting
paths and sampling paths produce different kinds of results in your
design, the currency has failed and pillar 1 owes you a fight.

## Deliverables

1. `GLOSSARY.md` — your nouns and verbs, each in two sentences. We will
   diff vocabularies before we diff code.
2. `DESIGN.md` — the type inventory and how the scenarios flow through it.
3. A working prototype with tests: all eight scenarios runnable.
4. `DECISIONS.md` — the five hardest calls you made, the alternatives you
   rejected, and why. At least two must be about pillar 1's edges: the
   places "everything is a game" resisted you and how you resolved it —
   or, per the pillar-rejection clause above, where you rejected it and
   what evidence earned the rejection. If nothing felt hard, you have not
   found the domain's real tensions yet; look again at S2, S5, and S6
   together, under pillar 1.

Do not aim for feature completeness across all published indices; aim for
the *shape* that could hold them. Cut scope anywhere except the scenarios.

## Reading (all references are fair game; these are the ones we like)

- Lundberg & Lee (2017), *A Unified Approach to Interpreting Model
  Predictions* (KernelSHAP) — https://arxiv.org/abs/1705.07874
- Covert & Lee (2021), *Improving KernelSHAP: Practical Shapley Value
  Estimation via Linear Regression* — https://arxiv.org/abs/2012.01536
- Fumagalli et al. (2023), *SHAP-IQ: Unified Approximation of any-order
  Shapley Interactions* — https://arxiv.org/abs/2303.01179
- Kolpaczki et al. (2024), *SVARM-IQ: Efficient Approximation of Any-order
  Shapley Interactions through Stratification* —
  https://arxiv.org/abs/2401.13371
- Tsai et al. (2023), *Faith-Shap: The Faithful Shapley Interaction
  Index* — https://arxiv.org/abs/2203.00870
- Lundberg et al. (2020), *From Local Explanations to Global
  Understanding with Explainable AI for Trees* (TreeSHAP) —
  https://arxiv.org/abs/1905.04610
- A perspective we value — https://openreview.net/forum?id=tp3Aw6t0QF
- A perspective we value — https://openreview.net/forum?id=gAO7AFSTJD
