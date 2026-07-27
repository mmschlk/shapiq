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
beyond ~15 players everything is estimation under a **budget** denominated
in game evaluations. The **Möbius transform** and **least-squares /
kernel-weighted** views of these indices are textbook. Explanations can be
judged by **fidelity**: how well the (sparser) attribution object accounts
for the model's behavior.

## Pillars (non-negotiable; the brief's one big bet is yours to elaborate)

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
   belongs in `DECISIONS.md`.

2. **Optax-shaped process.** Take the discipline of the JAX ecosystem
   (optax, equinox, flax) as your model: configuration is immutable
   values, computation is pure transformation, and any state that evolves
   between calls is a first-class value the *user* holds and passes —
   never hidden attributes mutating on a long-lived object.

3. **JAX-first numerics, open boundary.** The numerical core computes in
   JAX, batched — no per-coalition Python loops on hot paths. The model
   boundary stays backend-open (a torch model receives torch tensors;
   think array-api at the masking layer, not conversion ceremonies).

4. **Value axes and batching are a contract, not an accident.** Games may
   return vector values (class probabilities, margins) and be explained
   for many instances at once. Declare an explicit layout contract for
   how target axes, sample axes, and value axes compose — at the public
   boundary and internally — before writing estimators, and enforce it
   at the game boundary.

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

## Acceptance scenarios (the spec is here)

Your design is judged by whether these read naturally and run correctly in
it. Write each as a runnable script.

**S1 — On-ramps.** A trained torch tabular classifier, an image model over
superpixels, a text model over tokens, and a scikit-learn random forest are
each explained: Shapley values and order-2 interactions, exact for small
`n`, budgeted estimation for larger `n`. Exact and estimated agree in the
limit. The tabular classifier is also explained for both class
log-probabilities at once (vector values) and for a batch of ten
instances in one pass — the pillar-4 contract, exercised. Peak memory for
the image model is bounded regardless of how many coalitions an estimator
requests at once.

**S2 — Anytime, split-invariant, replayable.** Spending a budget of 4000
and then 4000 more yields *bit-identical* results to spending 8000 once —
including when budgets are awkward fractions of whatever internal step
size you choose. Any intermediate result can be revisited later and
continued, by the same process or a fresh one, with identical outcomes.
The same seed always reproduces the same evaluations. An explanation can
be audited after the fact: what was spent, on what, under which settings.

**S3 — Never pay twice.** Against an expensive model, a mode in which no
coalition is ever evaluated twice: repeats are free, the budget counts only
true model calls, and when the space is exhausted (or novelty dries up)
the surplus budget is not silently lost. S2's guarantees still hold in
this mode.

**S4 — Quality and comparison.** Quantify in one call how faithful an
explanation is to its model. Compare two different methods' explanations
of the same model on equal footing. Quantify what an explanation *missed*
— and explain the missed part itself.

**S5 — Correction at zero cost.** Given a cheap approximate model of `v`
(from domain knowledge or a fitted surrogate), produce an explanation that
uses the approximation for what it knows and the already-spent evaluations
of S2 for what it does not — with **zero** additional model calls — and
demonstrate when this beats the direct estimate and when it provably
cannot.

**S6 — Outsiders extend it.** A third party implements (a) a new
interaction index and (b) a new estimation strategy — an active-learning
loop that picks its next coalition by a posterior criterion and reports
per-attribution uncertainty — *without modifying your library*. Both
enjoy S2's guarantees. If your design needs the third party to subclass
more than one thing or register strings, revisit it.

**S7 — Fast where it counts.** On a cheap 12-player game, sampling-based
estimation sustains at least the order of one million model evaluations
per second end to end. Evaluating a fitted attribution object as a
function (if your design has such a thing) is vectorized, not a Python
loop over terms.

## Deliverables

1. `GLOSSARY.md` — your nouns and verbs, each in two sentences. We will
   diff vocabularies before we diff code.
2. `DESIGN.md` — the type inventory and how the scenarios flow through it.
3. A working prototype with tests: all seven scenarios runnable.
4. `DECISIONS.md` — the five hardest calls you made, the alternatives you
   rejected, and why. At least two must be about pillar 1's edges: the
   places "everything is a game" resisted you and how you resolved it.
   If nothing felt hard, you have not found the domain's real tensions
   yet; look again at S2, S5, and S6 together, under pillar 1.

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
- A perspective we value — https://openreview.net/forum?id=tp3Aw6t0QF
- A perspective we value — https://openreview.net/forum?id=gAO7AFSTJD
