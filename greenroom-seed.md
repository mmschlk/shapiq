# Green-room brief: a Shapley-interactions library, from zero

You are building a Python library for Shapley-value and Shapley-interaction
explanations of machine-learning models, from scratch. You get this brief,
the published literature, and nothing else.

**Rules of the room.** Do not consult the shapiq codebase (any version), its
documentation, or its issue tracker. Textbooks and papers are fair game —
in particular KernelSHAP (Lundberg & Lee 2017), Unbiased KernelSHAP (Covert
& Lee 2021), SHAP-IQ (Fumagalli et al. 2023), SVARM-IQ (Kolpaczki et al.
2024), Faithful Shapley Interactions (Tsai et al. 2023), and ProxySHAP.
Where this brief is silent, decide and write the decision down. Your
interpretation of the silence is a deliverable.

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
limit. Peak memory for the image model is bounded regardless of how many
coalitions an estimator requests at once.

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
   rejected, and why. If nothing felt hard, you have not found the domain's
   real tensions yet; look again at S2, S5, and S6 together.

Do not aim for feature completeness across all published indices; aim for
the *shape* that could hold them. Cut scope anywhere except the scenarios.
