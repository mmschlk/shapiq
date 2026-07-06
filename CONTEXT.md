# shapiq Context

`shapiq` is a Python library for explainable AI, focused on Shapley-based interaction explanations for machine learning models.

## Language

**Explainer**:
A strategy that takes a **Game** and configuration and produces an **ExplanationArray** when asked to explain.
_Avoid_: Computer, calculator, engine

**Approximator**:
An **Explainer** that estimates an **InteractionIndex** from sampled **Game** evaluations.
_Avoid_: approximate computer, estimator

**ExactExplainer**:
An **Explainer** that explains an **InteractionIndex** without sampling approximation.
_Avoid_: ExactComputer, exact calculator

**ApproximationState**:
The algorithm-specific accumulated evidence an **Approximator** uses to estimate an **InteractionIndex**, such as sampled **Coalitions**, **Values**, weights, duplicate-tracking data, or sufficient statistics. Only states that track evaluated game results need to carry value data.
_Avoid_: sample history, cache

**Approximation History**:
Optional bookkeeping that lets an **ApproximationState** restore or list value-equivalent earlier states after functional state transitions. It is mainly intended for post-hoc analysis of how approximations change with increasing budget; efficient history may retain shared backing storage from later states.
_Avoid_: previous-state pointer, undo stack

**SamplingState**:
An **ApproximationState** that records sampled **Coalitions** together with their evaluated **Values**.
_Avoid_: default state, raw sample cache

**Sampler**:
A component used by an **Approximator** to propose a **CoalitionArray** from an **ApproximationState** and sample budget. Samplers may evolve during sampling by returning a next sampler together with sampled coalitions; mutable samplers must not be used with **Approximation History**. Shape policy is sampler-owned and normally trusted by approximators rather than revalidated on every sample.
_Avoid_: generator, coalition generator

**Budget**:
The exact number of new **Game** evaluations an **Approximator** spends when asked to sample. Budgets are spent exactly rather than floored, rejected, or redistributed to align with a **Sampling Quantum**.
_Avoid_: permutation count, number of iterations

**Sampling Quantum**:
The smallest number of additional samples after which an **Approximator** can incorporate new evidence into its estimate, such as one full permutation walk. Samplers own and expose their quantum; budgets do not need to align with it.
_Avoid_: iteration cost, batch size

**Pending Samples**:
Sampled **Coalitions** and evaluated **Values** that belong to an incomplete **Sampling Quantum**. Pending samples remain in the **ApproximationState** and are completed by later sampling, but are masked when an **Explanation** is materialized to preserve unbiasedness.
_Avoid_: wasted evaluations, partial batch, leftover budget

**InsufficientSamplesError**:
An error raised when an **Approximator** cannot produce an **ExplanationArray** from its current **ApproximationState**.
_Avoid_: empty state error

**UnsupportedGameError**:
An error raised when an **Explainer** cannot work with the supplied **Game**.
_Avoid_: invalid game error

**SamplingError**:
An error raised when a **Sampler** cannot produce the requested **CoalitionArray**.
_Avoid_: sampler failure

**HistoryError**:
An index-style error raised when **Approximation History** is unavailable or cannot satisfy a requested history operation.
_Avoid_: rollback error, undo error

**Game**:
A cooperative-game base abstraction with a fixed number of **Players** and **Explanation Target** shape that accepts a **CoalitionArray** and returns **Values** for those coalitions. Its **ValueArray** shape follows the broadcasted shape of the targets and the **CoalitionArray**. Games validate player-count compatibility at the boundary where they receive coalitions.
_Avoid_: value function, model wrapper

**CallableGame**:
A **Game** adapter for a callable that already maps **CoalitionArrays** to **Values**, adding game metadata and backend conversion at the boundary.
_Avoid_: FunctionGame, WrappedGame

**MaskedGame**:
A **Game** composed from a **MaskedPredictor** and a **LinkFunction**.
_Avoid_: linked predictor game

**Value**:
The output assigned by a **Game** to a **Coalition**; it may be scalar-valued, vector-valued, or a structured array-like element of a supported value space. Model predictions are not automatically **Values**; they become **Values** only after the game maps them into the cooperative-game value space.
_Avoid_: payoff, prediction

**ValueArray**:
A scalar, array-like object, or specialized value container representing **Values** aligned with the relevant **CoalitionArray** shape, or with the broadcasted shape of explanation targets and coalitions. Its logical shape describes the array of value elements and excludes each value's internal shape; where possible, the value representation is tracked through **Game** and **Explainer** type parameters.
_Avoid_: output batch, predictions

**Array-Like Data Type**:
A shapiq-owned data container that exposes logical shape, dimensionality, and size while hiding internal representation axes.
_Avoid_: array API implementation

**Masker**:
A component that turns a **CoalitionArray** into model-native masked inputs by representing absent **Players**.
_Avoid_: imputer, perturbation function

**MaskedPredictor**:
A metadata-carrying abstraction with a fixed number of **Players** and **Explanation Target** shape that accepts a **CoalitionArray** and returns model-native predictions for those coalitions.
_Avoid_: masked model, prediction game

**ModelMaskedPredictor**:
A **MaskedPredictor** formed by composing a **Masker** with a model.
_Avoid_: masked model wrapper

**PredictionArray**:
A loose term for the model-native prediction structure returned by a **MaskedPredictor** and consumed by a **LinkFunction**. A **PredictionArray** is not a concrete type or protocol and does not become a **ValueArray** until a **LinkFunction** maps and normalizes it.
_Avoid_: ValueArray, model output when discussing the composed game contract

**LinkFunction**:
A component that maps model-native predictions into **Values** and normalizes them into the representation expected by a **Game**. A **LinkFunction** receives only model predictions when called; additional information is supplied when the link is constructed.
_Avoid_: ValueMapper, value processor, output processor

**Explanation Target**:
An input case or other subject for which an **Explanation** is produced.
_Avoid_: target, instance when used generically

**Sample Sharing**:
The sampler policy for sharing sampled **Coalitions** across **Explanation Target** axes by replacing selected target dimensions with size one before appending the sample budget axis. The default policy does not share samples and preserves the target shape.
_Avoid_: batch broadcast, broadcast flag

**CoalitionArray**:
An **Array-Like Data Type** whose elements are **Coalitions** for a fixed set of **Players**. Each **Coalition** records whether each **Player** is present or absent, and a **CoalitionArray** may have arbitrary array shape. Every **CoalitionArray** exposes its number of players independently of its storage representation; its logical shape describes the array of coalition elements and excludes the player dimension.
_Avoid_: coalition batch, mask array

**DenseCoalitionArray**:
A **CoalitionArray** backed by a dense boolean array whose final storage dimension represents player membership.
_Avoid_: dense mask array

**Coalition**:
A scalar element of a **CoalitionArray** representing one subset of **Players**.
_Avoid_: player mask, sample

**Player**:
An explainable unit whose presence or absence is represented in a **Coalition**.
_Avoid_: feature, variable, participant

**Interaction**:
A subset or ordered tuple of distinct **Players**, depending on the **Interaction Orientation**, that receives an **Attribution** in an **Explanation**. The empty interaction is allowed; fixed-size multi-interaction access may use an array-api-compatible integer array whose final axis stores player indices.
_Avoid_: explanation coalition, tuple key

**Interaction Orientation**:
Whether an **Interaction** treats player order as meaningful. Undirected interactions ignore player order; directed interactions preserve player order.
_Avoid_: direction flag, orderedness

**Explanation**:
A scalar element of an **ExplanationArray** for one **Explanation Target**, assigning **Attributions** to selected **Interactions** under a specified **InteractionIndex** and order.
_Avoid_: explanation map, result dict

**ExplanationArray**:
An **Array-Like Data Type** whose elements are **Explanations** for the same fixed set of **Players**, represented by batched internal data rather than as a Python array of explanation objects. Its logical shape describes the array of explanation elements and excludes interactions and value dimensions. An **ExplanationArray** records the number of players, **InteractionIndex**, **Order**, and **Interaction Orientation** of the represented attributions, inherits player metadata from the explained **Game** when available, can be called with an **Interaction** to return its **Attribution**, and can report for which explanation elements an attribution is available.
_Avoid_: explanation batch, list of explanations

**DenseExplanationArray**:
An **ExplanationArray** whose represented **Interactions** are shared across every **Explanation Target**.
_Avoid_: dense explanation

**SparseExplanationArray**:
An **ExplanationArray** whose stored **Interactions** may differ across **Explanation Targets**. Missing stored entries are not represented unless the sparse explanation defines an object-level default attribution.
_Avoid_: sparse explanation

**Attribution**:
A **Value**-shaped contribution assigned to an **Interaction** within an **Explanation**.
_Avoid_: score, importance

**InteractionIndex**:
A uniquely named rule, referred to by a string name, that defines which **Attributions** an **Explanation** assigns to **Interactions** and how those attributions relate to a **Game**. Initial names include SV, SII, k-SII, STII, and FSII.
_Avoid_: index, metric, method

**Order**:
The maximum size of **Interactions** included in an **Explanation**. Order may be zero, in which case only the empty interaction may be represented. A second-order explanation may include singleton and pairwise interactions.
_Avoid_: degree, exact order
