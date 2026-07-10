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
The exact number of new sampled **Coalitions** an **Approximator** evaluates on its **Game** when asked to sample. Budgets are spent exactly rather than floored, rejected, or redistributed to align with a **Sampling Quantum**.
_Avoid_: permutation count, number of iterations

**Sampling Quantum**:
The smallest number of additional samples after which an **Approximator** can incorporate new evidence into its estimate, such as one full permutation walk. A completed quantum is also called a sampled unit. Samplers own and expose their quantum; budgets do not need to align with it.
_Avoid_: iteration cost, batch size

**Pending Samples**:
Sampled **Coalitions** and evaluated **Values** that belong to an incomplete **Sampling Quantum**. Pending samples remain in the **ApproximationState** and are completed by later sampling, but are masked when an **Explanation** is materialized to preserve unbiasedness.
_Avoid_: wasted evaluations, partial batch, leftover budget

**Seed Samples**:
Deterministic evaluations an **Approximator** needs before sampled units can be interpreted, such as the empty and grand coalition. Seed samples are emitted first by the **Sampler** as a one-time prelude unit and are paid from the sample **Budget**; constructing an **Explainer** never evaluates the **Game**.
_Avoid_: initialization cost, setup evaluations, create step

**Empty State**:
The **ApproximationState** of an **Approximator** that has not sampled yet. The first sampled batch replaces it with an evidence-bearing state; **Approximation History** begins at that first evidence state, and an empty state with history enabled lists only itself.
_Avoid_: uninitialized state, null state

**Deduplication**:
An **Approximator** policy that evaluates each distinct **Coalition** on the **Game** at most once, reusing stored **Values** for repeats. Only novel evaluations count toward the **Budget**, and repeated coalitions become free evidence; the estimate is unchanged relative to sampling without deduplication.
_Avoid_: without-replacement sampling, caching flag

**SamplingStallWarning**:
A warning issued when **Deduplication** leaves **Budget** unspent because the **Sampler** cannot produce novel **Coalitions**.
_Avoid_: exhaustion error

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

**TreeModel**:
The unified node-array layout of one decision tree (children, split features, thresholds, leaf values); library-specific models convert to it through the dispatched ``to_tree_model``. Leaf values may carry trailing value axes.
_Avoid_: tree structure, tree dict

**InterventionalTreeGame**:
A **Game** over a tree ensemble realizing the interventional semantics of baseline masking exactly: present **Players** take the explained inputs' feature values, absent players the baseline's, decomposed into per-leaf present/absent reachability constraints. The game type carries the tree-explanation semantics — a path-dependent sibling game is the planned alternative — and closed-form tree explainers dispatch on it.
_Avoid_: tree wrapper, TreeSHAP game

**MaskedGame**:
A **Game** composed from a **MaskedPredictor** and a **LinkFunction**; without a link function, predictions become **Values** through the dispatched ``to_values`` conversion, whose backend handlers (torch) register lazily on first contact.
_Avoid_: linked predictor game

**Value**:
The output assigned by a **Game** to a **Coalition**; it may be scalar-valued, vector-valued, or a structured array-like element of a supported value space. Model predictions are not automatically **Values**; they become **Values** only after the game maps them into the cooperative-game value space.
_Avoid_: payoff, prediction

**ValueArray**:
A scalar, array-like object, or specialized value container representing **Values** aligned with the relevant **CoalitionArray** shape, or with the broadcasted shape of explanation targets and coalitions. Its logical shape describes the array of value elements and excludes each value's internal shape; where possible, the value representation is tracked through **Game** and **Explainer** type parameters. Dense value arrays store logical axes first, then the sample axis, then each value's internal axes; **Games** declare that internal value shape.
_Avoid_: output batch, predictions

**Array-Like Data Type**:
A shapiq-owned data container that exposes logical shape, dimensionality, and size while hiding internal representation axes.
_Avoid_: array API implementation

**Masker**:
A component that turns a **CoalitionArray** into model-native masked inputs by representing absent **Players**. Maskers are backend-general: they compute in the array backend they were built from (NumPy, JAX, torch — anything Array API compatible) and masked inputs stay in that backend on its device; per-backend execution policy (autograd, devices, chunking) belongs to predictors and callable games, never to maskers.
_Avoid_: imputer, perturbation function

**Superpixel**:
A group of pixels acting as one **Player** when explaining image models, represented by an integer label map assigning every pixel a superpixel id covering ``0 .. n_players - 1``. A grid is the simplest layout; irregular layouts (SLIC-style) are just different label maps.
_Avoid_: patch, segment

**Token**:
A position in a token-id sequence acting as one **Player** when explaining sequence models; absent tokens are replaced by a mask token id (a special kind of baseline). Grouping subword tokens into word players mirrors the **Superpixel** label-map pattern.
_Avoid_: word (unless tokens are grouped into words)

**MaskedPredictor**:
A metadata-carrying abstraction with a fixed number of **Players** and **Explanation Target** shape that accepts a **CoalitionArray** and returns model-native predictions for those coalitions.
_Avoid_: masked model, prediction game

**ModelMaskedPredictor**:
A **MaskedPredictor** formed by composing a **Masker** with a model.
_Avoid_: masked model wrapper

**ChunkedMaskedPredictor**:
A torch **MaskedPredictor** composing a **Masker** with a model that streams **Coalitions** through both in chunks, bounding the flat model batch, keeping at most one chunk of masked inputs alive per device, and moving chunks to the model's parameter device (overridable) so tensors otherwise never leave their device.
_Avoid_: dataloader, batcher

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
Whether an **Interaction** treats player order as meaningful. Undirected interactions ignore player order; directed interactions preserve player order. Orientation is a representation property of an **ExplanationArray**; every shipped **InteractionIndex** produces undirected explanations.
_Avoid_: direction flag, orderedness

**Explanation**:
A scalar element of an **ExplanationArray** for one **Explanation Target**, assigning **Attributions** to selected **Interactions** under a specified **InteractionIndex** and order.
_Avoid_: explanation map, result dict

**ExplanationArray**:
An **Array-Like Data Type** whose elements are **Explanations** for the same fixed set of **Players**, represented by batched internal data rather than as a Python array of explanation objects. Its logical shape describes the array of explanation elements and excludes interactions and value dimensions. An **ExplanationArray** records the number of players, the **InteractionIndex** object itself (parameters included, so a weighted index keeps its weighting), the resolved **Order**, and the **Interaction Orientation** of the represented attributions, carries the **Baseline** separately from attributions, inherits player metadata from the explained **Game** when available, can be called with an **Interaction** to return its **Attribution**, and can report for which explanation elements an attribution is available.
_Avoid_: explanation batch, list of explanations

**DenseExplanationArray**:
An **ExplanationArray** whose represented **Interactions** are shared across every **Explanation Target**.
_Avoid_: dense explanation

**SparseExplanationArray**:
An **ExplanationArray** whose stored **Interactions** may differ across **Explanation Targets**. Missing stored entries are not represented unless the sparse explanation defines an object-level default attribution.
_Avoid_: sparse explanation

**Attribution**:
A **Value**-shaped contribution assigned to an **Interaction** within an **Explanation**. Attributions are defined on the centered game, following the game-theoretic convention that the empty **Coalition** has value zero.
_Avoid_: score, importance

**Baseline**:
The **Value** of the **Game** at the empty **Coalition**, carried by an **ExplanationArray** separately from **Attributions**. An order-0 Attribution, where an **InteractionIndex** defines one on the centered game (FBII's fitted intercept, the Co-Moebius grand total), is conceptually distinct from the baseline.
_Avoid_: expected value, offset, order-0 attribution

**InteractionIndex**:
A uniquely named rule, represented by an immutable index object carrying a string name, an **Order**, **Order Semantics**, and any index-defining parameters (the weighted Banzhaf joining probability ``p``), that defines which **Attributions** an **Explanation** assigns to **Interactions** and how those attributions relate to a **Game**. Explainers select behavior by index type and **Index Capability**, never by name. Names include SV, BV, SII, BII, CHII, k-SII, STII, FSII, FBII, kADD-SHAP, the weighted Banzhaf family WeightedBV, WeightedBII, and WeightedFBII, the generalized values SGV, BGV, CHGV, IGV, EGV, and JointSV, and the Moebius and Co-Moebius transforms.
_Avoid_: index string, metric, method

**Order Semantics**:
Whether an **InteractionIndex** treats its **Order** as explanation coverage, leaving **Attributions** of shared **Interactions** unchanged across orders (SV, BV, SII, BII), or as part of the index identity, changing attribution values with the order (STII, FSII). Transforms with no inherent order cap (Moebius, Co-Moebius) default their order to all players.
_Avoid_: truncation flag, order mode

**Index Capability**:
A structural protocol an **InteractionIndex** implements to work with an **Explainer** family. The **Cardinal Interaction Index** capability supplies cardinality-dependent discrete-derivative weights; the **Generalized Value** capability supplies cardinality-dependent bloc-marginal weights; the regression capability supplies a per-size kernel (zero-weight endpoints mark exact constraints; nonzero endpoints mark a free-intercept fit).
_Avoid_: feature flag, supported-index list

**Cardinal Interaction Index**:
An **Index Capability** for indices whose **Attributions** are weighted sums of discrete derivatives over outside **Coalitions**, with weights depending only on cardinalities (SV, BV, WeightedBV, SII, BII, WeightedBII, CHII, STII, and the Moebius and Co-Moebius transforms).
_Avoid_: derivative index, CII when a reader may not know the acronym

**Generalized Value**:
An **Index Capability** for indices whose **Attributions** weight the marginal contributions of whole **Interactions** joining outside **Coalitions** (SGV, BGV, CHGV, IGV, EGV, JointSV).
_Avoid_: bloc value, group value

**Value Generalization**:
The declared relation between an **InteractionIndex** and the probabilistic value its order-1 restriction equals: SII, CHII, STII, k-SII, FSII, kADD-SHAP, SGV, CHGV, and JointSV generalize SV; BII, FBII, and BGV generalize BV; WeightedBII and WeightedFBII generalize WeightedBV with the same weighting, so the declared target follows the instance's parameter. Declarations are index metadata and are verified numerically by tests. An index constructed at order one **equals** the value it generalizes: index objects compare extensionally over nonempty **Interactions**, so ``SII(order=1) == SV() == CHII(order=1)``; order-0 conventions remain per-index. A declared ``None`` means no shipped value object equals the restriction, not that none exists.
_Avoid_: reduction, canonical form

**Value Preservation**:
Whether an **InteractionIndex** whose **Order Semantics** are identity still keeps its order-1 **Attributions** equal to its generalized value at every order. All coverage indices preserve trivially; kADD-SHAP preserves despite identity semantics; STII, k-SII, FSII, FBII, WeightedFBII, and JointSV do not — their order-1 attributions equal the value only when constructed at order one.
_Avoid_: order stability, value consistency

**Order**:
The maximum size of **Interactions** included in an **Explanation**. Order may be zero, in which case only the empty interaction may be represented. A second-order explanation may include singleton and pairwise interactions.
_Avoid_: degree, exact order
