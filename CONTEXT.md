# shapiq Context

`shapiq` is a Python library for explainable AI, focused on Shapley-based interaction explanations for machine learning models.

## Language

**Explainer**:
Any strategy that takes a **Game** and configuration and produces an **Estimate** when asked. Explainers are frozen policies: binding a game and an index is configuration, and all evolving state rides in the returned estimate.
_Avoid_: Computer, calculator, engine

**Approximator**:
An **Explainer** that estimates an **InteractionIndex** from sampled **Game** evaluations through the policy verbs — estimate opens a carry, refine spends more **Budget** on one, and a policy refuses to continue another policy's carry.
_Avoid_: approximate computer, estimator

**ExactExplainer**:
An **Explainer** whose **Estimate** carries complete evidence: the full coalition sweep is its provenance and spending is visible, never hidden.
_Avoid_: ExactComputer, exact calculator

**Evidence**:
The accumulated record an **Estimate** is derived from — the sufficient statistic for exact resume. Everything else about an estimate is either recomputed from evidence or checkpointed with it; a policy's proposals are memoryless given the carried evidence, which is what makes budget splits, rollback, and replay exact. (Rename of ApproximationState; lands with the engine rewrite.)
_Avoid_: sample history, cache, state (when speaking of the concept)

**Approximation History**:
The always-on record of value-equivalent earlier evidence an **Evidence** can restore or list after functional transitions — one checkpoint per sample call, carrying the sample count and the **Bank**. History is identity, not a feature: an **Approximator** is an anytime estimator, and rollback plus resampling replays the same evidence. Efficient history retains shared backing storage from later states.
_Avoid_: previous-state pointer, undo stack, track_history flag

**SampledEvidence**:
**Evidence** recording sampled **Coalitions** together with their evaluated **Values**, and owning coalition identity: distinct coalitions, first positions, multiplicities. (Rename of SamplingState; lands with the engine rewrite. The gradient bridge will add a sibling species: path points with gradients.)
_Avoid_: default state, raw sample cache

**Sampler**:
A stateless draw value used by an **Approximator**: permutation samplers draw permutations, coalition samplers draw coalitions, and every draw derives from its unit index, so samplers never evolve. Shape policy is sampler-owned; everything downstream — rendering, budgets, evaluation — is approximator logic. Samplers are vehicles: they own the sampling procedure, never estimator logic.
_Avoid_: generator, coalition generator

**Walk**:
The coalition block one permutation materializes into, declared by the **Approximator** family that decodes it: a length, a render from player positions to walk masks, and an optional deterministic prelude extending the **Seed Samples**. The layout has one owner and never rides in the **Sampler**.
_Avoid_: walk plan object, walk sampler subclass

**Measure**:
A probability weighting over **Coalitions** — the inner product of the game space. The measure is part of an index's identity: a projection index is a subspace plus a measure, projections compose into a tower only under a shared measure, and **Fidelity** is distance under one. A **Sampler**'s **Sampling Law** targets a measure, which is why kernel-matched sampling makes the unweighted solve correct.
_Avoid_: weighting scheme, kernel (when speaking of the concept)

**Fidelity**:
How faithfully one **Game** accounts for another: the weighted R-squared of a surrogate against the explained game under a **Measure**. An **Explanation**'s order is its fidelity dial — order n is exact.
_Avoid_: accuracy, faithfulness score

**Extension**:
A differentiable function on the unit cube agreeing with a **Game** at the vertices. Gradient explainers integrate along paths of an extension, and the extension is part of the method: the same diagonal integral yields Integrated Gradients on the model's own extension and the Shapley value on the multilinear one.
_Avoid_: interpolation, relaxation

**Sampling Law**:
The marginal probability distribution of one drawn **Coalition**, declared by a **Sampler** as an optional capability — log-space, after wrapper transformations such as pairing. Coalition samplers declare their law; permutation samplers do not, because their draws are permutations, not coalitions. **Seed Samples** sit outside the law.
_Avoid_: sampling weights, proposal distribution

**Budget**:
The number of new game evaluations an **Approximator** is asked to spend. Budgets are spent in whole sampled units; the remainder is carried in the **Bank** and spent first on the next call, so splits never change the sampled evidence and no evaluation is made that cannot inform an estimate.
_Avoid_: permutation count, number of iterations

**Bank**:
The integer budget remainder an **Approximator** carries between sample calls: whole-unit spending banks what is left over, and deduplication may borrow (a negative bank) when its final unit overshoots. Checkpoints record the bank, so rollback restores the exact resume point.
_Avoid_: pending budget, leftover counter

**Sampled Unit**:
The coalition rows one draw materializes into — one full permutation walk, or one drawn coalition (doubled under pairing). Approximators spend budgets in whole units, so every stored row belongs to a complete unit and estimates never see partial evidence.
_Avoid_: sampling quantum, iteration cost, batch size

**Seed Samples**:
Deterministic evaluations an **Approximator** needs before sampled units can be interpreted: the empty and grand coalition, plus any family prelude such as STII's lower-order anchors. The approximator evaluates the seed block once, paid from the first sample **Budget**; constructing an **Explainer** never evaluates the **Game**.
_Avoid_: initialization cost, setup evaluations, create step

**NoEvidence**:
The **Evidence** of an **Estimate** that has not sampled yet; banked **Budget** may still ride it. The first sampled batch replaces it, **Approximation History** begins at that first evidence, and no evidence is its own single-entry history. (Rename of EmptyState; lands with the engine rewrite.)
_Avoid_: uninitialized state, null state, empty state

**Deduplication**:
An **Approximator** policy that evaluates each distinct **Coalition** on the **Game** at most once, reusing stored **Values** for repeats. Only novel evaluations count toward the **Budget**, and repeated coalitions become free evidence; the estimate is unchanged relative to sampling without deduplication.
_Avoid_: without-replacement sampling, caching flag

**SamplingStallWarning**:
A warning issued when **Deduplication** leaves **Budget** unspent because the **Sampler** cannot produce novel **Coalitions**.
_Avoid_: exhaustion error

**InsufficientSamplesError**:
An error raised when reading an **Estimate** whose evidence cannot support coefficients yet; the carry itself stays legal, so banked budgets survive.
_Avoid_: empty state error

**UnsupportedGameError**:
An error raised when an **Explainer** cannot work with the supplied **Game**.
_Avoid_: invalid game error

**Game**:
A value function: the cooperative-game abstraction assigning **Values** to **Coalitions** over a fixed set of **Players**, with an **Explanation Target** shape. Its **ValueArray** shape follows the broadcasted shape of the targets and the **CoalitionArray**; the boundary also accepts plain dense masks and wraps them. Games are the library's one currency: models behind **Maskers** enter as games, and every **Explanation** produced is again a game.
_Avoid_: model wrapper

**Basis**:
A value object spanning the game space with its own atoms: Moebius atoms fire when all of an **Interaction**'s players are present (synergy), Co-Moebius atoms when any is present (redundancy), Fourier atoms by presence parity. Sparsity is basis-relative, so declaring the basis is a modeling statement; a basis is not an **InteractionIndex** — indices add a measure and semantics on top.
_Avoid_: basis string, transform flag

**BasisGame**:
A **Game** known through a finite coefficient vector on a declared **Basis**: evaluating it plays the surrogate, indexing it reads a coefficient. Unlisted **Interactions** are zero, the empty interaction included — the empty slot is an ordinary coefficient whose meaning each producer declares. Sparse explanations are basis games with fewer terms, not a separate kind.
_Avoid_: parametric game, coefficient table

**Estimate**:
A **BasisGame** carrying its estimation provenance: the evidence it was derived from, the **Bank**, the **InteractionIndex** it was made under, its producer's fingerprint, and optional per-interaction uncertainty. Estimates are inert — process verbs live on the **Approximator** — and every producer returns one: sampled estimates carry sampled evidence, exact estimates complete evidence, tree estimates none.
_Avoid_: result object, explanation array

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
A subset of distinct **Players** that receives an **Attribution** in an **Explanation**. The empty interaction is allowed and is an ordinary coefficient slot.
_Avoid_: explanation coalition, tuple key

**Explanation**:
A **Game** interpreted as the account of another game: readable in a declared **Basis**, its coefficients are the **Attributions**. An explanation of order k is a k-additive surrogate — a lossy compression of the explained game whose order is the fidelity dial.
_Avoid_: explanation map, result dict, explanation array

**Attribution**:
A **Value**-shaped coefficient assigned to an **Interaction** by an **Explanation** — one readable term of the surrogate game.
_Avoid_: score, importance

**Baseline**:
The **Value** of the explained **Game** at the empty **Coalition**. Each **InteractionIndex** family declares what the empty slot of its **Explanation** holds: the baseline where the surrogate interpolates the empty coalition (the efficiency family), a fitted intercept (FBII-style fits), or nothing (kADD-SHAP).
_Avoid_: expected value, offset

**InteractionIndex**:
A uniquely named rule, represented by an immutable index object carrying a string name, an **Order**, and any index-defining parameters (the weighted Banzhaf joining probability ``p``), that defines which **Attributions** an **Explanation** assigns to **Interactions** and how those attributions relate to a **Game**. Explainers select behavior by index type and **Index Capability**, never by name. Names include SV, BV, SII, BII, CHII, k-SII, STII, FSII, FBII, kADD-SHAP, the weighted Banzhaf family WeightedBV, WeightedBII, and WeightedFBII, the generalized values SGV, BGV, CHGV, IGV, EGV, and JointSV, and the Moebius and Co-Moebius transforms.
_Avoid_: index string, metric, method

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
