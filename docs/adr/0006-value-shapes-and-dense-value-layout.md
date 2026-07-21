# Value Shapes And The Dense Value Layout

Games declare the internal shape of their values: `value_shape` on the game, defaulting to scalar. Dense value arrays carry the broadcast of the target shape and the coalition array's leading axes first, then the sample axis, then the declared value shape, and the game validates this contract at its boundary whenever it is called. Declaration was chosen over inference because lazy seeding defers the first evaluation until sampling — inference would leave estimators unable to validate or allocate before that point, degrading validation to trust — and because a misdeclared game should fail at its first evaluation with an error that names the fix rather than mislabeling axes silently.

The layout follows from contracts that already existed: a ValueArray's logical shape excludes each value's internal axes, so logical axes lead and internal axes trail; `SamplingState` was built for exactly this, appending on the sample axis by position and slicing history with a trailing ellipsis, and needed no changes; `DenseExplanationArray` mirrors it with the interaction axis at the target position and value axes trailing. Explanation arrays now record their `value_shape` and validate every attribution block against targets, interaction count, and value shape — previously blocks were not validated at all.

Estimators are linear in values, so vector estimation is per-component identical to scalar runs over the same sampled stream; the tests pin this for the exact explainer, permutation sampling, and the faithful regression, whose least squares solve simply gains right-hand sides. Internally, estimators move the value axes to the front for accumulation and back to trailing when assembling explanations: leading value axes broadcast against mask-derived arrays by left-padding, which is always correct, whereas trailing value axes align from the right and can silently pair a value dimension with a walk or window dimension of the same size. The moves are transposes at the evidence and explanation boundaries; sampling, deduplication, pending masking, and history never see value axes because they are coalition-sided.

`MaskedGame` declares its `value_shape` at the composition site rather than on the link: the glossary already assigns the link function the job of normalizing predictions into the representation the game expects, so the game owns the value-space declaration and links stay plain callables. Evidence memory scales with value size — a hundred thousand samples of thousand-dimensional float32 values is roughly 400 MB — so a link that reduces predictions to the outputs of interest remains the recommended composition; vector values are for when per-component attributions are the point.

## Amendment (2026-07-21): one canonical internal layout

The boundary contract above is unchanged — games return and validate the public layout
(broadcast targets, then samples, then value axes). What changed is everything behind it:
the internal world now has ONE canonical layout — value axes leading, then target axes,
with the sample axis last — and values cross between the two layouts at exactly two seams.
On entry, `EvidenceApproximator._evaluate` (sampled evidence) and
`ExactExplainer._game_values` (the cached powerset) convert boundary values to canonical;
on exit, explanation construction converts attribution blocks and baselines back to the
public trailing layout. `SamplingState` stores canonical values, which *simplifies* it:
the sample axis is always `[..., :n]`, so alignment validation, chunk concatenation, and
history slicing lost their target-shape axis arithmetic. The scattered
`to_leading`/`to_trailing` churn inside estimators is gone — an estimator author never
moves value axes, and the silent-misalignment class the previous paragraph's "moves are
transposes at the boundaries" sentence defended against is now unrepresentable between
the seams. Riding along: the `Game` docstring states the vector-space contract explicitly
— estimators are linear in values, so `ValueT` must support addition, scalar
multiplication, and centering `v − v(empty)`; anything nonlinear belongs in the link
function before predictions become values.
