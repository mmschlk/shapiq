# Core Array Abstractions

shapiq uses nominal array-like abstractions for coalitions and explanations instead of plain arrays or Python lists of objects. `CoalitionArray` and `ExplanationArray` expose logical `shape`, `ndim`, and `size` while hiding representation axes such as the player axis, interaction storage, and value dimensions; scalar coalitions and scalar explanations are zero-dimensional instances of those same abstractions rather than separate public types.

This is deliberately stricter than accepting arbitrary arrays everywhere, because coalitions and explanations need domain metadata such as `n_players`, `interaction_index`, `order`, and orientation. It is also deliberately looser than a full Python Array API implementation, because these objects are domain containers whose logical elements are coalitions or explanations, not numeric scalars.

Dense and sparse explanation variants share the same public abstraction. Dense explanations represent the full interaction structure implied by their metadata, while sparse explanations may store different interactions per explanation target and may provide an object-level default attribution for missing stored entries.
