# Latent Player Space

**Status: Accepted.** Extends the finite-player representation in ADR 0006.

## Decision

`FinitePlayers.n_players` counts positions in each current set;
`latent_n_players` counts the shared ID space. IDs are nonnegative integers
strictly below the latent count. Local coalition positions follow each set's
sorted IDs, while latent coalition positions are indexed directly by ID.

Count-only and metadata-only construction default to the generated local count
as the latent count. With metadata, the latent count is always the catalog length.
Without metadata, explicit IDs infer one batch-wide `max(ids) + 1`, or zero for
an empty array. The keyword-only `latent_n_players` argument with IDs can reserve
unreferenced IDs and must cover all references. A positional integer always means
the local `n_players` count, avoiding ambiguity between counts. Every constructor
variant accepts a redundant explicit latent count. With metadata it must match
the catalog length; with generated IDs it may equal or exceed the local count.
A known count skips max-based inference while retaining ID bounds validation.
This also lets ordinary dataclass reconstruction forward the existing count
without a special override to clear it.

Removal, indexing, and reshaping preserve the original latent count and metadata
catalog rather than inferring a smaller space from the remaining IDs.
Expansion broadcasts player sets and coalitions symmetrically, scatters local
membership into latent ID positions, and zero-fills absent IDs. Repeated local
IDs combine membership by logical OR. Packed input produces packed output using
the existing little-endian bit convention. Latent admissibility checks only
type, dtype, width, and batch broadcasting, not coalition values.

## Trade-Off

Keeping a stable latent space gives different player subsets common coalition
coordinates and preserves metadata identity after removal. It retains unused
slots, so expanded coalitions may be substantially wider than local coalitions.
Latent-size inference and bounds validation inspect concrete IDs during
construction; expansion and shape-only validation remain usable under JIT on
constructed player batches.
