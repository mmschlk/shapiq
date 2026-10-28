# Read-Only Latent Player Catalogs

**Status: Accepted.** Refines the shared metadata behavior in ADR 0007 for
ndlike 0.2.4's explicit pytree/static-metadata contracts.

## Decision

`FinitePlayers.players` holds an optional `LatentPlayers[P]` catalog. Raw inputs
are normalized once; existing catalogs are reused through construction,
indexing, reshaping, and removal. Catalogs implement the read-only `Sequence`
interface, plus one-dimensional boolean filtering. Integer indexing returns a
player; slicing and filtering return catalogs. No mutation, arithmetic,
reshaping, or ordering operations are exposed. Membership, count, and index
compare array-valued players as whole arrays.

The catalog is a one-dimensional sequence abstraction over array-like storage;
indexing uses the backing's first logical axis without exposing array operations
on the catalog. NumPy and JAX array inputs retain their backend. ndlike
`JaxLike` and `NumpyLike` backings preserve their logical axes and structured
representations without coercion. Python lists and other non-array iterables
always become one-dimensional NumPy object arrays, preserving each entry by
reference. No stacking, numeric conversion, or dtype promotion is inferred
from their contents. Callers who want packed storage provide an array-like
backing explicitly.

Existing array-like backings are reused directly, including one-dimensional
NumPy object arrays. Construction and indexing do not defensively copy storage,
freeze buffers, or change write flags. Indexing retains the backing's normal
view/allocation behavior; normalizing sequences and boolean indexing may still
allocate as required by the representation. Read-only describes the catalog's
interface, not ownership of its backing. Callers must not mutate supplied
backings, wrapper state, metadata objects, or returned players/views.

Catalog equality and hashing use identity, independently of contents.
`FinitePlayers` therefore uses ndlike's default pytree implementation: IDs are
dynamic children, while counts and the catalog are static auxiliary data.
There is no custom flattening or additional metadata-reference wrapper.

## Trade-Off

JAX requires static auxiliary data to support stable hashing and equality.
Identity-keyed catalogs retain arbitrary Python metadata without deep hashing
or requiring a tuple of hashable entries. This gives up list-style catalog
equality and sharing of the original mutable input list. Equal-content catalogs
with different identities produce different JIT cache keys; reusing a catalog
avoids that duplication. Avoiding defensive snapshots keeps wrapping and access
cheap, at the cost of relying on the caller's no-mutation promise. Mutations of
shared data are unsupported and do not invalidate compiled computations: this metadata is for Python-side
interpretation, not mutable input to compiled numerical behavior. Data that
should participate in transformations must be explicit dynamic fields instead.
