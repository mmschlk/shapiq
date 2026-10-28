# Sorted Finite Player IDs

**Status: Accepted.** Refines the concrete finite-player representation without
changing the logical batch contracts in ADRs 0001 and 0002.

## Decision

`FinitePlayers` stores IDs in ascending order along each set's final axis.
Construction sorts explicit ID arrays by default. The keyword-only
`sorted=True` flag skips sorting when the caller guarantees that IDs are already
sorted; this promise is not checked. Count-only and metadata-only construction
already generate sorted IDs. Sorting neither renumbers nor deduplicates IDs. The
optional metadata catalog remains shared and indexed by ID.

Coalition membership positions refer to this canonical sorted order, not the
order of the constructor's input array. Game adapters must therefore arrange
their player-dependent storage consistently; sorting players does not reorder
an independently supplied game's values.

## Trade-Off

Canonical order gives up arbitrary caller-selected player order in exchange for
efficient batched lookup and removal. Removing `k` IDs from `n` sorted players
uses binary search and linear compaction rather than an `n`-by-`k` match array
and a full-player sort. Only removal IDs need sorting. Removal reconstructs its
sorted subset through the normal constructor with `sorted=True`, avoiding
redundant sorting while retaining constructor validation.

Missing removal IDs, duplicate removal IDs, and requested IDs occurring more
than once in a set remain errors, preserving a fixed output width for every set
in the batch.
