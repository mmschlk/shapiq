# Large benchmark reports

The optional `partitioned-v1` format stores results in small, authenticated
column blocks. It is a data-export foundation; the live website still uses its
existing format until browser integration and a real release audit pass.

```python
with RecordStore(work / "records.sqlite") as records:
    panel = compose_matrix(plan, plan_sha256=reviewed_sha, record_store=records)
    write_partitioned_report(panel, work / "new-report")
```

The output directory must be new. The writer removes incomplete output on a
Python exception and writes `data.json` last. It does not submit jobs, approve
measurements, copy website assets, or publish a release.

## What is stored

The manifest identifies the snapshot, methods, budget protocol and block
directory. Every block descriptor records its filename, SHA256, byte size, row
count, kind and snapshot. Target, family and method partitions repeat those
identifiers inside the block. All blocks use the same lossless `columns-v2`
encoding, including missing fields, explicit nulls and full numeric precision.

| Block kind | Purpose |
| --- | --- |
| `games` | Game IDs, filter fields, weighting groups, eligibility and planned budgets |
| `metrics` | Scores, statuses, query counts and timing fields for interactive comparisons |
| `summaries` | Exact existing presets, including Elo, confidence intervals and history |
| `raw` | Complete sanitized result rows for downloads and inspection |
| `profiles` | Shared worker and source information |
| `runs` | Original run IDs and provenance |
| `details` | Full game and report metadata loaded only when needed |

Internal `sequence` values restore original result order after loading multiple
partitions. They are not measurements. Worker diagnostic scalars stay on their
original rows; shared profiles preserve the remaining worker information.
Original run IDs are retained.

The manifest also contains a small global `catalog` and a `catalog.targets`
entry for each explanation target. These list filter options, player bounds,
distinct real case counts, controls, score orders, available relative budgets,
and estimated-cost availability. Planned cells count all declared methods,
budgets and estimator seeds, including unsupported cells. Preparation exclusion
counts refer to exclusion entries. Real case counts use `metadata.case_id` or
the game ID, excluding synthetic games; they are not sums across targets.

Game blocks carry a `target` header. Their `row_zero_truth_energy` flag includes
the entire record store, so hiding a method cannot restore a zero-energy game.

The writer consumes the disk-backed store and streams existing summaries. It
does not average per-batch medians, change weights or recompute estimator output.
The legacy report writer remains available for small and private local reports.

## Reading a block

`benchmark/site/partitions.js` runs in either a page or a worker:

```javascript
const block = await BenchmarkPartitions.read(manifest, descriptor, {
  baseURL: new URL("./", location.href),
  signal: controller.signal,
});
for (let i = 0; i < block.count; i++) {
  const score = block.get("nmse", i);
  // Process the column value without expanding the entire block into objects.
}
```

`has(field, i)` distinguishes an absent field from explicit null. `row(i)`
restores one row when needed. Nested dictionary values are shared and must be
treated as read-only. A local upload can supply `files: Map<filename, File>`.
The reader verifies size, checksum, snapshot and partition identifiers before
returning data. Network reads stop if they exceed the declared byte size.

## Loading one detail or exact preset

Detail descriptors contain `objects: [[type, id], ...]`. Only matching blocks
need to be fetched for a game or report section. Small values retain the row
`{type, id, value}`. Oversized maps and lists use ordered fragments:

```javascript
{type: "report", id: "suite", fragment: {path: [], kind: "object", length: 2}}
{type: "report", id: "suite", fragment: {path: ["name"]}, value: "example"}
{type: "report", id: "suite", fragment: {path: ["budgets"]}, value: [11, 22]}
```

Nested containers declare their own path, kind and exact child count before
their children. Paths contain string object keys or integer array positions.
Empty containers, nulls and signed zero are preserved. Reconstruction must
reject repeated or missing children, undeclared parents and invalid key types;
object keys must be defined as own properties, including `__proto__`.

Summary descriptors contain `selectors`, a list of exact-selection SHA256s.
Every summary row is `{id, selector_sha256, value: preset}` or uses the same
fragment form. The original preset ID and every original preset field remain
unchanged after reconstruction. A selector hashes UTF-8 compact JSON without
ASCII escaping, with this array as its preimage:

```text
[score_order_or_null, include_controls, panel,
 sorted_methods, [[sorted_game_id, sorted_integer_budgets], ...]]
```

String ordering is by Unicode code point, not JavaScript's default UTF-16 sort.
The selector intentionally describes the same order-insensitive comparison as
the existing page; it is separate from the scientific preset ID. Fetch only
descriptors advertising the requested selector, filter rows by
`selector_sha256`, and reconstruct the first matching preset ID. An absent
selector requires no block reads. The consumer must still verify that the
reconstructed preset matches the requested games, methods, budgets and filters.

## Worker calculations

`query.js` computes the table, budget/time curves, coverage and issue summaries
without DOM access. It reads the selected target and one method at a time,
retains exact weighted medians, and keeps timing profiles separate. An exact
preset supplies Elo and history; arbitrary selections keep their calculated
scores without inventing an Elo panel.

`query-worker.js` connects this calculation to the authenticated block reader.
Each query has an ID; a replacement query or cancel message aborts the previous
one and suppresses stale replies. `partition-details.js` reconstructs only the
requested metadata object or preset. By default it rejects assemblies exceeding
one million fragments or 33,554,432 serialized UTF-16 code units. These are
logical limits, not a measured browser heap bound; exceeding one returns an
error rather than a partial answer.

The worker and lookup APIs are tested but are not connected to the live page.
The remaining integration must preserve controls, charts, details and downloads,
then qualify memory and responsiveness with actual large reports.

## Gates before enabling the format

The block byte cap bounds encoded assets, not parsed JSON memory. A browser
consumer must limit concurrent requests and retained blocks, handle canceled
queries, and measure memory on realistic selections. Complete table, plot,
filter, timing-profile and download parity with the current site still needs
qualification. In particular, arbitrary filters must keep exact weighted
medians; loading every block and concatenating all rows defeats this design.

Large maps/lists in details and presets are split without increasing the block
cap. An individual scalar or an excessively long path can still exceed it and
is rejected explicitly. Reconstructing one requested object can require more
memory than a block. Full-campaign payload size, manifest size and summary
lookup costs must be measured; splitting alone does not establish GitHub Pages suitability.
The reader is copied with website assets but is not enabled by the existing
page. The Pages manifest validator and browser entry points must be updated
together before a partitioned manifest can be deployed.

Reader checks run with:

```bash
node --test tests/shapiq_benchmark/{partitions,partition-details,query}.test.cjs
```
