# Large benchmark reports

The optional `partitioned-v1` format stores results in small, authenticated
column blocks. The browser supports this format alongside legacy reports. The
live dataset remains in its existing format until real-release parity and
resource checks pass.

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
distinct real case counts, controls, score orders, planned and observed relative
budgets, common-panel availability and estimated-cost availability. Planned cells count all declared methods,
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

The page uses `partition-client.js` to run queries and pass their summaries to
the existing table and chart renderers. Switching reports closes the old worker;
sorting and display-only changes reuse the latest result. Legacy local reports
keep their existing loading path.

`partition-download.js` reads selected raw blocks into a temporary IndexedDB
store, then emits rows in their original order. Writes wait for the destination.
Browsers with a file-save API can stream directly to disk; others use an explicit
64 MiB Blob limit and report an error if the selection is larger. The temporary
store also has explicit one-million-row and 1 GiB limits, including with a
streaming destination. Larger complete downloads still require the reproduction
archives; these limits must be profiled before publishing the full matrix.
Cancellation and errors clean up the temporary store. JSON retains original game metadata,
run IDs and provenance; CSV retains the existing columns. Internal `sequence`
and `worker_id` encoding fields are omitted after restoring the full worker
object. Older compact-report downloads can retain that redundant worker ID;
its omission changes no recorded worker field or scientific value.

The About page streams game details once and keeps only the fields used by its
tables. Counts replace unused row-index arrays. The displayed metadata has an
explicit size cap; full original metadata remains available in downloads.
Large-report memory and responsiveness still need qualification.

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
`benchmark/fetch_report.py` stages the pinned manifest and only its listed
assets in a fresh directory. It verifies checksums, descriptor bindings, column
encoding and public-field guards, and rejects payloads above its Pages packaging
limit. These checks do not replace the independent scientific release audit.
No matrix data is deployed simply by adding browser support.

Reader checks run with:

```bash
node --test tests/shapiq_benchmark/{partitions,partition-details,partition-client,partition-about,partition-download,query}.test.cjs
```
