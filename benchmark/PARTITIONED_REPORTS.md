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

## Gates before enabling the format

The block byte cap bounds encoded assets, not parsed JSON memory. A browser
consumer must limit concurrent requests and retained blocks, handle canceled
queries, and measure memory on realistic selections. Complete table, plot,
filter, timing-profile and download parity with the current site still needs
qualification. In particular, arbitrary filters must keep exact weighted
medians; loading every block and concatenating all rows defeats this design.

Large individual metadata or preset objects can exceed a block cap and are
rejected explicitly. Full-campaign payload size and summary lookup costs must
be measured; block splitting alone does not establish GitHub Pages suitability.
The reader is copied with website assets but is not enabled by the existing
page. The Pages manifest validator and browser entry points must be updated
together before a partitioned manifest can be deployed.

Reader checks run with:

```bash
node --test tests/shapiq_benchmark/partitions.test.cjs
```
