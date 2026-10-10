# Joining completed matrix waves to published history

`shapiq_benchmark.matrix_publication.compose_matrix` assembles an authenticated
panel in a caller-owned `RecordStore`. It does not submit jobs, approve results,
write a website, or publish anything. Keep the store open while using its bounded
selectors or `iter_summaries`; the existing report writer still requires lists.

```python
with RecordStore(work / "rows.sqlite") as records:
    panel = compose_matrix(plan_path, plan_sha256=reviewed_sha256,
                           record_store=records, cache_dir=work / "normalized")
    # A bounded writer and independent final publication audit follow separately.
```

The external, independently reviewed plan is JSON with these fields. Every
`pin` is `{"path": "...", "sha256": "..."}`; relative paths resolve beside the
plan. Receipt input maps use absolute, resolved local paths. Pins authenticate
bytes, not the truth of an unreviewed claim.

| Field | Value |
| --- | --- |
| `version` | `1` |
| `campaign` | Pin for the frozen matrix `campaign.json` |
| `source` | Exact campaign source object; original sources are never relabeled |
| `through_wave` | Last included wave ordinal |
| `reuse` | Pin for the approved historical reuse ledger |
| `required_corrections` | Method name → allowed historical software hash → source hash |
| `history.index` | Pin for the original published `data.json` |
| `history.shards` | Exact target filename → SHA256 map |
| `history.archives` | Exact reproduction ZIP filename → pin map |
| `history.bridge` | Pin for the final independent public-row/archive bridge review |
| `source_equivalence` | Pin for the final independent source/runtime/parameter review |
| `waves` | Ordered `{phase, audit: pin, review: pin}` entries for every included wave |

All final independent reviews require `status: "PASS"`, `complete: true`,
`blocking_issues: []`, and a nonempty `authenticated_inputs` path → SHA256 map.
They must bind any machine-generated probe or comparison evidence in that map.
The original automatic wave audit need not contain `blocking_issues`.

Each wave audit must identify the exact campaign identity (`plan_sha256`), full
`source`, `phase`, and complete batch-ID list. Its independent review repeats
`phase` and `plan_sha256`, supplies `audit_sha256`, and authenticates the audit
file plus every audit input. The review must additionally authenticate files
needed by the collector, including root `jobs.json` and the selected
`phase-N-inventory.json`; the automatic auditor does not necessarily read these.
Missing qualification, snapshot, artifact, allocation or checkpoint-companion
pins stop assembly before measurement journals are collected. Existing complete
cell, source, score, qualification and stopped-writer validation still runs.

The historical bridge review additionally requires:

- `scope: "public-measurements-run-ids-and-provenance"`;
- exact `history_index_sha256`, `history_shards`, `reuse_sha256`, and
  `archives` (ZIP filename → SHA256);
- `preserved_run_ids: true`, `preserved_run_provenance: true`, and exact
  `record_count` / `game_count` from the public index;
- `comparison_fields` equal to sorted `report.ROW_FIELDS` plus `run_id`;
- the index, all shards and all archives in its authenticated closure.

The source-equivalence review additionally requires:

- `scope: "estimator-code-runtime-and-parameters"`;
- exact `history_index_sha256` and `matrix_source`;
- the full `approved_historical_methods` catalog and its canonical identity as
  `historical_methods_sha256`;
- `required_corrections` exactly matching the plan. Every historical version of
  each corrected method must be allowed by that map.

History is imported with its original run IDs, worker details and published
quality roles. The ledger supplies authenticated table fingerprints and semantic
recipe evidence; deferred historical recipes remain available but do not count
as current matrix reuse. Conflicting game IDs, recipe configurations, run IDs,
method parameters or software/source pairs reject the union. Exact duplicate
resolution runs only after old and new panels are joined. Source versions remain
separate even when an independent review establishes estimator equivalence.

All input hashes are checked again before the caller's initially empty store is
filled. A failed import leaves that store empty. Public composition contains
hashes, ordinals, original historical settings and planned grids; private receipt
paths remain outside it. Scores, medians, confidence intervals and Elo must be
recomputed over the union, never averaged from old or per-batch summaries.

`tests/shapiq_benchmark/test_matrix_publication.py::fixture` is a small executable
plan example using real campaign collection and the current public writer.
Production bridge/source reviews and completed matrix-wave output reviews must
exist before constructing a production plan; the synthetic fixture grants no
approval to any real campaign.
