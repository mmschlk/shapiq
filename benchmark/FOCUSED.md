# Focused benchmark

The previous 38-wave expansion is canceled. Its completed artifacts and the
published cohort remain preserved. The replacement has passed its preparation
pilots; full game preparation is running. The website still shows the previous
audited cohort until the replacement passes its final audit.

| Application | Intended instances | Admitted after pilots | Players |
|---|---:|---:|---|
| Individual predictions | 32 recipes × 4 seeds = 128 | 124 | 12–512 features |
| Data valuation | 32 × 4 = 128 | 120 | 12 training-data groups; 32–512 neighbor examples |
| Feature selection | 32 × 4 = 128 | 116 | 12 features |

Six recipes failed the model-quality check against a dummy predictor on at least
one construction seed: `local-24`, `data-11`, `data-15`, `features-21`,
`features-24`, and `features-29`. They remain explicit exclusions. The 360 admitted
instances still need complete exact preparation and estimator evaluation.

[One readable CSV](suites/focused.csv) lists every dataset, model, construction and
player count. There are no synthetic games. Models are primarily random forests,
XGBoost and LightGBM, with selected SVM, MLP, GP and TabPFN comparisons.

Generate a suite from a checkout containing the approved estimator fixes (PR
#611), without changing the shared environment:

```bash
UV_NO_SYNC=1 uv run python -m shapiq_benchmark.focused benchmark/suites/focused.csv /path/on/lab/storage/suite.json
```

This writes a new suite, checks adapter and constructor compatibility, and never
submits jobs. The suite fixes nine budgets (0.5d through 128d), four construction
seeds and one estimator seed. LeverageSHAP/OddSHAP explicitly use the approved
two-query low-budget fallback. Worker limits are 30 seconds, extended to 120
seconds when both d ≥ 128 and the requested budget ≥ 32d; startup counts.

Enumerated games request all six targets. Native path-dependent trees request
SV, SII and k-SII; interventional trees request all six; neighbors request SV only.
Actual size, exact reference, signal and model quality still require qualification.
Matched Bioresponse/XGBoost and QSAR-TID11/RF sequences at 32/128/512 features use
nested seeded feature subsets, fixed splits and held-out points, and refitted
models. Other recipes retain their original feature-selection rules.

Python and browser summaries balance applications equally, then subtypes,
recipes and instances. Data-valuation SV gives equal weight to group retraining
and neighbor examples. Unsupported interaction references are excluded before
estimator evaluation; estimator failures remain in coverage denominators.

Next: qualify the frozen suite with `benchmark/focused_campaign.py preflight`,
measure useful completion under the proposed limits, then prepare once and run
ready batches in parallel. Native tree queries reuse the existing compiled prediction
kernel or vectorize the shipped recursion. Unsupported tree forms retain the original
traversal. Frozen payoffs and exact references stay unchanged; all 68 native tree
instances passed sampled bitwise equivalence checks before adopting this backend. Reserve and reconcile every Slurm task using the
same script's `budget` command. The total is **2,048 allocated CPU-hours**, with
**128 CPU cores maximum and no GPUs**; this includes pilots, idle allocation,
checks and export. The helper reports commitments; submission must enforce them.
Independent audits precede publication and live verification. Recurring watcher
wake-ups keep work moving; neither timers nor successful Slurm exits establish
scientific completion. Coverage shortfalls remain explicit.

## Code map

The focused campaign reuses the existing benchmark pipeline:

| Step | Source |
|---|---|
| Choose dataset, model, construction and player count | [suites/focused.csv](suites/focused.csv) |
| Translate that table into a suite | [focused.py](../src/shapiq_benchmark/focused.py) |
| Qualify costs and account for CPU allocations | [focused_campaign.py](focused_campaign.py) |
| Prepare exact games once | [prepare_matrix.py](prepare_matrix.py) |
| Traverse frozen native trees efficiently | [native_tree_backend.py](../src/shapiq_benchmark/native_tree_backend.py) |
| Evaluate estimators with counted queries | [runner.py](../src/shapiq_benchmark/runner.py) |
| Export results for the static website | [report.py](../src/shapiq_benchmark/report.py) |
| Deliver completion and recurring reminders | [watch_campaign.py](watch_campaign.py) |

Campaign-specific manifests, caches, job receipts and audit reports live on lab
storage, outside Git. Estimator changes stay in the separate draft PR #611;
production runs use a frozen checkout containing those changes.
