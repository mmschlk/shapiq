# Focused benchmark

The previous 38-wave expansion is canceled. Its completed artifacts and the
published cohort remain preserved. This replacement is being implemented and
qualified; its manifest is not a claim that all 384 instances have qualified.

| Application | Recipes × construction seeds | Players |
|---|---:|---|
| Individual predictions | 32 × 4 = 128 | 12–512 features |
| Data valuation | 32 × 4 = 128 | 12 training-data groups; 32–512 neighbor examples |
| Feature selection | 32 × 4 = 128 | 12 features |

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
ready batches in parallel. Reserve and reconcile every Slurm task using the
same script's `budget` command. The total is **2,048 allocated CPU-hours**, with
**128 CPU cores maximum and no GPUs**; this includes pilots, idle allocation,
checks and export. The helper reports commitments; submission must enforce them.
Independent audits precede publication and live verification. Recurring watcher
wake-ups keep work moving; neither timers nor successful Slurm exits establish
scientific completion. Coverage shortfalls remain explicit.
