# Roadmap to the full benchmark

This is the implementation plan for the next benchmark generation. The dataset
loaders, model profiles and compatible construction adapters are implemented.
Production qualification and execution still proceed phase by phase; an adapter
being implemented does not mean its full dataset grid has completed. Superseded jobs were cancelled;
published results keep their identities. This roadmap
supersedes executing the entire new dataset matrix with the old shallow models.

**Completed: phases 1–2.** The audited first release contains 64 games, all at
12 players, and 27,919 successful evaluations with no pending cells. Corrected
LeverageSHAP, OddSHAP, ProxySPEX and SVARM have full supported coverage.
[Release and reproduction files](https://github.com/rtealwitter/shapiq/releases/tag/benchmark-phase2-2026-10-01).

**Current: the selected phases three through seven are live; the full matrix is running.**
The corrected bounded rollout covers phases 3–7. Frozen source `5001ba42`
implements the original quality checks and selects up to sixteen new recipes
per phase. Corrected regression results use `e608ec8e` (PR #610), preserving
all frozen games. The immutable plan has six preparation batches and six
associated evaluation tasks, plus separate Wine and CPU-backend recovery batches.

The cumulative audited release contains **248 instances, 1,188 target definitions
and 90,070 successful evaluations** across 235,224 recorded cells. The earlier
109 real core instances, 55 real controls and 24 synthetic controls retain their
classification. The 60 new native instances have no table-quality role; existing
default comparisons include them, with the qualification limits below. Each
recipe has four construction seeds. All seven reproduction archives and 21,360
ranking presets passed independent checks.
[Phase-seven release and reproduction files](https://github.com/rtealwitter/shapiq/releases/tag/benchmark-phase7-svd-2026-10-04).

The full dataset × model × construction expansion is queued in a separate
campaign with **38 waves and 1,468 preparation batches**. The first wave is
preparing 11-player games. Nine enumeration waves cover compatible sizes up to
20 players; 29 structured-solver waves cover larger sizes up to 1,776, subject
to qualification. The plan retains all 63 datasets, compatible models and
constructions, all nine relative budgets and four construction seeds. Completed
recipes are reused only after authentication. Preparation and evaluation tasks
use 16 CPU cores each, with at most eight running (128 cores); no production GPUs are reserved.
Each wave's numerical audit gates the next wave. Independent review and website
publication follow separately. Matrix publication still needs bounded-memory
export and browser loading work; the live selected cohort remains unchanged.

Phase four contributed 48 instances from twelve recipes. Phase five added 40
CPU instances and four Mushroom TabPFN instances from CPU recovery. Three other
recovery recipes exceeded the unchanged eight-hour preparation-cost limit.
The original GPU preparation stopped under the 80% utilization guard; its
cancelled evaluation and the CPU replacement remain separately authenticated.

Phase six prepared **40 instances and 240 target definitions** from ten retained
recipes, with 11–17 players and four seeds each. All exact answers, Fourier
spectra and 577 preparation inputs passed independent numerical/integrity checks.
Six other recipes failed the existing cost, model-validation or imputation gates.
All 47,520 evaluation cells and the six-method correction passed independent
audits. Phase seven prepared **60 instances from fifteen recipes**, with four
seeds each and 21–256 players. Preparation, all 11,880 evaluation cells, corrected
regression results and final publication artifacts passed independent audits. The requested
512-player Iris recipe was excluded because it has too few training rows.
These larger structured games use native exact solvers and measured oracle
timings. Their saved model artifacts do not provide full coalition tables or
Fourier spectra; the audits record this distinction explicitly.

The phase-six scientific review found four Bike Sharing clustering instances
whose enormous scores come from near-zero within-cluster variance. They must
remain controls, giving **19 core and 21 control instances** for publication,
rather than the original frozen labels of 23 and 17. A versioned, payoff-only
numerical check supplements publication metadata without altering any saved
measurements. Future clustering recipes use `cluster_continuous_v1`; see the
[construction safeguards](README.md). Fourier spectra describe the frozen games;
numerical spikes do not establish meaningful interaction diversity.

Corrected-source reruns replace all nine budgets for the six regression classes
covered by [PR #610](https://github.com/mmschlk/shapiq/pull/610). Corrected results
through phase seven are published.
Elo fitting certifies accuracy within 0.001 points for the same objective. See
[replacement exports](README.md#replacing-a-corrected-estimator).

| Phase | Selected recipes before qualification | Preparation batches | Evaluation tasks |
| --- | ---: | ---: | ---: |
| 3: more constructions | 16 | 1 | 1 |
| 4: more models and player counts | 16 | 1 | 1 |
| 5: specialized models and inputs | 16 | 2 (CPU/CUDA) | 2 |
| 6: broader dataset coverage | 16 | 1 | 1 |
| 7: larger structured games | 16 | 1 | 1 |

These are planned batches, not completed games. Phase three qualified twelve
recipes initially and recovered two Wine recipes after download failures:
**56 instances and 336 target definitions** across four seeds, including controls.
Two recipes remain excluded by the imputation-noise gate. The Wine supplement
adds one preparation/evaluation pair; it retries existing recipes rather than
expanding the sixteen-recipe selection.

On Hopper, `QUALITY` is
`/hopper/groups/witterlab/rwitter/shapiq-benchmark-quality-rollout`.
`QUALITY/campaign/campaign.json` records the scientific plan;
`QUALITY/state.json` records progress. Original job IDs in
`QUALITY/campaign/jobs.json` have explicit recovery overrides: use
`QUALITY/operational/future-runtime-jobs.json` for phases 4–7 and the evaluator
recovery journals listed in `state.json` for phase three. The verified operational
wrapper stages identical packages on local disk to avoid shared-filesystem import
stalls. See the [continuation procedure](README.md#continuing-through-the-phases).

**Original campaign, tracked separately:** all twenty phase-three preparation and
evaluation batches passed independent audit on frozen source `3f6b9b50`:
1,492,128 recorded cells and 548,583 successful evaluations. The cancelled GPU-node
batch's completed cells and its 56-cell CPU supplement passed the final union audit.
These older-source measurements remain separate from the corrected release.
Its original phases 4–7 were cancelled. The former 733-batch full-matrix submission
is not the active expansion plan; use the new matrix campaign and its recovery
journals, as described in the [continuation procedure](README.md#continuing-through-the-phases).

The full benchmark is intended to cover **all 63 datasets in the target catalog**,
the model profiles below, every shipped game-construction family, four construction
seeds, and all
nine relative budgets. We expand every **compatible** pairing in the declared
model-to-construction mapping. A dataset being loadable is not sufficient to
claim its benchmark is complete.

## Fixed experiment protocol

- Four game instances, construction seeds 0–3; one estimator evaluation per
  instance, target and budget, with a recorded estimator seed. No rerolling to
  select a difficult observation or a favorable estimator result.
- Budgets: **0.5, 1, 2, 4, 8, 16, 32, 64 and 128 times d**.
- At least **11 players** everywhere. Enumerated games use 11, 12, 16 and 20,
  plus compatible native widths between 11 and 20. Never pad missing features.
  An early phase can use only one size before adding the complete grid.
- Above 20 players, require a separately qualified structured exact solver.
  Feature games retain real feature counts, such as 30, 33, 81 or 101; row-player
  neighbor games can use 32, 64, 128, 256 and 512 examples where data permit.
- Standard targets remain SV and the existing order-two k-SII, SII, STII, FSII
  and FBII profiles, wherever supported. Richer underlying games can contain
  interactions above order two even when the scored target is pairwise. This
  roadmap does not claim every higher-order target has an implemented exact solver.
- Freeze and cache each enumerated game once; derive all supported exact targets
  from the same payoffs. Retain existing normalization, signal-ratio threshold
  1e-6, per-query cost accounting, and actual/estimated timing distinction.
- Compare only compatible hardware profiles; retain single-thread execution,
  CPU affinity and source/model/data hashes. Continue using the matching Hopper
  EPYC 9754 node pool and label concurrent timings diagnostic.
- Each phase ends with an independent audit, a versioned export and live-site
  verification. Local candidate estimators use the same snapshots without
  publication. Failed, excluded and pending configurations remain visible.

### CPU and GPU preparation

Use GPUs when representative measurements show a worthwhile gain. The first
Hopper L40S pilot found fixed TabPFN prediction about 91× faster warm (9× on
the first call) than one EPYC 9754 core; XGBoost's small fit gained only 1.1×
and warm prediction 2×. These are model-call timings, not end-to-end campaign
speedups. A second pilot of the actual shipped TabPFN contextualization game
(Adult Census, 12 players, 64 context rows, 32 coalitions) measured warm batches
of 8.94 seconds on one CPU core versus 3.64 seconds on an L40S: about **2.5×**.
Enable explicit CUDA float32 preparation for this recipe, with one worker per
GPU. Keep the initial RF/XGBoost cohort on CPUs. GP and neural-model acceleration
require their own measurements.

Device selection is explicit in each recipe. CPU and GPU realizations may differ
numerically and must have separate authenticated snapshots. Record device,
backend versions and precision; retain one pinned standardized CPU per estimator.
Cached-query charges describe the hardware that generated the table, not a
counterfactual CPU cost. Compare timing within the same frozen game and hardware
profile. Do not silently combine CPU and GPU oracle costs as standardized runtime.

For enumerated games, compute the uniform-coalition Boolean Fourier spectrum
from the cached payoffs, with no additional model queries. Report variance by
interaction degree and effective order alongside predictive validation scores.
This describes the frozen game (including any frozen imputation noise), not
Shapley interaction indices. It does not automatically discard easy games or
select examples based on estimator rankings.

## Datasets and rollout order

**First four:** Adult Census, Breast Cancer, shapiq Wine Quality, Communities and
Crime. They cover classification/regression, continuous/mixed inputs and different
native widths. Start local explanations at 12 selected features.

**Next four:** Mushroom, Ionosphere, TabArena Miami Housing and TabArena
Superconductivity. This adds categorical structure, small-sample radar data and
larger regression feature spaces.

**Full rollout:** all remaining entries below, including every one of the 51
shipped TabArena loaders. Native widths below 11 exclude feature-player games,
but can still support row, group and model-player games. Task restrictions still
apply. NHANES uses the loader's signed survival target as an explicitly labeled
regression surrogate, not a censoring-aware survival model.

Prefer `shapiq_games.datasets`/`shapiq.datasets` loaders; sklearn remains only
where the current inventory lacks a shipped equivalent. The source/task/encoding
registry is [dataset_catalog.py](../src/shapiq_benchmark/dataset_catalog.py), with
loading in [datasets.py](../src/shapiq_benchmark/datasets.py).

| Dataset ID | Task | Native features | Rollout |
| --- | --- | ---: | --- |
| `california_housing` | regression | 8 | Full rollout |
| `diabetes` | regression | 10 | Full rollout |
| `bike_sharing` | regression | 12 | Full rollout |
| `iris` | classification | 4 | Full rollout |
| `breast_cancer` | classification | 30 | First four |
| `digits` | classification | 64 | Full rollout |
| `adult_census` | classification | 14 | First four |
| `mushroom` | classification | 22 | Next four |
| `ionosphere` | classification | 33 | Next four |
| `nhanesi` | regression | 79 | Full rollout |
| `communities_and_crime` | regression | 101 | First four |
| `wine_quality` | regression | 12 | First four |
| `tabarena_airfoil_self_noise` | regression | 5 | Full rollout |
| `tabarena_amazon_employee_access` | classification | 9 | Full rollout |
| `tabarena_anneal` | classification | 38 | Full rollout |
| `tabarena_fiat_500` | regression | 7 | Full rollout |
| `tabarena_aps_failure` | classification | 170 | Full rollout |
| `tabarena_bank_marketing` | classification | 13 | Full rollout |
| `tabarena_bank_customer_churn` | classification | 10 | Full rollout |
| `tabarena_bioresponse` | classification | 1776 | Full rollout |
| `tabarena_blood_transfusion` | classification | 4 | Full rollout |
| `tabarena_churn` | classification | 19 | Full rollout |
| `tabarena_coil2000` | classification | 85 | Full rollout |
| `tabarena_concrete_strength` | regression | 8 | Full rollout |
| `tabarena_credit_g` | classification | 20 | Full rollout |
| `tabarena_credit_card_default` | classification | 23 | Full rollout |
| `tabarena_airline_satisfaction` | classification | 21 | Full rollout |
| `tabarena_diabetes` | classification | 8 | Full rollout |
| `tabarena_diabetes130us` | classification | 47 | Full rollout |
| `tabarena_diamonds` | regression | 9 | Full rollout |
| `tabarena_ecommerce_shipping` | classification | 10 | Full rollout |
| `tabarena_fitness_club` | classification | 6 | Full rollout |
| `tabarena_food_delivery` | regression | 9 | Full rollout |
| `tabarena_give_me_credit` | classification | 10 | Full rollout |
| `tabarena_hazelnut` | classification | 30 | Full rollout |
| `tabarena_health_insurance` | regression | 6 | Full rollout |
| `tabarena_heloc` | classification | 23 | Full rollout |
| `tabarena_hiva_agnostic` | classification | 1617 | Full rollout |
| `tabarena_houses` | regression | 8 | Full rollout |
| `tabarena_hr_analytics` | classification | 12 | Full rollout |
| `tabarena_coupon_recommendation` | classification | 24 | Full rollout |
| `tabarena_good_customer` | classification | 13 | Full rollout |
| `tabarena_kddcup09` | classification | 212 | Full rollout |
| `tabarena_marketing_campaign` | classification | 25 | Full rollout |
| `tabarena_maternal_health` | classification | 6 | Full rollout |
| `tabarena_miami_housing` | regression | 15 | Next four |
| `tabarena_online_shoppers` | classification | 17 | Full rollout |
| `tabarena_protein` | regression | 9 | Full rollout |
| `tabarena_bankruptcy` | classification | 64 | Full rollout |
| `tabarena_qsar_biodeg` | classification | 41 | Full rollout |
| `tabarena_qsar_tid11` | regression | 1024 | Full rollout |
| `tabarena_qsar_fish_toxicity` | regression | 6 | Full rollout |
| `tabarena_sdss17` | classification | 11 | Full rollout |
| `tabarena_seismic_bumps` | classification | 15 | Full rollout |
| `tabarena_splice` | classification | 60 | Full rollout |
| `tabarena_students_dropout` | classification | 36 | Full rollout |
| `tabarena_superconductivity` | regression | 81 | Next four |
| `tabarena_taiwanese_bankruptcy` | classification | 94 | Full rollout |
| `tabarena_website_phishing` | classification | 9 | Full rollout |
| `tabarena_wine_quality` | regression | 12 | Full rollout |
| `tabarena_naticusdroid` | classification | 86 | Full rollout |
| `tabarena_jm1` | classification | 21 | Full rollout |
| `tabarena_mic` | classification | 111 | Full rollout |

Wine Quality and TabArena Wine Quality, and California Housing and TabArena
Houses, share underlying datasets. Keep their loader/preprocessing identities,
but group these source datasets when weighting headline scores. Extra wrappers
or model variants must not give one source dataset disproportionate weight.
Historical sklearn `wine` remains reproducible but is outside these 63 entries.

## Models and where they are used

These are initial named profiles to qualify, not claims that untuned defaults
perform well. Parameters and output scale become part of each game identity.
Use the existing [model builders](../src/shapiq_benchmark/setup.py), game
constructors and optional backends wherever possible.

| Profile | Initial configuration | Game constructions | Introduced |
| --- | --- | --- | --- |
| Random forest | 100 trees; no depth cap; minimum leaf size 5; one CPU thread | Local/global prediction, uncertainty, feature/data/group valuation, heterogeneous/forest ensemble, tree explanations | Phase 1 |
| XGBoost | Up to 200 trees; depth 8; learning rate 0.05; validation early stopping; one CPU thread | Local/global prediction, feature/data/group valuation, heterogeneous ensemble, tree explanations | Phase 1 |
| LightGBM | Up to 200 trees; 63 leaves; minimum leaf size 10; learning rate 0.05; validation early stopping; one CPU thread | Same eligible constructions as XGBoost | Phase 4 |
| MLP | Two hidden layers, 128 and 64 units; scaled inputs; validation early stopping | Local/global prediction, feature/data/group valuation, heterogeneous ensemble | Phase 4 |
| RBF SVM | SVC/SVR; training-only scaling; small validation grid for C/gamma | Product-kernel games, baseline/marginal local prediction, heterogeneous ensemble | Phase 3 |
| Fixed TabPFN predictor | One ensemble member initially; measured row cap; GPU where qualified faster | Baseline/marginal local prediction with a fixed fitted context | Phase 5 |
| TabPFN contextualization | One ensemble member; record backend/device and contextualization rows | Remove-and-contextualize classifier/regressor games; existing causal construction models | Phase 5 |
| Gaussian-process surrogate | Training-only scaling; validation-chosen kernel and bounded fitting rows; qualify GPU backend separately | Baseline/marginal local prediction | Phase 5 |
| Linear control | Logistic classification / ridge regression with scaled inputs | Baseline/marginal local prediction, feature/data/group valuation, heterogeneous ensemble | Phase 3 |
| Neighbor models | Unweighted 3-NN, distance-weighted 3-NN, radius neighbors | KNN, TNN, weighted KNN and binary-weighted KNN games | Phase 3 |
| K-means / no model | Three clusters / total-correlation statistic | Clustering / unsupervised dependence | Phase 3 |
| Pretrained image models | Shipped ResNet-18 and ViT with 16 patches | Image explanation | Phase 5 |
| Pretrained text model | Shipped DistilBERT sentiment model | Token-based sentiment explanation | Phase 5 |
| Diagnostic decision tree | Existing shallow profile, explicitly labeled | Reproduction and simple controls | Retained |

Classification/regression select the appropriate model variant. An ensemble game
has d member-model players; its model count therefore follows d, rather than the
100-tree setting for ordinary forest explanations. The heterogeneous ensemble's
member families are RF, XGBoost, RBF SVM and linear initially, adding LightGBM and
MLP in Phase 4. Fill d slots by cycling through those families in that fixed order,
with deterministic member seeds and frozen validated parameters; record every
member. Existing heterogeneous games keep their original member definitions.
Logistic regression probabilities need not define an additive game, so the
linear profile is a model-family control, not a guarantee of additive payoffs.

The mapping is intentional: an MLP cannot instantiate a tree game, a forest cannot
replace a product-kernel model, and regression cannot instantiate a
classification-only neighbor game. Phase 4 implements each model's eligible
construction set on the first eight datasets; Phase 6 applies that same mapping
to all 63 datasets.

### Training and qualification

For ordinary prediction models, use up to **5,000 fitting rows**, up to **1,000
validation rows**, and a separate held-out pool, limited by available data.
Replace the current 512/128 caps explicitly. Use deterministic, disjoint splits,
stratification when feasible, and training-only additional scaling/imputation.
Record any full-data preprocessing already performed by a shipped loader.
Keep original columns as players; any extra one-hot encoding must preserve grouped
player identities and qualify its exact solver rather than increasing d silently.

TabPFN and other costly constructions get a separately named, documented row cap
based on a measured preparation pilot. Feature/player subsets are selected before
model fitting, and are recorded. Match the trained model across constructions
when they explain the same dataset, feature subset, output and seed.

Validate log loss and balanced accuracy for classification, MSE/R2 for regression,
and comparison with a dummy baseline. Tune only using the validation partition
and a small fixed search budget. Fix multiclass objectives and single-thread
settings before reusing the existing tuning helper; its current binary-objective
and `n_jobs=-1` defaults are not suitable for the full matrix.

Record observed depth, features used along paths, payoff variation, player
influence and sampled higher-order differences. Strong predictive performance,
many declared features and deep trees are separate properties. Stratify easy and
complex games; do not retain only instances on which particular estimators fail.
Never reroll the four seeds to force a desired interaction profile.

For valuation/retraining games, freeze hyperparameters and validation decisions
before enumerating coalitions. The game uses only the rows/features selected by
its coalition. Preserve the shipped empty-coalition utility; explicitly implement
and test constant-label behavior and class-label mapping for classifiers on
single-class/subset-class training coalitions. Otherwise mark that pairing
unsupported before launching enumeration. Do not tune a new model per coalition.

Structured tree ground truth must match the exact prediction scale. Forest class
probabilities and boosted-tree margins are different profiles. A sigmoid/softmax
of a tree sum does not inherit its linear exact solver; probability games need
enumeration or another qualified exact method. Compare each new solver/model
pair against enumeration at small d before using it above 20 players.

## Game constructions

The listed variants remain distinct games, even when grouped together on the
website. Links lead to the shipped implementations used by the adapters.

| Construction | Players and payoff | Models/data | Phase |
| --- | --- | --- | --- |
| [Baseline local prediction](../src/shapiq/imputer/baseline_imputer.py) | Features; prediction after fixed-baseline replacement | RF/XGBoost first; LightGBM/MLP/SVM/linear later | 2 |
| [Marginal local prediction](../src/shapiq/imputer/marginal_imputer.py) | Features; background-averaged prediction | Same local model profiles | 2 |
| [Gaussian](../src/shapiq/imputer/gaussian_imputer.py), [copula](../src/shapiq/imputer/gaussian_copula_imputer.py), [generative conditional](../src/shapiq/imputer/generative_conditional_imputer.py) | Features; conditionally imputed prediction | RF/XGBoost/LightGBM/MLP; Gaussian variants need enough eligible continuous columns | 3–4 |
| [Global fidelity](../src/shapiq_games/benchmark/global_xai/base.py) | Features; loss against full-model predictions across observations | RF/XGBoost/LightGBM/MLP; retain the shipped fidelity definition | 3–4 |
| [Feature selection](../src/shapiq_games/benchmark/feature_selection/base.py) | Features; held-out performance after refitting | RF/XGBoost/LightGBM/MLP/linear | 3–4 |
| [Data valuation](../src/shapiq_games/benchmark/data_valuation/base.py) | Training examples; held-out utility after refitting | Same retraining profiles; small row-player games are explicitly small-data problems | 3–4 |
| [Dataset/group valuation](../src/shapiq_games/benchmark/dataset_valuation/base.py) | Groups of training rows; held-out utility | Same retraining profiles; groups allow realistic training-set sizes | 3–4 |
| [Ensemble / forest ensemble selection](../src/shapiq_games/benchmark/ensemble_selection/base.py) | Models/trees; selected ensemble's held-out utility | Fixed heterogeneous member lists / forest trees | 3–4 |
| [Uncertainty](../src/shapiq_games/benchmark/uncertainty/base.py) | Features; predictive uncertainty | Forest classifier; do not assume every model supports the same decomposition | 3 |
| [Clustering](../src/shapiq_games/benchmark/unsupervised_cluster/base.py) | Features; K-means clustering score | Eligible noncategorical columns; no labels | 3 |
| [Unsupervised dependence](../src/shapiq_games/benchmark/unsupervised_data/base.py) | Features; total correlation | No prediction model | 3 |
| [Path-dependent tree](../src/shapiq_games/benchmark/treeshapiq_xai/base.py) / [interventional tree](../src/shapiq/tree/interventional/game.py) | Features; tree prediction with the respective missing-feature convention | Qualified RF/XGBoost/LightGBM output profiles | 3–4 |
| [Product kernel](../src/shapiq/explainer/product_kernel/game.py) | Features; restricted kernel-model score | RBF SVC/SVR; binary classification or regression in the current adapter | 3 |
| [KNN](../src/shapiq/explainer/nn/games/knn.py), [TNN](../src/shapiq/explainer/nn/games/tnn.py), [weighted/binary-weighted KNN](../src/shapiq/explainer/nn/games/wknn.py) | Training examples; respective neighbor utility | Classification datasets | 3 |
| [TabPFN contextualization](../src/shapiq/imputer/tabpfn_imputer.py) | Features; predict after restricting training context to a coalition | TabPFN; add and qualify the currently missing regression adapter | 5 |
| [Image](../src/shapiq_games/benchmark/local_xai/benchmark_image.py) | Superpixels/patches; image-classification score | Four predetermined bundled images per configuration; ResNet-18/ViT | 5 |
| [Text](../src/shapiq_games/benchmark/local_xai/benchmark_language.py) | Tokens; sentiment score | Four fixed sentences with 11–20 actual tokenizer tokens; mask/remove variants | 5 |
| [Local/global confounding](../src/shapiq_games/benchmark/causal_xai/base.py) | Covariates; confounding attribution | Four seeded shipped Curth-VDS instances; treatment/outcome inputs required | 5 |
| [Unanimity, SOUM, dummy, random](../src/shapiq_games/synthetic) | Synthetic diagnostic games | No real dataset; separate from real-data headline weighting | Retained |

## Implementation phases and release gates

| Phase | Concrete scope | Code/artifacts to deliver | Gate before moving on |
| --- | --- | --- | --- |
| **1. Shared model profiles** | RF and XGBoost on the first four datasets | Reuse `setup.py` builders; add explicit model/profile/training/output fields to suite identities, model snapshots and recipe metadata; disjoint splits; quality report | Reproducible four-seed fits, held-out baseline comparisons, model/feature-use diagnostics and independent audit |
| **2. First complete stronger-model release** | Four datasets × two models × baseline/marginal × four seeds, all at d=12: **64 games** | Freeze 64 payoff tables; exact supported targets; run every registered estimator at all nine relative budgets; add model filter/details to website | Account for every planned cell, verify exact truth and cache charges, audit and publish |
| **3. All ordinary tabular constructions** | First eight datasets; RF/XGBoost plus the fixed SVM/NN/K-means/linear profiles needed by the constructions | Add every phase-3 construction above, classifier subset/empty behavior, quality/exclusion inventory; start each enumerated construction at its first feasible d≥11 | Every applicable family represented or explicitly excluded; retraining compute pilot measured; independently audited website release |
| **4. Model diversity and full size grid** | First eight datasets; add LightGBM and MLP to every eligible construction; complete 11/12/native/16/20 grid | Extend the same model factory, shared model artifacts, canonical IDs and compatibility rules; qualify converters/output scales | Complete declared model–construction–size mapping with four seeds, no duplicate overweighting, audited release |
| **5. Expensive/special constructions** | TabPFN classifier/regressor, image, text, local/global causal, retained synthetic diagnostics | Dedicated manifests, deterministic input selection, pretrained asset hashes, measured row/background caps; all listed variants | Each special family has four valid instances or documented exclusions; separate coverage, costs and audit; publish |
| **6. Full dataset rollout** | All 63 datasets across the qualified model/construction mapping and feasible enumerated sizes | Run remaining datasets in fixed catalog-order batches of about eight; warm caches, generate exact compatibility inventory, checkpoint/resume jobs and completion watchers | Inventory every requested combination; each finished batch has no pending cells and records exclusions/failures explicitly; update site after each audited batch |
| **7. Larger-player coverage and final release** | Native-width structured tree/kernel games and 32–512-row neighbor games across compatible datasets/models | Extend existing structured adapters beyond their current dataset restrictions; validate each target/model/output pairing against small enumeration, then run larger games | Publish a coverage matrix showing exactly which large-player families/targets are supported; no pending evaluations in the final declared suite; full independent audit and reproduction ZIP |

Phase 7 can begin after Phase 4 while Phase 6 runs, because its structured games
have separate caches and exact-reference qualification. It is a required part of
the full target. Generic games remain capped at 20 players; the qualified
structured extension supplies the larger-player coverage.

## Simple code organization and publication

Keep three declarative lists: dataset metadata, model profiles and construction
compatibility. Expand those into a frozen manifest; route construction through
existing shapiq loaders and game classes. Add one shared model-building path and
one classifier-subset handling path, rather than a class for every combination.
Prefer existing preparation, chunk cache, execution, reporting and website code.

The manifest must distinguish selected, qualified, unsupported, failed and pending
work. Family/dataset weighting prevents extra sizes, models or duplicate source
wrappers from dominating the headline. Keep median nMSE, mean nMSE, Elo, coverage,
relative-budget/time plots and historical plot; expose model profiles as a filter
and in expandable details. Historical scores remain tied to one declared cohort.

After each phase/batch: authenticate the snapshot and estimator revisions
(including corrected LeverageSHAP/OddSHAP), independently audit results, create
checksummed JSON/ZIP release assets, update the GitHub Pages manifest, and verify
the live deployment. Keep the current website available until its replacement
passes these checks. A local candidate uses the same frozen snapshot and reports
without uploading anything.

Completion means every declared compatible combination has been attempted and
audited, with no pending evaluations and explicit coverage for failures/timeouts.
Unsupported/excluded pairings retain reasons. A loader inventory or a completed
Slurm job alone does not establish completion.
