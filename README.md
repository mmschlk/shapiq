# Supplementary Material for "PolySHAP: Extending KernelSHAP with Interaction-Informed Polynomial Regression"

## All experiments can be found in the `experiments` folder.


## Implementations
This repository contains the code and scripts for reproducing the experiments presented in the paper. The implementation builds on the **shapiq** library and includes two new approximators:

1. **PolySHAP** (`shapiq.approximators.polyshap.PolySHAP`)  
   - Uses `ExplanationFrontierGenerator` to select interactions for PolySHAP.  

2. **RegressionMSR** (`shapiq.approximators.regressionMSR.RegressionMSR`)  
   - Baseline method using XGBoost and `UnbiasedKernelSHAP` (MSR) from shapiq.  
3. **Sampling without replacement**  
   - Implemented in CoalitionSampler (`shapiq.approximator.sampling.CoalitionSampler`)
---

## Folder Structure

Before running any scripts, ensure the following folders exist:

- /experiments/ground_truth/exhaustive
- /experiments/ground_truth/pathdependent
- /experiments/approximations/exhaustive
- /experiments/approximations/pathdependent
- /experiments/plots/exhaustive
- /experiments/plots/pathdependent
- /experiments/plots/runtime_analysis


> Files like `datasets.py` and `init_approximator.py` are included and do not need manual creation.  

---

## Environment

To ensure reproducibility, load the environment from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Scripts Overview

### 1. Approximation Scripts
- **Configuration flags inside the script:**
```python
    RANDOM_STATE = 40 # Random seed for reproducibility
    ID_CONFIG_APPROXIMATORS = 39 # Uses sampling without replacement, and standard sampling
    ID_CONFIG_APPROXIMATORS = 37 # Uses sampling without replacement, and with paired subset sampling
    RUN_GROUND_TRUTH = True       # Compute ground-truth Shapley values, =False=skip its computation
    RUN_APPROXIMATION = True      # Compute approximations, =False=skip its computation
```


- **`approximation_baseline.py`**  
  - Used for all non-tabular explanation games, i.e. ViT9, ViT16, ResNet18, DistillBERT  
  - Uses baseline imputation and pre-computed games from shapiq.
  - Parallelization can be activated by uncommenting the corresponding line and commenting the current `explain_instance` function.  
    **Note:** Parallelization does not work well with RegressionMSR.

- **`approximation_pathdependent.py`**  
  - Used for all tabular explanation games  
  - Fits a random forest and uses path-dependent feature perturbations  
  - Same configuration flags as above


  - Ground-truth values are stored in `/experiments/ground_truth/ID`  
    - ID = `exhaustive` (baseline imputation)  
    - ID = `pathdependent` (tabular datasets)
  - Approximated values are stored in `/experiments/approximations/ID`  
    - ID = `exhaustive` (baseline imputation)  
    - ID = `pathdependent` (tabular datasets)
---

### 2. Metric Computation

- **`computation_of_approximation_metrics.py`**  
  - Loads all approximations and ground-truth values  
  - Computes metrics and generates `results_benchmark.csv`  

- **`plot_approximation.py`**  
  - Generates plots for selected metrics:  
    - MSE  
    - Precision@5  
    - Spearman Correlation  
  - Saves plots in `/experiments/plots/ID`  
    - ID = `exhaustive` (baseline imputation)  
    - ID = `pathdependent` (tabular datasets)

---

### 3. Runtime Analysis

- **`runtime_analysis.py`**  
  - Variant of `approximation_pathdependent.py`  
  - Stores results for:  
    - PolySHAP → `runtime_analysis.csv`  
    - RegressionMSR → `runtime_analysis_baselines.csv`  

- **`plot_runtime_analysis.py`**  
  - Creates runtime plots and stores them in `/experiments/plots/runtime_analysis`


### 4. Performance Table
- **`performance_table.py`**  
  - Generates a LaTeX table summarizing the performance of all methods across datasets.  
  - Outputs the table to `performance_table.csv` for easy inclusion in LaTeX documents.  
  - The table includes metrics Mean, Q1, Q2, and Q3 for MSE for each method and dataset combination.