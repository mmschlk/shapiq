# Computing the data of "lab-stepper"

`compute.py` explains one passenger of the Titanic, step by step as the illustration tells it. It
loads the Titanic passengers from OpenML (`fetch_openml("titanic", version=1)`, parsed with
`liac-arff`; scikit-learn caches the download in `~/scikit_learn_data`), keeps the 1,045 with a known
age and fare, and trains a gradient boosting classifier (`GradientBoostingClassifier`, 100 trees of
depth 3) for the probability to survive on five features: sex, age, passenger class, fare, and the
family aboard (siblings, spouses, parents, children). Three quarters train it; on the other quarter
(262 passengers) its AUC is 0.84 and its accuracy 0.79.

The explained passenger is Mrs. Edward Ford (Margaret Ann Watson), 48, in third class with four of
her family aboard (`PASSENGER`): being a woman raises her chance, third class lowers it, and the two
interact more than any other pair. The model gives her 18%, against an average of 41%. All
explanations are in percentage points (pp) of her predicted probability. It writes
`../assets/data.js`:

- the explanation game with marginal masking (all 783 training passengers as the background), and
  from it, with shapiq's `ExactComputer`, the **Shapley values** of the five features and the
  pairwise **Shapley interactions** (k-SII of order 2); and the **Shapley values of the groups** of
  a partition, each group one player (`partition_values`): who she is (sex, age), her ticket (class,
  fare), and her family;
- her **ICE curves** (her probability as one feature varies), the **curves of the groups** (one
  curve over the second feature for each value of the first, a category), and the **pure
  interaction surfaces** of the pairs (the surface minus what each feature does alone), on grids
  through her own values.

The Shapley values, and the values of the groups, add up to her prediction minus the average
(−22.67 pp); the illustration rounds them to whole points that add up to the rounded difference.
Numbers are rounded to 3 decimals, the curves and surfaces to 2 (the file is about 8 kB).

Run it from the repository's root, with shapiq and scikit-learn installed. It takes a few seconds
(and the download from OpenML once); with other jobs on the machine, `OMP_NUM_THREADS=1` keeps it
fast:

```bash
OMP_NUM_THREADS=1 python docs/source/user_guide/illustrations/lab-stepper/compute/compute.py
```

It prints the held-out AUC and accuracy, the passenger, her prediction and the average, the Shapley
values, the values of the groups, and the interactions, the check that the Shapley values and the
values of the groups add up to the prediction minus the average, and the size of the file written.
