# Computing the data of "lab-hulls"

`compute.py` trains a gradient boosting model on six features of scikit-learn's diabetes data in raw
units (`load_diabetes(scaled=False)`: 442 patients, the target is a measure of disease progression
one year after baseline): age, BMI, blood pressure, glucose (`s6`), HDL (`s3`), and log
triglycerides (`s5`). Sex is left out: the data codes it as 1 or 2 without saying which is which,
and a curve over it would be two points. The model has shallow trees, a slow learning rate, and
subsampling, as befits a small data set; its R² on held-out patients is estimated by 5-fold
cross-validation (0.43 on average, from 0.24 to 0.56 over the folds), and the model is then fitted
on all patients. It explains patient 400 of the data (its row index), chosen for an explanation
with both signs: the body (BMI 30.6, blood pressure 121) raises the prediction, the lipids
(HDL 69, log triglycerides 4.26) lower it. It writes `../assets/data.js` (about 40 kB):

- the explanation game of the patient with marginal masking (all 442 patients as background), and
  from it, with shapiq's `ExactComputer`, the Shapley values and the pairwise Shapley interactions
  (k-SII of order 2); and the Shapley values of the units of a partition, BMI and blood pressure as
  one unit (the body), HDL and log triglycerides as one (the lipids), age and glucose alone, each
  unit one player, computed by the script from the game (`partition_values`);
- the patient's prediction as each feature varies (its ICE curve, 40 points), as the two features
  of each group vary (a surface, 20 × 20 points), and the pure interaction of each pair (its
  surface minus what each feature does alone), on evenly spaced grids through the patient's own
  values, so that the patient lies on them.

The illustration shows values to one decimal. The Shapley values of the features and those of the
units are also stored rounded so that they add up to the shown prediction minus the shown average
(largest remainders): 174.8 − 152.2 = 22.6. To keep the file small, the explanations are stored to
2 decimals, the curves' predictions and the surfaces to 1, the pure interactions to 2, and the
grids' values to 3 (the surfaces store only the first and last value of each feature's grid).

Run it from the repository's root, with shapiq and scikit-learn installed (it takes about 4
seconds):

```bash
python docs/source/user_guide/illustrations/lab-hulls/compute/compute.py
```

It prints the R² of each fold and their mean, the patient's prediction and the average
prediction, the Shapley values of the features and of the units with their sums (both 22.65, the
prediction minus the average), the rounded values shown, the four strongest interactions, and the
size of the file it wrote.
