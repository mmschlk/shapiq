# Computing the data of "units-and-forms"

`compute.py` trains a gradient boosting model on six features of the California housing data
(`MedInc`, `HouseAge`, `AveRooms`, `Population`, `Latitude`, `Longitude`) and explains the first
house of the data. It writes `../assets/data.js`:

- the explanation game of the house with marginal masking (256 background houses), and from it,
  with shapiq's `ExactComputer`, the Shapley values and the pairwise Shapley interactions (k-SII of
  order 2), and the Shapley value of every group of features taken as one player;
- the house's prediction as each feature varies (40 points), as each pair of features varies
  (20 × 20 points), and the pure interaction of each pair (the surface minus what each feature does
  alone).

Run it from the repository's root, with shapiq and scikit-learn installed (it takes a few seconds):

```bash
python docs/source/user_guide/illustrations/units-and-forms/compute/compute.py
```

It prints the model's R² on held-out houses, the house's prediction, and the check that the
Shapley values add up to the prediction minus the average prediction.
