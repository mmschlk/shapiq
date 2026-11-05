# Computing the data of "lab-waterfall"

`compute.py` trains the gradient boosting model of the illustration "units-and-forms" on six
features of the California housing data (`MedInc`, `HouseAge`, `AveRooms`, `Population`,
`Latitude`, `Longitude`) and explains the first house of the data, as that illustration does: its
explanation game with marginal masking (256 background houses), for all 2^6 = 64 subsets of the
features. From the game, it computes four explanations that all add up to the same total, the
prediction minus the average prediction:

| View       | Units 𝒮                                      | Explanation                                   |
| ---------- | -------------------------------------------- | --------------------------------------------- |
| `features` | each feature alone (6)                       | Shapley values (`ExactComputer`, `SV`)        |
| `groups`   | the location (latitude, longitude) as one, the others alone (5) | Shapley values of the parts, in the game whose players are the parts |
| `pairs`    | every feature and every pair (21)            | Shapley interactions of order 2 (`k-SII`)     |
| `all`      | every non-empty subset (63)                  | Möbius transform (`Moebius`)                  |

It writes `../assets/data.js` (about 2.5 KB): the average and the prediction, and for each view its
units (in the order the illustration shows them: by size, then by features), their values in
$1,000s rounded to 0.01, and the values shown, in whole $1,000s, rounded so that they add up to the
total shown ($447k − $211k = $236k) by largest remainders.

Run it from the repository's root, with shapiq and scikit-learn installed:

```bash
python docs/source/user_guide/illustrations/lab-waterfall/compute/compute.py
```

It takes about 3 seconds (on a machine busy with other work, `OMP_NUM_THREADS=1` keeps the model's
threads from waiting on each other). It prints, for each view, the number of units, the sum of the
values, and the sum of the values shown, then the model's R² on held-out houses (0.835), the
average (211.23), the prediction (446.86), and the total (235.63, shown as 236).
