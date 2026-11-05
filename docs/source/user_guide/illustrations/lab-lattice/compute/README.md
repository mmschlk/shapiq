# Computing the data of "lab-lattice"

`compute.py` downloads the hourly UCI bike sharing data once (from
`https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip`, into scikit-learn's data
folder, `sklearn.datasets.get_data_home()`), trains a gradient boosting model
(`HistGradientBoostingRegressor`) that predicts the rentals of an hour from four features (`hr`,
`workingday`, `temp`, `hum`: hour, working day, temperature, humidity), and explains one hour, a
humid working-day morning: Thursday, 19 May 2011, 8:00. It writes `../assets/data.js`:

- the explanation game of the hour with marginal masking (1,000 background hours of the training
  data): its prediction for every subset of the features, all 2^4 = 16;
- numbers, with shapiq's `ExactComputer`: the Shapley values of the features, the Shapley
  interactions up to pairs (k-SII of order 2), and the Möbius transform of all 16 subsets (the
  empty set holds the average prediction); and the Shapley values of the two groups of a
  partition, time (hour, working day) and weather (temperature, humidity), each group one player,
  computed by the script from the game;
- curves: the hour's prediction as the features of a set vary, the others as they are (the ICE
  curves of the single features and of the two groups), and the components f_S of the functional
  decomposition of the model with marginal masking, f = Σ f_S over all subsets S (the partial
  dependence of S minus that of its subsets, by inclusion and exclusion). A component of three or
  more features is a slice over two of them, the others at the hour's values. The grids pass
  through the hour's own values, so that each component at the hour is its Möbius value.

Displayed numbers are whole rentals, rounded so that the parts add up to the shown total (largest
remainders); the raw values are kept with 2 decimals, the curves with 1.

Run it from the repository's root, with shapiq and scikit-learn installed (10 to 20 seconds after
the download; on a busy machine, `OMP_NUM_THREADS=1` keeps it fast):

```bash
python docs/source/user_guide/illustrations/lab-lattice/compute/compute.py
```

It prints the model's R² on held-out hours (0.82), the hour's prediction (482 rentals; 487 were
rented) and the average prediction (193), the check that the shown parts of each explanation add up
(to 482 − 193 = 289, and to 482 for all 16 Möbius values), and the check that each component at the
hour equals its Möbius value.
