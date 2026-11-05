# Computing the data of "lab-digits"

`compute.py` trains a multi-layer perceptron (`MLPClassifier`, one hidden layer of 128 units, L2
penalty 0.3) on scikit-learn's handwritten digits (8 × 8 pixels, ink from 0 to 16; 1,347 digits to
train, 450 held out, split with `random_state=0`), and explains the probability it gives the class it
predicts for one held-out digit, a 7 (the 436th of the held-out digits, `DIGIT = 435`). The features
are the 16 patches of 2 × 2 pixels, a 4 × 4 grid numbered 1 to 16 row by row. It writes
`../assets/data.js`:

- the explanation game of the digit: the probability of 7 with the patches of a coalition kept and
  the others **erased** (their pixels set to 0, blank paper), for all 2^16 = 65,536 coalitions, in
  batches; from it, with shapiq's `ExactComputer`, the Shapley values of the patches and their
  pairwise Shapley interactions (k-SII of order 2, all 120 pairs), and the Shapley values of the
  four rows of patches, each row one player, computed by the script from the game
  (`partition_values`);
- the digit's ICE curves: the probability as the ink of one patch (its four pixels), or of one row,
  is set to 0, 1, …, 16 and the rest of the digit stays;
- for the 6 pairs with the strongest interactions, the pure interaction surface: the probability as
  the ink of both patches goes over 0, 2, …, 16, minus what each patch does alone (9 × 9 points).

The pixels of the digit are stored as they are; probabilities are rounded to 4 decimals, the curves
and surfaces to 3. The file is about 8 kB.

Run it from the repository's root, with shapiq and scikit-learn installed:

```bash
OMP_NUM_THREADS=1 python docs/source/user_guide/illustrations/lab-digits/compute/compute.py
```

It takes about 20 seconds (the model's training and the k-SII of 16 players take most of it;
`OMP_NUM_THREADS=1` keeps a busy machine from slowing it down), and prints the model's accuracy on
the held-out digits (98.4%), the digit's label and predicted probability (91.8%), the probability
with every patch erased (16.9%), the check that the Shapley values of the patches and of the rows
add up to the prediction minus the erased digit (0.7495), and the strongest pairs.
