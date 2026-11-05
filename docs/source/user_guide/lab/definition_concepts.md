---
orphan: true
---

# Lab: Five Concepts for Definition 2.1

% A page of the lab, not in the guide's table of contents: concepts for the illustration of
% Definition 2.1 (2.1 Feature-Based Explanations), to compare and to pick from. Each lives in
% user_guide/illustrations/lab-<name>/ with its compute script; delete the ones not chosen.

Five ways to show the core message of {ref}`Definition 2.1 <def-feature-based-explanation>`: the
**units** $\units$ (a collection of sets of features), the **explanation space** $\explspace$, and
the **explanation** $\expl$, one $\expl(S)$ for each unit $S$. Each concept tries one idea, on real
data with a real model, and computes every number it shows. The current illustration of the
chapter, for comparison, is on {doc}`../perspectives/feature_based_explanations`.

| Concept | The idea | Data and model |
| --- | --- | --- |
| {ref}`1. Waterfall <lab-waterfall>` | the same prediction, cut into different units | California housing, gradient boosting |
| {ref}`2. Subset lattice <lab-lattice>` | the units are a pick from all subsets | bike sharing (UCI), gradient boosting |
| {ref}`3. Hulls <lab-hulls>` | a unit is a shape drawn around features | diabetes, gradient boosting |
| {ref}`4. Image patches <lab-digits>` | the same definition on an image model | handwritten digits, a neural network |
| {ref}`5. Guided steps <lab-stepper>` | three questions, then the answer | Titanic (OpenML), gradient boosting |

(lab-waterfall)=
## 1. Waterfall: the Same Prediction, Cut Into Different Units

- **The idea.** A waterfall climbs from the average price to this house's price; each brick is one
  unit $S$, its height $\expl(S)$. One switch re-cuts it: single features (Shapley values), the
  location as one unit (Shapley values of the parts), all sets of at most two features (k-SII), all
  subsets (the Möbius transform). The bricks merge, split, appear, and vanish, and the total stays:
  the units change what is explained, not what is explained away.
- **The three parts.** A dot matrix under the bricks names each unit as a set ($\units$); the price
  axis is $\explspace = \mathbb{R}$; the brick in focus reads $S \mapsto \expl(S)$.
- **Data.** The guide's running example: the first house of the California housing data, gradient
  boosting on six features ($R^2$ 0.84), \$447k against \$211k on average.
- **Look at.** Switch to "Groups": latitude and longitude fuse into the location. Switch to "All
  subsets": the path overshoots before it lands.
- **Limits.** Numbers only (curves do not stack); with all 63 subsets the columns are thin on a
  phone.

```{shapiq-illustration} lab-waterfall
:anchor: illustration-lab-waterfall
:alt: A waterfall from the average price of a house to its prediction, whose bricks are the units of an explanation; switching between single features, groups, pairs, and all subsets re-cuts the bricks while the total stays.

The same prediction, cut into different units: every explanation adds up to the same total.
```

(lab-lattice)=
## 2. Subset Lattice: the Units Are a Pick From All Subsets

- **The idea.** All 16 subsets of four features in their lattice ($2^{\players}$, by size). Choosing
  the units lights up some of them: single features, groups, sets of at most two features, all
  subsets, or sets the reader clicks. The explanation space decides what hangs on each lit set: a
  number, a curve, or a map. The lit lattice is the explanation.
- **The three parts.** A switch for $\units$ (with the collection written out), a switch for
  $\explspace$, and the name of $\expl$; a panel below shows $S$ and $\expl(S)$ for the set in
  focus, with its curve large.
- **Data.** The UCI bike sharing data, hourly: gradient boosting on hour, working day,
  temperature, and humidity ($R^2$ 0.82), for a working-day morning at 8:00 (482 rentals against
  193 on average). Working day alone adds +2, but together with the hour +112.
- **Look at.** "Pairs" with numbers: the interaction of the hour and the working day. Click sets to
  build your own collection.
- **Limits.** Tall; with custom collections, the explanation switches to the Möbius transform,
  which may surprise.

```{shapiq-illustration} lab-lattice
:anchor: illustration-lab-lattice
:alt: The lattice of all 16 subsets of four features of the bike sharing data; choosing the units lights up a collection of them, and each lit set carries its explanation, a number, a curve, or a map.

The units are a pick from all subsets of the features: the lit lattice is the explanation.
```

(lab-hulls)=
## 3. Hulls: a Unit Is a Shape Drawn Around Features

- **The idea.** The features are dots; each unit is a smooth shape drawn around its features: a ring
  around one, a capsule around two. Switching the units morphs the shapes over the same dots; the
  explanation lives on each shape, as its tint and a tag (a number, or a small curve).
- **The three parts.** Three layers: the dots are the features $\players$, the outlines are the units
  $\units$, the tints and tags are $\expl(S) \in \explspace$. A line spells out the definition as it
  stands, and a box the map for one unit.
- **Data.** The diabetes data in raw units: gradient boosting on six features ($R^2$ 0.43 in
  cross-validation, an honest small dataset), one patient, 174.8 against 152.2 on average.
- **Look at.** "Groups": the body (BMI and blood pressure) and the lipids as capsules. The morphs
  between the collections.
- **Limits.** The pairs are busy; small interactions next to the Shapley values.

```{shapiq-illustration} lab-hulls
:anchor: illustration-lab-hulls
:alt: Six features of a patient as dots, each unit of an explanation a shape drawn around its features, tinted red or blue by its explanation; switching the units morphs the shapes.

A unit is a shape drawn around features; its explanation is its tint and its tag.
```

(lab-digits)=
## 4. Image Patches: the Same Definition on an Image Model

- **The idea.** Definition 2.1 is not about tables. A handwritten 7 is cut into 16 patches (or its
  rows, or pairs of patches), and the definition is laid out left to right: the units on the
  image, the explanation $\expl$ as an arrow, and the explanation space $\explspace$ as its own panel
  (a number line, or the curves overlaid). Each value flies from its unit to its place in
  $\explspace$.
- **The three parts.** Literally three columns: $\units \rightarrow \expl \rightarrow \explspace$.
- **Data.** scikit-learn's digits and a neural network (98.4% on held-out digits), one 7 (91.8%); a
  removed patch is erased, and the game is computed exactly over all $2^{16}$ coalitions.
- **Look at.** "Rows": the bar of the 7 carries +51.9. The values flying across $\expl$ when the
  units change.
- **Limits.** The colors break up the digit's shape; tall on a phone.

```{shapiq-illustration} lab-digits
:anchor: illustration-lab-digits
:alt: A handwritten 7 cut into patches, rows, or pairs of patches, the units of an explanation of a neural network; each unit's explanation is painted on the image and placed in the explanation space beside it.

The same definition on an image model: the units on the image, the explanation space beside it.
```

(lab-stepper)=
## 5. Guided Steps: Three Questions, Then the Answer

- **The idea.** The definition told as a story in four steps, with Back and Next: (1) the passenger
  and her features, (2) the units, as rows of dots across the features, (3) the explanation space,
  with one unit answered and the others waiting, (4) the explanation, every unit answered. Then the
  units and the space change freely.
- **The three parts.** One per step, each with its symbol and one sentence; at the end, the switch
  for $\units$ sits over the units and the switch for $\explspace$ over the answers, so the layout
  reads as $\expl : \units \to \explspace$.
- **Data.** The Titanic passengers from OpenML: gradient boosting on five features (AUC 0.84), a
  third-class passenger, 18% against 41% on average, in percentage points. Sex × class is her
  strongest interaction (−14 points).
- **Look at.** Step 3 with "Pairs": one answer, nine questions. The rows dropping out of the
  features.
- **Limits.** Pairs with curves (ten maps) is busy.

```{shapiq-illustration} lab-stepper
:anchor: illustration-lab-stepper
:alt: A guided story in four steps about one Titanic passenger: her features, the units of an explanation as rows of dots, the explanation space for one unit, and the explanation for all units.

Three questions, then the answer: the parts of the definition one at a time, then the whole.
```
