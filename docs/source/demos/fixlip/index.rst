FIxLIP: Which Words Go With Which Parts of a Photo?
===================================================

.. rst-class:: shapiq-examples-lead

How well does a vision–language model match a photo and its caption, and what makes it match
them? This demo explains how similar CLIP and SigLIP 2 find five COCO photos and their captions by
the interactions of image regions and words, as in FIxLIP :cite:p:`Baniecki.2025`, computed with
``shapiq``.

.. shapiq-demo:: fixlip
   :fallback: images/hero_siglip2.png
   :alt: A photo of a giraffe drinking at a river and the text 'a giraffe drinking water from a river', explained by a network of image regions and words, linked by the interactions that raise (red) or lower (blue) the similarity that SigLIP 2 assigns them.

What you see
------------

**The models.** CLIP :cite:p:`Radford.2021` and SigLIP 2 :cite:p:`Tschannen.2025` embed a photo
and a text in the same space; the closer the two embeddings (their cosine similarity), the better
the text describes the photo, as the model sees it. Both models here are ViT-B/32, as in FIxLIP's
figures: they cut the photo into patches of 32 x 32 pixels, 7 x 7 patches for CLIP and 8 x 8 for
SigLIP 2. Their similarities have different scales, so the demo compares each prediction to those
of 200 COCO photos with their own captions and with unrelated ones.

**The players** are the photo's patches and the caption's tokens. A coalition of players is what
the model gets to see: the patches outside the coalition are masked (replaced by patches of the
mean color), and the tokens outside it are masked out of the text model's attention. The
explanation splits the similarity of the whole photo and caption into the **main effect** of every
player and the **interaction** of every pair of players. An interaction is red when the two players
raise the similarity together by more than they do alone, and blue when they raise it by less.
Point at a region or a word (or tap it) to see all its interactions.

**The views.** *Clique* shows the players that together raise the similarity the most, found by
FIxLIP's greedy search, on the photo and as a graph of their interactions; − and + change its
size. *Full explanation* tints every player by its main effect and draws the strongest
interactions. *No explanation* shows only what the model sees and what it predicts: one number.

How it was computed
-------------------

The explanations are the Faithful Banzhaf interaction index (FBII) up to order 2, FIxLIP's index
with :math:`p = 1/2`. Each photo and model has between 59 and 83 players, too many to evaluate
all coalitions of them, so :class:`~shapiq.approximator.ProxySHAP` :cite:p:`Thies.2026` approximates
the index: it fits an XGBoost model to 10,000 coalitions, each player present with probability 1/2,
and reads the interactions off the model's trees exactly.

.. code-block:: python

   import math

   import numpy as np
   from xgboost import XGBRegressor

   import shapiq

   n = game.n_players  # the photo's patches and the caption's tokens
   proxy = XGBRegressor(  # the HPO-informed settings of ProxySHAP's proxy
       n_estimators=2000, max_depth=3, learning_rate=0.05, reg_lambda=5.0
   )
   approximator = shapiq.approximator.ProxySHAP(
       n=n,
       index="FBII",
       max_order=2,
       proxy_model=proxy,
       adjustment=False,
       # coalitions of players present with probability 1/2, where FBII puts its weight
       sampling_weights=np.array([math.comb(n, k) for k in range(n + 1)], dtype=float),
   )
   interactions = approximator.approximate(budget=10_000, game=game)

Every coalition costs one pass of the image model and one of the text model, so an explanation
takes 5 to 10 minutes on four CPU cores. How faithful an explanation is shows in the demo, under
each photo: the :math:`R^2` of its main effects and interactions on 512 coalitions it was not fit
to.

Run it again
------------

The scripts that computed the demo's data are in
`docs/source/demos/fixlip/compute <https://github.com/mmschlk/shapiq/tree/main/docs/source/demos/fixlip/compute>`__,
with a README on how to run them. ``config.toml`` lists the models and the photos with their
captions; change it to explain other photos, and then run:

.. code-block:: bash

   python explain.py  # the explanations of every photo and model
   python norms.py    # the scale of the similarities, on 200 COCO photos
   python pack.py     # the demo's data, ../assets/data.js

The photos are from `COCO <https://cocodataset.org>`__, and the demo credits each one with its
author and license.
