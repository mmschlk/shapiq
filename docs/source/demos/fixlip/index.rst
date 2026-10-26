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

The FIxLIP game
^^^^^^^^^^^^^^^

FIxLIP explains a vision–language model as a cooperative game: the players are the photo's
patches and the caption's tokens, and the value of a coalition is the similarity that the model
finds between what it sees of the photo and of the caption, with only the coalition's patches and
tokens. For CLIP, the game is a few lines of ``shapiq`` and ``transformers``:

.. code-block:: python

   import numpy as np
   import torch
   from PIL import Image
   from transformers import CLIPModel, CLIPProcessor

   import shapiq


   class FIxLIPGame(shapiq.Game):
       """How well CLIP matches a photo and a caption when it sees only some of them.

       The players are the photo's 7 x 7 patches and the caption's tokens. The value of
       a coalition is the cosine similarity of the photo with only the coalition's
       patches (the others painted in the mean color, which the model sees as nothing)
       and the caption with only its tokens (the others masked out of the attention).
       """

       def __init__(self, photo: Image.Image, caption: str) -> None:
           checkpoint = "openai/clip-vit-base-patch32"  # 224 x 224 pixels, 32 x 32 a patch
           self.model = CLIPModel.from_pretrained(checkpoint).eval()
           processor = CLIPProcessor.from_pretrained(checkpoint)
           inputs = processor(images=photo, text=caption, return_tensors="pt")
           self.pixels = inputs["pixel_values"]  # normalized: 0 is the mean color
           self.tokens = inputs["input_ids"]  # the caption's, between a start and an end
           tokens = processor.tokenizer.convert_ids_to_tokens(self.tokens[0, 1:-1])
           self.words = [token.removesuffix("</w>") for token in tokens]
           self.n_patches = 7 * 7
           n_players = self.n_patches + len(self.words)
           # the similarity of no patch and no word, which the players' effects add to
           empty = self.value_function(np.zeros((1, n_players), dtype=bool))[0]
           super().__init__(n_players, normalize=True, normalization_value=empty)

       @torch.no_grad()
       def value_function(self, coalitions: np.ndarray) -> np.ndarray:
           values = []
           for batch in np.array_split(coalitions, max(1, len(coalitions) // 64)):
               patches = torch.as_tensor(batch[:, : self.n_patches], dtype=torch.float32)
               # every patch covers 32 x 32 pixels: the absent ones get the mean color
               masks = torch.kron(patches.reshape(-1, 1, 7, 7), torch.ones(32, 32))
               # the absent tokens are masked out of the attention (start and end stay)
               attention = torch.ones(len(batch), self.tokens.shape[1], dtype=torch.long)
               attention[:, 1:-1] = torch.as_tensor(batch[:, self.n_patches :])
               outputs = self.model(
                   pixel_values=self.pixels * masks,
                   input_ids=self.tokens.repeat(len(batch), 1),
                   attention_mask=attention,
               )
               # the embeddings are normalized: their dot product is the cosine similarity
               similarity = (outputs.image_embeds * outputs.text_embeds).sum(dim=1)
               values.append(similarity.numpy())
           return np.concatenate(values)

This is the game of the demo's CLIP explanations: on the same photo, its values equal those of the
demo's scripts up to rounding.

Explain it yourself
^^^^^^^^^^^^^^^^^^^

The explanations are the Faithful Banzhaf interaction index (FBII) up to order 2, FIxLIP's index
with :math:`p = 1/2`: the main effect of every player and the interaction of every pair. Each photo
and model has between 59 and 83 players, too many to evaluate all coalitions of them, so
:class:`~shapiq.approximator.ProxySHAP` :cite:p:`Thies.2026` approximates the index: it fits an
XGBoost model to coalitions of the game, each player present with probability 1/2, and reads the
interactions off the model's trees exactly. Here it explains the demo's elephant, with the game
above (it takes a minute or two on a laptop):

.. code-block:: python

   import io
   import math
   import urllib.request

   from xgboost import XGBRegressor

   # the demo's elephant: a COCO photo (Wilson Kwo, CC BY-NC-SA 2.0), cropped to a square
   url = "https://s3.amazonaws.com/images.cocodataset.org/train2017/000000046761.jpg"
   photo = Image.open(io.BytesIO(urllib.request.urlopen(url).read()))
   photo = photo.crop((76, 0, 503, 427))
   game = FIxLIPGame(photo, "a small elephant standing next to a white bird.")
   n = game.n_players  # 49 patches and 10 tokens

   approximator = shapiq.approximator.ProxySHAP(
       n=n,
       index="FBII",
       max_order=2,
       proxy_model=XGBRegressor(  # the HPO-informed settings of ProxySHAP's proxy
           n_estimators=2000, max_depth=3, learning_rate=0.05, reg_lambda=5.0
       ),
       adjustment=False,
       # coalitions of players present with probability 1/2, where FBII puts its weight
       sampling_weights=np.array([math.comb(n, k) for k in range(n + 1)], dtype=float),
       random_state=0,
   )
   interactions = approximator.approximate(budget=2_000, game=game)  # the demo: 10,000

   # the strongest interactions of a patch (row, column) and a word
   pairs = {
       players: value
       for players, value in interactions.dict_values.items()
       if len(players) == 2 and players[0] < game.n_patches <= players[1]
   }
   for (patch, token), value in sorted(pairs.items(), key=lambda pair: -abs(pair[1]))[:5]:
       word = game.words[token - game.n_patches]
       print(f"patch ({patch // 7}, {patch % 7}) and {word!r}: {value:+.4f}")

.. code-block:: text

   patch (4, 0) and 'white': +0.0069
   patch (1, 6) and 'elephant': +0.0049
   patch (2, 4) and 'elephant': +0.0045
   patch (4, 6) and 'elephant': +0.0040
   patch (3, 5) and 'elephant': +0.0040

The egret's patch, at the left edge of the photo's fifth row, goes with "white", and the
elephant's patches go with "elephant". Try another caption, or ask the game what the model makes of
a coalition, e.g. ``game.value_function(coalitions)`` with your own rows of ``True`` and ``False``.

Every coalition costs one pass of the image model and one of the text model, so the demo's
explanations, with 10,000 coalitions each, took 5 to 10 minutes on four CPU cores. How faithful an
explanation is shows in the demo, under each photo: the :math:`R^2` of its main effects and
interactions on 512 coalitions it was not fit to.

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
