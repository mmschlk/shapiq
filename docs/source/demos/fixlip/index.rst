Explaining Similarity in Vision-Language Encoders
=================================================

.. rst-class:: shapiq-examples-lead

FIxLIP provides faithful interaction explanations of language–image pre-training models
:cite:p:`Baniecki.2025`. It explains the similarity that encoders such as CLIP and SigLIP 2 predict
for an image–text pair. This demo lets you explore FIxLIP explanations for five image–text pairs
from MS COCO. All explanations are computed with ``shapiq``.

.. shapiq-demo:: fixlip
   :fallback: images/hero_siglip2.png
   :alt: A photo of a giraffe drinking at a river and the text 'a giraffe drinking water from a river', explained by a network of image regions and words, linked by the interactions that raise (red) or lower (blue) the similarity that SigLIP 2 assigns them.

----

.. _fixlip-sec-summary:

.. rst-class:: fixlip-summary

Summary
-------

FIxLIP :cite:p:`Baniecki.2025` explains the similarity that a vision–language encoder predicts for
an image–text pair. It decomposes this prediction into *main effects* of single tokens and
*interactions* between pairs of tokens, i.e. image patches and text tokens. The explanation is
defined as the most faithful additive model of a cooperative game that masks tokens, which leads to
the weighted Banzhaf interaction index.

This page explains FIxLIP step by step on the five examples of the demo. It motivates why
similarity calls for interactions, defines the game and the explanation, describes how it is
computed, and evaluates it with three metrics. **For all five examples, the second-order explanation
is more faithful to the model than the first-order explanation derived from it.** This agrees with
the experiments of the authors on 1,000 image–text pairs from MS COCO. Finally, we compare CLIP and
SigLIP 2 and show how to explain an image–text pair with ``shapiq``.

.. _fixlip-sec-introduction:

Introduction
------------

Contrastive language–image pre-training (CLIP) :cite:p:`Radford.2021` revolutionized computer
vision by learning data representations that perform well in many downstream tasks. Scaling the
training to predict similarity between image and text, combined with continuous architectural
improvements such as SigLIP :cite:p:`Zhai.2023` and SigLIP 2 :cite:p:`Tschannen.2025` and
increasing the size of datasets, leads to the development of powerful *vision–language encoders*.
These rather opaque encoders become the backbone components in large vision–language models capable
of actual conversation and reasoning. They are increasingly applied in high-stakes decision-making,
like in the case of medical imaging :cite:p:`Ikezogwo.2023,Zhao.2025` and remote sensing
:cite:p:`Liu.2024`.

One way of validating similarity predictions at inference time is explaining the encoders with
*saliency maps* :cite:p:`Chefer.2021,Zhao.2024`. However, these first-order importance scores are
limited by their unimodal view of the model's output. **Vision–language encoders cannot be
faithfully explained without taking into account the cross-modal interactions between image–text
pairs** :cite:p:`Liang.2023`. **FIxLIP** :cite:p:`Baniecki.2025` addresses this limitation from a
game-theoretic perspective. Input tokens, i.e. image patches and text tokens, form coalitions of
players in a *cooperative game*. The resulting explanation assigns *attribution scores* to each
token and *interaction scores* to each pair of tokens in an image–text pair. FIxLIP thus shows not
only how similar a model finds an image and a text, but also **what the model sees as similar**.

.. _fixlip-sec-background:

Background
----------

This section describes the encoders, argues why their similarity predictions call for interactions,
and reviews how related work explains them.

.. _fixlip-sec-encoders:

Vision–Language Encoders in a Nutshell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A vision–language encoder :math:`f: \mathbb{R}^{n_\mathcal{I}} \times \mathbb{R}^{n_\mathcal{T}}
\to [-1, 1]` maps an image–text pair to a similarity score, as illustrated in
:shapiq-figure:`encoders`. It consists of a *vision encoder* :math:`f_\mathcal{I}:
\mathbb{R}^{n_\mathcal{I}} \to \mathbb{R}^{d}` with :math:`n_\mathcal{I}` image patches and a
*language encoder* :math:`f_\mathcal{T}: \mathbb{R}^{n_\mathcal{T}} \to \mathbb{R}^{d}` with
:math:`n_\mathcal{T}` text tokens. Each encoder maps its input to an *embedding* vector of dimension
:math:`d`. For an image–text pair :math:`(x_\mathcal{I}, x_\mathcal{T})`, the model then predicts
how well the image and the text *match* as the *cosine similarity* of the two embeddings.

.. math::
   :label: fixlip-vle

   f(x_\mathcal{I},x_\mathcal{T})
   := \cos\!\big(f_\mathcal{I}(x_\mathcal{I}),f_\mathcal{T}(x_\mathcal{T})\big)
   = \frac{f_\mathcal{I}(x_\mathcal{I}) \cdot f_\mathcal{T}(x_\mathcal{T})}
   {\Vert f_\mathcal{I}(x_\mathcal{I}) \Vert \cdot \Vert f_\mathcal{T}(x_\mathcal{T}) \Vert}

Vision–language encoders are trained on large collections of image–text pairs from the web. The
training *aligns* the two encoders. The embeddings of an image and of its own text then point in a
similar direction, whereas those of unrelated pairs do not. CLIP :cite:p:`Radford.2021`, SigLIP
:cite:p:`Zhai.2023` and SigLIP 2 :cite:p:`Tschannen.2025` are instances of this idea, which differ
mainly in their training objective. CLIP contrasts each matching pair with all other pairs of a
batch. SigLIP instead decides for every pair independently whether it matches, and SigLIP 2 extends
this recipe with further objectives. The demo uses the ViT-B/32 variant of CLIP and SigLIP 2, which
split an image into 7 × 7 and 8 × 8 patches, respectively.

.. shapiq-demo-figure:: fixlip encoders
   :alt: A photo, cut into 8 by 8 patches, goes through the vision encoder. Its text, cut into tokens, goes through the language encoder. Each encoder returns an embedding of 768 values, of which the first four are shown. The cosine similarity of the two embeddings is the prediction of the model.

   **From an image–text pair to a similarity.** CLIP and SigLIP 2 encode the patches of the image
   and the tokens of the text separately. The cosine similarity of the two embedding vectors is the
   prediction of the model.

Two models can predict very different similarity scores for the same image–text pair. These
scores cannot be compared directly, because every model has its own scale. A score is therefore
best read *relative* to a reference set of image–text pairs. In such a set, each image forms a
*matching pair* with its own caption and an *unrelated pair* with the caption of any other image. A
score in the range of the matching pairs and above the unrelated pairs tells that the model finds
the text to describe the image well. :shapiq-figure:`scale` shows these ranges for 200 images from
MS COCO :cite:p:`Lin.2014`, where all five examples of the demo score at least as high as a typical
matching pair for both models. A single score tells how similar the model finds the pair, but not
why.

.. shapiq-demo-figure:: fixlip scale
   :alt: For SigLIP 2, matching pairs have similarities around 0.15 and unrelated pairs around minus 0.03. For CLIP, matching pairs are around 0.31 and unrelated pairs around 0.15. A dot marks the chosen example on the scale of each model.

   **Every model has its own scale.** Thick bars cover the middle half of the pairs and thin lines
   cover the middle 90%. Gaps mark the medians and dots mark the chosen example.

.. _fixlip-sec-interactions:

Interactions Are the Natural Language of Similarity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Explanations of a classifier usually attribute a score to the parts of *one* input. A similarity
prediction is *different*, because it is a property of an image–text *pair*. :cite:t:`Liang.2023`
argue that such multimodal models cannot be understood without their cross-modal interactions. To
see why, consider what happens when one half of the pair is missing. :shapiq-figure:`masks` shows
the prediction of SigLIP 2 with the image alone and with the text alone. In both cases, it stays
close to the prediction for an *empty input*, in which all patches and all text tokens are masked.
Often it is even lower. Only both modalities together reach the similarity of the pair. Neither
modality has a similarity on its own.

.. shapiq-demo-figure:: fixlip masks
   :alt: Four versions of the example with the similarity that SigLIP 2 predicts. All tokens, no tokens, the image only and the text only. The similarity is high only with all tokens.

   **One modality alone.** The prediction of SigLIP 2 for the example with all tokens, with no
   tokens, with the image only and with the text only. Gray patches and crossed-out text tokens are
   masked.

This behavior follows from how the prediction is computed. The dot product in :eq:`fixlip-vle`
multiplies the coordinates of the image embedding with those of the text embedding. What an image
patch contributes to the prediction therefore *depends* on the text it is compared with, and vice
versa. This dependence is also what makes the model useful. If the prediction were the sum of one
score for the image and one score for the text, the best text would be the same for every image,
and the best image the same for every text. Retrieval and zero-shot classification work only
because the prediction is more than such a sum. Everything beyond the sum is an *interaction*
between the two modalities.

An explanation that assigns one score to each token misses exactly this part. A saliency map can
highlight an object in the image and the words of the text. However, it cannot tell *which* of the
words the model matched with the object. Only an interaction between an image patch and a text
token can express such a relation. **A faithful explanation of similarity thus has to include
cross-modal interactions**, of which FIxLIP explains the pairwise part. This conclusion is not
specific to CLIP, since any model that scores an image–text pair has to relate tokens across the
two modalities. It also agrees with the user studies of :cite:t:`Liang.2023`, in which visualizing
second-order attributions was necessary for understanding multimodal models.

.. _fixlip-sec-related:

Related Work and Its Shortcomings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most explanation methods for vision–language encoders compute first-order attributions and
visualize them as saliency maps. This includes *gradient-based* methods such as Grad-ECLIP
:cite:p:`Zhao.2024`, *attention-based* methods such as GAME :cite:p:`Chefer.2021`, and
*information-based* methods :cite:p:`Wang.2023b,Zhu.2025`. Shapley and Banzhaf values offer a
game-theoretic alternative, which MM-SHAP :cite:p:`Parcalabescu.2023` aggregates to measure how
much each modality contributes. All of these attributions are of *first order*, i.e. one score per
token. Thus, they cannot disentangle which words relate to which image regions.

A second line of work approximates *second-order* attributions. :cite:t:`Joukovsky.2023` approximate
the interactions with bilinear models, which :cite:t:`Sammani.2024` apply to image–image encoders.
In concurrent work to FIxLIP, exCLIP :cite:p:`Moeller.2025` approximates only the cross-modal
interactions, based on Integrated Hessians :cite:p:`Janizek.2021`. It omits the *main effects* of
single tokens and the *intra-modal* interactions, e.g. between two words. Its explanation is thus a
bipartite graph without weights on its nodes. In the experiments of :cite:t:`Baniecki.2025`,
exCLIP fails to recover the subsets of tokens that matter most for the prediction, which the
authors attribute to these missing terms. :shapiq-figure:`paper-methods` compares the methods on
one input.

.. figure:: figures/paper-methods.webp
   :name: paper-methods
   :alt: The explanations of GAME, Grad-ECLIP, exCLIP and FIxLIP for a photo of a dog next to a yellow hydrant and the text 'black dog next to a yellow hydrant'. GAME and Grad-ECLIP tint image patches and text tokens. exCLIP draws lines between patches and tokens. FIxLIP shows both.
   :width: 100%
   :figclass: fixlip-paper-figure

   **The methods on one input.** Explanations of CLIP (ViT-B/32) by two first-order methods, by
   exCLIP, and by FIxLIP. In this example, the text token *ant* is the most important for the
   prediction.

.. _fixlip-sec-method:

Explaining Similarity with FIxLIP
---------------------------------

FIxLIP formalizes the problem with game theory. Its explanation covers main effects, cross-modal
interactions, and intra-modal interactions. Moreover, it is model-agnostic, since it only needs the
predictions for masked inputs.

.. _fixlip-sec-game:

The FIxLIP Game
~~~~~~~~~~~~~~~

FIxLIP views the input tokens as players of a cooperative game. We index the image patches with
:math:`N_\mathcal{I} := \{1, \dots, n_\mathcal{I}\}` and the text tokens with :math:`N_\mathcal{T}
:= \{n_\mathcal{I} + 1, \dots, n_\mathcal{I} + n_\mathcal{T}\}`. For SigLIP 2 and the examples of
the demo, these are :math:`n_\mathcal{I} = 64` image patches and between 10 and 19 text tokens, e.g.
74 players for the first example (the elephant image). A *mask* :math:`M \subseteq N_\mathcal{I}
\cup N_\mathcal{T}` selects the tokens that the model receives. All other tokens are replaced with a
baseline. For inputs :math:`x, b` and a subset :math:`M_o \subseteq N` of the indices :math:`N \in
\{N_\mathcal{I}, N_\mathcal{T}\}` of one modality, the *masking operator* is defined as

.. math::
   :label: fixlip-masking

   x \oplus_{M_o} b := \begin{cases}
   x_i, & \text{if } i \in M_o, \\
   b_i, & \text{if } i \in N \setminus M_o.
   \end{cases}

For an image–text pair :math:`(x_\mathcal{I}, x_\mathcal{T})` and a baseline :math:`(b_\mathcal{I},
b_\mathcal{T})`, the *FIxLIP game* :math:`\nu: 2^{N_\mathcal{I} \cup N_\mathcal{T}} \to \mathbb{R}`
assigns to each mask the similarity that the model predicts for the masked image–text pair.

.. math::
   :label: fixlip-game

   \nu(M) := f\big(x_\mathcal{I} \oplus_{M \cap N_\mathcal{I}} b_\mathcal{I}, \,
   x_\mathcal{T} \oplus_{M \cap N_\mathcal{T}} b_\mathcal{T}\big)

For the image, a masked patch is set to 0 after normalization, which propagates no signal forward.
The figures show it as a gray patch. For the text, a masked token is excluded through the attention
mask of the encoder, while special tokens and padding are always kept.

:shapiq-figure:`removal` shows four of these masks. For example, removing an object from the image
lowers the similarity for all five examples, although the text still names the object. With
:math:`2^{n_\mathcal{I} + n_\mathcal{T}}` masks, the game cannot be inspected directly. An
explanation has to *summarize* it.

.. shapiq-demo-figure:: fixlip removal
   :alt: Four masked versions of the example with the similarity that SigLIP 2 predicts. The image without one object, the image without another object, and two random masks, of which the second keeps fewer tokens.

   **Four masks of the game.** The prediction of SigLIP 2 for the example without one of its
   objects and for two random masks. Gray patches and crossed-out text tokens are masked.

:cite:t:`Baniecki.2025` explain the logit-scaled similarity :math:`C \cdot f(x_\mathcal{I},
x_\mathcal{T})`, where the learnt constant :math:`C` is about 100 for CLIP and about 112 for SigLIP
2. The demo explains the cosine similarity itself, so its values are smaller by this factor.

.. _fixlip-sec-explanation:

The FIxLIP Explanation
~~~~~~~~~~~~~~~~~~~~~~

A FIxLIP explanation :math:`\mathbf{e}` assigns a constant :math:`\mathbf{e}_0`, a *main effect*
:math:`\mathbf{e}_i` to each token :math:`i \in N_\mathcal{I} \cup N_\mathcal{T}`, and an
*interaction* :math:`\mathbf{e}_{\{i,j\}}` to each pair of tokens. Together, they approximate the
game :math:`\nu(M) \approx \hat\nu_\mathbf{e}(M)` with a simple additive model.

.. math::
   :label: fixlip-additive

   \hat\nu_{\mathbf{e}}(M) := \mathbf{e}_{0} + \sum_{i \in M} \mathbf{e}_{i}
   + \sum_{\{i,j\} \subseteq M: i \neq j} \mathbf{e}_{\{i,j\}}
   \quad \text{for all masks } M \subseteq N_\mathcal{I} \cup N_\mathcal{T}

An interaction is *cross-modal* if it links an image patch with a text token. It is *intra-modal*
if it links two image patches or two text tokens. The explanation can be viewed as a complete graph
with the tokens as weighted nodes and the interactions as weighted edges. The demo calls the image
patches *regions* and the text tokens *words*, and both together *features*.

**What an interaction measures.** A positive interaction means that two tokens raise the similarity
more together than the sum of their individual effects. :shapiq-figure:`interaction` measures this
directly on the model, for the strongest interaction of the explanation between an image patch and
a word. We sample 256 random masks of all other tokens, in which each token is kept with
probability 0.5, and add the patch, the word, or both. What both add beyond the sum of their
individual effects is their interaction. The second panel pairs the same patch with the word that
has the largest main effect. Here, both together add just about the sum of their individual
effects, so their interaction is close to zero. The last bar of each panel shows the estimate of
the explanation. For all five examples, it agrees with the interaction measured on the model up to
0.002.

.. shapiq-demo-figure:: fixlip interaction
   :alt: Two bar charts. Each shows how much adding a patch, adding a word and adding both change the similarity on average, the interaction, which is what both add beyond the sum of the two, and the estimate of this interaction in the explanation. The interaction is large in the first chart and close to zero in the second.

   **An interaction measured on the model.** The average change of the prediction of SigLIP 2 when
   tokens are added to 256 random masks. The interaction is the change for both together minus the
   changes for the patch and for the word. The FIxLIP estimate is the interaction of the two tokens
   in the explanation.

**Reading the explanation.** The demo shows the whole explanation as *Full explanation*, and
:shapiq-figure:`explanation` shows it for the chosen example. **Red** marks what raises the
similarity and **blue** marks what lowers it. Typically, the words that name the objects have the
largest main effects, and the strongest interactions link these words with patches of their objects.
The main effects of the image patches are small in comparison. For SigLIP 2 and all five examples,
their sum stays below 0.03 in absolute value, while the cross-modal interactions sum to about 0.1.
The list of the figure gives both sums for the chosen example. Thus, **the image contributes to the
similarity almost exclusively through interactions with the text**.

.. shapiq-demo-figure:: fixlip explanation
   :alt: The photo and its tokens. Tints mark the main effects, and lines link the pairs of tokens with the strongest interactions. A list gives the strongest main effects and the strongest interactions with their values, the sum of the main effects of all patches, and the sum of all cross-modal interactions.

   **The explanation of the example for SigLIP 2.** Tints show the main effects and lines show the
   12 strongest interactions, as in the demo. **Red** raises the similarity and **blue** lowers it.
   Lines under the text are interactions between two text tokens. The list also sums the main
   effects of all patches and all cross-modal interactions.

Two properties of the explanation are worth keeping in mind. First, a main effect is a coefficient
of the additive model :eq:`fixlip-additive` and not the effect of a token alone. For the example
with the elephant, the word *elephant* has the largest main effect with +0.064. With all other
tokens masked, however, this word changes the prediction of SigLIP 2 by only +0.002. The two numbers
differ for every example. The main effect absorbs what the word contributes together with many
patches of its object at once, which no single pair can express. Second, the terms of the
explanation do not have to add up to the prediction for the full input. FIxLIP optimizes how well
the additive model predicts all masks, which :ref:`the section on faithfulness
<fixlip-sec-faithfulness>` makes precise.

**First-order conversion.** A second-order explanation can always be simplified into a first-order
one. For a token :math:`i \in N_\mathcal{I} \cup N_\mathcal{T}`, the first-order attribution is
given by

.. math::
   :label: fixlip-first-order

   \mathbf{e}_i + p \sum_{j \in N_\mathcal{I} \cup N_\mathcal{T}: j \neq i} \mathbf{e}_{\{i,j\}}

Here, :math:`p \in (0, 1)` is the parameter of FIxLIP that :ref:`the section on faithfulness
<fixlip-sec-faithfulness>` introduces. In the demo, the attribution of a token is its main effect
plus one half of the sum of its interactions. :cite:t:`Baniecki.2025` show that these attributions
are the weighted Banzhaf values of :math:`\hat\nu_\mathbf{e}`. They can be visualized as a
traditional saliency map. However, the conversion keeps only the sum of the interactions of a token
and no longer says with which tokens it interacts.

.. _fixlip-sec-conditioning:

Conditioning on a Token
~~~~~~~~~~~~~~~~~~~~~~~

A complete graph with more than 70 nodes is hard to read as a whole. A token can therefore be
selected for *conditioning*, which shows only its interactions with all other tokens as a heatmap.
In the demo, hover over a region or a word, or tap it. :shapiq-figure:`conditioning` conditions on
three tokens of the chosen example.

.. shapiq-demo-figure:: fixlip conditioning
   :alt: Three panels with the photo and its tokens. Each panel frames one token, a word or an image patch, and tints all other tokens by their interaction with it.

   **Conditioning turns interactions into heatmaps.** Every token is tinted by its interaction with
   the framed token. The framed tokens are the two words and the patch of
   :shapiq-figure:`interaction`.

Conditioning on a word highlights the patches it goes with, and conditioning on a patch highlights
its words. Intra-modal interactions show up as well, and they can differ between the models. For
the example with the elephant, the words *elephant* and *small* interact negatively for SigLIP 2 and
positively for CLIP. :cite:t:`Baniecki.2025` report the same positive interaction for CLIP on this
example.

.. _fixlip-sec-cliques:

Cliques of Tokens
~~~~~~~~~~~~~~~~~

The explanation also allows searching for the subsets of tokens that raise the similarity the most.
In the graph view, such a subset is a *clique*. Its value is the sum of the main effects and
interactions of its tokens, i.e. :math:`\hat\nu_\mathbf{e}(M) - \mathbf{e}_0` for a clique
:math:`M`. Finding the best clique of a given size by brute force is infeasible. Instead, FIxLIP
uses a greedy search that starts from each token and repeatedly adds the token that raises the
value the most.

The demo opens with this view, called *Clique*, and its slider changes the size of the clique.
:shapiq-figure:`cliques` shows the cliques of the chosen example, with a slider of its own.
Typically, a small clique consists of words that name the objects and of patches that show them.
Larger cliques add neighboring patches and further words.

.. shapiq-demo-figure:: fixlip cliques
   :alt: Left, the photo and its tokens with the clique of six tokens. The tokens of the clique are framed, and lines show the interactions between them. Right, a line chart of the value of the clique in the explanation and of the gain measured on the model for every size. The value lies above the gain. A slider under the chart sets the size of the clique from 2 to 10.

   **Cliques of the example for SigLIP 2.** The slider sets the size of the clique, whose tokens are
   framed. The value is the sum of their main effects and interactions. The gain is the prediction
   of SigLIP 2 for the clique alone minus its prediction for an empty input.

The value of a clique is computed from the explanation and not from the model. The figure therefore
also reports what SigLIP 2 itself predicts for each clique alone, as a gain over an empty input. For
all five examples and all sizes, this gain is smaller than the value. For two examples, the clique
of two even consists of two words. Without any image patch, the model then predicts less than for an
empty input. The explanation thus *overestimates* small cliques, which we revisit in :ref:`the
evaluation <fixlip-sec-correlation>`.

.. _fixlip-sec-computation:

Computing FIxLIP
----------------

The FIxLIP explanation is defined as the most faithful additive model of the game. This section
defines this notion of faithfulness and describes how the explanation is estimated from sampled
masks.

.. _fixlip-sec-faithfulness:

Faithfulness
~~~~~~~~~~~~

A natural requirement is that the explanation predicts the game well. FIxLIP measures the squared
error :math:`\big(\nu(M) - \hat\nu_\mathbf{e}(M)\big)^2` over all masks and weights each mask
according to its size. For :math:`p \in (0, 1)`, the *p-faithfulness* is defined as

.. math::
   :label: fixlip-faithfulness

   \mathfrak{F}_p(\nu,\hat\nu) := \sum_{M \subseteq N_\mathcal{I} \cup N_\mathcal{T}}
   p^{\vert M \vert} (1-p)^{n_\mathcal{I} + n_\mathcal{T} - \vert M \vert}
   \big(\nu(M) - \hat\nu(M)\big)^2

The weight :math:`\mathbb{P}_p(M) := p^{\vert M \vert} (1-p)^{n_\mathcal{I} + n_\mathcal{T} - \vert
M \vert}` is the probability of the mask when every token is kept independently with probability
:math:`p`. The *FIxLIP-p* explanation is the most faithful approximation of the game with respect
to :math:`\mathfrak{F}_p`.

.. math::
   :label: fixlip-explanation

   \mathbf{e}^{\text{FIxLIP-}p}
   := \operatorname*{arg\,min}_{\mathbf{e}} \, \mathfrak{F}_p(\nu,\hat\nu_\mathbf{e})

Its interactions are the *weighted Banzhaf interactions* :cite:p:`Marichal.2011`. For :math:`p =
0.5`, all masks have equal weight. FIxLIP then recovers the faithful Banzhaf interaction index
:cite:p:`Tsai.2022`, which has a known analytical solution :cite:p:`Hammer.1992`.

**Masking with a purpose.** The parameter :math:`p` addresses a known problem of removal-based
explanations :cite:p:`Hase.2021`. Inputs with many masked tokens are *out-of-distribution*, i.e.
images become unrecognizable and texts become ambiguous. A larger :math:`p` prioritizes masks that
keep most tokens and thus stay closer to real inputs, whereas a smaller :math:`p` prioritizes
sparse masks. :shapiq-figure:`masking` shows the effect for the chosen example. At :math:`p = 0.3`,
many masks fall back to the similarity of an empty input. At :math:`p = 0.7`, the predictions move
towards the similarity of the full input. The colors separate the masks that keep the word with
the largest main effect from all others.

.. shapiq-demo-figure:: fixlip masking
   :alt: Three columns for p equal to 0.3, 0.5 and 0.7. Each shows one random mask and a histogram of the similarity of 512 random masks. The histogram moves from the similarity of no tokens towards the similarity of all tokens as p grows. Masks that keep the word with the largest main effect have higher similarities.

   **The parameter p decides which masks matter.** One random mask for each :math:`p` and the
   predictions of SigLIP 2 for 512 random masks. The dashed lines mark the empty and the full
   input.

.. _fixlip-sec-sampling:

Sampling Masks
~~~~~~~~~~~~~~

Evaluating all :math:`2^{n_\mathcal{I} + n_\mathcal{T}}` masks is infeasible. Instead, the
*model-agnostic estimator* samples masks :math:`M^{(1)}, \dots, M^{(m)}` independently from
:math:`\mathbb{P}_p` and evaluates the model on them. It then minimizes the following estimate of
:math:`\mathfrak{F}_p`, which is a least-squares problem.

.. math::
   :label: fixlip-model-agnostic

   \hat{\mathfrak{F}}^{(m)}_p(\nu,\hat\nu_\mathbf{e}) := \frac{1}{m} \sum_{\ell=1}^m
   \big(\nu(M^{(\ell)}) - \hat\nu_\mathbf{e}(M^{(\ell)})\big)^2

For :math:`p = 0.5`, this estimator reduces to Faith-Banzhaf :cite:p:`Tsai.2022`, a variant of
KernelSHAP :cite:p:`Lundberg.2017`. Each sampled mask costs one pass of both encoders.

**Cross-modal sampling.** FIxLIP exploits that the two encoders process their inputs independently.
Since every token is kept independently, it can sample :math:`m_\mathcal{I}` masks
:math:`M_\mathcal{I}^{(1)}, \dots, M_\mathcal{I}^{(m_\mathcal{I})}` of the image and
:math:`m_\mathcal{T}` masks :math:`M_\mathcal{T}^{(1)}, \dots, M_\mathcal{T}^{(m_\mathcal{T})}` of
the text separately. The *cross-modal estimator* minimizes the squared error over all combinations
of the two.

.. math::
   :label: fixlip-cross-modal

   \hat{\mathfrak{F}}_p^{(m_\mathcal{I},m_\mathcal{T})}(\nu,\hat\nu_\mathbf{e}) :=
   \frac{1}{m_\mathcal{I} \cdot m_\mathcal{T}}
   \sum_{\ell_\mathcal{I}=1}^{m_\mathcal{I}} \sum_{\ell_\mathcal{T}=1}^{m_\mathcal{T}}
   \big(\nu(M_\mathcal{I}^{(\ell_\mathcal{I})} \cup M_\mathcal{T}^{(\ell_\mathcal{T})})
   - \hat\nu_\mathbf{e}(M_\mathcal{I}^{(\ell_\mathcal{I})}
   \cup M_\mathcal{T}^{(\ell_\mathcal{T})})\big)^2

Each masked image and each masked text is encoded once, and the similarities of all
:math:`m_\mathcal{I} \cdot m_\mathcal{T}` combinations follow from cheap dot products. For example,
six masked images and six masked texts cost twelve passes of an encoder and give 36 predictions.
This is also a reason to prefer weighted Banzhaf interactions over Shapley interactions, for which
it is unknown if the weights factor into one distribution per modality. :cite:t:`Baniecki.2025`
show that :math:`\hat{\mathfrak{F}}_p^{(m_\mathcal{I},m_\mathcal{T})}` is an unbiased estimate of
:math:`\mathfrak{F}_p`. Its variance is at most of the same order as for the model-agnostic
estimator with :math:`m \approx m_\mathcal{I} \approx m_\mathcal{T}` masks, and at least the
variance of the model-agnostic estimator with :math:`m_\mathcal{I} \cdot m_\mathcal{T}` independent
masks.

Empirically, the cross-modal estimator speeds up model inference by over 20× and the whole
explanation pipeline by about 5×. :shapiq-figure:`paper-runtime` shows the runtime that the
authors measure for both estimators. In their experiments, FIxLIP with weighted Banzhaf
interactions uses the cross-modal estimator with a budget of :math:`2^{21}` masks, whereas FIxLIP
with Shapley interactions uses the model-agnostic estimator with :math:`2^{17}` masks.

.. shapiq-demo-figure:: fixlip paper-runtime
   :alt: A line chart of the runtime in seconds against the number of sampled masks, from 2 to the power of 15 to 2 to the power of 21, on a logarithmic scale. The cross-modal estimator is several times faster than the model-agnostic estimator for the whole explanation, and far faster in model inference.

   **Runtime of the two estimators.** Seconds on one GPU (A100) for SigLIP 2 (ViT-B/32). Redrawn
   from the data of Figure 5 of :cite:t:`Baniecki.2025`.

**Scaling up.** The number of interactions grows quadratically with the number of tokens, from 2,701
for the first example of the demo to 25,425 for a ViT-B/16 version of CLIP with 196 patches and 30
text tokens. For such models, :cite:t:`Baniecki.2025` first compute cheap first-order attributions
and then restrict the interactions to the most important tokens or to cross-modal pairs.

.. _fixlip-sec-demo-estimator:

The Estimator of the Demo
~~~~~~~~~~~~~~~~~~~~~~~~~

The demo computes FIxLIP for :math:`p = 0.5` with a different estimator, which needs fewer model
evaluations. :class:`~shapiq.approximator.ProxySHAP` :cite:p:`Thies.2026` fits a gradient-boosted
tree model to 10,000 sampled masks. It then computes the faithful Banzhaf interactions of this tree
model up to order 2 exactly. The explanation therefore targets the same index as the cross-modal
estimator, but it inherits the approximation error of the tree model.

.. _fixlip-sec-evaluation:

Evaluating the Explanation
--------------------------

:cite:t:`Baniecki.2025` derive three evaluation metrics that also apply to second-order
explanations. We illustrate each metric on inputs of the demo and show the results of the authors
next to it.

.. _fixlip-sec-correlation:

Faithfulness Correlation
~~~~~~~~~~~~~~~~~~~~~~~~

The first metric samples new masks and compares the predictions of the explanation with those of
the model. The *p-faithfulness correlation* is their Spearman rank correlation over :math:`m` masks
sampled from :math:`\mathbb{P}_p`.

.. math::
   :label: fixlip-correlation

   \mu_{\mathrm{corr}}(\mathbf{e};p) :=
   \underset{M^{(1)},\dots,M^{(m)} \overset{\mathrm{iid}}{\sim} \mathbb{P}_p}
   {\mathrm{correlation}}\big(\nu(M^{(i)}),\hat\nu_\mathbf{e}(M^{(i)})\big)

It relies on ranks, since methods like saliency maps do not estimate the similarity itself. For
game-theoretic explanations, the :math:`R^2` coefficient applies as well.
:shapiq-figure:`faithfulness` shows 512 held-out masks for the chosen example. For all five
examples, the second-order explanation predicts them better than its conversion to first order.

.. shapiq-demo-figure:: fixlip faithfulness
   :alt: Two scatter plots of the similarity predicted by the explanation against the similarity predicted by SigLIP 2 for 512 masks. The dots of the second-order explanation lie closer to the diagonal than those of its first-order conversion.

   **Predicting the model on held-out masks.** Each dot is one of 512 random masks with
   :math:`p = 0.5`. The constant of the explanation is refit on these masks.

The demo reports this :math:`R^2` for every explanation, with the constant :math:`\mathbf{e}_0`
refit on the held-out masks. It ranges from 0.74 to 0.96 across the five examples and the two
models. Note that the explanation is optimized for :math:`p = 0.5`, i.e. for masks that keep about
half of the tokens. On sparser masks sampled with :math:`p = 0.3`, its :math:`R^2` drops for every
example. A small clique is an even sparser mask, since it keeps only a few of the tokens. This is
why the value of a small clique differs from the prediction of the model for the clique alone.

In the evaluation of :cite:t:`Baniecki.2025`, FIxLIP with the matching :math:`p` is the most
faithful method for CLIP, as :shapiq-figure:`paper-faithfulness` shows. This evaluation uses the
1,000 image–text pairs from the MS COCO test set for which the model predicts the highest
similarity. GAME and Grad-ECLIP stay at a median correlation of about 0.5, and exCLIP has the
lowest median with 0.40. Shapley and Banzhaf values rank the masks well, with median correlations
of about 0.8. However, they predict the similarity itself poorly. Their median :math:`R^2` is below
0 for Shapley values and 0.58 for Banzhaf values, against 0.85 for FIxLIP with :math:`p = 0.5`.
FIxLIP with Shapley interactions (SI) lies in between with 0.73.

The authors also study the choice of :math:`p`. An explanation computed with :math:`p = 0.7`
predicts masks sampled with :math:`p = 0.3` much better than vice versa. The demo only shows
:math:`p = 0.5`.

.. shapiq-demo-figure:: fixlip paper-faithfulness
   :alt: Box plots of the p-faithfulness correlation for p equal to 0.7 and of R squared for p equal to 0.5. FIxLIP with the matching p has the highest median and the narrowest box in both. GAME, Grad-ECLIP and exCLIP have median correlations between 0.40 and 0.51.

   **Faithfulness of the methods.** The distribution of both metrics over 1,000 image–text pairs
   for CLIP (ViT-B/32). Boxes cover the middle half of the pairs and gaps mark the medians. FIxLIP
   is shown with Shapley interactions (SI) and with weighted Banzhaf interactions for three values
   of :math:`p`. Values below 0 are cut off at the axis. Redrawn from the data of Figure 4 of
   :cite:t:`Baniecki.2025`.

.. _fixlip-sec-aid:

Insertion and Deletion
~~~~~~~~~~~~~~~~~~~~~~

The second metric evaluates whether the explanation finds the subsets of tokens that matter. For
every size :math:`k`, it selects the subsets with the highest and the lowest value under the
explanation, i.e. :math:`M_{\mathbf{e},\max,k} := \operatorname*{arg\,max}_{M: \vert M \vert = k}
\hat\nu_\mathbf{e}(M)` and :math:`M_{\mathbf{e},\min,k} := \operatorname*{arg\,min}_{M: \vert M
\vert = k} \hat\nu_\mathbf{e}(M)`, and evaluates the model on both. The *insertion curve*
:math:`\mathcal{C}_{\mathrm{insert}}(\mathbf{e}) := \{(k, \nu(M_{\mathbf{e},\max,k}))\}_k` inserts
:math:`k` tokens into the empty input. The *deletion curve*
:math:`\mathcal{C}_{\mathrm{delete}}(\mathbf{e}) := \{(k, \nu(M_{\mathbf{e},\min,n_\mathcal{I} +
n_\mathcal{T} - k}))\}_k` deletes :math:`k` tokens from the full input, so that the tokens with the
lowest value remain. The *AID* score is the area between the two curves.

.. math::
   :label: fixlip-aid

   \mu_{\mathrm{AID}}(\mathbf{e}) := \sum_{k=1}^{n_\mathcal{I}+n_\mathcal{T}}
   \big(\nu(M_{\mathbf{e},\max,k}) - \nu(M_{\mathbf{e},\min,k})\big)

It generalizes insertion and deletion curves :cite:p:`Hama.2023` to interactions. For first-order
attributions, the subsets follow from ranking the tokens. The insertion curve then adds the tokens
with the highest attribution first, and the deletion curve removes them first. With interactions,
the best subset of size :math:`k + 1` does not have to contain the best subset of size :math:`k`.
The subsets are therefore searched for every size.

:shapiq-figure:`curves` shows both curves for the chosen example, on the cosine scale of the demo.
Its horizontal axis counts the tokens that are kept. The insertion curve is thus read from left to
right, and the deletion curve from right to left. The slider sets the number of tokens kept and
shows the subsets of both curves at this point. The figure reports :eq:`fixlip-aid` as an average
over all sizes, normalized by the difference between the predictions for the full and the empty
input. For all five examples, the second-order explanation reaches a larger AID than its conversion
to first order. It also finds five tokens whose removal lowers the similarity below the prediction
for an empty input. Moreover, it finds subsets with a higher similarity than the full input.

.. shapiq-demo-figure:: fixlip curves
   :alt: Insertion and deletion curves for the example over the number of tokens kept. The curves of the second-order explanation enclose a larger area than those of its first-order conversion. A slider sets the number of tokens kept. Below the chart, the subset of the insertion curve at this point, with a high similarity, and the subset that the deletion curve leaves, with a low similarity.

   **Insertion and deletion for the example.** The prediction of SigLIP 2 over the number of tokens
   that are kept. Solid lines are insertion curves, which add the tokens valued the most from left
   to right. Dotted lines are deletion curves, which remove them from right to left. The slider
   sets the number of tokens kept. The two masks below the chart are the subsets of the
   second-order explanation at this point, which the dots mark on its curves.

:cite:t:`Baniecki.2025` report the same behavior on 1,000 pairs from MS COCO, as
:shapiq-figure:`paper-curves` shows. Its upper plot overlays the methods that are ticked, so that
they can be compared directly. FIxLIP finds tokens whose removal increases the prediction, contrary
to gradient-based methods. The curves also depend on :math:`p`. In these results, :math:`p = 0.7`
is the most faithful while up to half of the tokens are deleted, and :math:`p = 0.3` when more than
half are deleted.

.. shapiq-demo-figure:: fixlip paper-curves
   :alt: A plot that overlays the insertion and deletion curves of the methods that are ticked. Below it, seven small plots, one for each method, with a tick box, its insertion curve, its deletion curve and the area between them. The three variants of FIxLIP enclose the largest areas, between 0.94 and 1.02. Shapley values reach 0.77, Grad-ECLIP 0.37, GAME 0.33 and exCLIP 0.25.

   **Insertion and deletion of the methods.** The normalized prediction of CLIP (ViT-B/32) from no
   tokens on the left to all tokens on the right, on average over 1,000 image–text pairs. A value
   of 0 is the prediction for the empty input and 1 is the prediction for the full input. The upper
   plot overlays the methods that are ticked in the small plots below it. The AID is given with
   its standard deviation. Redrawn from the data of Figure 3 of :cite:t:`Baniecki.2025`, who draw
   the curves in the other direction.

.. _fixlip-sec-pointing:

Pointing Game
~~~~~~~~~~~~~

The third metric is a sanity check with a designed ground truth. Four images from ImageNet-1k
:cite:p:`Deng.2009` are combined into one, and the text names up to four of their classes. The
*pointing game recognition* (PGR) measures which share of the absolute cross-modal interactions is
correct.

.. math::
   :label: fixlip-pgr

   \mu_{\mathrm{PGR}}(\mathbf{e};N_\mathcal{T}) := \frac
   {\sum_{k \in N_\mathcal{T}} \big(\Vert \mathbf{e}_{\mathrm{in},k,>0} \Vert_1
   + \Vert \mathbf{e}_{\mathrm{out},k,<0} \Vert_1\big)}
   {\sum_{k \in N_\mathcal{T}} \big(\Vert \mathbf{e}_{\mathrm{in},k} \Vert_1
   + \Vert \mathbf{e}_{\mathrm{out},k} \Vert_1\big)} \in [0, 1]

Here, :math:`\mathbf{e}_{\mathrm{in},k}` and :math:`\mathbf{e}_{\mathrm{out},k}` denote the
interactions that belong and do not belong to object :math:`k`, i.e. a class name with the patches
of its image. The indices :math:`>0` and :math:`<0` select the positive and the negative
interactions, and :math:`\Vert \mathbf{e} \Vert_1 := \sum_{i,j} \vert \mathbf{e}_{\{i,j\}} \vert`.
In other words, a positive interaction is correct if it links a class name with the patches of its
object. A negative interaction is correct if it links a class name with the patches of another
object. A random baseline scores 0.25. The left panel of :shapiq-figure:`pointing` plays this game
for SigLIP 2 on four photos of the demo, explained as in the demo. For this input, 0.92 of the
absolute interactions are correct, which is in line with the 0.89 that :cite:t:`Baniecki.2025`
report for SigLIP 2 with four objects.

.. shapiq-demo-figure:: fixlip pointing
   :alt: Left, four photos of the demo in one image, an elephant, a giraffe, a pineapple and a clock tower, with the text 'elephant giraffe pineapple clock'. The patches are tinted by their interaction with one of the four names, which can be chosen. Right, the reported pointing game recognition of CLIP for one to four objects. First-order methods fall from between 0.61 and 0.70 for one object to between 0.28 and 0.37 for four objects. exCLIP reaches 0.92 and FIxLIP 0.83 for four objects.

   **The pointing game.** The left panel shows a pointing game on four photos of the demo. The
   patches are tinted by their interaction with the framed name in the explanation of SigLIP 2.
   **Red** raises the similarity and **blue** lowers it. The right panel shows Table 1 of
   :cite:t:`Baniecki.2025` for CLIP (ViT-B/32), where FIxLIP uses :math:`p = 0.5`. Each score is
   an average over 500 inputs.

The right panel shows the results of the authors for CLIP. First-order methods fail to distinguish
between several objects and approach the random baseline. With four objects, Grad-ECLIP and GAME
reach 0.28 and Shapley values reach 0.37. In contrast, the two second-order methods pass the check
with 0.83 for FIxLIP and 0.92 for exCLIP. exCLIP scores higher for two or more objects, which the
authors attribute to its specialization in cross-modal interactions.

.. _fixlip-sec-models:

Comparing CLIP and SigLIP 2
---------------------------

FIxLIP can compare different encoders, since it treats them as black boxes. In the demo, switch
between *SigLIP 2* and *CLIP* for the same image–text pair. The cliques of the two models often
differ, as :shapiq-figure:`models` shows for the chosen example.

.. shapiq-demo-figure:: fixlip models
   :alt: The cliques of six tokens for CLIP and for SigLIP 2 on the same example. The framed image patches and words of the two cliques differ.

   **The same example explained for two models.** The cliques of six tokens for CLIP and SigLIP 2.
   The width of a line is relative to the strongest interaction of each clique. The two models
   have different tokenizers and image resolutions.

:cite:t:`Baniecki.2025` also compare the two models quantitatively with the pointing game. In this
separate experiment on a subset of the games, the explanations of SigLIP 2 (ViT-B/32) reach a
recognition of about 0.90 for one to four objects, while the explanations of CLIP reach 0.81 to
0.85. Thus, SigLIP 2 can be explained more faithfully with cross-modal interactions than CLIP. In
general, the authors find that smaller models such as ViT-B/32 can be explained more faithfully
than larger ones.

.. _fixlip-sec-limitations:

Limitations
-----------

- **Second order only.** Interactions between three or more tokens are not explained. Main effects
  and pairs absorb them as far as they can. For the example with the elephant, the main effect of
  *elephant* is +0.064, while the word alone changes the prediction by only +0.002.
  :cite:t:`Baniecki.2025` themselves point to a possible third-order interaction in one of their
  examples.
- **Greedy cliques.** The search for cliques is greedy and is not guaranteed to find the best
  subset.
- **Masked inputs.** The explanation depends on the masking baseline, and sparse masks remain
  out-of-distribution inputs for the model. For every example of the demo, the :math:`R^2` drops on
  masks sampled with :math:`p = 0.3`.
- **Many tokens.** In the experiments of the authors, the faithfulness of the explanations drops
  when the number of input tokens increases.
- **Approximation.** The explanations of the demo are approximated from 10,000 masks for each
  example. Their :math:`R^2` on held-out masks, between 0.74 and 0.96, shows how far each of them
  can be trusted.
- **Human evaluation.** How useful the visualizations are for humans is left to future user
  studies.

.. _fixlip-sec-code:

Explain It Yourself
-------------------

The FIxLIP game can be implemented with ``shapiq`` and ``transformers`` in about 40 lines. The
following game masks the image patches and text tokens of CLIP as described in :ref:`the section on
the game <fixlip-sec-game>`.

.. code-block:: python

   import numpy as np
   import torch
   from PIL import Image
   from transformers import CLIPModel, CLIPProcessor

   import shapiq


   class FIxLIPGame(shapiq.Game):
       """The similarity that CLIP predicts for a masked image-text pair.

       The players are the 7 x 7 patches of the image and the tokens of the text.
       """

       def __init__(self, image: Image.Image, text: str) -> None:
           checkpoint = "openai/clip-vit-base-patch32"
           self.model = CLIPModel.from_pretrained(checkpoint).eval()
           processor = CLIPProcessor.from_pretrained(checkpoint)
           inputs = processor(images=image, text=text, return_tensors="pt")
           self.pixels = inputs["pixel_values"]  # normalized, so 0 is the mean color
           self.tokens = inputs["input_ids"]  # between a start and an end token
           tokens = processor.tokenizer.convert_ids_to_tokens(self.tokens[0, 1:-1])
           self.words = [token.removesuffix("</w>") for token in tokens]
           self.n_patches = 7 * 7
           n_players = self.n_patches + len(self.words)
           # the similarity of the empty input is the baseline of all values
           empty = self.value_function(np.zeros((1, n_players), dtype=bool))[0]
           super().__init__(n_players, normalize=True, normalization_value=empty)

       @torch.no_grad()
       def value_function(self, coalitions: np.ndarray) -> np.ndarray:
           values = []
           for batch in np.array_split(coalitions, max(1, len(coalitions) // 64)):
               patches = torch.as_tensor(batch[:, : self.n_patches], dtype=torch.float32)
               # every patch covers 32 x 32 pixels and masked patches get the mean color
               masks = torch.kron(patches.reshape(-1, 1, 7, 7), torch.ones(32, 32))
               # masked tokens are excluded through the attention mask
               attention = torch.ones(len(batch), self.tokens.shape[1], dtype=torch.long)
               attention[:, 1:-1] = torch.as_tensor(batch[:, self.n_patches :])
               outputs = self.model(
                   pixel_values=self.pixels * masks,
                   input_ids=self.tokens.repeat(len(batch), 1),
                   attention_mask=attention,
               )
               # embeddings are normalized, so their dot product is the cosine similarity
               similarity = (outputs.image_embeds * outputs.text_embeds).sum(dim=1)
               values.append(similarity.numpy())
           return np.concatenate(values)

:class:`~shapiq.approximator.ProxySHAP` then approximates the faithful Banzhaf interaction index up
to order 2, i.e. FIxLIP with :math:`p = 0.5`. The following code explains the first example of the
demo, which takes a minute or two on a laptop.

.. code-block:: python

   import io
   import math
   import urllib.request

   from xgboost import XGBRegressor

   # the image of the first example is a COCO photo by Wilson Kwo under CC BY-NC-SA 2.0
   url = "https://s3.amazonaws.com/images.cocodataset.org/train2017/000000046761.jpg"
   image = Image.open(io.BytesIO(urllib.request.urlopen(url).read()))
   image = image.crop((76, 0, 503, 427))  # the square that the demo shows
   game = FIxLIPGame(image, "a small elephant standing next to a white bird.")
   n = game.n_players  # 49 patches and 10 tokens

   approximator = shapiq.approximator.ProxySHAP(
       n=n,
       index="FBII",
       max_order=2,
       proxy_model=XGBRegressor(
           n_estimators=2000, max_depth=3, learning_rate=0.05, reg_lambda=5.0
       ),
       adjustment=False,
       # every token is kept with probability 0.5
       sampling_weights=np.array([math.comb(n, k) for k in range(n + 1)], dtype=float),
       random_state=0,
   )
   interactions = approximator.approximate(budget=2_000, game=game)  # the demo uses 10,000

   # the strongest interactions between an image patch and a word
   pairs = {
       players: value
       for players, value in interactions.dict_values.items()
       if len(players) == 2 and players[0] < game.n_patches <= players[1]
   }
   for (patch, token), value in sorted(pairs.items(), key=lambda pair: -abs(pair[1]))[:5]:
       word = game.words[token - game.n_patches]
       row, column = patch // 7 + 1, patch % 7 + 1
       print(f"patch ({row}, {column}) and {word!r}: {value:+.4f}")

.. code-block:: text

   patch (5, 1) and 'white': +0.0069
   patch (2, 7) and 'elephant': +0.0049
   patch (3, 5) and 'elephant': +0.0045
   patch (5, 7) and 'elephant': +0.0040
   patch (4, 6) and 'elephant': +0.0040

For CLIP, the patch of the egret in row 5 and column 1 goes with *white*, and the patches of the
elephant go with *elephant*. Try another text, or evaluate the game on your own masks with
``game.value_function(coalitions)``.

The scripts that computed the demo and the data of the figures on this page are in
`docs/source/demos/fixlip/compute
<https://github.com/mmschlk/shapiq/tree/main/docs/source/demos/fixlip/compute>`__, with a README on
how to run them. The original code of FIxLIP is available at `github.com/hbaniecki/fixlip
<https://github.com/hbaniecki/fixlip>`__. The images are from MS COCO :cite:p:`Lin.2014`, and the
demo credits each one with its author and license.
