Explaining Graph Neural Networks with Exact Interactions
========================================================

.. The write-up of this page (what you see, what it shows, how it was computed) follows once the
   demos are settled. The last one is in the repository's history: this file at commit 20c526c.
   Until then, the figure of the receptive fields stands for it, below the first screen.

.. container:: graphshapiq-opening

   .. shapiq-demo:: graphshapiq-explorer
      :fallback: images/hero_graphshapiq_explorer_demo.png
      :alt: The molecule 1,3-dinitrobenzene explained by a GNN for the question whether it is mutagenic: each nitrogen and its two oxygens are joined by red bands, the pairs of atoms that together raise the odds the most.

   .. rst-class:: graphshapiq-more

   :ref:`Read the write-up <graphshapiq-sec-summary>`

----

.. role:: graphshapiq-red
.. role:: graphshapiq-blue

.. _graphshapiq-sec-summary:

.. rst-class:: graphshapiq-summary graphshapiq-writeup

Summary
-------

GraphSHAP-IQ :cite:p:`Muschalik.2025` explains the prediction of a graph neural network (GNN) for a
single graph. It decomposes the prediction into Shapley interactions (SIs), which assign a
contribution to every node and to every group of nodes up to an *explanation order*. The SIs range
from Shapley values (SVs) for single nodes to Möbius interactions (MIs) for all groups. GraphSHAP-IQ
computes them exactly, without evaluating the GNN on all :math:`2^n` maskings of the :math:`n`
nodes.

This page explains GraphSHAP-IQ step by step on the seven molecules of the demo. It motivates why
predictions for graphs call for interactions, defines the game and the explanation, shows how to
interpret it, and describes how it is computed. **For a message passing GNN with a linear global
pooling and a linear output layer, all MIs outside the receptive fields are zero, so exact SIs of
any order need the GNN only on the maskings inside the receptive fields.** For indigo with 20 atoms
and a GNN with three layers, these are 37,888 instead of 1,048,576 model calls. Finally, the page
states where this guarantee ends and shows how to explain a molecule with ``shapiq``.

.. _graphshapiq-sec-introduction:

Introduction
------------

Graph-structured data appears in many domains, such as molecular chemistry :cite:p:`Gilmer.2017`,
water distribution networks, sociology and physics. Graph neural networks emerged as the leading
family of architectures that exploit the topology of such data. A major drawback of GNNs is the
opacity of their predictive mechanism, which they share with most deep learning architectures.
Reliable explanations of their predictions are crucial when a decision has significant consequences
or leads to a new discovery, for example when a GNN flags a molecule as mutagenic.

The Shapley value :cite:p:`Shapley.1951` is a prominent concept to assign contributions to the
entities of a black box model :cite:p:`Lundberg.2017,Covert.2021`. However, the SV gives no insight
into joint contributions of entities, known as interactions. Shapley interactions
:cite:p:`Grabisch.1999,Bordt.2023` extend the SV to groups of entities up to an explanation order.
SVs, SIs and MIs are limited by exponential complexity. For a molecule with 30 atoms, an exact
computation requires :math:`2^{30} \approx 10^9` model calls. Existing explanations of GNNs
therefore approximate the SV or apply it as a heuristic on subgraphs. GraphSHAP-IQ
:cite:p:`Muschalik.2025` instead includes knowledge about the GNN and computes exact SIs of any
order. It is a model-specific computation of SIs for GNNs, akin to TreeSHAP
:cite:p:`Lundberg.2020,Muschalik.2024a` for tree-based models.

.. _graphshapiq-sec-background:

Background
----------

This section describes GNNs for graph prediction, argues why their predictions call for
interactions, and reviews how related work explains them.

.. _graphshapiq-sec-gnn:

Graph Neural Networks in a Nutshell
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A graph :math:`g = (V, E, \mathbf{X})` consists of nodes :math:`V = \{v_1, \dots, v_n\}`, edges
:math:`E \subset V \times V` and node features :math:`\mathbf{X} = [\mathbf{x}_1, \dots,
\mathbf{x}_n]^t \in \mathbb{R}^{n \times d_0}`. A *message passing* GNN iteratively aggregates the
node features of a node :math:`v` within its neighborhood :math:`\mathcal{N}(v) := \{u \in V \mid
e_{uv} \in E\}`, as illustrated in :shapiq-figure:`pipeline`. In each iteration :math:`k \in \{1,
\dots, \ell\}`, the hidden node features are computed node-wise by

.. math::
   :label: graphshapiq-message-passing

   \mathbf{h}_i^{(0)} := \mathbf{x}_i, \quad
   \mathbf{h}_i^{(k)} := \rho^{(k)}\big(\mathbf{h}_i^{(k-1)},
   \psi(\{\{\varphi^{(k)}(\mathbf{h}_j^{(k-1)}) \mid u_j \in \mathcal{N}(v_i)\}\})\big)

Here, :math:`\{\{\cdot\}\}` indicates a multiset, :math:`\rho^{(k)}` and :math:`\varphi^{(k)}` are
arbitrary functions, and :math:`\psi` is a permutation-invariant function. In words, every layer
updates the features of a node with the features of its neighbors. After :math:`\ell` layers, the
node embedding is :math:`f_i(\mathbf{X}) := \mathbf{h}_i^{(\ell)}`. For *graph prediction*, a
permutation-invariant pooling function :math:`\Psi` aggregates the node embeddings, and an output
layer :math:`\sigma` maps the result to the prediction.

.. math::
   :label: graphshapiq-gnn

   f_g(\mathbf{X}) := \sigma\big(\Psi(\{\{f_i(\mathbf{X}) \mid v_i \in V\}\})\big)

Graph Convolutional Networks :cite:p:`Kipf.2017`, Graph Attention Networks
:cite:p:`Velickovic.2018` and Graph Isomorphism Networks (GIN) :cite:p:`Xu.2019` are instances of
:eq:`graphshapiq-message-passing`, which differ in how a layer aggregates the neighbors. The demo
uses one GIN with :math:`\ell = 3` layers for each of three questions. Two GINs are trained on
B-XAIC :cite:p:`Proszewska.2025` and predict whether a molecule contains an indole and whether it
is a pan-assay interference compound (PAINS) :cite:p:`Baell.2010`. The third is trained on
Mutagenicity :cite:p:`Kazius.2005,Morris.2020` and predicts whether a molecule is mutagenic. The
nodes are the atoms without hydrogens, the edges are the bonds, and the node features encode the
element of an atom. All three GINs use a sum as :math:`\Psi` and a dense layer as :math:`\sigma`.

.. shapiq-demo-figure:: graphshapiq-explorer pipeline
   :alt: A molecule passes through three layers of message passing, in which the receptive field of one atom grows by one bond per layer. Sum pooling and a linear output layer turn the node embeddings into the log-odds of the GNN. The molecule is shown again with the term that each atom adds to the log-odds.

   **From a molecule to a prediction.** Three layers of message passing, sum pooling and a linear
   output layer yield the log-odds of the GNN for the question of the molecule. Since pooling and
   output layer are linear, the log-odds are a sum of one term per atom and a bias.

The value explained on this page is the log-odds of a GNN, i.e. its logit for yes minus its logit
for no. For 1,3-dinitrobenzene, the GNN for mutagenicity predicts log-odds of 3.98, which is a
probability of 98.2%. Since :math:`\Psi` and :math:`\sigma` are linear, these log-odds are a sum
of one term per atom. Each nitrogen atom adds +0.59 and each oxygen atom adds +0.50. However, these
terms do not explain the prediction. The embedding of an atom is computed from all atoms within
three bonds, so its term cannot be attributed to the atom itself.

.. _graphshapiq-sec-interactions:

Substructures Call for Interactions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Properties of a graph are often carried by substructures and not by single nodes. In chemistry, a
nitro group signals mutagenicity :cite:p:`Kazius.2005`, and an indole is a system of two fused
rings. What one atom contributes to the prediction then depends on its neighbors.
:shapiq-figure:`nitro` measures this dependence on the model. It adds one atom, another atom, or
both to maskings of all other atoms and compares the changes of the log-odds.

.. shapiq-demo-figure:: graphshapiq-explorer nitro
   :alt: Two bar charts on one axis for the two atoms with the strongest pairwise interaction of the molecule. The first chart has all other atoms masked. Its bars show the change of the log-odds when the first atom is added, when the second atom is added and when both are added, the interaction, which is the change for both minus the two single changes, and the Möbius interaction of the pair from GraphSHAP-IQ, which equals this interaction. The second chart shows the same changes on average over 256 random maskings of the other atoms, the average interaction, and the 2-SII of the pair from GraphSHAP-IQ, which the average interaction estimates.

   **An interaction measured on the model.** The change of the log-odds when the two atoms with
   the strongest 2-SII are added, alone and together, with all other atoms masked and on average
   over 256 random maskings of the other atoms. The interaction is the change for both minus the
   changes for each atom alone. With all other atoms masked, it is the MI of the pair. The random
   maskings first draw the number of unmasked atoms uniformly and then the atoms, so their average
   interaction estimates the 2-SII of the pair.

An interaction is what two nodes add together beyond the sum of what each of them adds alone. For
1,3-dinitrobenzene, the two atoms of the figure are a nitrogen and an oxygen of a nitro group. With
all other atoms masked, the nitrogen atom alone changes the log-odds by +0.03 and the oxygen atom
alone by −0.97. Both together change them by +1.85. The pair thus adds 2.79 beyond the sum of the
two single changes. On average over 256 random maskings of the other atoms, the pair adds 1.86
beyond the sum, with a standard error of 0.04. The interaction of two nodes thus depends on the
masking of the others, and for other molecules it even changes its sign. An explanation has to
summarize it over all maskings.

A score per node cannot express such a surplus. It can only split the surplus among the nodes,
which hides that they act together. For 1,3-dinitrobenzene, the SVs assign +1.86 to each nitrogen
atom and only +0.07 to each oxygen atom. Yet masking the four oxygen atoms lowers the log-odds from
3.98 to 0.93. The oxygen atoms matter, but only together with their nitrogen atom. An explanation
of a graph prediction thus has to include interactions between nodes.

.. _graphshapiq-sec-related:

Related Work and Its Shortcomings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most local explanations of GNNs are perturbation-based and output isolated subgraphs. This
includes GNNExplainer :cite:p:`Ying.2019`, PGExplainer :cite:p:`Luo.2020` and SubgraphX
:cite:p:`Yuan.2021`, whereas :cite:t:`Pope.2019` introduce node attributions. Several methods use
the SV. GraphSVX :cite:p:`Duval.2021` approximates SVs of single nodes, and for graph prediction
this approximation is model-agnostic. SubgraphX, GStarX :cite:p:`Zhang.2022` and SAME
:cite:p:`Ye.2023` use the SV or a related value as a heuristic score to assess the quality of
subgraphs. :cite:t:`Bui.2024` approximate pairwise SIs with model-agnostic sampling to improve the
detection of subgraphs. None of these methods decomposes the prediction exactly into contributions
of nodes and groups of nodes.

A second line of work computes or approximates SVs and SIs for arbitrary models. Exact values
require all :math:`2^n` maskings, which is infeasible beyond a few dozen nodes. Model-agnostic
approximators such as KernelSHAP :cite:p:`Lundberg.2017`, SHAP-IQ :cite:p:`Fumagalli.2023`,
SVARM-IQ :cite:p:`Kolpaczki.2024a` and KernelSHAP-IQ :cite:p:`Fumagalli.2024` sample maskings and
return estimates. L-Shapley :cite:p:`Chen.2019` exploits the structure of the data and restricts
the SV of a node to a neighborhood. It is exact for a single node embedding, but not for the graph
prediction, where interactions that involve a node also appear in other neighborhoods
:cite:p:`Muschalik.2025`. Exact computation has so far been feasible only for models with a known
structure, such as tree ensembles :cite:p:`Lundberg.2020,Muschalik.2024a`.

.. _graphshapiq-sec-method:

Shapley Interactions for Graph Neural Networks
----------------------------------------------

GraphSHAP-IQ explains a graph prediction with SIs of the nodes. This requires a cooperative game on
the nodes and an explanation that summarizes the game at a chosen explanation order.

.. _graphshapiq-sec-game:

The GNN-Induced Graph Game
~~~~~~~~~~~~~~~~~~~~~~~~~~

Formally, a cooperative game :math:`\nu: \mathcal{P}(N) \to \mathbb{R}` assigns a payout to every
group of players in the power set :math:`\mathcal{P}(N)` of the players :math:`N = \{1, \dots,
n\}`. GraphSHAP-IQ views the nodes as the players and the prediction of the GNN as the payout. A
subset :math:`T \subseteq N` selects the nodes that keep their node features. All other nodes are
masked with a baseline, as :shapiq-figure:`maskings` shows.

.. shapiq-definition:: GNN-induced Graph and Node Game
   :name: graphshapiq-def-game

   For a graph :math:`g = (V, E, \mathbf{X})` and a GNN :math:`f_g`, let :math:`N := \{i : v_i \in
   V\}` be the node indices. The *graph game* :math:`\nu_g: \mathcal{P}(N) \to \mathbb{R}` is
   defined as

   .. math::

      \begin{gathered}
      \nu_g(T) := f_{g,\hat{y}}(\mathbf{X}^{(T)}) \text{ with }
      \mathbf{X}^{(T)} := (\mathbf{x}_1^{(T)}, \dots, \mathbf{x}_n^{(T)})^t
      \in \mathbb{R}^{n \times d_0} \\
      \text{and } \mathbf{x}_i^{(T)} := \begin{cases}
      \mathbf{x}_i & \text{if } i \in T, \\
      \mathbf{b} & \text{if } i \notin T,
      \end{cases}
      \end{gathered}

   with :math:`i \in N` and baseline :math:`\mathbf{b} \in \mathbb{R}^{d_0}`. In graph regression
   :math:`f_{g,\hat{y}} \equiv f_g` and for graph classification :math:`f_{g,\hat{y}}` is the
   component of the predicted class :math:`\hat{y}` of :math:`f_g`. The (multi-dimensional) *node
   game* :math:`\nu_i: \mathcal{P}(N) \to \mathbb{R}^{d_\ell}` is defined as :math:`\nu_i(T) :=
   f_i(\mathbf{X}^{(T)})` for :math:`i \in N` and each node :math:`v_i \in V`.

The GNNs of the demo have two outputs, one for yes and one for no. In place of the component of
the predicted class :math:`f_{g,\hat{y}}`, this page explains their log-odds :math:`f_{g,\text{yes}}
- f_{g,\text{no}}`. The log-odds are still linear in the pooled node embeddings, so everything that
follows holds for them. The explanation then describes the log-odds also for a molecule that the
GNN predicts as no.

Masking replaces node features and keeps all edges, so the topology of the graph is preserved.
Computing SVs on such a game is known as Baseline Shapley :cite:p:`Sundararajan.2020a`. As a
baseline :math:`\mathbf{b}`, GraphSHAP-IQ proposes the average of each node feature over the whole
graph. The demo instead uses the average node features of the training set of each GNN, so that all
molecules share one baseline.

.. shapiq-demo-figure:: graphshapiq-explorer maskings
   :alt: Four copies of the molecule with the probability and the log-odds that the GNN predicts for each. In the first, no atom is masked. In the second, all atoms are masked. In the third, a meaningful substructure is masked. The fourth is a random masking. Masked atoms are gray.

   **Maskings of the graph game.** The prediction of the GNN for the molecule with all atoms, with
   no atoms, with a substructure masked and for a random masking. Masked atoms are gray and receive
   the baseline.

A graph with :math:`n` nodes has :math:`2^n` maskings. For 1,3-dinitrobenzene, these are
:math:`2^{12} = 4{,}096`. With all atoms, the GNN predicts log-odds of 3.98. With no atoms, it
predicts 0.18, which is a probability of 54.4%. With both nitro groups masked, the log-odds drop to
−0.28, i.e. 43.0%.

The graph and the node game are directly linked by :eq:`graphshapiq-gnn` as

.. math::
   :label: graphshapiq-link

   \begin{gathered}
   \nu_g(T) = f_{g,\hat{y}}(\mathbf{X}^{(T)})
   = \sigma_{\hat{y}}\big(\Psi(\{\{f_i(\mathbf{X}^{(T)}) \mid v_i \in V\}\})\big) \\
   = \sigma_{\hat{y}}\big(\Psi(\{\{\nu_i(T) \mid i \in N\}\})\big)
   \end{gathered}

where :math:`\sigma_{\hat{y}}` outputs the component of :math:`\sigma` for the predicted class
:math:`\hat{y}`. For the log-odds, it outputs the difference of the two components of
:math:`\sigma`. In words, the graph game pools the node games of all nodes. This link is what
GraphSHAP-IQ later exploits.

.. _graphshapiq-sec-orders:

From Shapley Values to Möbius Interactions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A game with :math:`2^n` values cannot be inspected directly, so an explanation has to summarize it.
The most detailed summary is given by the Möbius interactions :cite:p:`Harsanyi.1963,Rota.1964`.
The MI :math:`m: \mathcal{P}(N) \to \mathbb{R}` of a set :math:`S \subseteq N` is

.. math::
   :label: graphshapiq-mi

   \begin{gathered}
   m(S) := \sum_{T \subseteq S} (-1)^{\vert S \vert - \vert T \vert} \nu(T) \\
   \text{and they recover} \quad
   \nu(T) = \sum_{S \subseteq T} m(S) \text{ for all } S, T \subseteq N
   \end{gathered}

The MI of :math:`S` can be interpreted as the *pure additive contribution* that is exclusively
achieved by a coalition of all players in :math:`S` and cannot be attributed to any subgroup of
:math:`S`. Every game value is the sum of the MIs of the players that are present. For
1,3-dinitrobenzene, the MI of a nitrogen atom with one of its oxygen atoms is the surplus of +2.79
from above.

The least complex summary is given by the Shapley values, a weighted average over marginal
contributions :math:`\Delta_i(T) := \nu(T \cup i) - \nu(T)`.

.. math::
   :label: graphshapiq-sv

   \phi^{\text{SV}}(i) := \sum_{T \subseteq N \setminus i}
   \frac{1}{n \cdot \binom{n-1}{\vert T \vert}} \Delta_i(T)

The SVs satisfy *efficiency*, i.e. they sum to the difference between the prediction
:math:`\nu(N)` and the prediction :math:`\nu(\emptyset)` without any node features. The game may
be normalized such that :math:`\nu(\emptyset) = 0`, which does not affect the SVs. The rest of this
section assumes this normalization. In terms of MIs, the SV distributes every MI equally among its
players.

.. math::
   :label: graphshapiq-conversion

   \phi^{\text{SV}}(i) = \sum_{\tilde{S} \subseteq N: i \in \tilde{S}}
   \frac{m(\tilde{S})}{\vert \tilde{S} \vert}

Shapley interactions fill the range between these two extremes. Given an explanation order
:math:`k = 1, \dots, n`, the SIs :math:`\Phi_k: \mathcal{P}_k(N) \to \mathbb{R}` assign
contributions to all subsets of :math:`N` up to size :math:`k`, summarized in
:math:`\mathcal{P}_k(N)`. The SIs decompose the prediction with :math:`\nu(N) = \sum_{S \subseteq
N, \vert S \vert \leq k} \Phi_k(S)`. SIs are based on discrete derivatives
:cite:p:`Grabisch.1999`, which extend the marginal contribution to groups. For two players
:math:`i, j \in N` and a subset :math:`T \subseteq N \setminus ij`, the discrete derivative
:math:`\Delta_{ij}(T) := \nu(T \cup ij) - \nu(T) - \Delta_i(T) - \Delta_j(T)` is the quantity that
:shapiq-figure:`nitro` measures. A positive value of the discrete derivative :math:`\Delta_S(T)`
indicates synergistic effects, and a negative value indicates redundancy. The Shapley interaction
index (SII) summarizes the discrete derivatives of a set :math:`S` in the presence of all subsets
:math:`T`.

.. math::
   :label: graphshapiq-sii

   \begin{gathered}
   \phi^{\text{SII}}(S) = \sum_{T \subseteq N \setminus S}
   \frac{1}{(n - \vert S \vert + 1) \cdot \binom{n - \vert S \vert}{\vert T \vert}} \Delta_S(T) \\
   \text{with} \quad
   \Delta_S(T) := \sum_{L \subseteq S} (-1)^{\vert S \vert - \vert L \vert} \nu(T \cup L)
   \end{gathered}

The :math:`k`-Shapley values (:math:`k`-SII) :cite:p:`Lundberg.2020,Bordt.2023` construct SIs
recursively based on the SII, such that the interactions of SII and :math:`k`-SII for the highest
order coincide. For :math:`k = 1`, the
:math:`k`-SII is the SV, and for :math:`k = n`, it is the MI. All explanations on this page are
:math:`k`-SIIs. Like the SV in :eq:`graphshapiq-conversion`, every SI is a weighted sum of MIs, so
SIs of any order follow from the MIs. GraphSHAP-IQ visualizes the SIs on the graph itself.

.. shapiq-definition:: SI-Graph
   :name: graphshapiq-def-si-graph

   The SI-Graph is an undirected hypergraph :math:`g_k^{\text{SI}} := (N, \mathcal{P}_k(N),
   \Phi_k)` with node attributes :math:`\Phi_k(i)` for :math:`i \in N` and hyperedge attributes
   :math:`\Phi_k(S)` for :math:`2 \leq \vert S \vert \leq k`.

The simplest SI-Graph displays the SVs as node attributes, whereas the most complex SI-Graph
displays the MIs as node and hyperedge attributes. The explanation order :math:`k` adjusts the
complexity in between, as :shapiq-figure:`order` shows. For every order, the sum of all
contributions in the SI-Graph yields the prediction of the normalized game, which is the log-odds
minus the prediction with no atoms. For 1,3-dinitrobenzene, this sum is 3.8052, i.e. the log-odds
of 3.9826 minus 0.1774. It is distributed among 12 SVs, among 78 contributions of order 2, or among
4,095 MIs.

.. shapiq-demo-figure:: graphshapiq-explorer order
   :alt: The SI-Graph of the molecule, drawn on its atoms and bonds. A slider sets the explanation order to 1, 2, 3, 4 or n. At order 1 only the atoms carry contributions. At higher orders, hyperedges between atoms appear. A reading shows that the contributions sum to the same value for every order.

   **The explanation order.** The SI-Graph of the molecule for the explanation orders 1 to 4 and
   for the MIs. :graphshapiq-red:`Red` raises the log-odds and :graphshapiq-blue:`blue` lowers
   them. For every order, the contributions sum to the log-odds minus the prediction with no atoms.

.. _graphshapiq-sec-interpretation:

Interpreting the SI-Graph
-------------------------

An SI-Graph is read like the graph it explains. :graphshapiq-red:`Red` marks contributions that
raise the prediction and :graphshapiq-blue:`blue` marks contributions that lower it. A
node attribute of order :math:`k \geq 2` is not an SV. It is what remains for the node after its
interactions up to order :math:`k` are accounted for. For 1,3-dinitrobenzene, each nitrogen atom
has an SV of +1.86. At order 2, its node attribute is only +0.15, while each of its two pairs with
an oxygen atom carries +1.83. Each oxygen atom has a node attribute of −0.70. The GNN thus reads
the nitro group through its N–O pairs, and an oxygen atom without its nitrogen atom speaks against
mutagenicity.

The explanation order is a trade-off between accuracy and complexity. SVs are quick to read but
hide such structure. MIs describe the game completely, but 4,095 contributions for 12 atoms are
hard to read. The demo therefore offers the orders 2, 3 and 4 in between.

Interactions also explain predictions that differ although the atoms are the same.
:shapiq-figure:`arrangement` shows catechol, resorcinol and hydroquinone, which all consist of a
benzene ring with two hydroxy groups. Only the positions of the two oxygen atoms differ. The PAINS
GNN predicts 99.8% for catechol, 10.9% for resorcinol and 25.2% for hydroquinone. At order 2, the
interaction of the two oxygen atoms is +1.12 for catechol, where they are neighbors on the ring. It
is −0.43 for resorcinol and −0.15 for hydroquinone. Catechol is a known pattern of PAINS
:cite:p:`Baell.2010`, and the sign of the interaction of the two oxygen atoms separates it from the
other two arrangements. This pair is not the strongest contribution of catechol. The two carbon
atoms that carry the hydroxy groups interact with +1.82, and each of them has a node attribute of
−2.48. The figure therefore frames the pair of the two oxygen atoms and fades all other groups.

.. shapiq-demo-figure:: graphshapiq-explorer arrangement
   :alt: Three molecules side by side. Catechol, resorcinol and hydroquinone are benzene rings with two hydroxy groups in neighboring, alternate and opposite positions. Each is shown with the probability that the PAINS GNN predicts and with its SI-Graph of order 2. The pair of the two oxygen atoms is framed and all other groups are faded. The pair is red for catechol and blue for the other two. Catechol also has strong red interactions between its ring atoms and blue node attributes on them.

   **Same atoms, other arrangement.** The predictions of the PAINS GNN for catechol, resorcinol
   and hydroquinone, and their SI-Graphs of order 2. The three molecules have the same eight atoms.
   The interaction of the two oxygen atoms is framed, and all other groups are faded.

.. _graphshapiq-sec-computation:

Computing Exact Interactions with GraphSHAP-IQ
----------------------------------------------

By :eq:`graphshapiq-mi`, the MIs of the graph game require the GNN on all :math:`2^n` maskings.
This section shows that most of these model calls are unnecessary, since most MIs of a GNN are
zero by its architecture.

.. _graphshapiq-sec-receptive-fields:

Receptive Fields
~~~~~~~~~~~~~~~~

In :eq:`graphshapiq-message-passing`, every layer passes messages between neighbors only. The
number of layers :math:`\ell` therefore determines the *receptive field* of a node, i.e. the
message-passing range defined by its :math:`\ell`-hop neighborhood

.. math::
   :label: graphshapiq-receptive-field

   \begin{gathered}
   \mathcal{N}_i^{(\ell)} := \{j \in N \mid d_g(i,j) \leq \ell\} \\
   \text{with } d_g(i,j) := \text{length of shortest path from } v_j \text{ to } v_i \text{ in } g
   \end{gathered}

The embedding of a node cannot depend on the features of a node outside its receptive field.
Consequently, the node game is unaffected by maskings outside the :math:`\ell`-hop neighborhood.
For a graph :math:`g` and an :math:`\ell`-Layer GNN :math:`f_g`, the GNN-induced node game
:math:`\nu_i` with :math:`i \in N` satisfies the *node game invariance* :cite:p:`Muschalik.2025`

.. math::
   :label: graphshapiq-invariance

   \nu_i(T) = \nu_i(T \cap \mathcal{N}_i^{(\ell)}) \quad \text{for } T \subseteq N

This holds for every message passing technique. The receptive-field figure below shows the
receptive fields of four molecules for one to four layers. For cholesterol with 28 atoms and three
layers, the largest receptive field contains 16 atoms.

.. shapiq-demo:: graphshapiq-exact
   :anchor: receptive-fields

.. _graphshapiq-sec-sparsity:

Sparse Möbius Interactions
~~~~~~~~~~~~~~~~~~~~~~~~~~

The invariance :eq:`graphshapiq-invariance` concerns single node embeddings. To transfer it to the
graph prediction, GraphSHAP-IQ requires a structural assumption on how the GNN combines them.

.. shapiq-definition:: GNN Architecture
   :name: graphshapiq-def-architecture

   The global pooling :math:`\Psi` and the output layer :math:`\sigma` are required to be linear
   functions, e.g. :math:`\Psi` is a mean or sum pooling operation and :math:`\sigma` is a dense
   layer.

This assumption is common in practice :cite:p:`Xu.2019`, and the GNNs of the demo satisfy it. Due
to the node game invariance, the MIs of a node game are zero for every set that is not fully
contained in the receptive field of its node. Formally, let :math:`m_i: \mathcal{P}(N) \to
\mathbb{R}^{d_\ell}` be the MIs of the GNN-induced node game :math:`\nu_i` for :math:`i \in N`
under :shapiq-statement:`graphshapiq-def-architecture`. Then, these trivial node game interactions
are :cite:p:`Muschalik.2025`

.. math::
   :label: graphshapiq-trivial-node

   m_i(S) = \mathbf{0} \quad \text{for all } S \not\subseteq \mathcal{N}_i^{(\ell)}

Under :shapiq-statement:`graphshapiq-def-architecture`, the graph game in :eq:`graphshapiq-link`
is a linear combination of the node games, and MIs are linear in the game. The zeros of the node
games therefore carry over to the graph game as trivial graph game interactions
:cite:p:`Muschalik.2025`. Let :math:`m_g: \mathcal{P}(N) \to \mathbb{R}` be the MIs of the
GNN-induced graph game :math:`\nu_g` and :math:`\mathcal{I}` the set of *non-trivial
interactions*. Then

.. math::
   :label: graphshapiq-sparsity

   \begin{gathered}
   m_g(S) = 0 \text{ for all } S \subseteq N \text{ with } S \notin \mathcal{I}, \\
   \text{where} \quad
   \mathcal{I} := \bigcup_{i \in N} \mathcal{P}(\mathcal{N}_i^{(\ell)})
   \end{gathered}

In words, a group of nodes can only interact if all of its nodes lie in the receptive field of one
node. :shapiq-figure:`sparsity` confirms this on the GNNs of the demo. ``shapiq`` computes all
:math:`2^n` MIs by brute force and compares them with the set :math:`\mathcal{I}`. For indigo,
1,010,688 of the 1,048,576 MIs lie outside :math:`\mathcal{I}`. With the GNN in double precision,
their largest absolute value is :math:`6 \cdot 10^{-13}`. With the GNN in single precision, as in
the demo, it is 0.0004, which is the level of its rounding errors. The largest MI inside
:math:`\mathcal{I}` is 9.13. For 1,3-dinitrobenzene, the receptive fields of three layers
cover the whole molecule, so that :math:`\mathcal{I}` contains all sets.

.. shapiq-demo-figure:: graphshapiq-explorer sparsity
   :alt: A scatter plot of all Möbius interactions of the molecule. The horizontal axis is the size of the set. The vertical axis is the absolute value on a logarithmic scale. Interactions of sets inside the set I reach values above 1. Interactions of sets outside the set I stay far below a dashed line, which marks the rounding errors of the GNN in single precision.

   **Möbius interactions vanish outside the receptive fields.** All :math:`2^n` MIs of the
   molecule, computed by brute force with the GNN in double precision, by the size of their set.
   The colors separate the sets inside :math:`\mathcal{I}` from those outside. The dashed line
   marks the rounding errors of the GNN in single precision.

.. _graphshapiq-sec-complexity:

Complexity
~~~~~~~~~~

The set :math:`\mathcal{I}` contains every subset of each of its sets. The MIs of all sets in
:math:`\mathcal{I}` therefore follow from the graph game on :math:`\mathcal{I}` alone. Computing
MIs and SIs on the graph game requires :math:`\vert \mathcal{I} \vert` *model calls*
:cite:p:`Muschalik.2025`, which are bounded by

.. math::
   :label: graphshapiq-complexity

   \vert \mathcal{I} \vert \leq \sum_{i \in N} 2^{\vert \mathcal{N}_i^{(\ell)} \vert}
   \leq n \cdot 2^{n_{\max}^{(\ell)}}
   \leq n \cdot 2^{\frac{d_{\max}^{\ell+1}-1}{d_{\max}-1}}

where :math:`n_{\max}^{(\ell)} := \max_{i \in N} \vert \mathcal{N}_i^{(\ell)} \vert` is the size of
the largest :math:`\ell`-hop neighborhood and :math:`d_{\max}` is the maximum degree of the graph.
In other words, the complexity depends at most linearly on the size of the graph. It depends
exponentially on the connectivity of the graph and on the number of layers. GraphSHAP-IQ turns this
result into the four steps of :shapiq-figure:`algorithm`.

.. shapiq-demo-figure:: graphshapiq-explorer algorithm
   :alt: Algorithm 1 of GraphSHAP-IQ in a box. Its Require line holds sliders for the number of layers and the explanation order, beside a drawing of the molecule with its largest receptive field highlighted. Below, line 1 of the model-agnostic computation, which sets I to all subsets of N, is struck out and followed by line 1 of GraphSHAP-IQ, which sets I to the union of the power sets of the receptive fields. Lines 2 to 4 follow. Two columns count, for each line, the sets, model calls, Möbius interactions and Shapley interactions of the model-agnostic computation and of GraphSHAP-IQ. A bar shows the share of model calls that GraphSHAP-IQ needs.

   **Algorithm 1 against the model-agnostic computation.** The struck line is line 1 of the
   model-agnostic exact computation, which evaluates all :math:`2^n` maskings. GraphSHAP-IQ changes
   only this line. The columns count the sets, model calls, MIs and SIs of each line for the chosen
   molecule. They depend on the graph only, so the counts for :math:`\ell \neq 3` need no GNN.

   .. container:: graphshapiq-algorithm-lines

      **Algorithm 1** (GraphSHAP-IQ). **Require:** Graph :math:`g = (V, E, \mathbf{X})`,
      :math:`\ell`-Layer GNN :math:`f_g`, SI order :math:`k`.

      1. :math:`\mathcal{I} \leftarrow \bigcup_{i \in N} \mathcal{P}(\mathcal{N}_i^{(\ell)})`
      2. :math:`\nu_g \leftarrow [f_g(\mathbf{X}^{(T)})]_{T \in \mathcal{I}}`
      3. :math:`m_g \leftarrow [\texttt{MI}(\nu_g, S)]_{S \in \mathcal{I}}`
      4. :math:`\Phi_k \leftarrow \texttt{MItoSI}(m_g, k)`

      **Return:** MIs :math:`m_g`, SIs :math:`\Phi_k`.

GraphSHAP-IQ identifies the set :math:`\mathcal{I}` from the graph, evaluates the GNN for all
maskings in :math:`\mathcal{I}`, computes the MIs with :eq:`graphshapiq-mi`, and converts them to
SIs. Note that a higher explanation order needs no further model calls.
:shapiq-figure:`complexity` compares :math:`\vert \mathcal{I} \vert` with :math:`2^n` for the
molecules of this page. For indigo, GraphSHAP-IQ needs 128, 2,366, 37,888 and 458,752 model calls
for one to four layers, instead of 1,048,576. For cholesterol, it needs 129,664 model calls for
three layers instead of 268,435,456. The saving grows with the size of the graph and shrinks with
the number of layers.

.. shapiq-demo-figure:: graphshapiq-explorer complexity
   :alt: A chart of the number of model calls on a logarithmic scale against the number of atoms, for the seven molecules of the demo and the four molecules of the receptive-field figure. A dashed line marks 2 to the power of n. One dot per molecule marks the number of maskings that GraphSHAP-IQ needs for the number of layers that a slider sets from one to four. The molecules of the demo are filled dots and the molecules of the receptive-field figure are rings. The chosen molecule is highlighted, and a dotted line leads from its dot up to the dashed line. For large molecules, the dots lie far below the line.

   **Maskings against** :math:`2^n`\ **.** The number of model calls :math:`\vert \mathcal{I}
   \vert` of GraphSHAP-IQ for the seven molecules of the demo (dots) and the four molecules of the
   receptive-field figure (rings). The slider sets the number of layers. The dashed line marks all
   :math:`2^n` maskings, and the dotted line shows the saving for the chosen molecule.

For densely connected graphs and GNNs with many layers, :math:`\vert \mathcal{I} \vert` can still
be infeasible. An approximation variant of GraphSHAP-IQ introduces a parameter :math:`\lambda`,
which limits the size of the computed MIs, and assigns the remaining value of each larger
receptive field to one MI, so that efficiency is maintained. GraphSHAP-IQ computes exact SIs if
:math:`\lambda \geq n_{\max}^{(\ell)} - 1` :cite:p:`Muschalik.2025`.

.. _graphshapiq-sec-evaluation:

Evaluation
----------

Exact SIs need no faithfulness metric, since they are the SIs of the graph game by construction.
The evaluation therefore checks that the computation returns the exact values, and compares its
cost with model-agnostic approximation.

The first check is a computation by brute force. For each of the seven molecules, ``shapiq``
evaluates the GNN on all :math:`2^n` maskings and computes all MIs with :eq:`graphshapiq-mi`. The
MIs of GraphSHAP-IQ agree with them on every set in :math:`\mathcal{I}`, and no MI outside
:math:`\mathcal{I}` exceeds the rounding errors of the GNN in single precision. For 1,3-dinitrobenzene, benzimidazole and
aspirin, :math:`\mathcal{I}` contains all sets, so that GraphSHAP-IQ evaluates all :math:`2^n`
maskings as well.

The comparison runs GraphSHAP-IQ and the model-agnostic approximators of ``shapiq`` with the same
number of model calls. :shapiq-figure:`approximation` shows the mean squared error (MSE) of their
2-SII estimates for each budget of the approximation variant. For indigo, GraphSHAP-IQ is exact
after 37,888 model calls, which are 3.6% of all maskings. With the same budget, KernelSHAP-IQ
reaches an MSE of 0.002, on average over three runs. Below this budget, the picture is mixed. With
:math:`\lambda = 1` and 110 model calls, the approximation variant has an MSE of 0.52 and SVARM-IQ
of 5.4. From about :math:`\lambda = 3` on, the approximation variant is worse than the
approximators. With :math:`\lambda = 8` and 30,248 model calls, its MSE is 113, while KernelSHAP-IQ
reaches 0.003. The reason is that the GNNs of the demo have large MIs of high order and opposite sign.
The approximation variant reassigns their remaining value to sets of the size of a receptive
field, which spreads it onto the pairs.

.. shapiq-demo-figure:: graphshapiq-explorer approximation
   :alt: A line chart of the mean squared error of the 2-SII estimates on a logarithmic scale against the number of model calls. Each curve is the mean of three runs with a band from the smallest to the largest value. For the smallest budgets, the approximation variant of GraphSHAP-IQ is level with or better than most approximators. For larger budgets it lies above KernelSHAP-IQ, SVARM-IQ, SHAP-IQ and Permutation Sampling, whose curves decrease. At the budget of the exact computation it is exact.

   **Approximation at equal budget.** The MSE of 2-SII estimates against the exact values of
   GraphSHAP-IQ, for the approximation variant of GraphSHAP-IQ with growing :math:`\lambda` and
   for model-agnostic approximators of ``shapiq``, on the graph game of the molecule.

The same comparison on 10 molecules with 30 to 40 atoms of Mutagenicity and a 2-layer GIN is shown
in :shapiq-figure:`paper-approximation`. There, the approximation variant falls below an MSE of
:math:`10^{-6}` with about 2,500 model calls and reaches about :math:`10^{-10}` with about 7,800.
All model-agnostic baselines stay above :math:`10^{-6}`. GraphSHAP-IQ thus outperforms the
baselines in settings with a majority of lower-order MIs. The GNNs of the demo are not such a
setting. The interaction-informed variants of the baselines, which set all SIs outside
:math:`\mathcal{I}` to zero, also change little for them. For indigo, only 28 of the 190 pairs lie
outside :math:`\mathcal{I}`, and the MSE of SVARM-IQ at the exact budget improves from 0.0022 to
0.0020. At the exact budget, GraphSHAP-IQ takes about 2 seconds on average for such a molecule and a 2-layer
GCN, and its runtime does not increase with the explanation order.

.. shapiq-demo-figure:: graphshapiq-explorer paper-approximation
   :alt: A line chart of the mean squared error of 2-SII estimates on a logarithmic scale from 10 to the power of minus 10 to 0.1, against up to 8,000 model evaluations. The curve of GraphSHAP-IQ falls to about 10 to the power of minus 10. The curves of KernelSHAP-IQ, inconsistent KernelSHAP-IQ, Permutation Sampling, SHAP-IQ and SVARM-IQ end between 10 to the power of minus 6 and 10 to the power of minus 3.

   **Approximation on Mutagenicity.** The MSE of 2-SII estimates for a 2-layer GIN on molecules of
   Mutagenicity, for GraphSHAP-IQ and model-agnostic baselines, redrawn from Figure 4 (left) of
   GraphSHAP-IQ :cite:p:`Muschalik.2025`. Dashed lines are the interaction-informed variants of
   the baselines, which set all SIs outside :math:`\mathcal{I}` to zero.

The saving in model calls is largest for large and sparse graphs. On the 4,337 molecules of
Mutagenicity with 30.3 nodes on average, GraphSHAP-IQ reduces the model calls by a factor of about
:math:`10^5` for one layer, :math:`10^4` for two layers and :math:`10^2` for three layers. For
graphs with more than 100 nodes, where :math:`2^n` exceeds :math:`10^{30}`, many instances can be
computed exactly for one and two layers with fewer than :math:`10^5` model calls.

.. _graphshapiq-sec-insights:

What the Interactions Reveal
----------------------------

With exact SIs of every order at hand, the orders can be compared. This section reports what they
show about the GNNs of the demo.

The first finding concerns the nitro group. For 1,3-dinitrobenzene, the SVs point at the two
nitrogen atoms. The 2-SIIs and the MIs both show that not the nitrogen atom but its interactions
with the oxygen atoms contribute the most. The same holds for a molecule of Mutagenicity with 30
atoms and three nitro groups, explained for a 2-layer GCN. Two differently trained GNNs thus rely
on the same substructure in the same way.

The second finding concerns the order of the MIs. :shapiq-figure:`orders` splits the MIs of each
molecule by the size of their set. For all seven molecules, the MIs of four or more atoms sum to
more than +100 in their positive part and to less than −100 in their negative part. The two parts
nearly cancel. The explained difference of the log-odds is below 17 in absolute value. For
serotonin, the largest MI is a group of seven atoms of the indole with +9.96, and the next largest
is a group of eight atoms with −8.53. The GNNs thus spread their prediction over many terms of
high order and opposite sign. SIs of low order are the readable summary of these terms.

.. shapiq-demo-figure:: graphshapiq-explorer orders
   :alt: Horizontal bars on a symmetric logarithmic axis, in one group per molecule. Each group has four rows for the Möbius interactions of size 1, of size 2, of size 3 and of size 4 or more. In each row, a red bar to the right shows the sum of the positive interactions and a blue bar to the left shows the sum of the negative interactions. A line marks the total of each molecule, and dashed lines mark plus and minus 100. For every molecule, the bars of size 4 or more are the longest in both directions and reach beyond the dashed lines.

   **Möbius interactions by order.** The :graphshapiq-red:`positive` and the
   :graphshapiq-blue:`negative` sum of the MIs of size 1, 2, 3 and 4 or more, for the seven
   molecules and the question of each, on a symmetric logarithmic axis. The line marks the sum of
   all bars of a molecule, which is the log-odds minus the prediction with no atoms. The dashed
   lines mark ±100.

The third finding is an error of a GNN. Indigo contains no indole, but the GNN for indoles predicts
96.6%. The SVs assign +2.09 to each nitrogen atom and −1.38 to each oxygen atom. At order 2, each
oxygen atom has an interaction of −3.18 with the nitrogen atom of its ring. The GNN thus weighs
evidence against an indole and still decides for it. Each half of indigo has the ring skeleton of
an indole, but not its bond orders, which the GNN does not see.

.. _graphshapiq-sec-limitations:

Limitations
-----------

- **Linear readout.** The guarantee requires :shapiq-statement:`graphshapiq-def-architecture`. For
  a 2-layer GCN on Mutagenicity with a 2-layer perceptron as output layer, the MIs are not
  restricted to the receptive fields. For such a GNN, GraphSHAP-IQ is not
  exact.
- **Dense graphs and deep GNNs.** The number of model calls grows exponentially with the size of
  the receptive fields. For indigo, four layers need 458,752 model calls, which is 44% of all
  :math:`2^{20}` maskings. For three of the seven molecules, three layers already cover the whole
  molecule, and GraphSHAP-IQ saves nothing. ``shapiq`` supports receptive fields of up to 62 nodes
  and warns above :math:`2^{25}` maskings.
- **Approximation.** Below the exact budget, the approximation variant can be far less accurate
  than model-agnostic approximators when the GNN has large MIs of high order. For indigo and 30,248
  model calls, its MSE is 113 against 0.003 for KernelSHAP-IQ. For a molecule whose receptive
  fields cover all atoms, the exact budget is :math:`2^n` and the comparison is trivial.
- **Masking with a fixed baseline.** The explanation depends on the baseline, and masked graphs are
  not molecules. For the GNN for mutagenicity, a masked atom is 73% carbon, 15% oxygen and 8%
  nitrogen. Other masking strategies, such as induced subgraphs or removed edges, could emphasize
  other properties of the GNN.
- **Elements only.** The GNNs of the demo see the element of each atom and the bonds, but no bond
  orders and no hydrogens. This is why the GNN for indoles predicts 96.6% for indigo.
- **The model, not the chemistry.** The explanation is exact for the GNN and not for the property.
  The GNN for mutagenicity has a weighted F1 score of 0.794 on its test set, so its explanations
  can describe wrong predictions.
- **Small molecules in the browser.** The demo explains a molecule that the reader builds with at
  most 15 atoms, i.e. at most 32,768 maskings.

.. _graphshapiq-sec-code:

Explain It Yourself
-------------------

GraphSHAP-IQ is available in ``shapiq`` as :class:`~shapiq.explainer.GraphExplainer`. It explains
any PyTorch Geometric model that is called with the node features, the edges and the batch vector,
and that states its number of message passing layers as ``num_layers``. The following model is the
GIN of the demo with sum pooling and a linear output layer. It returns the log-odds as its single
output.

.. code-block:: python

   import base64
   import json
   import urllib.request

   import numpy as np
   import torch
   from torch_geometric.data import Data
   from torch_geometric.nn import GIN, global_add_pool

   import shapiq


   class GNN(torch.nn.Module):
       """A GIN with sum pooling and a linear output layer, which returns the log-odds."""

       def __init__(self, n_features: int = 11, hidden: int = 32, num_layers: int = 3) -> None:
           super().__init__()
           self.num_layers = num_layers  # determines the receptive fields
           self.gin = GIN(n_features, hidden, num_layers, out_channels=hidden)
           self.readout = torch.nn.Linear(hidden, 1)

       def forward(self, x, edge_index, batch):
           return self.readout(global_add_pool(self.gin(x, edge_index), batch))

The next lines load the weights of the GNN for mutagenicity from the data of the demo, where the
batch normalization of the trained GNN is folded into the linear layers. A GNN of your own needs
only its trained weights. The code then explains 1,3-dinitrobenzene with 2-SII.

.. code-block:: python

   url = (
       "https://raw.githubusercontent.com/mmschlk/shapiq/main/"
       "docs/source/demos/graphshapiq-explorer/assets/data.js"
   )
   data = urllib.request.urlopen(url).read().decode()
   task = json.loads(data[data.index("{") : data.rindex("}") + 1])["tasks"][1]
   model = GNN()
   steps = task["gnn"]["steps"]  # seven linear layers, in the order of the model
   arrays = [np.frombuffer(base64.b64decode(step[key]), "<f4") for step in steps for key in "wb"]
   for parameter, array in zip(model.parameters(), arrays):
       parameter.data = torch.tensor(array).reshape(parameter.shape)

   # 1,3-dinitrobenzene: a benzene ring (atoms 0 to 5) with two nitro groups
   symbols = ["C", "N", "O", "F", "Cl", "Br", "P", "S", "B", "I", "Unk"]
   atoms = ["C", "C", "C", "C", "C", "C", "N", "O", "O", "N", "O", "O"]
   bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
   bonds += [(4, 6), (6, 7), (6, 8), (2, 9), (9, 10), (9, 11)]
   graph = Data(
       x=torch.eye(len(symbols))[[symbols.index(atom) for atom in atoms]],
       edge_index=torch.tensor(bonds + [(j, i) for i, j in bonds]).T,
   )

   explainer = shapiq.explainer.GraphExplainer(
       model,
       data=np.array(task["baseline"]),  # the baseline of masked atoms
       index="k-SII",
       max_order=2,
   )
   interactions = explainer.explain(graph)

   n = graph.num_nodes
   print(f"{interactions.estimation_budget:,} of {2**n:,} maskings")
   print(f"exact: {not interactions.estimated}")
   print(f"sum of all contributions: {interactions.values.sum():+.2f}")
   pairs = {S: v for S, v in interactions.dict_values.items() if len(S) == 2}
   for (i, j), value in sorted(pairs.items(), key=lambda pair: -abs(pair[1]))[:5]:
       print(f"{atoms[i]}{i} and {atoms[j]}{j}: {value:+.2f}")

.. code-block:: text

   4,096 of 4,096 maskings
   exact: True
   sum of all contributions: +3.98
   N6 and O8: +1.83
   N9 and O11: +1.83
   N6 and O7: +1.83
   N9 and O10: +1.83
   C2 and C3: +0.28

The four strongest interactions are the N–O pairs of the two nitro groups. The contributions
include the prediction with no atoms and therefore sum to the log-odds of 3.98. For this small
molecule, the receptive fields cover all atoms. For indigo, the same code evaluates 37,888 of
1,048,576 maskings. Try another molecule, another explanation order with ``max_order``, or the MIs
with ``index="Moebius"``.

The scripts that computed the data of the figures on this page are in
`docs/source/demos/graphshapiq-explorer/compute
<https://github.com/mmschlk/shapiq/tree/main/docs/source/demos/graphshapiq-explorer/compute>`__,
with a README on how to run them. The scripts that trained the GNNs and computed the demo are in
`docs/source/demos/graphshapiq-common/compute
<https://github.com/mmschlk/shapiq/tree/main/docs/source/demos/graphshapiq-common/compute>`__.

.. _graphshapiq-sec-citation:

.. rst-class:: graphshapiq-citation

Citation
--------

This page is based on *Exact Computation of Any-Order Shapley Interactions for Graph Neural
Networks* :cite:p:`Muschalik.2025`, the work that introduces GraphSHAP-IQ. The definitions, the
theoretical results, the algorithm and the results on Mutagenicity are taken from it. This includes
the data of :shapiq-figure:`paper-approximation`, which is redrawn here from Figure 4 (left). Everything shown for the
molecules of the demo is computed for this page with ``shapiq``, with GNNs that were trained for
the demo.

The original code of GraphSHAP-IQ is available at
`github.com/FFmgll/GraphSHAP-IQ <https://github.com/FFmgll/GraphSHAP-IQ>`__. If you use
GraphSHAP-IQ in your work, please cite it.

.. code-block:: bibtex

   @inproceedings{Muschalik.2025,
     title     = {Exact Computation of Any-Order Shapley Interactions for
                  Graph Neural Networks},
     author    = {Maximilian Muschalik and Fabian Fumagalli and Paolo Frazzetto and
                  Janine Strotherm and Luca Hermes and Alessandro Sperduti and
                  Eyke H{\"u}llermeier and Barbara Hammer},
     booktitle = {International Conference on Learning Representations (ICLR)},
     year      = {2025},
     url       = {https://openreview.net/forum?id=9tKC0YM8sX}
   }
