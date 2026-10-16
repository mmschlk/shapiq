# Why Use `shapiq`?

`shapiq` is a library for both the people who explain models and the people who research
game-theoretic explanations. Here is what it offers.

````{grid} 1 1 2 2
:gutter: 3
:class-container: shapiq-cards

~~~{grid-item-card} {octicon}`git-merge` Shapley Interactions of Any Order
:class-card: shapiq-card
:link: index
:link-type: doc

`shapiq` extends `shap` by Shapley interactions: k-SII, the Shapley-Taylor and the faithful
interaction indices, and more. Its data structure and algorithms handle interactions of any
order.
~~~

~~~{grid-item-card} {octicon}`person` Shapley Values, Too
:class-card: shapiq-card
:link: start
:link-type: doc

Explain models with Shapley values as in `shap`, with established and recent approximators
such as KernelSHAP, Leverage SHAP, permutation sampling, and SVARM, which also scale to many
features.
~~~

~~~{grid-item-card} {octicon}`workflow` Exact Algorithms
:class-card: shapiq-card
:link: /auto_examples/trees/index
:link-type: doc

Exact Shapley values and interactions of tree ensembles, and exact Shapley values of
nearest-neighbor models, Gaussian processes, and other product-kernel models, without sampling.
~~~

~~~{grid-item-card} {octicon}`apps` Explanation and Game Theory
:class-card: shapiq-card
:link: /auto_examples/exact_computation/index
:link-type: doc

A `Game` maps any problem to a cooperative game, without the design
decisions of an explanation method. The `ExactComputer` computes
the Shapley value, the Banzhaf value, interactions, the Möbius transform, the core, and more;
the explainers turn models into games.
~~~

~~~{grid-item-card} {octicon}`trophy` Benchmarks
:class-card: shapiq-card
:link: /api_reference_benchmark
:link-type: doc

A platform for new methods: `shapiq_games` defines games from data valuation to vision
transformers, and `shapiq_benchmark` scores approximators against their exact values with
error, ranking, and faithfulness metrics.
~~~

~~~{grid-item-card} {octicon}`eye` Visualization
:class-card: shapiq-card
:link: /auto_examples/visualization/index
:link-type: doc

Force, waterfall, bar, beeswarm, and scatter plots as known from `shap`, and plots made for
interactions: network, SI graph, and UpSet plots.
~~~
````
