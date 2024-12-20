# Why Use ``shapiq``?

There are a couple of reasons why you might want to use ``shapiq``:

## Explanations with Shapley Interactions

``shapiq`` directly extends on ``shap`` but also allows for computation of Shapley interactions.
These interactions can be used to explain models in more detail.
To facilitate any-order interactions, ``shapiq`` requires specific data structure and sets of algorithms.

## Explanations with Shapley values

Similar to ``shap``, ``shapiq`` can also be used to explain models with the well-established Shapley values.
Many algorithms that are available in ``shap`` are also available in ``shapiq``.
Often, this is beneficial when you are looking into a higher-number of features.

## Two Independent Perspectives: Explanation and Game Theory

``shapiq`` offers two independent perspectives on the same problem: explanation and game theory.
We introduce the notion of a general ``game``, which maps any machine learning problem (also outside the scope of machine learning) to a cooperative game without design decisions of explanation methods.
This allows for easy computation of many game-theoretic concepts, such as the Shapley value, Shapley interactions, or the Banzhaf value.
The explanation perspective is similar to ``shap`` and includes established mechanisms to transform any machine learning model into a cooperative game.
``shapiq`` offers a unified interface to both perspectives.

## Benchmarking of Novel Approaches

``shapiq`` is a platform for benchmarking novel approaches in the field of Shapley values and Shapley interactions.
We implement many state-of-the-art algorithms and provide a unified interface to compare them.
Further, we provide a set of tools to evaluate the performance of these algorithms on pre-computed benchmarks tasks.
