Minimal TextImputer Example

The imputer_example.py script explains the prediction of a DistilBERT
sentiment classifier for the sentence:

The movie is not bad.

The example:

* creates a word-level TextImputer;
* uses word removal as the perturbation strategy;
* explains the positive sentiment probability;
* computes first-order Shapley values with KernelSHAP;
* computes second-order interactions with KernelSHAPIQ;
* displays force plots for both explanations.

Run the example from the project root:

uv run python examples/language/imputer_example.py

The first run automatically downloads the Hugging Face model.
