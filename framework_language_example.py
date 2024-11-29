"""This script runs the framework on a language example."""

import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch
from transformers import pipeline

import shapiq
from framework_explanations import compute_explanation_int_val
from framework_si_graph import si_graph_plot
from shapiq.plot._config import BLUE, RED


class SentimentClassificationGame(shapiq.Game):
    """The sentiment analysis classifier modeled as a cooperative game.

    Args:
        classifier: The sentiment analysis classifier.
        tokenizer: The tokenizer of the classifier.
        test_sentence: The sentence to be explained.
    """

    def __init__(self, classifier, tokenizer, test_sentence):
        self.classifier = classifier
        self.tokenizer = tokenizer
        self.test_sentence = test_sentence
        self.mask_token_id = tokenizer.mask_token_id
        self.tokenized_input = np.asarray(tokenizer(test_sentence)["input_ids"][1:-1])
        self.n_players = len(self.tokenized_input)

        empty_coalition = np.zeros((1, len(self.tokenized_input)), dtype=bool)
        self.normalization_value = float(self.value_function(empty_coalition)[0])
        super().__init__(
            n_players=self.n_players, normalization_value=self.normalization_value, verbose=True
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Computes the value of the coalitions.

        Args:
            coalitions: A numpy matrix of shape (n_coalitions, n_players).

        Returns:
            A vector of the value of the coalitions.
        """
        texts = []
        for coalition in coalitions:
            tokenized_coalition = self.tokenized_input.copy()
            # all tokens not in the coalition are set to mask_token_id
            tokenized_coalition[~coalition] = self.mask_token_id
            coalition_text = self.tokenizer.decode(tokenized_coalition)
            texts.append(coalition_text)

        # get the sentiment of the texts (call the model as defined above)
        sentiments = self._model_call(texts)

        return sentiments

    def _model_call(self, input_texts: list[str]) -> np.ndarray:
        """Calls the sentiment classification model with a list of texts.

        Args:
            input_texts: A list of input texts.

        Returns:
            A vector of the sentiment of the input texts.
        """
        outputs = self.classifier(input_texts)
        outputs = [
            output["score"] * 1 if output["label"] == "POSITIVE" else output["score"] * -1
            for output in outputs
        ]
        sentiments = np.array(outputs, dtype=float)

        return sentiments


def _get_color_and_alpha(max_value: float, value: float) -> tuple[str, float]:
    """Gets the color and alpha value for an interaction value."""
    color = RED.hex if value >= 0 else BLUE.hex
    ratio = value / max_value
    return color, abs(ratio)


def sentence_plot(
    interaction_values: shapiq.InteractionValues,
    feature_names: list[str],
    fontsize: int = 15,
    xlim: tuple[float, float] = (0.0, 1.2),
    word_spacing: float = 0.03,
) -> None:
    """Plots the first order effects of a sentence.

    The effects are plotted as a color behind the words of the sentence. The words are text in black
    and behind them the color indicates the effect of the word on the sentiment of the sentence.

    Args:
        interaction_values: The interaction values of the sentence
        feature_names: The words of the sentence.
        fontsize: The fontsize of the words.
        xlim: The x-axis limits of the plot.
        word_spacing: The spacing between the words.
    """
    feature_names = [feature_name.strip() for feature_name in feature_names]

    # first draw the words on a plot to get the width of the words
    fig_dummy, ax_dummy = plt.subplots()
    x_pos = 0
    x_positions, widths, heights = [], [], []
    for i, feature_name in enumerate(feature_names):
        r = fig_dummy.canvas.get_renderer()
        t = ax_dummy.text(x_pos, 0.5, feature_name, fontsize=fontsize, ha="left", va="bottom")
        bb = t.get_window_extent(renderer=r).transformed(plt.gca().transData.inverted())
        widths.append(bb.width)
        heights.append(bb.height)
        x_positions.append(x_pos)
        x_pos += bb.width + word_spacing
    plt.close(fig_dummy)

    # now draw the real plot
    fig, ax = plt.subplots()
    ax.set_xlim(xlim)
    ax.axis("off")

    attributions = [interaction_values[(i,)] for i in range(len(feature_names))]
    max_abs_value = max([abs(value) for value in attributions])

    x_pos = 0
    for i, (feature_name, attribution) in enumerate(zip(feature_names, attributions)):
        color, alpha = _get_color_and_alpha(max_abs_value, attribution)
        text_color = "black" if alpha < 0.7 else "white"
        ax.text(
            x_pos, 0.5, feature_name, fontsize=fontsize, ha="left", va="bottom", color=text_color
        )

        # draw the background color
        padding = 0.04 * alpha
        height = heights[i] + padding
        width = widths[i]
        patch = FancyBboxPatch(
            xy=(x_pos, 0.5 - padding / 2),
            width=width,
            height=height,
            color=color,
            alpha=alpha,
            zorder=-1,
            boxstyle="Round, pad=0, rounding_size=0.005",
        )
        ax.add_patch(patch)
        x_pos += width + word_spacing


if __name__ == "__main__":

    PLOT_TITLE = False

    # framework settings ---------------------------------------------------------------------------
    # for the language model only baseline fanova is available
    feature_influence = "full"  # "pure", "partial", "full"
    entity = "interaction"  # "individual", "joint", "interaction"
    order = 2  # 1, 2, 3, ...

    print("Feature influence:", feature_influence)
    print("Entity:", entity)
    print("Order:", order)
    print()

    # sentence -------------------------------------------------------------------------------------
    review = "The acting was bad but the movie enjoyable"
    save_name = review.replace(" ", "_")
    values_name = f"language_game_{save_name}.npz"
    plot_save_name = f"language_game_{save_name}_{feature_influence}_{entity}_{order}"

    # get the model and tokenizer ------------------------------------------------------------------
    classifier = pipeline(task="sentiment-analysis", model="lvwerra/distilbert-imdb")
    tokenizer = classifier.tokenizer

    print(f"Loaded classifier: {classifier}")
    print(f"Special tokens: {tokenizer.special_tokens_map}")
    print(f"Mask token id: {tokenizer.mask_token_id}")

    tokenized_sentence = tokenizer(review)
    print(f"Classifier output: {classifier(review)} for the sentence: '{review}'")
    print(f"Tokenized sentence: {tokenized_sentence}")

    # setup the game -------------------------------------------------------------------------------
    game = SentimentClassificationGame(classifier, tokenizer, review)
    print(f"Game for the full coalition: {game(game.grand_coalition)[0]}")

    # pre-compute and store the game values --------------------------------------------------------
    if os.path.exists(values_name):
        print(f"Loading values from {values_name}")
        game.load_values(values_name)
    else:
        print(f"Precomputing and saving values to {values_name}")
        game.precompute()
        game.save_values(values_name)

    # compute the explanation ----------------------------------------------------------------------
    explanation = compute_explanation_int_val(
        game=game,
        entity_type=entity,
        influence=feature_influence,
        explanation_order=order,
    )
    int_values = explanation.get_n_order(min_order=1, order=order)
    print("Sum of all effects:", sum(explanation.values) + explanation.baseline_value)
    print(int_values)

    # plot the explanation -------------------------------------------------------------------------
    feature_names = [tokenizer.decode([token_id]) for token_id in game.tokenized_input]
    title = f"Entity: {entity.capitalize()}, Influence: {feature_influence.capitalize()}, Order: {order}"
    if order != 1:
        si_graph_nodes = list(shapiq.powerset(range(int_values.n_players), min_size=2, max_size=2))
        si_graph_plot(
            int_values,
            graph=si_graph_nodes,
            draw_original_edges=False,
            circular_layout=True,
            label_mapping={i: feature_names[i] for i in range(len(feature_names))},
            compactness=1e20,
            size_factor=3,
            node_size_scaling=1.5,
            n_interactions=100,
            node_area_scaling=False,
        )
        if PLOT_TITLE:
            plt.title(title)
    else:
        sentence_plot(explanation, feature_names)

    plt.tight_layout()
    plt.savefig(plot_save_name + ".pdf")
    plt.show()
