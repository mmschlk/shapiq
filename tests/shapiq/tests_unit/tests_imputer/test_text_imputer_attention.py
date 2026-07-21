from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from shapiq.imputer.text_imputer import (  # noqa: E402
    AttentionMaskPerturbation,
    TextImputer,
    create_tensor_perturbation_strategy,
)


class TinyTokenizer:
    """Tiny tokenizer for attention-mask unit tests.

    It avoids downloading HuggingFace models/tokenizers while still exposing the
    methods and attributes used by TextImputer.
    """

    def __init__(self) -> None:
        self.vocab: dict[str, int] = {
            "<eos>": 0,
            "<pad>": 1,
        }
        self.inv_vocab: dict[int, str] = {
            0: "<eos>",
            1: "<pad>",
        }
        self.eos_token = "<eos>"
        self.eos_token_id = 0
        self.pad_token = "<pad>"
        self.pad_token_id = 1
        self.mask_token = "[MASK]"
        self.mask_token_id = 2
        self.padding_side = "right"

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        """Encode text by whitespace tokens."""
        token_ids: list[int] = []

        for token in text.split():
            if token not in self.vocab:
                token_id = len(self.vocab)
                self.vocab[token] = token_id
                self.inv_vocab[token_id] = token

            token_ids.append(self.vocab[token])

        return token_ids

    def tokenize(self, text: str) -> list[str]:
        """Return whitespace tokens."""
        return text.split()

    def decode(
        self,
        token_ids: list[int] | torch.Tensor,
        *,
        skip_special_tokens: bool = False,
    ) -> str:
        """Decode token ids."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()

        return " ".join(self.inv_vocab[int(token_id)] for token_id in token_ids)

    def convert_ids_to_tokens(
        self,
        token_ids: list[int] | torch.Tensor,
    ) -> list[str]:
        """Convert token ids to token strings."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()

        return [self.inv_vocab[int(token_id)] for token_id in token_ids]


class StaticPlayerStrategy:
    """Player strategy with predefined players."""

    def __init__(self, players: list[str]) -> None:
        self.players = players

    def get_players(self) -> list[str]:
        return self.players

    @property
    def n_players(self) -> int:
        return len(self.players)

    def coalition_to_text(self, coalition: np.ndarray, perturbation_strategy) -> str:
        output: list[str] = []

        for keep, player in zip(coalition, self.players, strict=False):
            if keep:
                output.append(player)
            else:
                output.append(perturbation_strategy.perturb(player))

        return " ".join(output)


class FakeEncoderClassifier(torch.nn.Module):
    """Fake encoder classifier whose score depends on visible tokens."""

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: object,
    ) -> SimpleNamespace:
        visible_count = attention_mask.float().sum(dim=1)
        logits = torch.stack([-visible_count, visible_count], dim=1)

        return SimpleNamespace(logits=logits)


class FakeCausalLM(torch.nn.Module):
    """Fake causal LM whose target score depends on visible prompt tokens."""

    def __init__(self, target_token_id: int, vocab_size: int = 128) -> None:
        super().__init__()
        self.target_token_id = target_token_id
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: object,
    ) -> SimpleNamespace:
        batch_size, sequence_length = input_ids.shape
        logits = torch.zeros(
            batch_size,
            sequence_length,
            self.vocab_size,
            device=input_ids.device,
        )

        visible_count = attention_mask.float().sum(dim=1)
        logits[:, :, self.target_token_id] = visible_count[:, None]

        return SimpleNamespace(logits=logits)


def test_attention_mask_builds_inputs_for_one_coalition() -> None:
    """Attention masking keeps input_ids fixed and masks missing player spans."""
    tokenizer = TinyTokenizer()
    players = ["Paris", "is", "beautiful", "."]

    coalition = np.array([False, True, False, True], dtype=bool)

    masked_inputs = AttentionMaskPerturbation.build_inputs_for_coalitions(
        tokenizer=tokenizer,
        players=players,
        coalitions=coalition,
        player_separator=" ",
    )

    encoded = masked_inputs[0]

    input_ids = encoded["input_ids"].tolist()[0]
    attention_mask = encoded["attention_mask"].tolist()[0]

    assert tokenizer.decode(input_ids) == "Paris is beautiful ."
    assert attention_mask == [0, 1, 0, 1]


def test_attention_mask_builds_inputs_for_batch_coalitions() -> None:
    """Attention masking supports a matrix of coalitions."""
    tokenizer = TinyTokenizer()
    players = ["Paris", "is", "beautiful", "."]

    coalitions = np.array(
        [
            [True, True, True, True],
            [False, False, True, True],
        ],
        dtype=bool,
    )

    masked_inputs = AttentionMaskPerturbation.build_inputs_for_coalitions(
        tokenizer=tokenizer,
        players=players,
        coalitions=coalitions,
        player_separator=" ",
    )

    assert len(masked_inputs) == 2
    assert masked_inputs[0]["attention_mask"].tolist()[0] == [1, 1, 1, 1]
    assert masked_inputs[1]["attention_mask"].tolist()[0] == [0, 0, 1, 1]

    # The token ids stay unchanged across attention-masked coalitions.
    assert masked_inputs[0]["input_ids"].tolist() == masked_inputs[1]["input_ids"].tolist()


def test_attention_mask_prompt_keeps_prompt_tokens_visible() -> None:
    """Prompt tokens should remain visible while player tokens are maskable."""
    tokenizer = TinyTokenizer()
    players = ["Paris", "is", "beautiful"]

    coalition = np.array([False, True, True], dtype=bool)

    masked_inputs = AttentionMaskPerturbation.build_prompt_inputs_for_coalitions(
        tokenizer=tokenizer,
        players=players,
        coalitions=coalition,
        prompt_template="Question: {text} Answer:",
        player_separator=" ",
    )

    encoded = masked_inputs[0]

    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    attention_mask = encoded["attention_mask"].tolist()[0]

    assert tokens == ["Question:", "Paris", "is", "beautiful", "Answer:"]
    assert attention_mask == [1, 0, 1, 1, 1]


def test_attention_mask_wrong_coalition_width_raises() -> None:
    """Coalition width must match the number of players."""
    tokenizer = TinyTokenizer()
    players = ["Paris", "is", "beautiful"]

    wrong_coalition = np.array([[True, False]], dtype=bool)

    with pytest.raises(ValueError, match="Expected coalition width"):
        AttentionMaskPerturbation.build_inputs_for_coalitions(
            tokenizer=tokenizer,
            players=players,
            coalitions=wrong_coalition,
            player_separator=" ",
        )


def test_text_imputer_attention_mask_encoder_path_returns_scores() -> None:
    """TextImputer should score attention-masked encoder inputs."""
    tokenizer = TinyTokenizer()
    model = FakeEncoderClassifier()
    player_strategy = StaticPlayerStrategy(["Paris", "is", "beautiful", "."])

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="Paris is beautiful.",
        player_strategy=player_strategy,
        perturbation_type="attention_mask",
        model_type="encoder_classifier",
        class_idx=1,
        output_type="logit",
        batch_size=2,
        device="cpu",
    )

    coalitions = np.array(
        [
            [True, True, True, True],
            [False, False, True, True],
        ],
        dtype=bool,
    )

    scores = imputer(coalitions)

    assert scores.shape == (2,)
    assert scores[0] > scores[1]


def test_text_imputer_attention_mask_causal_lm_path_returns_scores() -> None:
    """TextImputer should score attention-masked causal-LM prompt inputs."""
    tokenizer = TinyTokenizer()
    target_token_id = tokenizer.encode("yes")[0]
    model = FakeCausalLM(target_token_id=target_token_id)
    player_strategy = StaticPlayerStrategy(["Paris", "is", "beautiful"])

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="Paris is beautiful.",
        player_strategy=player_strategy,
        perturbation_type="attention_mask",
        model_type="causal_lm",
        target_label="yes",
        prompt_template="Question: {text} Answer:",
        batch_size=2,
        device="cpu",
    )

    coalitions = np.array(
        [
            [True, True, True],
            [False, False, True],
        ],
        dtype=bool,
    )

    scores = imputer(coalitions)

    assert scores.shape == (2,)
    assert np.all(np.isfinite(scores))
    assert scores[0] > scores[1]


def test_attention_mask_full_prediction_uses_full_coalition() -> None:
    """full_prediction should work for attention-mask perturbation."""
    tokenizer = TinyTokenizer()
    model = FakeEncoderClassifier()
    player_strategy = StaticPlayerStrategy(["Paris", "is", "beautiful", "."])

    imputer = TextImputer(
        model=model,
        tokenizer=tokenizer,
        text="Paris is beautiful.",
        player_strategy=player_strategy,
        perturbation_type="attention_mask",
        model_type="encoder_classifier",
        class_idx=1,
        output_type="logit",
        batch_size=2,
        device="cpu",
    )

    score = imputer.full_prediction

    assert isinstance(score, float)
    assert np.isfinite(score)


def test_attention_mask_rejects_mismatched_coalition_and_player_spans() -> None:
    """Coalition length must match the number of player spans."""
    base_attention_mask = torch.ones((1, 3), dtype=torch.long)
    player_spans = [(0, 1), (1, 2), (2, 3)]
    coalition = np.array([True, False], dtype=bool)

    with pytest.raises(ValueError, match="does not match number of player spans"):
        AttentionMaskPerturbation.build_attention_mask_for_coalition(
            base_attention_mask=base_attention_mask,
            player_spans=player_spans,
            coalition=coalition,
        )


def test_attention_mask_prompt_template_requires_text_placeholder() -> None:
    """Prompt templates must contain the text placeholder."""
    tokenizer = TinyTokenizer()
    players = ["Paris", "is", "beautiful"]
    coalition = np.array([True, False, True], dtype=bool)

    with pytest.raises(ValueError, match=r"prompt_template must contain '\{text\}'"):
        AttentionMaskPerturbation.build_prompt_inputs_for_coalitions(
            tokenizer=tokenizer,
            players=players,
            coalitions=coalition,
            prompt_template="Question: Paris is beautiful Answer:",
            player_separator=" ",
        )


def test_attention_mask_prompt_rejects_wrong_coalition_width() -> None:
    """Prompt coalition width must match the number of players."""
    tokenizer = TinyTokenizer()
    players = ["Paris", "is", "beautiful"]
    wrong_coalition = np.array([[True, False]], dtype=bool)

    with pytest.raises(ValueError, match="Expected coalition width"):
        AttentionMaskPerturbation.build_prompt_inputs_for_coalitions(
            tokenizer=tokenizer,
            players=players,
            coalitions=wrong_coalition,
            prompt_template="Question: {text} Answer:",
            player_separator=" ",
        )


def test_attention_mask_causal_lm_requires_prompt_template() -> None:
    """Causal-LM attention masking requires a prompt template."""
    tokenizer = TinyTokenizer()
    perturbation = AttentionMaskPerturbation(tokenizer=tokenizer)

    with pytest.raises(ValueError, match="prompt_template is required for causal_lm"):
        perturbation.evaluate(
            players=["Paris", "is", "beautiful"],
            coalitions=np.array([[True, False, True]], dtype=bool),
            model_type="causal_lm",
        )


def test_attention_mask_seq2seq_requires_prompt_template() -> None:
    """Seq2seq attention masking requires a prompt template."""
    tokenizer = TinyTokenizer()
    perturbation = AttentionMaskPerturbation(tokenizer=tokenizer)

    with pytest.raises(ValueError, match="prompt_template is required for seq2seq"):
        perturbation.evaluate(
            players=["Paris", "is", "beautiful"],
            coalitions=np.array([[True, False, True]], dtype=bool),
            model_type="seq2seq",
        )


def test_attention_mask_seq2seq_builds_prompt_inputs() -> None:
    """Seq2seq attention masking should build prompt-based masked inputs."""
    tokenizer = TinyTokenizer()
    perturbation = AttentionMaskPerturbation(tokenizer=tokenizer)

    masked_inputs = perturbation.evaluate(
        players=["Paris", "is", "beautiful"],
        coalitions=np.array([[False, True, True]], dtype=bool),
        model_type="seq2seq",
        prompt_template="Question: {text} Answer:",
        player_separator=" ",
    )

    assert len(masked_inputs) == 1
    assert masked_inputs[0]["attention_mask"].tolist()[0] == [1, 0, 1, 1, 1]


def test_attention_mask_rejects_unknown_model_type() -> None:
    """Attention masking should reject unsupported model types."""
    tokenizer = TinyTokenizer()
    perturbation = AttentionMaskPerturbation(tokenizer=tokenizer)

    with pytest.raises(ValueError, match="Unknown model_type for attention masking"):
        perturbation.evaluate(
            players=["Paris"],
            coalitions=np.array([[True]], dtype=bool),
            model_type="unsupported",
        )


def test_create_tensor_perturbation_strategy_rejects_unknown_strategy() -> None:
    """Tensor perturbation factory should reject unknown strategies."""
    tokenizer = TinyTokenizer()

    with pytest.raises(ValueError, match="Unknown tensor perturbation strategy"):
        create_tensor_perturbation_strategy(
            "unsupported",
            tokenizer=tokenizer,
        )
