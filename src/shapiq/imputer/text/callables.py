"""Model callables for coalition-based text explanations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

try:
    import torch
except ImportError as err:
    from ._error import _text_import_error

    raise _text_import_error from err

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase
    from transformers.modeling_outputs import BaseModelOutput

# =============================================================================
# TARGET CALLABLES
# =============================================================================


class BaseTargetCallable(ABC):
    """Abstract interface for model-specific scoring."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
    ) -> None:
        """Abstract interface for model-specific scoring."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @abstractmethod
    def predict(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Return scalar scores."""

    def predict_from_inputs(
        self,
        inputs: list[dict[str, torch.Tensor]],
    ) -> np.ndarray:
        """Return scalar scores from pre-tokenized model inputs."""
        msg = f"{self.__class__.__name__} does not support pre-tokenized inputs."
        raise NotImplementedError(msg)


# =============================================================================
# Encoder only support
# =============================================================================


class EncoderClassifierCallable(BaseTargetCallable):
    """Score text with an encoder-only sequence-classification model.

    The callable returns either the selected class logit or its softmax probability.
    It is suitable for models such as BERT, RoBERTa, or DistilBERT with a sequence-classification head.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        class_idx: int = 1,
        output_type: str = "logit",
    ) -> None:
        """Encoder classifier scoring."""
        super().__init__(model, tokenizer, device)

        self.class_idx = class_idx
        self.output_type = output_type

        if output_type not in {"logit", "probability"}:
            msg = "output_type must be 'logit' or 'probability'"
            raise ValueError(msg)

    def predict(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Return one score per text from the configured classifier output.

        Texts are tokenized as a padded batch and evaluated in inference mode.
        Depending on ``output_type``, the returned score is either the raw logit or
        the softmax probability for ``class_idx``.
        """
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits

        if self.output_type == "logit":
            scores = logits[:, self.class_idx]

        else:
            probs = torch.softmax(logits, dim=-1)
            scores = probs[:, self.class_idx]

        return scores.detach().cpu().numpy()

    def predict_from_inputs(
        self,
        inputs: list[dict[str, torch.Tensor]],
    ) -> np.ndarray:
        """Run encoder classifier inference from pre-tokenized inputs."""
        input_ids = torch.cat(
            [item["input_ids"] for item in inputs],
            dim=0,
        ).to(self.device)

        attention_mask = torch.cat(
            [item["attention_mask"] for item in inputs],
            dim=0,
        ).to(self.device)

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if "token_type_ids" in inputs[0]:
            model_inputs["token_type_ids"] = torch.cat(
                [item["token_type_ids"] for item in inputs],
                dim=0,
            ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**model_inputs)
            logits = outputs.logits

        if self.output_type == "logit":
            scores = logits[:, self.class_idx]
        else:
            probs = torch.softmax(logits, dim=-1)
            scores = probs[:, self.class_idx]

        return scores.detach().cpu().numpy()


# =============================================================================
# Causal LM support
# =============================================================================


class CausalLMCallable(BaseTargetCallable):
    """Score text using the log-probability of a target causal-LM continuation.

    Each input text is inserted into ``prompt_template``. The score is the
    autoregressive log-probability of ``target_label`` after that prompt.
    Multi-token target labels are scored token by token, conditioned on the
    prompt and all preceding target tokens.

    This supports decoder-only models such as Gemma, Llama, GPT, and Qwen,
    provided their tokenizer defines either a padding token or an EOS token.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        target_label: str = "good",
        prompt_template: str = ("Review: {text}\n\nSentiment:"),
    ) -> None:
        """Causal language model scoring."""
        super().__init__(model, tokenizer, device)
        self.prompt_template = prompt_template
        self.target_token_ids = tokenizer.encode(target_label, add_special_tokens=False)

        if len(self.target_token_ids) == 0:
            msg = f"Target label '{target_label}' produced no tokens."
            raise ValueError(msg)

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                msg = "Tokenizer must define either a pad token or eos token."
                raise ValueError(msg)

            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "left"

    def _build_prompt(
        self,
        text: str,
    ) -> str:
        """Construct a prompt for causal LM scoring."""
        return self.prompt_template.format(text=text)

    def _score_target_sequence(
        self,
        prompt: str,
    ) -> float:
        """Compute the autoregressive log-probability of the target label.

        For each target token, the model receives the prompt followed by preceding target tokens.
        The final-position distribution is then used to score the next target token.
        """
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.target_token_ids
        total_log_prob = 0.0

        for i in range(len(target_ids)):
            prefix_ids = target_ids[:i]
            input_ids = prompt_ids + prefix_ids
            encoded = {"input_ids": torch.tensor([input_ids], device=self.device)}

            with torch.no_grad():
                outputs = self.model(**encoded)

            logits = outputs.logits
            next_token_logits = logits[:, -1, :]
            log_probs = torch.log_softmax(next_token_logits, dim=-1)

            total_log_prob += log_probs[0, target_ids[i]].item()

        return total_log_prob

    def predict(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """Score texts using target-sequence log probabilities."""
        scores = []
        for text in texts:
            prompt = self._build_prompt(text)
            score = self._score_target_sequence(prompt)
            scores.append(score)

        return np.asarray(scores, dtype=np.float32)

    def predict_from_inputs(
        self,
        inputs: list[dict[str, torch.Tensor]],
    ) -> np.ndarray:
        """Score target sequence from pre-tokenized causal-LM prompt inputs."""
        scores = []

        for item in inputs:
            prompt_input_ids = item["input_ids"].to(self.device)
            prompt_attention_mask = item["attention_mask"].to(self.device)

            total_log_prob = 0.0

            for i, target_token_id in enumerate(self.target_token_ids):
                prefix_ids = self.target_token_ids[:i]

                if len(prefix_ids) > 0:
                    prefix_tensor = torch.tensor(
                        [prefix_ids],
                        dtype=torch.long,
                        device=self.device,
                    )

                    input_ids = torch.cat(
                        [prompt_input_ids, prefix_tensor],
                        dim=1,
                    )

                    prefix_attention_mask = torch.ones_like(prefix_tensor)

                    attention_mask = torch.cat(
                        [prompt_attention_mask, prefix_attention_mask],
                        dim=1,
                    )
                else:
                    input_ids = prompt_input_ids
                    attention_mask = prompt_attention_mask

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                logits = outputs.logits
                next_token_logits = logits[:, -1, :]
                log_probs = torch.log_softmax(next_token_logits, dim=-1)

                total_log_prob += log_probs[0, target_token_id].item()

            scores.append(total_log_prob)

        return np.asarray(scores, dtype=np.float32)


# =============================================================================
# seq2seq support
# =============================================================================


class Seq2SeqCallable(BaseTargetCallable):
    """Score a fixed target sequence with an encoder-decoder model.

    For each input text, this callable computes the conditional log-probability
    of generating ``target_label`` using teacher forcing:

        log P(target_label | input_text)

    A multi-token target is scored token by token. By default, the final score
    is the mean token log-probability, so targets of different lengths are more
    comparable.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        device: str,
        target_label: str = "positive",
        prompt_template: str = "{text}",
        *,
        normalize: bool = True,
    ) -> None:
        """Initialize seq2seq target-sequence scoring."""
        super().__init__(model, tokenizer, device)

        if not getattr(model.config, "is_encoder_decoder", False):
            msg = (
                "Seq2SeqCallable requires an encoder-decoder model with "
                "model.config.is_encoder_decoder=True."
            )
            raise ValueError(msg)

        self.target_label = target_label
        self.prompt_template = prompt_template
        self.normalize = normalize

        self.target_token_ids: list[int] = tokenizer.encode(
            target_label,
            add_special_tokens=False,
        )
        if not self.target_token_ids:
            msg_0 = f"Target label {target_label!r} produced no tokens after encoding."
            raise ValueError(msg_0)

        decoder_start_token_id = model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.pad_token_id

        if decoder_start_token_id is None:
            msg_1 = (
                "Cannot determine decoder_start_token_id: neither "
                "model.config.decoder_start_token_id nor tokenizer.pad_token_id "
                "is available."
            )
            raise ValueError(msg_1)

        self.decoder_start_token_id = decoder_start_token_id

    def _build_prompt(self, text: str) -> str:
        """Wrap the original text into a prompt template."""
        return self.prompt_template.format(text=text)

    def _encode_inputs(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Encode a list of texts into encoder input tensors."""
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _compute_log_prob_for_target(
        self,
        encoder_outputs: BaseModelOutput,
        attention_mask: torch.Tensor,
        batch_size: int,
    ) -> np.ndarray:
        """Compute the log-probability of the decoder generating the target token sequence."""
        total_log_probs = torch.zeros(batch_size, device=self.device)

        decoder_input_ids = torch.full(
            (batch_size, 1),
            self.decoder_start_token_id,
            dtype=torch.long,
            device=self.device,
        )

        for target_token_id in self.target_token_ids:
            with torch.no_grad():
                outputs = self.model(
                    encoder_outputs=encoder_outputs,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                )

            log_probs = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)
            total_log_probs += log_probs[:, target_token_id]

            next_token = torch.full(
                (batch_size, 1),
                target_token_id,
                dtype=torch.long,
                device=self.device,
            )
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=1)

        if self.normalize:
            total_log_probs /= len(self.target_token_ids)

        return total_log_probs.cpu().numpy()

    def predict(self, texts: list[str]) -> np.ndarray:
        """Compute log-probability scores of the Seq2Seq target sequence for a batch of texts."""
        prompts = [self._build_prompt(text) for text in texts]
        encoder_inputs = self._encode_inputs(prompts)

        encoder = self.model.get_encoder()
        with torch.no_grad():
            encoder_outputs = encoder(
                input_ids=encoder_inputs["input_ids"],
                attention_mask=encoder_inputs["attention_mask"],
                return_dict=True,
            )

        return self._compute_log_prob_for_target(
            encoder_outputs=encoder_outputs,
            attention_mask=encoder_inputs["attention_mask"],
            batch_size=len(prompts),
        )

    def predict_from_inputs(
        self,
        inputs: list[dict[str, torch.Tensor]],
    ) -> np.ndarray:
        """Score target sequence from pre-tokenized encoder inputs."""
        input_ids = torch.cat(
            [item["input_ids"] for item in inputs],
            dim=0,
        ).to(self.device)

        attention_mask = torch.cat(
            [item["attention_mask"] for item in inputs],
            dim=0,
        ).to(self.device)

        encoder = self.model.get_encoder()

        with torch.no_grad():
            encoder_outputs = encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        return self._compute_log_prob_for_target(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            batch_size=input_ids.shape[0],
        )
