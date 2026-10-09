"""Shared helpers (fakes and skip markers) for the shapiq_games tests."""

from __future__ import annotations

import doctest
import importlib
import importlib.util
import pkgutil
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from types import ModuleType


def is_installed(package: str) -> bool:
    """Return whether ``package`` can be imported."""
    return importlib.util.find_spec(package) is not None


def package_modules(package: ModuleType) -> list[str]:
    """Return the names of a package and all its modules."""
    prefix = f"{package.__name__}."
    names = [name for _, name, _ in pkgutil.walk_packages(package.__path__, prefix)]
    return sorted([package.__name__, *names])


def run_docstring_examples(name: str) -> None:
    """Run the docstring examples of a module and fail if one of them fails."""
    module = importlib.import_module(name)
    result = doctest.testmod(
        module, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE, report=True
    )
    assert result.failed == 0, f"{result.failed} docstring example(s) failed in {name}"


skip_if_no_skimage = pytest.mark.skipif(
    not is_installed("skimage"), reason="scikit-image is not installed"
)


class FakeTokenizer:
    """A whitespace tokenizer mimicking the parts of a Hugging Face tokenizer the games use."""

    mask_token_id = 0

    def __init__(self) -> None:
        self.vocabulary: dict[str, int] = {"[MASK]": 0, "[CLS]": 1, "[SEP]": 2}

    def __call__(self, text: str, *, add_special_tokens: bool = True) -> dict[str, list[int]]:
        ids = [self.vocabulary.setdefault(word, len(self.vocabulary)) for word in text.split()]
        return {"input_ids": [1, *ids, 2] if add_special_tokens else ids}

    def decode(self, ids: np.ndarray) -> str:
        words = {index: word for word, index in self.vocabulary.items()}
        return " ".join(words[int(i)] for i in ids)


class FakeSentimentPipeline:
    """A sentiment 'model': P(positive) is 0.5 plus 0.1 per 'good' and minus 0.1 per 'bad'."""

    def __init__(self, labels: tuple[str, str] = ("POSITIVE", "NEGATIVE")) -> None:
        self.tokenizer = FakeTokenizer()
        self.labels = labels
        self.calls = 0
        self.batch_sizes: list[int | None] = []

    def __call__(
        self,
        texts: list[str],
        *,
        top_k: int | None = 1,
        batch_size: int | None = None,
        **_: object,
    ) -> list[list[dict[str, float | str]]] | list[dict[str, float | str]]:
        self.calls += len(texts)
        self.batch_sizes.append(batch_size)
        outputs = []
        for text in texts:
            words = text.split()
            positive = min(max(0.5 + 0.1 * (words.count("good") - words.count("bad")), 0.0), 1.0)
            scores = sorted(
                [
                    {"label": self.labels[0], "score": positive},
                    {"label": self.labels[1], "score": 1.0 - positive},
                ],
                key=lambda out: -float(out["score"]),
            )
            outputs.append(scores if top_k is None else scores[0])
        return outputs


def mean_brightness_classifier(images: np.ndarray) -> np.ndarray:
    """A two-class 'image classifier': the probability of class 0 is the mean brightness."""
    brightness = np.asarray(images, dtype=float).mean(axis=(1, 2, 3)) / 255.0
    return np.stack([brightness, 1.0 - brightness], axis=1)
