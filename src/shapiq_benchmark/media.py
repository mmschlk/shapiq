"""Optional shipped text, image, TabPFN, and causal family representatives."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from shapiq import Game

import numpy as np

CATALOG = {
    "text": (
        "shapiq_games.benchmark.local_xai.benchmark_language.SentimentAnalysis",
        "text_explanation",
        False,
    ),
    "image": (
        "shapiq_games.benchmark.local_xai.benchmark_image.ImageClassifier",
        "image_explanation",
        False,
    ),
    "tabpfn": ("shapiq.imputer.tabpfn_imputer.TabPFNImputer", "local_explanation", False),
    "causal_global": ("shapiq_games.benchmark.causal_xai.benchmark.CurthVDS", "synthetic", True),
    "causal_local": (
        "shapiq_games.benchmark.causal_xai.base.LocalConfoundingXAI",
        "synthetic",
        True,
    ),
}
EXTRA_CATALOG = {
    name: {
        "class": cls,
        "source": "src/" + cls.rsplit(".", 1)[0].replace(".", "/") + ".py",
        "application_family": family,
        "synthetic": synthetic,
        "coverage": "one bounded representative; dataset wrappers not implied",
    }
    for name, (cls, family, synthetic) in CATALOG.items()
}


class ActiveImage:
    """Remove the unused final player reported by the legacy SLIC clipping path."""

    def __init__(self, game: Game, active: list[int]) -> None:
        """Retain the existing game and its observed nonempty segment labels."""
        self.game, self.active, self.n_players = game, active, len(active)

    def __call__(self, coalitions: np.ndarray) -> np.ndarray:
        """Lift reduced coalitions into the original reported player space."""
        full = np.zeros((len(coalitions), self.game.n_players), dtype=bool)
        full[:, self.active] = coalitions
        return self.game(full)


def make_extra(name: str, *, instance_seed: int = 0) -> tuple:
    """Prepare actual downloaded-model games; missing access propagates as coverage failure."""
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(instance_seed)
    metadata = {
        **EXTRA_CATALOG[name],
        "recipe": name,
        "parameters": {},
        "random_state": instance_seed,
        "instance_seed": instance_seed,
    }
    if name == "text":
        from shapiq_games.benchmark.local_xai.benchmark_language import SentimentAnalysis

        sentences = (
            "A thoughtful film with a moving ending.",
            "A dull film with a weak ending.",
            "A funny story with a warm heart.",
            "A slow story with a strong cast.",
        )
        sentence = sentences[instance_seed]
        game = SentimentAnalysis(sentence, device="cpu", mask_strategy="mask")
        metadata.update(
            dataset="authored sentiment examples",
            input_id=f"sentence-{instance_seed}",
            cluster_id="pretrained-lvwerra-distilbert-imdb",
            replicate_unit="explanation input",
            text=sentence,
            model="lvwerra/distilbert-imdb",
            model_revision=game._classifier.model.config._commit_hash,  # noqa: SLF001 -- record downloaded revision
            player_unit="token",
            semantics="signed sentiment confidence after tokenizer masking",
            stochastic_frozen=False,
        )
    elif name == "image":
        from shapiq_games.benchmark.local_xai.benchmark_image import ImageClassifier

        directory = Path(__file__).resolve().parents[1] / "shapiq_games/benchmark/imagenet_examples"
        path = sorted(directory.glob("*.JPEG"))[instance_seed]
        original = ImageClassifier(
            model_name="resnet_18", n_superpixel_resnet=9, x_explain_path=str(path)
        )
        original.model_function.batch_size = 1
        mask = original.model_function.superpixels
        active = [int(label) - 1 for label in np.unique(mask)]
        game = ActiveImage(original, active)
        # The removed reported players must genuinely be null players.
        full = np.ones((1, original.n_players), dtype=bool)
        lifted = np.zeros_like(full)
        lifted[:, active] = True
        np.testing.assert_allclose(original(full), original(lifted), rtol=0, atol=0)
        metadata.update(
            dataset="ImageNet bundled examples",
            input_id=path.name,
            cluster_id="pretrained-resnet18-imagenet1k-v1",
            replicate_unit="explanation input",
            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            model="torchvision ResNet18 IMAGENET1K_V1",
            player_unit="superpixel",
            parameters={
                "requested_superpixels": 9,
                "inference_batch_size": 1,
                "active_player_indices": active,
            },
            segmentation_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
            semantics="shipped fixed-gray (127) superpixel masking; unused reported players removed",
            stochastic_frozen=False,
        )
    elif name == "tabpfn":
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from tabpfn import TabPFNClassifier

        from shapiq.imputer.tabpfn_imputer import TabPFNImputer

        x, y = load_iris(return_X_y=True)
        train, test = train_test_split(
            np.arange(len(x)), test_size=0.2, random_state=instance_seed, stratify=y
        )
        train = train[:64]
        model = TabPFNClassifier(device="cpu", n_estimators=1, random_state=instance_seed)
        game = TabPFNImputer(
            model,
            x[train],
            y[train],
            x_test=x[test],
            predict_function=lambda model, rows: model.predict_proba(rows)[:, 1],
        )
        game.fit(x[test[0]])
        metadata.update(
            dataset="iris",
            data_sha256=hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
            train_indices=train.tolist(),
            test_indices=test.tolist(),
            point_row=int(test[0]),
            model="TabPFNClassifier",
            parameters={"n_estimators": 1, "device": "cpu", "class_index": 1},
            player_unit="feature",
            semantics="remove-and-contextualize class-one probability",
            stochastic_frozen=True,
        )
    else:
        from shapiq_games.benchmark.causal_xai.base import LocalConfoundingXAI
        from shapiq_games.benchmark.causal_xai.benchmark import CurthVDS

        base = CurthVDS(n=64, d=4, seed=instance_seed, n_estimators=1, device="cpu")
        game = (
            base
            if name == "causal_global"
            else LocalConfoundingXAI(
                X=base.X,
                A=base.A,
                Y=base.Y,
                tau_hat=base.tau_hat,
                x_i=base.X[0],
                device="cpu",
                n_estimators=1,
            )
        )
        metadata.update(
            dataset="Curth-VDS synthetic",
            model="TabPFNRegressor",
            data_sha256=hashlib.sha256(
                base.X.tobytes() + base.A.tobytes() + base.Y.tobytes()
            ).hexdigest(),
            parameters={
                "n": 64,
                "d": 4,
                "seed": instance_seed,
                "n_estimators": 1,
                "mode": "signed",
                "device": "cpu",
            },
            player_unit="synthetic feature",
            semantics="shipped signed confounding attribution",
            stochastic_frozen=True,
        )
    metadata["n_players"] = game.n_players
    return game, metadata
