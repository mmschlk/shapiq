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


def preparation_backend(device: str) -> dict:
    """Describe an explicitly requested CUDA oracle; never silently fall back to CPU."""
    import torch

    if device != "cuda" or not torch.cuda.is_available():
        message = "CUDA preparation requires an available, explicitly requested GPU."
        raise ValueError(message)
    current = torch.cuda.current_device()
    return {
        "device": "cuda",
        "gpu_model": torch.cuda.get_device_name(current),
        "compute_capability": list(torch.cuda.get_device_capability(current)),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "inference_precision": "float32",
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


def make_extra(
    name: str,
    *,
    instance_seed: int = 0,
    dataset: str | None = None,
    n_players: int | None = None,
    device: str = "cpu",
) -> tuple:
    """Prepare actual downloaded-model games; missing access propagates as coverage failure."""
    import torch

    if device not in ("cpu", "cuda") or (device != "cpu" and name != "tabpfn"):
        message = "Only TabPFN preparation supports an explicit CUDA device."
        raise ValueError(message)
    backend = preparation_backend(device) if device == "cuda" else None
    if name != "tabpfn" and dataset is not None:
        message = "Dataset overrides are supported only for TabPFN media recipes."
        raise ValueError(message)
    if (
        name != "tabpfn"
        and n_players is not None
        and (type(n_players) is not int or n_players not in (11, 12))
    ):
        message = "Text, image, and causal player overrides must be eleven or twelve."
        raise ValueError(message)
    torch.set_num_threads(1)
    torch.manual_seed(instance_seed)
    metadata = {
        **EXTRA_CATALOG[name],
        "recipe": name,
        "parameters": {},
        "random_state": instance_seed,
        "instance_seed": instance_seed,
    }
    if backend is not None:
        metadata["preparation_hardware"] = backend
    if name == "text":
        from shapiq_games.benchmark.local_xai.benchmark_language import SentimentAnalysis

        sentences = (
            "A thoughtful film with a moving ending.",
            "A dull film with a weak ending.",
            "A funny story with a warm heart.",
            "A slow story with a strong cast.",
        )
        if n_players is not None:
            sentences = (
                "A thoughtful film with a moving and deeply satisfying ending.",
                "A dull film with a weak and very predictable ending.",
                "A funny story with a warm and truly lovely heart.",
                "A slow story with a strong and surprisingly good cast.",
            )
        sentence = sentences[instance_seed]
        if n_players == 12:
            sentence = sentence.replace("A ", "A really ", 1)
        game = SentimentAnalysis(sentence, device="cpu", mask_strategy="mask")
        metadata.update(
            dataset="authored sentiment examples",
            input_id=f"sentence-{instance_seed}" + (f"-{n_players}tokens" if n_players else ""),
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
        requested_segments = n_players + 1 if n_players is not None else 9
        original = ImageClassifier(
            model_name="resnet_18",
            n_superpixel_resnet=requested_segments,
            x_explain_path=str(path),
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
                "requested_superpixels": requested_segments,
                "inference_batch_size": 1,
                "active_player_indices": active,
            },
            segmentation_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
            semantics="shipped fixed-gray (127) superpixel masking; unused reported players removed",
            stochastic_frozen=False,
        )
    elif name == "tabpfn":
        from tabpfn import TabPFNClassifier

        from shapiq.imputer.tabpfn_imputer import TabPFNImputer
        from shapiq_benchmark.datasets import dataset_details
        from shapiq_benchmark.families import DATASETS, _dataset, feature_subset

        dataset = dataset or "iris"
        if dataset not in DATASETS or DATASETS[dataset]["task"] != "classification":
            message = "TabPFN recipes require a registered classification dataset."
            raise ValueError(message)
        x, y, train, test, names = _dataset(dataset, instance_seed)
        data_hash = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
        train = train[:64]
        features = feature_subset(x, n_players, instance_seed)
        parameters = {"n_estimators": 1, "device": device, "class_index": 1}
        eligible = np.flatnonzero(np.ptp(x[train], axis=0) > 0)
        if n_players is not None and len(eligible) < x.shape[1]:
            features = eligible[feature_subset(x[:, eligible], n_players, instance_seed)]
            parameters["feature_rule"] = (
                "seeded subset of columns nonconstant on the 64 TabPFN training rows"
            )
        x = x[:, features]
        model_options = {"inference_precision": torch.float32} if device == "cuda" else {}
        if device == "cuda":
            parameters["inference_precision"] = "float32"
        model = TabPFNClassifier(
            device=device, n_estimators=1, random_state=instance_seed, **model_options
        )
        game = TabPFNImputer(
            model,
            x[train],
            y[train],
            x_test=x[test],
            predict_function=lambda model, rows: model.predict_proba(rows)[:, 1],
        )
        game.fit(x[test[0]])
        metadata.update(
            dataset=dataset,
            **dataset_details(dataset),
            data_sha256=data_hash,
            feature_indices=features.tolist(),
            feature_names=[str(names[i]) for i in features],
            train_indices=train.tolist(),
            test_indices=test.tolist(),
            point_row=int(test[0]),
            model="TabPFNClassifier",
            parameters=parameters,
            player_unit="feature",
            semantics="remove-and-contextualize class-one probability",
            stochastic_frozen=True,
        )
    else:
        from shapiq_games.benchmark.causal_xai.base import LocalConfoundingXAI
        from shapiq_games.benchmark.causal_xai.benchmark import CurthVDS

        dimension = n_players if n_players is not None else 4
        base = CurthVDS(n=64, d=dimension, seed=instance_seed, n_estimators=1, device="cpu")
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
                "d": dimension,
                "seed": instance_seed,
                "n_estimators": 1,
                "mode": "signed",
                "device": "cpu",
            },
            player_unit="synthetic feature",
            semantics="shipped signed confounding attribution",
            stochastic_frozen=True,
        )
    if n_players is not None and game.n_players != n_players:
        message = f"Requested {n_players} active players but the game has {game.n_players}."
        raise ValueError(message)
    metadata["n_players"] = game.n_players
    return game, metadata
