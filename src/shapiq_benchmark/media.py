"""Optional shipped text, image, TabPFN, and causal family representatives."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from typing import Any

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
CATALOG["text_remove"] = CATALOG["text"]
CATALOG["image_vit"] = CATALOG["image"]

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


def _weight_hash(modules: list) -> str:
    """Identify the exact downloaded tensor weights, independently of mutable model names."""
    digest = hashlib.sha256()
    for number, module in enumerate(modules):
        for name, tensor in sorted(module.state_dict().items()):
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(f"{number}:{name}:{array.dtype}:{array.shape}".encode())
            digest.update(array.tobytes())
    return digest.hexdigest()


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
        name not in ("tabpfn", "image_vit")
        and n_players is not None
        and (type(n_players) is not int or n_players not in (11, 12, 16, 20))
    ):
        message = "Media player overrides must be 11, 12, 16 or 20."
        raise ValueError(message)
    torch.set_num_threads(1)
    torch.manual_seed(instance_seed)
    metadata: dict = {
        **EXTRA_CATALOG[name],
        "recipe": name,
        "parameters": {},
        "random_state": instance_seed,
        "instance_seed": instance_seed,
    }
    if backend is not None:
        metadata["preparation_hardware"] = backend
    if name in ("text", "text_remove"):
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
        if n_players not in (None, 11, 12):
            message = "Fixed text examples support 11 or 12 tokenizer players."
            raise ValueError(message)
        sentence = sentences[instance_seed]
        if n_players == 12:
            sentence = sentence.replace("A ", "A really ", 1)
        strategy = "remove" if name == "text_remove" else "mask"
        game = SentimentAnalysis(sentence, device="cpu", mask_strategy=strategy)
        text_model = game._classifier.model  # noqa: SLF001 -- authenticate the shipped pretrained model
        if hasattr(text_model, "state_dict"):
            metadata["model_sha256"] = _weight_hash([text_model])
        metadata.update(
            dataset="authored sentiment examples",
            input_id=f"sentence-{instance_seed}" + (f"-{n_players}tokens" if n_players else ""),
            cluster_id="pretrained-lvwerra-distilbert-imdb",
            replicate_unit="explanation input",
            text=sentence,
            model="lvwerra/distilbert-imdb",
            model_revision=game._classifier.model.config._commit_hash,  # noqa: SLF001 -- record downloaded revision
            player_unit="token",
            parameters={"mask_strategy": strategy, "device": "cpu"},
            semantics=f"signed sentiment confidence after tokenizer {strategy}",
            stochastic_frozen=False,
        )
    elif name in ("image", "image_vit"):
        from shapiq_games.benchmark.local_xai.benchmark_image import ImageClassifier

        directory = Path(__file__).resolve().parents[1] / "shapiq_games/benchmark/imagenet_examples"
        path = sorted(directory.glob("*.JPEG"))[instance_seed]
        if name == "image_vit":
            if n_players not in (None, 16):
                message = "The qualified ViT recipe has exactly 16 patches."
                raise ValueError(message)
            if torch.cuda.is_available():
                message = "Shipped ViT auto-selects CUDA; use a CPU-only preparation worker."
                raise ValueError(message)
            game = ImageClassifier(model_name="vit_16_patches", x_explain_path=str(path))
            wrapper = cast("Any", game.model_function)
            metadata["model_sha256"] = _weight_hash(
                [
                    wrapper._embedding_layer,  # noqa: SLF001 -- hash shipped component weights
                    wrapper._encoder,  # noqa: SLF001 -- hash shipped component weights
                    wrapper._classifier,  # noqa: SLF001 -- shipped model exposes component modules
                ]
            )
            metadata.update(
                model="google/vit-base-patch32-384",
                cluster_id="pretrained-vit-base-patch32-384",
                player_unit="patch",
                parameters={"model_name": "vit_16_patches", "device": "cpu"},
                semantics="shipped ViT patch-mask token classification probability",
            )
        else:
            requested_segments = n_players + 1 if n_players is not None else 9
            original = ImageClassifier(
                model_name="resnet_18",
                n_superpixel_resnet=requested_segments,
                x_explain_path=str(path),
            )
            resnet = cast("Any", original.model_function)
            if hasattr(resnet, "model"):
                metadata["model_sha256"] = _weight_hash([resnet.model])
            resnet.batch_size = 1
            mask = resnet.superpixels
            active = [int(label) - 1 for label in np.unique(mask)]
            game = ActiveImage(original, active)
            full = np.ones((1, original.n_players), dtype=bool)
            lifted = np.zeros_like(full)
            lifted[:, active] = True
            np.testing.assert_allclose(original(full), original(lifted), rtol=0, atol=0)
            metadata.update(
                cluster_id="pretrained-resnet18-imagenet1k-v1",
                model="torchvision ResNet18 IMAGENET1K_V1",
                player_unit="superpixel",
                parameters={
                    "requested_superpixels": requested_segments,
                    "inference_batch_size": 1,
                    "active_player_indices": active,
                },
                segmentation_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
                semantics="shipped fixed-gray (127) superpixel masking; unused reported players removed",
            )
        metadata.update(
            dataset="ImageNet bundled examples",
            input_id=path.name,
            replicate_unit="explanation input",
            data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            stochastic_frozen=False,
        )
    elif name == "tabpfn":
        import tabpfn

        from shapiq.imputer.tabpfn_imputer import TabPFNImputer
        from shapiq_benchmark.datasets import dataset_details
        from shapiq_benchmark.families import DATASETS, _dataset, feature_subset
        from shapiq_benchmark.models import _tabpfn_checkpoint

        dataset = dataset or "iris"
        if dataset not in DATASETS:
            message = "TabPFN recipes require a registered dataset."
            raise ValueError(message)
        classification = DATASETS[dataset]["task"] == "classification"
        x, y, train, test, names = _dataset(dataset, instance_seed)
        data_hash = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
        train = train[:64]
        features = feature_subset(x, n_players, instance_seed)
        parameters = {"n_estimators": 1, "device": device}
        if classification:
            parameters["class_index"] = 1
        eligible = np.flatnonzero(np.ptp(x[train], axis=0) > 0)
        if n_players is not None and len(eligible) < x.shape[1]:
            features = eligible[feature_subset(x[:, eligible], n_players, instance_seed)]
            parameters["feature_rule"] = (
                "seeded subset of columns nonconstant on the 64 TabPFN training rows"
            )
        x = x[:, features]
        model_options: dict = {"inference_precision": torch.float32} if device == "cuda" else {}
        if device == "cuda":
            parameters["inference_precision"] = "float32"
        # New configured recipes bind an actual checkpoint file. Legacy recipes
        # retain the shipped default to preserve their existing realization.
        if n_players is not None:
            checkpoint = _tabpfn_checkpoint("classification" if classification else "regression")
            model_options["model_path"] = checkpoint
            with checkpoint.open("rb") as stream:
                metadata["checkpoint_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
            metadata["checkpoint_name"] = checkpoint.name
        constructor = tabpfn.TabPFNClassifier if classification else tabpfn.TabPFNRegressor
        model = constructor(
            device=device, n_estimators=1, random_state=instance_seed, **model_options
        )
        game = TabPFNImputer(
            model,
            x[train],
            y[train],
            x_test=x[test],
            predict_function=(
                (lambda model, rows: model.predict_proba(rows)[:, 1])
                if classification
                else (lambda model, rows: model.predict(rows))
            ),
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
            model="TabPFNClassifier" if classification else "TabPFNRegressor",
            parameters=parameters,
            player_unit="feature",
            semantics=(
                "remove-and-contextualize class-one probability"
                if classification
                else "remove-and-contextualize regression prediction"
            ),
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
