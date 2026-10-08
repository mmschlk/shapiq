"""Imagenette: ten easily classified ImageNet classes, for the image games.

`Imagenette <https://github.com/fastai/imagenette>`_ (fast.ai, Apache-2.0) is a subset of
ImageNet with full-size photos of ten classes. Every image keeps its ImageNet class, so pretrained
ImageNet classifiers (the vision transformer and ResNet-18 of the image games) classify them
directly. The official archive is downloaded on first use, verified by its SHA-256 checksum,
extracted once into the local data cache, and then deleted.
"""

from __future__ import annotations

import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from ._cache import RemoteFile, fetch, get_data_dir

__all__ = [
    "IMAGENETTE_CLASSES",
    "ImageDataset",
    "ImagenetteSize",
    "ImagenetteSplit",
    "load_imagenette",
]

type ImagenetteSplit = Literal["train", "val"]
type ImagenetteSize = Literal["160px", "320px"]

# WordNet id -> (ImageNet class index, class name)
IMAGENETTE_CLASSES: dict[str, tuple[int, str]] = {
    "n01440764": (0, "tench"),
    "n02102040": (217, "English springer"),
    "n02979186": (482, "cassette player"),
    "n03000684": (491, "chain saw"),
    "n03028079": (497, "church"),
    "n03394916": (566, "French horn"),
    "n03417042": (569, "garbage truck"),
    "n03425413": (571, "gas pump"),
    "n03445777": (574, "golf ball"),
    "n03888257": (701, "parachute"),
}

# The images are resized so that their shorter side is 160 or 320 pixels.
_ARCHIVES: dict[str, RemoteFile] = {
    "160px": RemoteFile(
        url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz",
        filename="imagenette2-160.tgz",
        sha256="64d0c4859f35a461889e0147755a999a48b49bf38a7e0f9bd27003f10db02fe5",
        subdir="imagenette",
    ),
    "320px": RemoteFile(
        url="https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
        filename="imagenette2-320.tgz",
        sha256="569b4497c98db6dd29f335d1f109cf315fe127053cedf69010d047f0188e158c",
        subdir="imagenette",
    ),
}
_COMPLETE = ".complete"


@dataclass(frozen=True)
class ImageDataset:
    """Images on disk with their labels. An image is loaded when it is accessed.

    Attributes:
        name: The dataset name.
        paths: The image files, sorted by class and file name.
        labels: The ImageNet class index of every image.
        class_names: The name of every ImageNet class index that occurs.
    """

    name: str
    paths: tuple[Path, ...]
    labels: np.ndarray
    class_names: dict[int, str]

    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.paths)

    def __getitem__(self, index: int) -> np.ndarray:
        """Return image ``index`` as an RGB ``uint8`` array of shape ``(height, width, 3)``."""
        with Image.open(self.paths[index]) as image:
            return np.asarray(image.convert("RGB"))

    def label_name(self, index: int) -> str:
        """Return the class name of image ``index``."""
        return self.class_names[int(self.labels[index])]


def _extract(archive: Path, target: Path) -> None:
    """Extract the JPEG images of ``archive`` into ``target``, all at once or not at all.

    The images are extracted into a staging directory that is renamed to ``target`` when complete.
    A complete ``target`` is never replaced, as another process may be reading it: if one appears
    meanwhile (another process extracted the same archive), the staging directory is discarded.
    """
    staging = Path(tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}."))
    try:
        with tarfile.open(archive) as tar:
            images = [
                member
                for member in tar.getmembers()
                if member.isfile() and member.name.lower().endswith(".jpeg")
            ]
            tar.extractall(staging, members=images, filter="data")
        (staging / _COMPLETE).touch()
        for _ in range(2):
            try:
                staging.rename(target)  # atomic; fails if target exists and is not empty
            except OSError:
                if (target / _COMPLETE).exists():
                    break
                shutil.rmtree(target, ignore_errors=True)  # an incomplete leftover
            else:
                return
        else:
            staging.rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)  # gone once renamed


def load_imagenette(
    *, split: ImagenetteSplit = "val", size: ImagenetteSize = "320px"
) -> ImageDataset:
    """Load the Imagenette images, downloading the archive on first use.

    The first call downloads the archive (about 99 MB for ``"160px"`` and 342 MB for ``"320px"``),
    verifies it, extracts its images into ``<data dir>/imagenette/``, and deletes the archive.

    Args:
        split: ``"val"`` (3,925 images, default) or ``"train"`` (9,469 images).
        size: ``"320px"`` (default) or ``"160px"``: the length of the shorter image side.

    Returns:
        The images of the split, sorted by class and file name, with their ImageNet classes.

    Raises:
        ValueError: If ``split`` or ``size`` is unknown.
        OSError: If the download fails or the checksum does not match.
    """
    if split not in ("train", "val"):
        msg = f"split must be 'train' or 'val', got {split!r}."
        raise ValueError(msg)
    if size not in _ARCHIVES:
        msg = f"size must be one of {sorted(_ARCHIVES)}, got {size!r}."
        raise ValueError(msg)
    remote = _ARCHIVES[size]
    root = get_data_dir() / remote.subdir / remote.filename.removesuffix(".tgz")
    if not (root / _COMPLETE).exists():
        archive = fetch(remote)
        try:
            if not (root / _COMPLETE).exists():  # another process may have extracted it meanwhile
                _extract(archive, root)
        except FileNotFoundError:  # the archive was deleted by a process that extracted it
            if not (root / _COMPLETE).exists():
                raise
        archive.unlink(missing_ok=True)  # the images are kept; the archive is not needed

    paths: list[Path] = []
    labels: list[int] = []
    for wnid, (imagenet_index, _) in sorted(IMAGENETTE_CLASSES.items()):
        files = sorted((root / root.name / split / wnid).glob("*.JPEG"))
        paths.extend(files)
        labels.extend([imagenet_index] * len(files))
    return ImageDataset(
        name=f"imagenette_{split}_{size}",
        paths=tuple(paths),
        labels=np.asarray(labels, dtype=int),
        class_names=dict(IMAGENETTE_CLASSES.values()),
    )
