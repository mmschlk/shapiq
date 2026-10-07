"""Tests for the dataset registry, the local data cache, and the Imagenette images."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Self

import numpy as np
import pytest

from shapiq_games.datasets import (
    TABARENA_DATASETS,
    _cache as cache,
    get_data_dir,
    get_dataset_spec,
    list_datasets,
    load_curthvds_synthetic,
    load_dataset,
)

if TYPE_CHECKING:
    from pathlib import Path

SYNTHETIC = ["chess", "condind", "corrgroups60", "cross", "disjunct", "group",
             "independentlinear60", "random", "sphere", "xor"]  # fmt: skip


def test_registry_contains_every_dataset() -> None:
    assert len(list_datasets(kind="tabular")) == 22
    assert sorted(list_datasets(kind="synthetic")) == SYNTHETIC
    assert len(list_datasets(kind="tabarena")) == 51 == len(TABARENA_DATASETS)
    assert len(list_datasets()) == 83


def test_task_types_are_declared() -> None:
    assert get_dataset_spec("california_housing").task == "regression"
    assert get_dataset_spec("adult_census").task == "classification"
    # TabArena task types come from the official metadata, not from the labels
    assert get_dataset_spec("tabarena_wine_quality").task == "regression"
    assert get_dataset_spec("tabarena_mic").task == "classification"
    assert list_datasets(task="regression")
    assert set(list_datasets(task="regression")).isdisjoint(list_datasets(task="classification"))


def test_unknown_dataset_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dataset"):
        load_dataset("not_a_dataset")


def test_parameters_are_only_accepted_by_synthetic_datasets() -> None:
    with pytest.raises(ValueError, match="does not take parameters"):
        load_dataset("breast_cancer", n_samples=10)


@pytest.mark.parametrize("name", SYNTHETIC)
def test_synthetic_datasets_are_seeded(name: str) -> None:
    first, second = load_dataset(name), load_dataset(name)
    other_seed = load_dataset(name, random_state=1)
    np.testing.assert_array_equal(first.x, second.x)
    np.testing.assert_array_equal(first.y, second.y)
    assert not np.array_equal(first.x, other_seed.x)
    assert first.x.dtype == np.float64
    assert first.n_samples == 1000


def test_synthetic_parameters_are_recorded() -> None:
    dataset = load_dataset("xor", n_samples=200, n_irrelevant=4, random_state=3)
    assert dataset.x.shape == (200, 6)
    assert dataset.params == {"n_samples": 200, "n_irrelevant": 4, "random_state": 3}


def test_classification_labels_are_encoded() -> None:
    dataset = load_dataset("breast_cancer")
    assert dataset.task == "classification"
    assert dataset.n_classes == 2
    assert set(np.unique(dataset.y)) == {0, 1}
    assert dataset.class_names == ("0", "1")
    assert len(dataset.feature_names) == dataset.n_features == 30


def test_split_is_deterministic_and_stratified() -> None:
    dataset = load_dataset("breast_cancer")
    first = dataset.split(test_size=0.2, random_state=0)
    second = dataset.split(test_size=0.2, random_state=0)
    other = dataset.split(test_size=0.2, random_state=1)
    np.testing.assert_array_equal(first.x_test, second.x_test)
    assert not np.array_equal(first.x_test, other.x_test)
    assert first.x_train.shape[0] + first.x_test.shape[0] == dataset.n_samples
    # stratified: the class balance of the test set matches the data
    assert abs(first.y_test.mean() - dataset.y.mean()) < 0.02


def test_split_keeps_at_least_thirty_test_samples() -> None:
    dataset = load_dataset("xor", n_samples=100)
    assert dataset.split(test_size=0.05).x_test.shape[0] == 30
    with pytest.raises(ValueError, match="too few"):
        load_dataset("xor", n_samples=30).split()


def test_curthvds_synthetic_study() -> None:
    frame = load_curthvds_synthetic(n=100, d=8, random_state=0)
    assert frame.shape == (100, 10)
    assert {"Treatment", "Outcome", "Instrument1", "Confounder1"} <= set(frame.columns)
    assert frame.equals(load_curthvds_synthetic(n=100, d=8, random_state=0))
    with pytest.raises(ValueError, match="at least 4"):
        load_curthvds_synthetic(d=3)


class _Response:
    """A streamed response, delivered in small chunks."""

    def __init__(self, content: bytes) -> None:
        self.content = content

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int) -> list[bytes]:
        return [self.content[i : i + 3] for i in range(0, len(self.content), 3)]


def _patch_download(monkeypatch: pytest.MonkeyPatch, content: bytes) -> list[str]:
    calls: list[str] = []

    def fake_get(url: str, *, stream: bool, timeout: int) -> _Response:
        assert stream
        calls.append(url)
        return _Response(content)

    monkeypatch.setattr(cache.requests, "get", fake_get)
    return calls


def test_data_dir_respects_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SHAPIQ_DATA_DIR", str(tmp_path / "custom"))
    assert get_data_dir() == tmp_path / "custom"
    monkeypatch.delenv("SHAPIQ_DATA_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert get_data_dir() == tmp_path / "xdg" / "shapiq"


def test_fetch_verifies_and_caches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SHAPIQ_DATA_DIR", str(tmp_path))
    content = b"a,b\n1,2\n"
    calls = _patch_download(monkeypatch, content)
    remote = cache.RemoteFile(
        "https://example.org/f.csv", "f.csv", hashlib.sha256(content).hexdigest()
    )

    path = cache.fetch(remote)
    assert path == tmp_path / "datasets" / "f.csv"
    assert path.read_bytes() == content
    assert cache.fetch(remote) == path
    assert len(calls) == 1  # the second fetch is served from the cache
    assert not list(path.parent.glob(".*.tmp"))  # no temporary files are left behind


def test_fetch_rejects_checksum_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SHAPIQ_DATA_DIR", str(tmp_path))
    _patch_download(monkeypatch, b"tampered")
    remote = cache.RemoteFile("https://example.org/f.csv", "f.csv", "0" * 64)
    with pytest.raises(OSError, match="Checksum mismatch"):
        cache.fetch(remote)
    assert not (tmp_path / "datasets" / "f.csv").exists()


def test_fetch_replaces_corrupted_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SHAPIQ_DATA_DIR", str(tmp_path))
    content = b"good"
    calls = _patch_download(monkeypatch, content)
    (tmp_path / "datasets").mkdir()
    (tmp_path / "datasets" / "f.csv").write_bytes(b"corrupted")
    remote = cache.RemoteFile(
        "https://example.org/f.csv", "f.csv", hashlib.sha256(content).hexdigest()
    )
    assert cache.fetch(remote).read_bytes() == content
    assert len(calls) == 1


def test_cached_files_are_readable_by_others(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("SHAPIQ_DATA_DIR", str(tmp_path))
    content = b"x"
    _patch_download(monkeypatch, content)
    remote = cache.RemoteFile(
        "https://example.org/f.csv", "f.csv", hashlib.sha256(content).hexdigest()
    )
    mode = cache.fetch(remote).stat().st_mode
    assert mode & 0o044 == 0o044


def test_tabarena_imputes_categories_before_encoding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing category is imputed with the most frequent category, not a non-existent code."""
    import sys
    import types

    import pandas as pd

    x = pd.DataFrame({
        "color": pd.Categorical(["red", "red", "blue", None, "red"]),
        "size": [1.0, None, 3.0, 4.0, 5.0],
    })  # fmt: skip
    y = pd.Series(["yes", "no", "yes", "no", "yes"])

    class _Dataset:
        def get_data(self, target: str, dataset_format: str) -> tuple:
            return x, y, None, None

    openml = types.ModuleType("openml")
    openml.datasets = types.SimpleNamespace(get_dataset=lambda *_, **__: _Dataset())
    monkeypatch.setitem(sys.modules, "openml", openml)
    monkeypatch.setenv("SHAPIQ_DATA_DIR", str(tmp_path))

    dataset = load_dataset("tabarena_blood_transfusion")
    color = dataset.x[:, 0]
    assert set(np.unique(color)) == {0.0, 1.0}  # blue = 0, red = 1, no 0.5 from a median
    assert color[3] == 1.0  # the mode, red
    assert dataset.x[1, 1] == pytest.approx(3.5)  # numeric median
    assert dataset.task == "classification"
    assert (tmp_path / "tabarena" / "blood_transfusion.csv").exists()


def test_tabular_data_comes_from_original_sources() -> None:
    """No dataset is served from this repository; upstream tables declare their shape."""
    from shapiq_games.datasets import _tabular

    remotes = [*_tabular._SHAP_FILES.values(), *_tabular._UCI_FILES.values()]
    assert all("mmschlk/shapiq" not in remote.url for remote in remotes)
    for name, upstream in _tabular._UPSTREAM.items():
        assert upstream.source in ("openml", "uci", "uci_files", "sklearn"), name
        assert get_dataset_spec(name).source.split()[0] in ("OpenML", "UCI", "scikit-learn")


def test_upstream_tables_are_cached_and_shape_checked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import pandas as pd

    from shapiq_games.datasets import _tabular

    monkeypatch.setenv("SHAPIQ_DATA_DIR", str(tmp_path))
    zoo = pd.DataFrame(np.arange(101 * 16).reshape(101, 16), columns=[f"f{i}" for i in range(16)])
    zoo["target"] = np.arange(101) % 7 + 1
    downloads: list[int] = []

    class FakeUCI:
        @staticmethod
        def fetch_ucirepo(id: int) -> object:  # noqa: A002
            downloads.append(id)
            data = type(
                "Data", (), {"features": zoo.drop(columns="target"), "targets": zoo[["target"]]}
            )
            return type("Repo", (), {"data": data})

    monkeypatch.setattr(_tabular, "require", lambda package, **_: FakeUCI())
    x, y = _tabular.load_zoo()
    assert x.shape == (101, 16)
    assert sorted(set(y)) == list(range(7))
    _tabular.load_zoo()
    assert downloads == [111]  # downloaded once, then read from the cache
    assert (tmp_path / "tabular" / "zoo.csv").exists()

    downloads.clear()
    zoo = zoo.iloc[:50]  # an upstream table that changed
    with pytest.raises(ValueError, match="upstream data changed"):
        _tabular._download_table("zoo")


def test_shap_dataset_downloads() -> None:
    """End-to-end check of a checksum-pinned file from shap's data (requires access to GitHub)."""
    dataset = load_dataset("communities_and_crime")
    assert dataset.x.shape == (1994, 101)


def _fake_imagenette(tmp_path: Path, extra: dict[str, bytes] | None = None) -> Path:
    """Write a tiny archive with Imagenette's layout (and its stray non-image files)."""
    import io
    import tarfile

    from PIL import Image

    def jpeg(color: tuple[int, int, int]) -> bytes:
        buffer = io.BytesIO()
        Image.new("RGB", (8, 6), color).save(buffer, format="JPEG")
        return buffer.getvalue()

    members = {
        "imagenette2-160/.DS_Store": b"junk",
        "imagenette2-160/noisy_imagenette.csv": b"path,label\n",
        "imagenette2-160/val/n01440764/a.JPEG": jpeg((200, 0, 0)),
        "imagenette2-160/val/n03888257/b.JPEG": jpeg((0, 0, 200)),
        "imagenette2-160/train/n01440764/c.JPEG": jpeg((0, 200, 0)),
        **(extra or {}),
    }
    archive = tmp_path / "fake.tgz"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return archive


def _patch_imagenette_fetch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, archive: Path
) -> list[Path]:
    import shutil

    from shapiq_games.datasets import _imagenette

    monkeypatch.setenv("SHAPIQ_DATA_DIR", str(tmp_path / "data"))
    fetched: list[Path] = []

    def fake_fetch(remote: cache.RemoteFile) -> Path:
        target = cache.get_data_dir() / remote.subdir / remote.filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(archive, target)
        fetched.append(target)
        return target

    monkeypatch.setattr(_imagenette, "fetch", fake_fetch)
    return fetched


def test_imagenette_is_extracted_once_with_imagenet_labels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from shapiq_games.datasets import load_imagenette

    fetched = _patch_imagenette_fetch(monkeypatch, tmp_path, _fake_imagenette(tmp_path))
    images = load_imagenette(split="val", size="160px")
    assert len(images) == 2
    assert images.labels.tolist() == [0, 701]  # tench and parachute, as ImageNet classes
    assert images.label_name(1) == "parachute"
    assert images[0].shape == (6, 8, 3)
    assert images[0].dtype == np.uint8
    assert images[0][..., 0].mean() > images[0][..., 2].mean()  # the red image
    assert len(load_imagenette(split="train", size="160px")) == 1
    assert len(fetched) == 1  # extracted once; later calls read the cache
    assert not fetched[0].exists()  # the archive is deleted after extraction
    with pytest.raises(ValueError, match="split"):
        load_imagenette(split="test")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="size"):
        load_imagenette(size="64px")  # type: ignore[arg-type]


def test_imagenette_extraction_refuses_paths_outside_the_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import tarfile

    from shapiq_games.datasets import load_imagenette

    escape = {"imagenette2-160/val/n01440764/../../../../evil.JPEG": b"x"}
    _patch_imagenette_fetch(monkeypatch, tmp_path, _fake_imagenette(tmp_path, escape))
    with pytest.raises(tarfile.FilterError):
        load_imagenette(split="val", size="160px")
    assert not list((tmp_path / "data").rglob("evil.JPEG"))
    assert not (tmp_path / "data" / "imagenette" / "imagenette2-160").exists()
