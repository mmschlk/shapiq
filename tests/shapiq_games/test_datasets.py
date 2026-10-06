"""Tests for the dataset registry, the local data cache, and the example images."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

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
    def __init__(self, content: bytes) -> None:
        self.content = content

    def raise_for_status(self) -> None:
        return None


def _patch_download(monkeypatch: pytest.MonkeyPatch, content: bytes) -> list[str]:
    calls: list[str] = []

    def fake_get(url: str, timeout: int) -> _Response:
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


def test_pinned_urls_point_to_a_commit() -> None:
    remote = cache.RemoteFile.pinned("datasets/data/zoo.csv", "0" * 64)
    assert cache.PINNED_COMMIT in remote.url
    assert remote.url.endswith("/src/shapiq_games/datasets/data/zoo.csv")


def test_small_pinned_dataset_downloads() -> None:
    """End-to-end check of one small pinned file (requires network access to GitHub)."""
    dataset = load_dataset("zoo")
    assert dataset.x.shape == (101, 16)
    assert dataset.n_classes == 7
