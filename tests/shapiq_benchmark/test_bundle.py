"""Public reproduction archives retain identities without exposing private candidates."""

from __future__ import annotations

import hashlib
import json
import zipfile
from typing import TYPE_CHECKING

import numpy as np
import pytest

from shapiq_benchmark.bundle import bundle, main
from shapiq_benchmark.report import merge_results
from shapiq_benchmark.runner import digest, identity, load_snapshot

if TYPE_CHECKING:
    from pathlib import Path


def inputs(tmp_path: Path) -> tuple[Path, Path]:
    """Build an authenticated tiny snapshot and a compatible successful baseline."""
    root = tmp_path / "frozen"
    root.mkdir()
    np.savez(root / "game.npz", values=np.array([0.0, 1.0]))
    snapshot = {
        "schema_version": 1,
        "provenance": {"git_commit": "abc123", "shapiq": "test"},
        "suite": {"name": "tiny", "methods": ["KernelSHAP"], "budgets": [2], "seeds": [0]},
        "artifacts": {"game.npz": digest(root / "game.npz")},
        "games": [
            {
                "id": "tiny",
                "family": "test",
                "stratum": "one",
                "n_players": 1,
                "index": "SV",
                "order": 1,
                "artifact": "game.npz",
                "truth": {"coordinates": [[0]], "values": [1.0]},
            }
        ],
    }
    snapshot["snapshot_id"] = identity(snapshot)
    (root / "snapshot.json").write_text(json.dumps(snapshot))
    result = {
        **{key: snapshot[key] for key in ("schema_version", "snapshot_id", "suite", "games")},
        "snapshot_provenance": snapshot["provenance"],
        "run_provenance": {"source_sha256": "baseline-code"},
        "methods": {"KernelSHAP": {"private": False, "source_sha256": "baseline-code"}},
        "records": [
            {
                "game_id": "tiny",
                "method": "KernelSHAP",
                "budget": 2,
                "seed": 0,
                "status": "ok",
                "mse": 0.0,
                "nmse": 0.0,
            }
        ],
    }
    path = tmp_path / "results.json"
    path.write_text(json.dumps(result))
    return root, path


def test_bundle_roundtrip_and_checksums(tmp_path: Path) -> None:
    """Downloaded artifacts load unchanged and baseline rows remain mergeable."""
    snapshot, results = inputs(tmp_path)
    (snapshot / "unrelated-private.txt").write_text("do not export")
    output = tmp_path / "download.zip"
    bundle(snapshot, results, output)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == {
            "snapshot/snapshot.json",
            "snapshot/game.npz",
            "baselines/results.json",
            "README.md",
            "SHA256SUMS",
        }
        for entry in archive.read("SHA256SUMS").decode().splitlines():
            checksum, name = entry.split("  ", 1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == checksum
        readme = archive.read("README.md").decode()
        assert "abc123" in readme
        assert "--candidate local_benchmark/candidate.py:factory" in readme
        archive.extractall(tmp_path / "download")
    restored, _ = load_snapshot(tmp_path / "download/snapshot")
    original, _ = load_snapshot(snapshot)
    assert restored == original
    assert merge_results([results]) == merge_results([tmp_path / "download/baselines/results.json"])


@pytest.mark.parametrize("field", ["snapshot_id", "suite", "games", "snapshot_provenance"])
def test_mismatched_baseline_rejected(tmp_path: Path, field: str) -> None:
    """A valid results format cannot attach itself to a different frozen panel."""
    snapshot, results = inputs(tmp_path)
    data = json.loads(results.read_text())
    data[field] = "wrong"
    results.write_text(json.dumps(data))
    output = tmp_path / "download.zip"
    with pytest.raises(ValueError, match="must match"):
        bundle(snapshot, results, output)
    assert not output.exists()


@pytest.mark.parametrize("private", [True, None])
def test_private_or_unspecified_method_rejected(tmp_path: Path, private: object) -> None:
    """Only an explicit public method flag authorizes inclusion in an archive."""
    snapshot, results = inputs(tmp_path)
    data = json.loads(results.read_text())
    data["methods"]["KernelSHAP"]["private"] = private
    results.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="private candidate"):
        bundle(snapshot, results, tmp_path / "download.zip")


def test_artifact_tampering_rejected(tmp_path: Path) -> None:
    """Do not package altered artifacts even when baseline metadata still matches."""
    snapshot, results = inputs(tmp_path)
    (snapshot / "game.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        bundle(snapshot, results, tmp_path / "download.zip")


def test_failed_baseline_omits_private_exception_text(tmp_path: Path) -> None:
    """Share a failure's type without disclosing paths embedded in its message."""
    snapshot, results = inputs(tmp_path)
    data = json.loads(results.read_text())
    data["records"][0].update(
        status="failed", mse=None, nmse=None, error="ValueError: /private/path/model.py"
    )
    results.write_text(json.dumps(data))
    output = tmp_path / "download.zip"
    bundle(snapshot, results, output)
    with zipfile.ZipFile(output) as archive:
        assert all(b"/private/path" not in archive.read(name) for name in archive.namelist())
        published = json.loads(archive.read("baselines/results.json"))
    assert published["records"][0]["error_type"] == "ValueError"
    assert "error" not in published["records"][0]
    assert published["snapshot_id"] == data["snapshot_id"]
    assert json.loads(results.read_text())["records"][0]["error"] == data["records"][0]["error"]


@pytest.mark.parametrize("relative", ["../outside.npz", "/tmp/outside.npz", "a/../game.npz"])
def test_unsafe_artifact_paths_rejected(tmp_path: Path, relative: str) -> None:
    """Reject unsafe names before reading or hashing their referenced files."""
    snapshot, results = inputs(tmp_path)
    path = snapshot / "snapshot.json"
    data = json.loads(path.read_text())
    data["artifacts"] = {relative: "irrelevant"}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsafe artifact"):
        bundle(snapshot, results, tmp_path / "download.zip")


def test_symlinks_and_input_replacement_rejected(tmp_path: Path) -> None:
    """An authenticated name must not traverse a symlink or overwrite its source."""
    snapshot, results = inputs(tmp_path)
    with pytest.raises(ValueError, match="replace an input"):
        bundle(snapshot, results, results)
    artifact = snapshot / "game.npz"
    real = snapshot / "real.npz"
    artifact.rename(real)
    artifact.symlink_to(real)
    with pytest.raises(ValueError, match="without symlinks"):
        bundle(snapshot, results, tmp_path / "download.zip")


def shard_inputs(tmp_path: Path) -> tuple[Path, list[Path]]:
    """Split two seed slots into separate compatible raw result files."""
    snapshot, first_path = inputs(tmp_path)
    manifest = snapshot / "snapshot.json"
    frozen = json.loads(manifest.read_text())
    frozen["suite"]["seeds"] = [0, 1]
    frozen["snapshot_id"] = identity(
        {key: value for key, value in frozen.items() if key != "snapshot_id"}
    )
    manifest.write_text(json.dumps(frozen))
    first = json.loads(first_path.read_text())
    first.update(snapshot_id=frozen["snapshot_id"], suite=frozen["suite"])
    first_path.write_text(json.dumps(first))
    second = json.loads(json.dumps(first))
    second["records"][0].update(
        seed=1, status="failed", mse=None, nmse=None, error="ValueError: /private/path/shard.py"
    )
    second_path = tmp_path / "second.json"
    second_path.write_text(json.dumps(second))
    return snapshot, [first_path, second_path]


def test_shard_bundle_cli_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple CLI inputs produce separately reusable, sanitized, checksummed shards."""
    snapshot, paths = shard_inputs(tmp_path)
    output = tmp_path / "download.zip"
    monkeypatch.setattr(
        "sys.argv",
        [
            "bundle",
            "--snapshot",
            str(snapshot),
            "--results",
            *map(str, paths),
            "--output",
            str(output),
        ],
    )
    main()
    with zipfile.ZipFile(output) as archive:
        assert "baselines/shard-0000.json" in archive.namelist()
        assert "baselines/shard-0001.json" in archive.namelist()
        assert "baselines/results.json" not in archive.namelist()
        assert all(b"/private/path" not in archive.read(name) for name in archive.namelist())
        assert (
            "--results local_benchmark/bundle/baselines/*.json"
            in archive.read("README.md").decode()
        )
        for entry in archive.read("SHA256SUMS").decode().splitlines():
            checksum, name = entry.split("  ", 1)
            assert hashlib.sha256(archive.read(name)).hexdigest() == checksum
        archive.extractall(tmp_path / "download")
    merged = merge_results(sorted((tmp_path / "download/baselines").glob("*.json")))
    assert len(merged["records"]) == 2
    assert {row["seed"] for row in merged["records"]} == {0, 1}
    assert merged["records"][1]["error_type"] == "ValueError"
    assert json.loads(paths[1].read_text())["records"][0]["error"].endswith(
        "/private/path/shard.py"
    )


@pytest.mark.parametrize("problem", ["snapshot", "private", "source", "duplicate"])
def test_later_shard_validation(tmp_path: Path, problem: str) -> None:
    """Validate every shard, including cross-file conflicts, before creating an archive."""
    snapshot, paths = shard_inputs(tmp_path)
    second = json.loads(paths[1].read_text())
    if problem == "snapshot":
        second["snapshot_provenance"] = {"git_commit": "other"}
    elif problem == "private":
        second["methods"]["KernelSHAP"]["private"] = True
    elif problem == "source":
        second["methods"]["KernelSHAP"]["source_sha256"] = "different"
    else:
        second["records"][0]["seed"] = 0
    paths[1].write_text(json.dumps(second))
    output = tmp_path / "download.zip"
    with pytest.raises(ValueError):
        bundle(snapshot, paths, output)
    assert not output.exists()


def test_shard_inputs_cannot_be_overwritten(tmp_path: Path) -> None:
    """The output protection includes later shards, not just the first result file."""
    snapshot, paths = shard_inputs(tmp_path)
    original = paths[1].read_bytes()
    with pytest.raises(ValueError, match="replace an input"):
        bundle(snapshot, paths, paths[1])
    assert paths[1].read_bytes() == original
    with pytest.raises(ValueError, match="At least one result"):
        bundle(snapshot, [], tmp_path / "download.zip")
