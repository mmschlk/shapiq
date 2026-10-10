"""Small stdlib fixtures exercise compression and the real Pages fetch closure."""

from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from benchmark.compress_report import compress_report, digest
from benchmark.fetch_report import fetch_report


def fixture(root: Path) -> dict:
    """Create one raw, metric and game block without scientific imports."""
    root.mkdir()
    identity = "a" * 64
    assets = {
        k: [] for k in ("raw", "metrics", "games", "details", "runs", "profiles", "summaries")
    }
    for kind in ("raw", "metrics", "games"):
        payload = {
            "snapshot_id": identity,
            "kind": kind,
            "count": 1,
            "codec": "columns-v2",
            "columns": {"id": {"values": ["original-id"]}},
        }
        path = root / f"partition-{kind}-0.json"
        path.write_text(json.dumps(payload))
        assets[kind].append(
            {
                "snapshot_id": identity,
                "kind": kind,
                "count": 1,
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "layout": "partitioned-v1",
        "snapshot_id": identity,
        "methods": {"test": {"private": False}},
        "record_count": 1,
        "game_count": 1,
        "assets": assets,
    }
    (root / "data.json").write_text(json.dumps(manifest))
    return manifest


def pin(root: Path, manifest: dict) -> dict:
    """Seal a modified public manifest for the download validator."""
    (root / "data.json").write_text(json.dumps(manifest))
    return {"url": (root / "data.json").as_uri(), "sha256": digest(root / "data.json")}


class CompressionTests(unittest.TestCase):
    """Transport changes never alter decoded payloads or descriptor identities."""

    def test_roundtrip_fetch_and_parallel_determinism(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            before = fixture(root / "a")
            fixture(root / "b")
            stats = compress_report(root / "a", workers=2)
            compress_report(root / "b")
            after = json.loads((root / "a/data.json").read_text())
            self.assertEqual(
                (root / "a/data.json").read_bytes(), (root / "b/data.json").read_bytes()
            )
            self.assertEqual(stats["blocks"], 3)
            for kind in ("raw", "metrics", "games"):
                old, new = before["assets"][kind][0], after["assets"][kind][0]
                self.assertEqual(old["sha256"], new["sha256"])
                self.assertEqual(old["bytes"], new["bytes"])
                self.assertFalse((root / "a" / old["file"]).exists())
                decoded = gzip.decompress((root / "a" / new["file"]).read_bytes())
                self.assertEqual(hashlib.sha256(decoded).hexdigest(), old["sha256"])
            result = fetch_report(pin(root / "a", after), root / "fetched")
            self.assertEqual(
                (root / "fetched/data.json").read_bytes(),
                (root / "fetched/about.json").read_bytes(),
            )
            self.assertEqual(
                result["bytes"], sum(p.stat().st_size for p in (root / "fetched").iterdir())
            )

    def test_bad_transport_and_decoded_pins(self):
        for field, value in (
            ("compressed_sha256", "f" * 64),
            ("sha256", "f" * 64),
            ("compressed_bytes", 1),
            ("bytes", 1),
            ("bytes", 64 * 1024 * 1024 + 1),
            ("encoding", "unknown"),
        ):
            with self.subTest(field=field, value=value), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture(root / "source")
                compress_report(root / "source")
                manifest = json.loads((root / "source/data.json").read_text())
                manifest["assets"]["raw"][0][field] = value
                with self.assertRaises(ValueError):
                    fetch_report(pin(root / "source", manifest), root / "bad")
                self.assertFalse((root / "bad/data.json").exists())

    def test_wrong_input_preserves_plain_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "report"
            manifest = fixture(root)
            before = (root / "data.json").read_bytes()
            descriptor = manifest["assets"]["raw"][0]
            (root / descriptor["file"]).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                compress_report(root)
            self.assertEqual((root / "data.json").read_bytes(), before)
            self.assertTrue((root / descriptor["file"]).exists())

    def test_compressed_private_columns_still_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = fixture(root / "source")
            descriptor = manifest["assets"]["raw"][0]
            path = root / "source" / descriptor["file"]
            payload = json.loads(path.read_text())
            payload["columns"]["error"] = {"values": ["private failure"]}
            path.write_text(json.dumps(payload))
            descriptor.update(bytes=path.stat().st_size, sha256=digest(path))
            pin(root / "source", manifest)
            compress_report(root / "source")
            manifest = json.loads((root / "source/data.json").read_text())
            with self.assertRaisesRegex(ValueError, "Private column"):
                fetch_report(pin(root / "source", manifest), root / "bad")


if __name__ == "__main__":
    unittest.main()
