"""Bounded stdlib fixtures for one-asset release archive transport."""

from __future__ import annotations

import json
import stat
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

from benchmark.compress_report import compress_report, digest
from benchmark.fetch_report import fetch_report

from tests.shapiq_benchmark.test_compress_report import fixture


def archive_pin(root: Path, fault: str | None = None) -> dict:
    """Package a tiny report, optionally altering its ZIP closure or transport pin."""
    source = root / "source"
    manifest = json.loads((source / "data.json").read_text())
    names = ["data.json", *(d["file"] for group in manifest["assets"].values() for d in group)]
    if fault == "missing":
        names.pop()
    path = root / "website-data.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            if fault == "symlink" and name == "data.json":
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(info, (source / name).read_bytes())
            else:
                archive.write(
                    source / name,
                    name,
                    compress_type=zipfile.ZIP_DEFLATED
                    if fault == "deflated"
                    else zipfile.ZIP_STORED,
                )
        if fault == "extra":
            archive.writestr("partition-raw-9999.json", b"extra")
        if fault == "path":
            archive.writestr("../outside.json", b"outside")
        if fault == "duplicate":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                archive.writestr("data.json", (source / "data.json").read_bytes())
    pin = {
        "url": (source / "data.json").as_uri(),
        "sha256": digest(source / "data.json"),
        "archive": {"url": path.as_uri(), "sha256": digest(path), "bytes": path.stat().st_size},
    }
    if fault == "archive_hash":
        pin["archive"]["sha256"] = "f" * 64
    elif fault == "manifest_hash":
        pin["sha256"] = "f" * 64
    elif fault == "oversize":
        pin["archive"]["bytes"] -= 1
    elif fault == "truncated":
        pin["archive"]["bytes"] += 1
    return pin


class ArchiveTests(unittest.TestCase):
    """An archive is only a delivery container for the exact existing public files."""

    def test_exact_plain_and_gzip_archive_closure(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture(root / "source")
                if compressed:
                    compress_report(root / "source")
                pin = archive_pin(root)
                individual = fetch_report(
                    {k: pin[k] for k in ("url", "sha256")}, root / "individual"
                )
                # Archive mode must not fall back to an unavailable individual asset URL.
                pin["url"] = (root / "not-uploaded/data.json").as_uri()
                result = fetch_report(pin, root / "archived")
                self.assertEqual(result, individual)
                for name in result["files"]:
                    self.assertEqual(
                        (root / "archived" / name).read_bytes(),
                        (root / "individual" / name).read_bytes(),
                    )

    def test_reject_changed_or_unsafe_archive(self):
        for fault in (
            "path",
            "extra",
            "missing",
            "symlink",
            "duplicate",
            "deflated",
            "archive_hash",
            "manifest_hash",
            "oversize",
            "truncated",
        ):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture(root / "source")
                with self.assertRaises(ValueError):
                    fetch_report(archive_pin(root, fault), root / "bad")
                self.assertFalse((root / "bad/data.json").exists())
                self.assertFalse((root / "outside.json").exists())

    def test_member_checksum_and_privacy_still_checked(self):
        for fault in ("member_hash", "private"):
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest = fixture(root / "source")
                descriptor = manifest["assets"]["raw"][0]
                if fault == "member_hash":
                    descriptor["sha256"] = "f" * 64
                else:
                    path = root / "source" / descriptor["file"]
                    payload = json.loads(path.read_text())
                    payload["columns"]["error"] = {"values": ["private failure"]}
                    path.write_text(json.dumps(payload))
                    descriptor.update(bytes=path.stat().st_size, sha256=digest(path))
                (root / "source/data.json").write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    fetch_report(archive_pin(root), root / "bad")
                self.assertFalse((root / "bad/data.json").exists())


if __name__ == "__main__":
    unittest.main()
