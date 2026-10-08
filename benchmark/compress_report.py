"""Add lossless gzip transport to a completed partitioned report before publication.

Decoded hashes, sizes, row counts and scientific identities stay unchanged.
Call before the final output hash inventory and Pages size check. No library
codec or estimator changes are required.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_BYTES = 64 * 1024 * 1024


def digest(path: Path) -> str:
    """Hash one file without loading it into memory."""
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


def require(value: object, message: str) -> None:
    """Reject incomplete or mismatched candidates."""
    if not value:
        raise ValueError(message)


def compress_report(directory: Path, *, workers: int = 1) -> dict:
    """Commit a gzip manifest only after every decoded input is authenticated."""
    directory = Path(directory)
    require(type(workers) is int and 1 <= workers <= 32, "Use 1-32 compression workers")
    manifest_path = directory / "data.json"
    require(not manifest_path.is_symlink(), "Manifest must be a regular file")
    original = manifest_path.read_bytes()
    manifest = json.loads(original)
    require(manifest.get("layout") == "partitioned-v1", "Need a partitioned report")
    descriptors = [d for group in manifest["assets"].values() for d in group]
    require(len({d["file"] for d in descriptors}) == len(descriptors), "Repeated asset filename")
    require(not any("encoding" in d for d in descriptors), "Report already has encoded assets")

    def compress(descriptor: dict) -> tuple[Path, int, int]:
        name = descriptor["file"]
        require(
            re.fullmatch(
                r"partition-(raw|metrics|games|details|runs|profiles|summaries)-\d+\.json", name
            ),
            "Unsafe partition filename",
        )
        source = directory / name
        require(source.is_file() and not source.is_symlink(), "Partition must be a regular file")
        require(
            type(descriptor["bytes"]) is int and 0 < descriptor["bytes"] <= MAX_BYTES,
            "Invalid decoded size",
        )
        require(source.stat().st_size == descriptor["bytes"], "Decoded size differs")
        output = directory / (name + ".gz")
        require(not output.exists() and not output.is_symlink(), "Encoded output already exists")
        actual, size = hashlib.sha256(), 0
        with (
            source.open("rb") as incoming,
            output.open("xb") as outgoing,
            gzip.GzipFile(filename="", mode="wb", fileobj=outgoing, mtime=0) as zipped,
        ):
            while block := incoming.read(1024 * 1024):
                size += len(block)
                require(size <= descriptor["bytes"], "Decoded input grew")
                actual.update(block)
                zipped.write(block)
        require(
            size == descriptor["bytes"] and actual.hexdigest() == descriptor["sha256"],
            "Decoded checksum/size differs",
        )
        transport_size = output.stat().st_size
        require(0 < transport_size <= MAX_BYTES, "Encoded block exceeds transport bound")
        descriptor.update(
            file=output.name,
            encoding="gzip",
            compressed_bytes=transport_size,
            compressed_sha256=digest(output),
        )
        return source, transport_size, size

    # zlib releases the GIL; threads avoid copying report blocks between processes.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        completed = list(executor.map(compress, descriptors))
    require(manifest_path.read_bytes() == original, "Manifest changed during compression")
    content = (json.dumps(manifest, separators=(",", ":"), allow_nan=False) + "\n").encode()
    # Existing consumers see either a complete plain or a complete gzip manifest.
    temporary = directory / "data.json.gzip.tmp"
    with temporary.open("xb") as stream:
        stream.write(content)
    temporary.replace(manifest_path)
    (directory / "about.json").write_bytes(content)
    for path, _encoded, _decoded in completed:
        path.unlink()
    return {
        "encoding": "gzip",
        "blocks": len(descriptors),
        "decoded_partition_bytes": sum(row[2] for row in completed),
        "compressed_partition_bytes": sum(row[1] for row in completed),
        "decoded_manifest_sha256": hashlib.sha256(original).hexdigest(),
        "manifest_sha256": hashlib.sha256(content).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    sys.stdout.write(json.dumps(compress_report(args.report, workers=args.workers)) + "\n")
