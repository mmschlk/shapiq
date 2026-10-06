"""Stage a frozen dependency archive on each node, checking its original inventory.

Run with the pinned interpreter's -S option: only the standard library is needed.
The archive is made once inside an accounted allocation, never an installation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import socket
import subprocess
import tarfile
from pathlib import Path


def digest(path: Path) -> str:
    """Hash a file without loading its contents into memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify(root: Path, expected: dict) -> None:
    """Check every copied file, including package metadata and startup hooks."""
    found = set()
    for base, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(base) / name
            if not path.is_file() and not path.is_symlink():
                continue
            relative = str(path.relative_to(root))
            found.add(relative)
            entry = expected.get(relative)
            if path.is_symlink():
                actual = {"link": os.readlink(path)}
            else:
                actual = {"sha256": digest(path), "bytes": path.stat().st_size}
            if actual != entry:
                msg = f"Dependency bytes differ: {relative}"
                raise ValueError(msg)
    if found != set(expected):
        msg = "Dependency file inventory differs"
        raise ValueError(msg)


def main() -> None:
    """Create the shared archive or stage its authenticated bytes on one node."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--archive-source", type=Path)
    parser.add_argument(
        "--local-root",
        type=Path,
        default=Path("/tmp/rwitter-shapiq-comprehensive-environment"),  # noqa: S108 -- owner checked
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Dependency staging requires an accounted Slurm allocation")
    receipt = json.loads(args.receipt.read_text())
    expected = receipt["files"]
    if args.create:
        if args.archive.exists():
            raise FileExistsError(args.archive)
        source = args.archive_source or Path(receipt["source"])
        verify(source, expected)
        temporary = args.archive.with_suffix(".partial")
        subprocess.run(  # noqa: S603 -- fixed tool and local campaign paths
            ["/usr/bin/tar", "-cf", str(temporary), "-C", str(source), "."], check=True
        )
        temporary.rename(args.archive)
        args.archive.with_suffix(".sha256").write_text(digest(args.archive) + "\n")
    else:
        root = args.local_root
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink() or root.stat().st_uid != os.getuid():
            msg = "Node-local directory ownership mismatch"
            raise ValueError(msg)
        filesystem = subprocess.check_output(  # noqa: S603 -- fixed read-only tool
            ["/usr/bin/findmnt", "-T", str(root), "-n", "-o", "FSTYPE"], text=True
        ).strip()
        if filesystem not in {"xfs", "ext4", "tmpfs"}:
            msg = "Dependencies require a node-local filesystem"
            raise ValueError(msg)
        with (root / "stage.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            destination = root / "site-packages"
            if not destination.exists():
                size = sum(entry.get("bytes", 0) for entry in expected.values())
                if shutil.disk_usage(root).free < size + 2 * 1024**3:
                    msg = "Insufficient node-local space"
                    raise ValueError(msg)
                temporary = root / ("copy-" + os.environ["SLURM_JOB_ID"])
                temporary.mkdir()
                with tarfile.open(args.archive) as archive:
                    archive.extractall(temporary, filter="data")
                verify(temporary, expected)
                temporary.rename(destination)
            else:
                verify(destination, expected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(
            {
                "job": os.environ["SLURM_JOB_ID"],
                "host": socket.gethostname(),
                "original_receipt_sha256": digest(args.receipt),
                "archive": str(args.archive),
                "created": args.create,
                "destination": str(args.local_root / "site-packages"),
            },
            handle,
        )
        handle.write("\n")


if __name__ == "__main__":
    main()
