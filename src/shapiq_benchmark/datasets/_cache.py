"""Local cache for downloaded data files.

No data file ships with the package. Files are downloaded on first use from their original
sources, verified against a SHA-256 checksum where one is pinned, and cached in a local directory:

- ``$SHAPIQ_DATA_DIR`` if set,
- otherwise ``$XDG_CACHE_HOME/shapiq`` if ``XDG_CACHE_HOME`` is set,
- otherwise ``~/.cache/shapiq``.

Writes are atomic (temporary file and rename), so concurrent test workers never read a
half-written file. Nothing is ever written into the installed package.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

__all__ = [
    "RemoteFile",
    "atomic_write_bytes",
    "fetch",
    "get_data_dir",
    "read_table",
    "write_table",
]

_TIMEOUT_SECONDS = 120


def get_data_dir() -> Path:
    """Return the root directory of the local shapiq data cache (created on demand)."""
    if env_dir := os.environ.get("SHAPIQ_DATA_DIR"):
        root = Path(env_dir)
    elif xdg_cache := os.environ.get("XDG_CACHE_HOME"):
        root = Path(xdg_cache) / "shapiq"
    else:
        root = Path.home() / ".cache" / "shapiq"
    root.mkdir(parents=True, exist_ok=True)
    return root


@dataclass(frozen=True)
class RemoteFile:
    """A file that is downloaded on demand and cached locally.

    Attributes:
        url: The download URL.
        filename: The file name in the cache directory.
        sha256: The expected SHA-256 hex digest; a download that does not match is rejected.
        subdir: The sub-directory of the cache directory the file is stored in.
    """

    url: str
    filename: str
    sha256: str
    subdir: str = "datasets"


_CHUNK_BYTES = 1 << 20


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically via a temporary file in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        Path(tmp_name).chmod(0o644)  # mkstemp creates owner-only files
        Path(tmp_name).replace(path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def write_table(path: Path, table: pd.DataFrame) -> None:
    """Write a table as CSV atomically and losslessly (every float to 17 significant digits).

    ``DataFrame.to_csv`` can drop the last digit of a float by default; see :func:`read_table`.
    """
    buffer = StringIO()
    table.to_csv(buffer, index=False, float_format="%.17g")
    atomic_write_bytes(path, buffer.getvalue().encode("utf-8"))


def read_table(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read a table written by :func:`write_table`, parsing floats correctly rounded.

    The default fast parser of ``pd.read_csv`` is not correctly rounded for 17-digit strings.
    """
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def fetch(remote: RemoteFile) -> Path:
    """Return the local path of ``remote``, downloading and verifying it on first use.

    The download is streamed to a temporary file in the cache directory while it is hashed, and
    only moved into place once the checksum matches, so large files never sit in memory and a
    failed download leaves nothing behind.

    Args:
        remote: The file to fetch.

    Returns:
        The path of the verified file in the local cache.

    Raises:
        OSError: If the download fails or the checksum does not match.
    """
    path = get_data_dir() / remote.subdir / remote.filename
    try:
        if _sha256_file(path) == remote.sha256:
            return path
    except FileNotFoundError:  # not cached (or deleted by another process meanwhile)
        pass
    # a corrupted or outdated entry is replaced atomically below, never deleted first: another
    # process may already have replaced it with the verified file and be reading it

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    digest = hashlib.sha256()
    try:
        with (
            os.fdopen(fd, "wb") as handle,
            requests.get(remote.url, stream=True, timeout=_TIMEOUT_SECONDS) as response,
        ):
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=_CHUNK_BYTES):
                handle.write(chunk)
                digest.update(chunk)
    except requests.RequestException as error:
        tmp_path.unlink(missing_ok=True)
        msg = f"Could not download {remote.url}: {error}"
        raise OSError(msg) from error
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    if (actual := digest.hexdigest()) != remote.sha256:
        tmp_path.unlink(missing_ok=True)
        msg = (
            f"Checksum mismatch for {remote.url}: expected {remote.sha256}, got {actual}. "
            "The file was not cached."
        )
        raise OSError(msg)
    tmp_path.chmod(0o644)  # mkstemp creates owner-only files
    tmp_path.replace(path)
    return path
