"""Local cache for downloaded data files.

No data file ships with :mod:`shapiq_games`. Files are downloaded on first use from pinned
sources, verified against a SHA-256 checksum, and cached in a local directory:

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
from pathlib import Path

import requests

__all__ = ["RemoteFile", "atomic_write_bytes", "fetch", "get_data_dir"]

# Commit of mmschlk/shapiq whose tree still contains the data files that used to be bundled with
# shapiq_games. Pinning a commit keeps the bytes immutable: the files live on in git history.
PINNED_COMMIT = "5ff37e6ce6ccefe4ca8ef938d16e61da3d779242"
PINNED_DATA_URL = (
    f"https://raw.githubusercontent.com/mmschlk/shapiq/{PINNED_COMMIT}/src/shapiq_games/"
)

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
        url: The pinned download URL.
        filename: The file name in the cache directory.
        sha256: The expected SHA-256 hex digest. ``None`` means the file could not be pinned yet
            and is cached without verification.
        subdir: The sub-directory of the cache directory the file is stored in.
    """

    url: str
    filename: str
    sha256: str | None
    subdir: str = "datasets"

    @classmethod
    def pinned(cls, path: str, sha256: str, *, subdir: str = "datasets") -> RemoteFile:
        """Create a file served from the pinned commit of this repository.

        Args:
            path: The path relative to ``src/shapiq_games/`` at :data:`PINNED_COMMIT`.
            sha256: The expected SHA-256 hex digest.
            subdir: The cache sub-directory.

        Returns:
            The remote file.
        """
        return cls(
            url=PINNED_DATA_URL + path,
            filename=Path(path).name,
            sha256=sha256,
            subdir=subdir,
        )


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
    if path.exists():
        if remote.sha256 is None or _sha256_file(path) == remote.sha256:
            return path
        path.unlink()  # corrupted or outdated cache entry, download again

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

    if remote.sha256 is not None and (actual := digest.hexdigest()) != remote.sha256:
        tmp_path.unlink(missing_ok=True)
        msg = (
            f"Checksum mismatch for {remote.url}: expected {remote.sha256}, got {actual}. "
            "The file was not cached."
        )
        raise OSError(msg)
    tmp_path.chmod(0o644)  # mkstemp creates owner-only files
    tmp_path.replace(path)
    return path
