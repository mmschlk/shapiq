"""Fetch pinned public report assets for Pages, validating each file before copying.

Uses only the standard library so Pages does not need the benchmark environment.
This checks packaging; scientific release audits remain a separate prerequisite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import urlopen

MAX_BLOCK_BYTES = 64 * 1024 * 1024
MAX_SITE_BYTES = 950 * 1024 * 1024  # Reserve room beneath Pages' limit for UI assets.
KINDS = {"raw", "metrics", "games", "details", "runs", "profiles", "summaries"}


def require(condition: object, message: str) -> None:
    """Reject incomplete or unexpected public packaging."""
    if not condition:
        raise ValueError(message)


def parse(content: bytes) -> dict:
    """Keep nonfinite numbers out of public JSON, including exponent overflow."""

    def number(value: str) -> float:
        parsed = float(value)
        require(math.isfinite(parsed), "Nonfinite public JSON number")
        return parsed

    return json.loads(content, parse_float=number, parse_constant=number)


def validate_columns(part: dict) -> None:
    """Validate the wire codec and private fields without retaining decoded rows."""
    count, columns = part["count"], part["columns"]
    require(type(count) is int and count >= 0, "Invalid row count")
    require(isinstance(columns, dict) and (count == 0 or bool(columns)), "Missing columns")
    decoded = []
    for field, column in columns.items():
        values, missing = column["values"], column.get("missing", [])
        require(isinstance(values, list) and len(values) == count, "Column length mismatch")
        require(
            isinstance(missing, list)
            and all(type(i) is int and 0 <= i < count for i in missing)
            and len(set(missing)) == len(missing),
            "Invalid missing positions",
        )
        dictionary = column.get("dictionary")
        require("dictionary" not in column or isinstance(dictionary, list), "Invalid dictionary")
        absent = set(missing)
        if dictionary is not None:
            require(
                all(
                    i in absent or v is None or (type(v) is int and 0 <= v < len(dictionary))
                    for i, v in enumerate(values)
                ),
                "Invalid dictionary reference",
            )
        decoded.append((field, values, absent, dictionary))
    if part.get("kind") != "details":
        return
    for i in range(count):
        row = {
            key: dictionary[values[i]]
            if dictionary is not None and values[i] is not None
            else values[i]
            for key, values, absent, dictionary in decoded
            if i not in absent
        }
        if row.get("type") == "game":
            if "fragment" in row:
                path = row["fragment"]["path"]
                require(
                    not path or path[0] not in {"truth", "artifact"},
                    "Private fragmented game fields",
                )
                if not path and "value" in row:
                    require(
                        not {"truth", "artifact"}.intersection(row["value"]), "Private game fields"
                    )
            else:
                require(not {"truth", "artifact"}.intersection(row["value"]), "Private game fields")


def download(url: str, sha256: str, expected_size: int | None = None) -> bytes:
    """Bound transfer size and authenticate the bytes before parsing them."""
    require(bool(re.fullmatch(r"[0-9a-f]{64}", sha256)), "Invalid asset checksum")
    limit = expected_size if expected_size is not None else MAX_BLOCK_BYTES
    require(isinstance(limit, int) and 0 < limit <= MAX_SITE_BYTES, "Invalid asset size")
    with urlopen(url, timeout=120) as response:  # noqa: S310 -- pinned public report URL
        content = response.read(limit + 1)
    require(len(content) <= limit, "Asset exceeds its declared size")
    require(expected_size is None or len(content) == expected_size, "Truncated asset")
    require(hashlib.sha256(content).hexdigest() == sha256, "Dataset checksum mismatch")
    return content


def fetch_report(source: dict, output: Path) -> dict:
    """Write only the authenticated closure of a legacy or partitioned report."""
    payload = download(source["url"], source["sha256"])
    data = parse(payload)
    require(data["schema_version"] == 1, "Unknown benchmark schema")
    require(
        bool(data["methods"]) and not any(m.get("private", True) for m in data["methods"].values()),
        "Private or absent public methods",
    )
    require(
        not output.exists() or not any(output.iterdir()), "Use a fresh report staging directory"
    )
    output.mkdir(parents=True, exist_ok=True)
    total = len(payload)
    files = {"data.json", "about.json"}
    partitioned = data.get("layout") == "partitioned-v1"
    if partitioned:
        require(
            isinstance(data["snapshot_id"], str)
            and bool(re.fullmatch(r"[0-9a-f]{64}", data["snapshot_id"])),
            "Invalid snapshot identity",
        )
        require(
            all(
                type(data[key]) is int and 0 <= data[key] <= 2**53 - 1
                for key in ("record_count", "game_count")
            ),
            "Invalid manifest counts",
        )
        require(set(data["assets"]) == KINDS, "Unknown partition kinds")
        descriptors = [d for kind in sorted(KINDS) for d in data["assets"][kind]]
        for kind in KINDS:
            require(all(d["kind"] == kind for d in data["assets"][kind]), "Wrong asset group")
    else:
        require(not data.get("layout"), "Unknown report layout")
        require(bool(data["records"] or data.get("record_shards")), "Empty public report")
        require(
            all(not {"truth", "artifact"}.intersection(g) for g in data["games"]),
            "Private game fields",
        )
        require(
            all(not {"estimate", "error", "truth"}.intersection(r) for r in data["records"]),
            "Private record fields",
        )
        descriptors = data.get("record_shards", [])
    for descriptor in descriptors:
        name = descriptor["file"]
        pattern = (
            r"partition-(?:raw|metrics|games|details|runs|profiles|summaries)-\d+\.json"
            if partitioned
            else r"records-[a-z-]+-\d+\.json"
        )
        require(
            bool(re.fullmatch(pattern, name)) and name not in files,
            "Invalid or repeated asset filename",
        )
        if partitioned:
            require(
                type(descriptor["count"]) is int and 0 <= descriptor["count"] <= 2**53 - 1,
                "Invalid descriptor row count",
            )
            require(type(descriptor["bytes"]) is int, "Invalid descriptor byte size")
            require(name.startswith(f"partition-{descriptor['kind']}-"), "Wrong partition filename")
            require(descriptor["snapshot_id"] == data["snapshot_id"], "Wrong descriptor snapshot")
            require(0 < descriptor["bytes"] <= MAX_BLOCK_BYTES, "Oversized block")
        content = download(
            urljoin(source["url"], name),
            descriptor["sha256"],
            descriptor.get("bytes") if partitioned else None,
        )
        total += len(content)
        require(total <= MAX_SITE_BYTES, "Report exceeds the Pages packaging limit")
        part = parse(content)
        require(part["snapshot_id"] == data["snapshot_id"], "Wrong asset snapshot")
        require(
            part["codec"] == ("columns-v2" if partitioned else "columns-v1"), "Wrong column codec"
        )
        keys = (
            ("count", "kind", "target", "family", "method", "objects", "selectors")
            if partitioned
            else ("count", "target")
        )
        require(all(part.get(k) == descriptor.get(k) for k in keys), "Asset descriptor mismatch")
        require(
            not {"estimate", "error", "truth", "artifact"}.intersection(part["columns"]),
            "Private column fields",
        )
        validate_columns(part)
        (output / name).write_bytes(content)
        files.add(name)
    if partitioned:
        for kind, count in [
            ("raw", data["record_count"]),
            ("metrics", data["record_count"]),
            ("games", data["game_count"]),
        ]:
            require(
                sum(d["count"] for d in data["assets"][kind]) == count, "Incomplete partition grid"
            )
        about = payload
    else:
        keys = ("schema_version", "snapshot_id", "snapshot_provenance", "suite", "games")
        about = (
            json.dumps({key: data[key] for key in keys}, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode()
    require(
        total + len(about) <= MAX_SITE_BYTES, "Report metadata exceeds the Pages packaging limit"
    )
    (output / "about.json").write_bytes(about)
    (output / "data.json").write_bytes(payload)
    return {
        "bytes": total + len(about),
        "files": sorted(files),
        "layout": data.get("layout", "legacy"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    sys.stdout.write(
        json.dumps(fetch_report(json.loads(args.source.read_text()), args.output)) + "\n"
    )
