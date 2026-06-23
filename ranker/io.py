"""Streaming I/O for the candidate pool."""

from __future__ import annotations

import gzip
import io
from typing import Iterator

try:  # orjson is ~2-3x faster than stdlib json on this workload
    import orjson as _json

    def _loads(b: bytes | str):
        return _json.loads(b)

except ImportError:  # pragma: no cover - fallback path
    import json as _json

    def _loads(b: bytes | str):
        return _json.loads(b)


def _open_text(path: str) -> io.TextIOBase:
    """Open .jsonl or .jsonl.gz transparently as a text stream."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def iter_candidates(path: str) -> Iterator[dict]:
    """Yield one candidate dict per non-blank line. Memory stays O(1) in the pool size."""
    with _open_text(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield _loads(line)


def count_candidates(path: str) -> int:
    """Cheap line count (used for sanity checks); streams, does not parse JSON."""
    n = 0
    with _open_text(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n
