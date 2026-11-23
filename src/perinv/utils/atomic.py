"""Atomic file writes: write to a temp file unique to the writer (`<name>.<pid>.<id>.tmp`), then rename.
Readers never see a half-written file, and concurrent writers of one path never share a temp file."""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TypeVar

T = TypeVar("T")

# On Windows, renaming over a file that another process has open (even just reading it) fails with
# PermissionError. Parallel runner workers read each other's manifests, so this happens in practice
# (seen as a run stuck at status "running"). Readers hold files for milliseconds: a short retry fixes it.
_RETRIES = 20
_FIRST_DELAY = 0.01


def retry_on_permission_error(fn: Callable[[], T]) -> T:
    """Call fn, retrying a few times on PermissionError (Windows sharing violations). Other errors pass through."""
    delay = _FIRST_DELAY
    for attempt in range(_RETRIES):
        try:
            return fn()
        except PermissionError:
            if attempt == _RETRIES - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 0.2)
    raise AssertionError("unreachable")


@contextmanager
def atomic_path(final: str | Path) -> Iterator[Path]:
    """Yield a temp path next to `final`; on success rename it to `final`, on error delete it.

    os.replace is atomic on the same filesystem (POSIX and Windows), so `final` is either the old file,
    the complete new file, or absent. It is never a partial file.
    """
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f"{final.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        yield tmp
        retry_on_permission_error(lambda: os.replace(tmp, final))
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_text(final: str | Path, text: str) -> None:
    with atomic_path(final) as tmp:
        tmp.write_text(text, encoding="utf-8", newline="\n")   # LF on every OS (.gitattributes)
