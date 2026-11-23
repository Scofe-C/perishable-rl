"""Publish generated config files all-or-nothing, and remove the ones a generator no longer produces.

Used by scripts/make_calibrated_configs.py and scripts/make_shock_configs.py. Two failure modes it closes:
- A generator that fails halfway used to leave a mix of new and old files. Now every file is validated in a
  temp dir first, and nothing in `out_dir` changes unless all of them pass.
- A preset or SKU that disappears used to leave its old file behind, still loadable and still referenced by
  matrices. Now generated files (recognised by their header) that this run owns but did not write are deleted.
  Hand-written files, and generated files outside `owns`, are never touched.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from perinv.utils.atomic import atomic_write_text


def publish(files: dict[str, str], out_dir: str | Path, *, header: str, validate: Callable[[Path], object],
            owns: Callable[[Path], bool]) -> tuple[list[Path], list[Path]]:
    """files: {file name: body}; header (one line, ends with a newline) is prepended to each.

    validate(path) must raise for an invalid file. owns(path) says whether a file in out_dir belongs to this
    run's scope, so it may be removed when stale. Returns (written, removed).
    """
    if not header.endswith("\n") or header.count("\n") != 1:
        raise ValueError("header must be exactly one line ending with a newline")
    with tempfile.TemporaryDirectory() as tmp:
        for name, body in files.items():
            p = Path(tmp) / name
            atomic_write_text(p, header + body)
            try:
                validate(p)
            except Exception as exc:
                raise ValueError(f"generated {name} is invalid, nothing was written: {exc}") from exc
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, body in files.items():
        atomic_write_text(out / name, header + body)
        written.append(out / name)
    removed = []
    for p in sorted(out.glob("*.yaml")):
        if p.name in files or not owns(p):
            continue
        with open(p, encoding="utf-8") as fh:
            generated = fh.readline() == header
        if generated:   # unlink after closing: Windows cannot delete an open file
            p.unlink()
            removed.append(p)
    return written, removed
