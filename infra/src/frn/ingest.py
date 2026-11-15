"""Raw layer: fetch the pinned FreshRetailNet revision and verify every byte.

    python -m frn.ingest --data-root /data/freshretailnet        # in Docker
    python -m frn.ingest --data-root D:/NEU/data/freshretailnet  # on the host

Layout: <data-root>/raw/<revision>/{README.md, data/train.parquet, data/eval.parquet, _manifest.json}

- Pinned: the URL contains the commit, so an upstream update cannot change what we get.
- Verified: a file counts only if its sha256 matches configs/source.yaml. A mismatch is fatal.
- Idempotent: verified files are not downloaded again. A corrupt local file is fetched once more.
- Atomic: download to `<name>.part`, verify, then rename. A crash never leaves a "valid-looking" file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

log = logging.getLogger("frn.ingest")
DEFAULT_SOURCE = Path(__file__).resolve().parents[2] / "configs" / "source.yaml"
CHUNK = 1 << 20


class IntegrityError(RuntimeError):
    """Downloaded bytes do not match the pinned hash."""


@dataclass(frozen=True)
class Source:
    dataset: str
    revision: str
    base_url: str
    files: dict[str, dict]  # relative path -> {sha256, bytes}

    @classmethod
    def load(cls, path: str | Path = DEFAULT_SOURCE) -> Source:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(raw["dataset"], raw["revision"], raw["base_url"].rstrip("/"), raw["files"])

    def url(self, rel: str) -> str:
        return f"{self.base_url}/{self.dataset}/resolve/{self.revision}/{rel}"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "frn-ingest"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out, CHUNK)


def fetch_file(source: Source, rel: str, raw_dir: Path) -> dict:
    """Make raw_dir/rel exist with the pinned hash. Returns its manifest entry."""
    want = source.files[rel]["sha256"]
    final = raw_dir / rel
    if final.exists():
        got = sha256_of(final)
        if got == want:
            log.info("ok (already verified): %s", rel)
            return {"sha256": got, "bytes": final.stat().st_size, "action": "verified"}
        log.warning("local %s has sha256 %s, expected %s: fetching again", rel, got[:12], want[:12])
    final.parent.mkdir(parents=True, exist_ok=True)
    part = final.with_name(final.name + ".part")
    try:
        log.info("downloading %s", source.url(rel))
        _download(source.url(rel), part)
        got = sha256_of(part)
        if got != want:
            raise IntegrityError(f"{rel}: downloaded sha256 {got} != pinned {want}")
        os.replace(part, final)
    finally:
        part.unlink(missing_ok=True)
    return {"sha256": got, "bytes": final.stat().st_size, "action": "downloaded"}


def ingest(data_root: str | Path, source: Source | None = None) -> Path:
    """Fetch and verify every pinned file, then write _manifest.json. Returns the raw directory."""
    source = source or Source.load()
    raw_dir = Path(data_root) / "raw" / source.revision
    entries = {rel: fetch_file(source, rel, raw_dir) for rel in source.files}
    manifest = {"dataset": source.dataset, "revision": source.revision, "verified_at": datetime.now(UTC).isoformat(),
                "files": entries}
    tmp = raw_dir / "_manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(tmp, raw_dir / "_manifest.json")
    return raw_dir


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch + verify the pinned FreshRetailNet revision.")
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--source", default=str(DEFAULT_SOURCE))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        out = ingest(args.data_root, Source.load(args.source))
    except IntegrityError as exc:
        log.error("%s", exc)
        return 2
    print(f"raw layer verified at {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
