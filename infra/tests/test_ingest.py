"""Raw layer: pinned, verified, idempotent, atomic. Runs offline against a file:// fake upstream."""

import hashlib
import json

import pytest

from frn import ingest as ing
from frn.ingest import IntegrityError, Source, ingest


@pytest.fixture
def upstream(tmp_path):
    """A fake Hugging Face layout on disk: <base>/<dataset>/resolve/<rev>/<file>."""
    rev, files = "abc123", {"README.md": b"hello", "data/train.parquet": b"PAR1" * 1000}
    base = tmp_path / "hf"
    for rel, data in files.items():
        p = base / "org" / "ds" / "resolve" / rev / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    pins = {rel: {"sha256": hashlib.sha256(d).hexdigest(), "bytes": len(d)} for rel, d in files.items()}
    return Source("org/ds", rev, base.as_uri(), pins), tmp_path / "data"


def test_fetch_verify_and_manifest(upstream):
    src, root = upstream
    raw = ingest(root, src)
    assert (raw / "data" / "train.parquet").read_bytes() == b"PAR1" * 1000
    m = json.loads((raw / "_manifest.json").read_text())
    assert m["revision"] == "abc123" and {e["action"] for e in m["files"].values()} == {"downloaded"}
    assert not list(raw.rglob("*.part")) and not list(raw.rglob("*.tmp"))


def test_second_run_downloads_nothing(upstream, monkeypatch):
    src, root = upstream
    ingest(root, src)
    monkeypatch.setattr(ing, "_download", lambda *a: pytest.fail("should not download verified files"))
    m = json.loads((ingest(root, src) / "_manifest.json").read_text())
    assert {e["action"] for e in m["files"].values()} == {"verified"}


def test_corrupt_local_file_is_fetched_again(upstream):
    src, root = upstream
    raw = ingest(root, src)
    (raw / "README.md").write_bytes(b"tampered")
    m = json.loads((ingest(root, src) / "_manifest.json").read_text())
    assert m["files"]["README.md"]["action"] == "downloaded"
    assert (raw / "README.md").read_bytes() == b"hello"


def test_hash_mismatch_is_fatal_and_leaves_nothing(upstream):
    src, root = upstream
    bad = Source(src.dataset, src.revision, src.base_url,
                 src.files | {"data/train.parquet": {"sha256": "0" * 64, "bytes": 4000}})
    with pytest.raises(IntegrityError, match="pinned"):
        ingest(root, bad)
    raw = root / "raw" / "abc123"
    assert not (raw / "data" / "train.parquet").exists()
    assert not list(raw.rglob("*.part")) and not (raw / "_manifest.json").exists()


def test_pinned_source_config_is_complete():
    src = Source.load()
    assert len(src.revision) == 40 and src.dataset == "Dingdong-Inc/FreshRetailNet-50K"
    for rel, pin in src.files.items():
        assert len(pin["sha256"]) == 64 and pin["bytes"] > 0, rel
    assert src.url("data/train.parquet").endswith(f"/resolve/{src.revision}/data/train.parquet")
