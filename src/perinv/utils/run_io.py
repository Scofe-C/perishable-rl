"""Run identity, manifests and status. Contract: docs/WAREHOUSE.md §2 and §4.

    tk = train_key(cfg, policy="ppo", hparams=hp, seed=0, train_env_steps=200_000)
    rk = run_key(tk, demand_hash=cfg.demand_hash(), crn_test_sha256=sha)
    with run_context(RUNS_ROOT, run_name("A", 1, 0.6, "ppo", 0), run_key=rk, train_key=tk,
                     config=resolved_config, meta=describe_env(cfg) | {"group": "A", ...}) as run:
        run.stage = "train"; ...; run.update(train={"train_key": tk, "status": "done", ...})
        run.stage = "eval";  ...; run.log("eval_done", rows=...)
    # on success: status=done; on any exception: status=failed + failure.json, then re-raised

Owner: Member 3.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import pyarrow.parquet as pq
import yaml

from perinv.env.config import EnvConfig, stable_hash
from perinv.utils.atomic import atomic_write_text, retry_on_permission_error

RUNS_ROOT = Path("runs")
MANIFEST_SCHEMA_VERSION = 1
EVAL_FILE = "eval_test.parquet"
REQUIRED_META = ("group", "policy", "seed", "n_skus", "kappa", "capacity", "horizon", "env",
                 "config_hash", "demand_hash", "crn_test_sha256", "train_env_steps")

log = logging.getLogger("perinv.run")


# ---------------------------------------------------------------- identity

def run_name(group: str, n_skus: int, kappa: float, policy: str, seed: int) -> str:
    """Readable folder name, e.g. A-N1-k0.6-ppo-s0. Not an identity: the same name can hold a stale run."""
    return f"{group}-N{n_skus}-k{kappa:g}-{policy}-s{seed}"


def event_order_version() -> str:
    """SPEC §3 version from the env package. 'unversioned' until M1 adds perinv.env.EVENT_ORDER_VERSION."""
    import perinv.env as env_pkg

    return str(getattr(env_pkg, "EVENT_ORDER_VERSION", "unversioned"))


def train_key(cfg: EnvConfig, *, policy: str, hparams: dict, seed: int, train_env_steps: int | None) -> str:
    """Everything that changes the trained model, and nothing else (no git SHA, host or time)."""
    return stable_hash({"config_hash": cfg.config_hash(), "policy": policy, "hparams": hparams, "seed": seed,
                        "train_env_steps": train_env_steps, "event_order_version": event_order_version()})


def run_key(train_key: str, *, demand_hash: str, crn_test_sha256: str, eval_config_hash: str | None = None) -> str:
    """train_key + what changes the evaluation output. The key stored on every fact_step row.

    demand_hash / crn_test_sha256 / eval_config_hash describe the world the policy is *evaluated* in. For a
    cross-evaluation (built on config A, evaluated on config B) they come from B, so it never shares a key with
    the native run on A.
    """
    from perinv.eval.writer import EVAL_SCHEMA_VERSION

    return stable_hash({"train_key": train_key, "demand_hash": demand_hash, "crn_test_sha256": crn_test_sha256,
                        "eval_config_hash": eval_config_hash, "eval_schema_version": EVAL_SCHEMA_VERSION})


def describe_env(cfg: EnvConfig) -> dict:
    """The env part of the manifest (WAREHOUSE.md §4.2), including per-SKU cost lists for the SQL views."""
    env = asdict(cfg)
    env.pop("name")
    return {"n_skus": cfg.n_skus, "kappa": cfg.kappa, "capacity": cfg.capacity, "horizon": cfg.horizon,
            "env": env, "config_hash": cfg.config_hash(), "demand_hash": cfg.demand_hash()}


# ---------------------------------------------------------------- status

def read_manifest(run_dir: Path) -> dict | None:
    path = Path(run_dir) / "manifest.json"
    try:
        return json.loads(retry_on_permission_error(lambda: path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return None


def run_status(run_dir: Path, expected_run_key: str) -> str:
    """'missing' | 'running' | 'failed' | 'stale' | 'done' (WAREHOUSE.md §4.3).

    'running' means a crash or a run in progress; the runner treats everything except 'done' as to-do.
    """
    m = read_manifest(run_dir)
    if m is None:
        return "missing"
    if m.get("run_key") != expected_run_key:
        return "stale"
    if m.get("status") != "done":
        return m.get("status", "running")
    f = Path(run_dir) / EVAL_FILE
    if not f.exists():
        return "stale"
    file_meta = pq.read_schema(f).metadata or {}
    if file_meta.get(b"perinv.run_key", b"").decode() != expected_run_key:
        return "stale"
    return "done"


def is_done(run_dir: Path, expected_run_key: str) -> bool:
    return run_status(run_dir, expected_run_key) == "done"


def train_reusable(run_dir: Path, expected_train_key: str) -> bool:
    """True if a finished training with the same train_key is recorded, so only evaluation has to run."""
    t = (read_manifest(run_dir) or {}).get("train") or {}
    return t.get("train_key") == expected_train_key and t.get("status") == "done"


def find_trained(runs_root: Path, expected_train_key: str, prefer: Path | None = None) -> Path | None:
    """A run directory holding a finished training with this train_key: `prefer` first, then any run.

    Lets a cross-evaluation reuse the model of the matching native run instead of training it again.
    """
    if prefer is not None and train_reusable(prefer, expected_train_key):
        return Path(prefer)
    for manifest in sorted(Path(runs_root).glob("*/manifest.json")):
        if train_reusable(manifest.parent, expected_train_key):
            return manifest.parent
    return None


# ---------------------------------------------------------------- lineage helpers

def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def installed_packages() -> list[str]:
    """`name==version` for every installed distribution. Uses importlib.metadata, so it works in venvs
    without pip (uv creates those; `pip freeze` would silently return nothing there)."""
    pkgs = {f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions() if d.metadata["Name"]}
    return sorted(pkgs, key=str.lower)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- the run context

class DirtyTreeError(RuntimeError):
    """Official runs need a clean git tree, so git_sha fully describes the code."""


@dataclass
class Run:
    dir: Path
    name: str
    run_key: str
    train_key: str
    stage: str = "setup"
    manifest: dict = field(default_factory=dict)

    def update(self, **fields) -> None:
        """Merge fields into manifest.json (atomic rewrite)."""
        self.manifest.update(fields)
        atomic_write_text(self.dir / "manifest.json", json.dumps(self.manifest, indent=2, default=str))

    def log(self, event: str, **fields) -> None:
        """Append one JSON line to log.jsonl; every line carries run_key, stage and a UTC timestamp."""
        line = {"ts": _now(), "run_key": self.run_key, "run_name": self.name, "stage": self.stage,
                "event": event, **fields}
        with open(self.dir / "log.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, default=str) + "\n")
        log.info("%s %s %s", self.name, event, fields)


@contextmanager
def run_context(root: Path, name: str, *, run_key: str, train_key: str, config: dict, meta: dict,
                official: bool = False) -> Iterator[Run]:
    """Mark the run `running`, yield it, then mark it `done`, or `failed` (with failure.json) and re-raise."""
    missing = [k for k in REQUIRED_META if k not in meta]
    if missing:
        raise ValueError(f"manifest meta is missing {missing} (WAREHOUSE.md §4.2)")
    dirty = bool(_git("status", "--porcelain"))
    if official and dirty:
        raise DirtyTreeError("official run on a dirty git tree: commit or stash first")

    run_dir = Path(root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    previous = read_manifest(run_dir) or {}
    (run_dir / "failure.json").unlink(missing_ok=True)
    for leftover in run_dir.glob("*.tmp"):  # from a process that was killed mid-write
        leftover.unlink()

    run = Run(dir=run_dir, name=name, run_key=run_key, train_key=train_key)
    atomic_write_text(run_dir / "config.yaml", yaml.safe_dump(config, sort_keys=True))
    run.update(
        schema_version=MANIFEST_SCHEMA_VERSION, run_name=name, run_key=run_key, train_key=train_key,
        status="running", **meta,
        actual_env_steps=None,
        git_sha=_git("rev-parse", "HEAD"), git_dirty=dirty, official=official,
        python=sys.version.split()[0], platform=platform.platform(), host=platform.node(),
        pip_freeze=installed_packages(), start_time=_now(),
        # keep a finished training if it is still valid, so a stale eval does not force retraining
        train=previous.get("train") if (previous.get("train") or {}).get("train_key") == train_key else None,
    )
    cpu0, wall0 = time.process_time(), time.perf_counter()
    run.log("start")
    try:
        yield run
        # Finishing is protected too: if marking the run done fails, it must end up "failed", not "running".
        run.stage = "finish"
        run.update(status="done", end_time=_now(), cpu_seconds=time.process_time() - cpu0,
                   wall_seconds=time.perf_counter() - wall0,
                   artifacts=sorted(p.name for p in run_dir.iterdir() if p.is_file() and not p.name.endswith(".tmp")))
    except BaseException as exc:  # includes KeyboardInterrupt: an interrupted run must not look done
        error = {"run_key": run_key, "run_name": name, "stage": run.stage, "type": type(exc).__name__,
                 "message": str(exc)}
        atomic_write_text(run_dir / "failure.json", json.dumps(error, indent=2))
        run.update(status="failed", error=error, end_time=_now(),
                   cpu_seconds=time.process_time() - cpu0, wall_seconds=time.perf_counter() - wall0)
        run.log("failed", type=error["type"], message=error["message"])
        raise
    run.log("done")
