"""Run directories and manifests (plan §3). Layout: runs/<run_id>/{config.yaml, manifest.json, ...}."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml

RUNS_ROOT = Path("runs")


def run_id(group: str, n_skus: int, kappa: float, algo: str, seed: int) -> str:
    return f"{group}-N{n_skus}-k{kappa:g}-{algo}-s{seed}"


def is_done(run_dir: Path) -> bool:
    return (run_dir / "eval_test.parquet").exists()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def start_run(rid: str, config: dict, root: Path = RUNS_ROOT) -> Path:
    run_dir = root / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=True))
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    manifest = {
        "run_id": rid,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "pip_freeze": freeze.stdout.splitlines(),
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "_cpu_start": time.process_time(),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return run_dir


def finish_run(run_dir: Path, **extra) -> None:
    """Record end time, CPU seconds, and anything else (e.g. actual_env_steps=...)."""
    path = run_dir / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest["cpu_seconds"] = time.process_time() - manifest.pop("_cpu_start")
    manifest.update(extra)
    path.write_text(json.dumps(manifest, indent=2))
