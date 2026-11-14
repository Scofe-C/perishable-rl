#!/usr/bin/env bash
# Fresh Ubuntu VM (or WSL2) -> ready to run experiments. Run from the repo root.
set -euo pipefail

PY=${PY:-python3.15}
if ! command -v "$PY" >/dev/null; then
  sudo apt-get update && sudo apt-get install -y software-properties-common
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get install -y "$PY" "$PY-venv"
fi

"$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
# CPU-only torch wheel (much smaller than the default CUDA build); same version as requirements.txt.
.venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -r requirements-dev.txt -e .
.venv/bin/pytest -q
echo "OK. Activate with: source .venv/bin/activate"
