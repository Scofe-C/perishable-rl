"""Publishing a finished layer folder: swap a fully built and checked staging folder into place.

Readers see the old complete folder or the new complete folder, never a mix. Between the two renames the
folder is briefly absent (rename of a non-empty folder over another is not atomic on Windows).
"""

from __future__ import annotations

import os
import shutil
import uuid
from pathlib import Path


def new_staging(parent: Path, label: str) -> Path:
    staging = parent / f"_staging_{label}_{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    return staging


def publish_dir(staging: Path, final: Path) -> None:
    old = final.with_name(final.name + f".old_{uuid.uuid4().hex[:8]}")
    if final.exists():
        os.replace(final, old)
    os.replace(staging, final)
    shutil.rmtree(old, ignore_errors=True)
