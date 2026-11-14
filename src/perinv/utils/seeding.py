"""Call `init_process(seed)` first thing in every entry point (inside `if __name__ == "__main__":`)."""

from __future__ import annotations

import os
import random

import numpy as np


def init_process(seed: int) -> None:
    # One thread per process: runs are launched in parallel and would otherwise fight over cores.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    set_all_seeds(seed)


def set_all_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
