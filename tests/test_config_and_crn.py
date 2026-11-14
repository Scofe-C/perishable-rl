from pathlib import Path

import numpy as np
import pytest

from perinv.data.crn import generate_paths, load_paths, write_paths
from perinv.env.config import load_env_config

ENV_DIR = Path(__file__).parents[1] / "configs" / "env"


@pytest.mark.parametrize("path", sorted(ENV_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_env_configs_load(path):
    cfg = load_env_config(path)
    assert cfg.pipeline_slots == cfg.lead_time
    assert cfg.capacity > 0


def test_capacity_does_not_change_demand_hash():
    a, b = load_env_config(ENV_DIR / "n1_k06.yaml"), load_env_config(ENV_DIR / "n1_k10.yaml")
    assert a.demand_hash() == b.demand_hash()
    assert a.config_hash() != b.config_hash()


def test_crn_reproducible_and_splits_disjoint():
    cfg = load_env_config(ENV_DIR / "n2_k06.yaml")
    a = generate_paths(cfg, "test", 5, root_seed=1)
    assert a.shape == (5, cfg.horizon, cfg.n_skus) and a.dtype == np.int32
    assert np.array_equal(a, generate_paths(cfg, "test", 5, root_seed=1))
    # path k does not depend on how many paths are generated
    assert np.array_equal(a[:3], generate_paths(cfg, "test", 3, root_seed=1))
    assert not np.array_equal(a, generate_paths(cfg, "tune", 5, root_seed=1))


def test_crn_manifest_detects_tampering(tmp_path):
    cfg = load_env_config(ENV_DIR / "n1_k06.yaml")
    out = write_paths(cfg, tmp_path, root_seed=1, sizes={"tune": 3, "test": 4})
    assert load_paths(cfg, "test", tmp_path).shape[0] == 4
    np.savez_compressed(out / "test.npz", demand=np.zeros((4, cfg.horizon, 1), np.int32))
    with pytest.raises(ValueError, match="manifest hash"):
        load_paths(cfg, "test", tmp_path)
