"""Planned guard tests, one per pitfall in the team draft §7. Each owner replaces its skip with the real test.

These show up as SKIPPED in every pytest run, so nobody forgets them.
"""

import pytest

PENDING = {
    "sb3 adapter: VecNormalize frozen at eval (training=False, norm_reward=False), deterministic=True": "M2",
    "baselines: (s,S) and newsvendor feasible + deterministic; newsvendor <= DP for N=1": "M2",
}


@pytest.mark.parametrize("name", list(PENDING))
def test_pending(name):
    pytest.skip(f"TODO ({PENDING[name]}): {name}")
