"""Planned guard tests, one per pitfall in the team draft §7. Each owner replaces its skip with the real test.

These show up as SKIPPED in every pytest run, so nobody forgets them.
"""

import pytest

PENDING = {
    "env: gymnasium check_env; reset(demand_path) replays exactly; capacity never exceeded": "M1",
    "conformance: env step == DP transition on random (state, action, demand), several (C,L,tau)": "M1",
    "dp: state count == C(C+L+P, L+P); tiny instance vs brute force; V[0,s0] ~ simulated DP policy": "M1",
    "action mapping: a0=-1 -> 0, a0=+1 -> R; sum <= R (hypothesis); integers; even split": "M1",
    "masking: action_masks() == capacity; MaskablePPO eval never clips": "M3",
    "sb3 adapter: VecNormalize frozen at eval (training=False, norm_reward=False), deterministic=True": "M2",
    "baselines: (s,S) and newsvendor feasible + deterministic; newsvendor <= DP for N=1": "M2",
}


@pytest.mark.parametrize("name", list(PENDING))
def test_pending(name):
    pytest.skip(f"TODO ({PENDING[name]}): {name}")
