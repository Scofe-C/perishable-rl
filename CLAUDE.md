# CLAUDE.md

A three-person course project. It compares deep RL algorithms (SB3: PPO, A2C, SAC, DQN; sb3-contrib:
MaskablePPO) on a perishable, multi-SKU inventory env with shared capacity, against exact DP (N=1) and
heuristics (N≥2).

## Read before doing anything
1. [docs/STATUS.md](docs/STATUS.md): what's done, the current step, and blockers.
2. [docs/SPEC.md](docs/SPEC.md): the event order, action mappings, CRN format and Policy interface.
3. [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): the experiment matrix and fairness rules.
4. README.md: the owner of each directory (M1 env/DP, M2 PPO/A2C + baselines, M3 SAC/DQN/MaskablePPO + CRN/eval).

Ask the user which member they are if it matters for the task. Stay inside that member's area unless asked.

## Hard rules
- **SPEC.md is the source of truth.** Do not change env behaviour, the action mapping, the CRN format or
  `configs/budget.yaml` on your own. Those changes need all three members. Propose them, and if agreed,
  log them in docs/DECISIONS.md.
- **Env and DP share `env/dynamics.py`.** Never add a second transition implementation without a
  conformance test against it.
- **One evaluation path.** Every agent is scored by `perinv.eval.evaluate` on the `test` CRN paths.
  Hyperparameter tuning uses the `tune` paths only. Never tune on `test`.
- **Don't rewrite RL algorithms.** Use SB3 / sb3-contrib with hparams from configs.
- **Pitfall guards from the team draft:**
  - Scale before the sigmoid (`sigmoid(4·a0)`).
  - Freeze VecNormalize at eval (`training=False`, `norm_reward=False`).
  - Evaluate MaskablePPO with masks.
  - Call `init_process(seed)`, which sets `torch.set_num_threads(1)`, in every entry point under `if __name__ == "__main__":`.
  - Seed stochastic baselines by path index.
  - Log actual env steps, not `total_timesteps`.
- Each pitfall has a placeholder in `tests/test_pending_guards.py`. Replace the skip with a real test
  when you implement the related code.
- CPU only. Run final experiments on Linux or WSL2, not native Windows.
- This is a proof-of-concept research project. Prefer simple, correct and tested code over abstraction.

## Commands
```bash
pip install -r requirements.txt -r requirements-dev.txt -e .
pytest -q              # fast tests (slow ones: pytest -m slow)
ruff check .
python scripts/make_crn_paths.py configs/env/*.yaml
```

## When you finish a step
Update docs/STATUS.md: move the current step forward and add one dated line under History.
