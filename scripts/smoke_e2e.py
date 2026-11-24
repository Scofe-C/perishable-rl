"""Week-1 end-to-end check (team draft §8): N=1 env -> PPO 10k steps -> test paths -> one plot.

    python scripts/smoke_e2e.py                    # runs_smoke/, results/smoke/smoke_e2e.png
    python scripts/smoke_e2e.py --steps 2000 --paths 5

Goes through the real runner (same builders, keys, manifest, writer and warehouse as the experiments), so a
pass means the interfaces fit together. The plot: PPO's learning curve on the tune paths against the DP,
(s,S) and newsvendor on the test paths. Rerunning skips finished runs.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from perinv.eval import metrics, warehouse  # noqa: E402
from perinv.runner import expand, run_matrix  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="configs/env/n1_k06.yaml")
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--runs-root", default="runs_smoke")
    ap.add_argument("--out", default="results/smoke")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    matrix = out / "smoke_matrix.yaml"
    budget = out / "smoke_budget.yaml"
    budget.write_text(yaml.safe_dump({"train_env_steps": args.steps, "seeds": [0], "profiles": {}}))
    matrix.write_text(yaml.safe_dump({"budget": budget.as_posix(), "groups": {"SMOKE": {
        "envs": [args.env], "policies": ["ppo", "dp", "ss", "newsvendor"]}}}))
    counts = run_matrix(expand(matrix, runs_root=args.runs_root))
    if counts.get("failed"):
        print(f"FAILED: {dict(counts)}; see {args.runs_root}/*/failure.json")
        return 1

    con = warehouse.connect(args.runs_root)
    runs = metrics.run_table(con).query("`group` == 'SMOKE'").set_index("policy")
    ppo_dir = Path(args.runs_root) / runs.loc["ppo", "run_name"]
    curve = pd.read_csv(ppo_dir / "learning_curve.csv")

    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(curve.env_steps, curve.tune_profit, marker="o", ms=3, label="PPO (tune paths, during training)")
    ax.scatter([curve.env_steps.iloc[-1]], [runs.loc["ppo", "mean_profit"]], color="C0", marker="*", s=120,
               zorder=3, label="PPO final (test paths)")
    for pol, color, ls in [("dp", "k", "-"), ("ss", "C2", "--"), ("newsvendor", "C3", ":")]:
        ax.axhline(runs.loc[pol, "mean_profit"], color=color, ls=ls, label=f"{pol} (test paths)")
    ax.set_xlabel("actual env steps")
    ax.set_ylabel("mean profit per 30-day episode")
    ax.set_title(f"Smoke test: {Path(args.env).stem}, PPO {args.steps:,} steps")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png = out / "smoke_e2e.png"
    fig.savefig(png, dpi=130)

    table = runs[["mean_profit", "pct_of_dp", "waste_rate", "fill_rate", "actual_env_steps"]].round(3)
    print(table.to_string())
    print(f"wrote {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
