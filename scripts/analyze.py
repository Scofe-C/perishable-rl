"""Aggregate finished runs into report tables and one figure per cell group.

    python scripts/analyze.py                         # reads runs/, writes results/
    python scripts/analyze.py --runs-root runs_smoke --out results/smoke

Outputs: runs.csv (one row per run), summary.csv (IQM + 95% CI over seeds per cell), profit_<group>.png.
Contract: docs/WAREHOUSE.md §5. Owner: Member 3.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no display needed (cloud VMs, CI)
import matplotlib.pyplot as plt  # noqa: E402

from perinv.eval import metrics, warehouse  # noqa: E402
from perinv.utils.seeding import init_process  # noqa: E402


def plot_profit(summary, out: Path) -> list[Path]:
    """One bar chart per group: IQM profit per policy for each (N, kappa) cell, with 95% CI whiskers."""
    written = []
    for group, g in summary.groupby("group"):
        cells = g[["n_skus", "kappa"]].drop_duplicates().sort_values(["n_skus", "kappa"]).to_records(index=False)
        policies = sorted(g.policy.unique())
        fig, axes = plt.subplots(1, len(cells), figsize=(3.2 * len(cells) + 1, 3.6), squeeze=False)
        for ax, (n, k) in zip(axes[0], cells, strict=True):
            c = g[(g.n_skus == n) & (g.kappa == k)].set_index("policy").reindex(policies)
            err = [c.mean_profit - c.mean_profit_low, c.mean_profit_high - c.mean_profit]
            ax.bar(range(len(policies)), c.mean_profit, yerr=err, capsize=3, color="#4C78A8")
            ax.set_xticks(range(len(policies)), policies, rotation=45, ha="right")
            ax.set_title(f"N={n}, κ={k:g}")
            ax.grid(axis="y", alpha=0.3)
        axes[0][0].set_ylabel("mean profit per episode (IQM over seeds)")
        fig.suptitle(f"Group {group}: test-path profit, 95% bootstrap CI")
        fig.tight_layout()
        path = out / f"profit_{group}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    con = warehouse.connect(args.runs_root)
    runs = metrics.run_table(con)
    if runs.empty:
        print(f"no finished runs under {args.runs_root}", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs.to_csv(out / "runs.csv", index=False)
    summary = metrics.summary(con)
    summary.to_csv(out / "summary.csv", index=False)
    figs = plot_profit(summary, out)
    print(f"{len(runs)} runs -> {out / 'runs.csv'}, {out / 'summary.csv'}, {len(figs)} figure(s)")
    return 0


if __name__ == "__main__":
    init_process(0)  # project rule; bootstrap CIs also use their own fixed seed
    sys.exit(main())
