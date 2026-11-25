"""Aggregate finished runs into report tables and one figure per cell group.

    python scripts/analyze.py                         # reads runs/, writes results/
    python scripts/analyze.py --runs-root runs_smoke --out results/smoke

Outputs: runs.csv (one row per run), summary.csv (IQM + 95% CI over seeds per cell), profit_<group>.png, and
the report tables and figures for the three questions (docs/EXPERIMENTS.md):
  q1_pct_of_dp.csv                      N = 1: % of DP per policy (IQM over seeds, 95% CI)
  q2_vs_best_heuristic.csv, q2_*.png    every cell: paired profit difference to the best heuristic of that cell
  q3_efficiency.csv, q3_curves_*.png    CPU hours, steps to 95% of the final tune profit, learning curves
Contract: docs/WAREHOUSE.md §5. Owner: Member 3.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")  # files only, no display needed (cloud VMs, CI)
import matplotlib.pyplot as plt  # noqa: E402

from perinv.eval import metrics, warehouse  # noqa: E402
from perinv.eval.stats import bootstrap_ci  # noqa: E402
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


HEURISTICS = ["ss", "newsvendor"]
RL = ["ppo", "a2c", "sac", "dqn", "maskppo"]
COLORS = {"ppo": "#4C78A8", "a2c": "#F58518", "sac": "#54A24B", "dqn": "#B279A2", "maskppo": "#E45756"}


def _ci(x) -> dict:
    est = bootstrap_ci(np.asarray(x, dtype=float))
    return {"iqm": est.value, "low": est.low, "high": est.high, "n_seeds": est.n}


def q1_pct_of_dp(runs: pd.DataFrame) -> pd.DataFrame:
    """N = 1 only: profit as % of the exact DP on the same test paths, IQM over seeds."""
    rows = []
    for (n, k, pol), g in runs[runs.n_skus == 1].groupby(["n_skus", "kappa", "policy"]):
        rows.append({"n_skus": n, "kappa": k, "policy": pol} | _ci(100 * g.pct_of_dp)
                    | {"mean_profit": g.mean_profit.mean(), "waste_rate": g.waste_rate.mean(),
                       "fill_rate": g.fill_rate.mean()})
    return pd.DataFrame(rows).sort_values(["kappa", "iqm", "policy"], ascending=[True, False, True])


def q2_vs_best_heuristic(con, summary: pd.DataFrame) -> pd.DataFrame:
    """Per cell, the stronger heuristic (by test profit) is the reference; each RL seed is paired path by path."""
    h = summary[summary.policy.isin(HEURISTICS)]
    if h.empty:
        return pd.DataFrame()
    best = h.loc[h.groupby(["group", "n_skus", "kappa"]).mean_profit.idxmax()].set_index(["n_skus", "kappa"])
    rows = []
    for pol in RL:
        for base in HEURISTICS:
            d = metrics.paired_vs(con, pol, base)
            if d.empty:
                continue
            for (n, k), g in d.groupby(["n_skus", "kappa"]):
                if (n, k) not in best.index or best.loc[(n, k), "policy"] != base:
                    continue
                ref = best.loc[(n, k), "mean_profit"]
                seeds_sig = int(((g.low > 0) | (g.high < 0)).sum())
                rows.append({"n_skus": n, "kappa": k, "policy": pol, "reference": base, "reference_profit": ref}
                            | {f"diff_{key}": v for key, v in _ci(g.mean_diff).items()}
                            | {"pct_of_reference": 100 * (1 + _ci(g.mean_diff)["iqm"] / ref),
                               "seeds_ci_excludes_0": seeds_sig, "seeds_better": int((g.low > 0).sum())})
    return pd.DataFrame(rows).sort_values(["n_skus", "kappa", "policy"]) if rows else pd.DataFrame()


def plot_q2(q2: pd.DataFrame, out: Path) -> Path:
    """% of the best heuristic against N, one panel per kappa: how each algorithm scales with the SKU count."""
    kappas = sorted(q2.kappa.unique())
    fig, axes = plt.subplots(1, len(kappas), figsize=(5 * len(kappas), 3.8), squeeze=False, sharey=True)
    for ax, k in zip(axes[0], kappas, strict=True):
        for pol, g in q2[q2.kappa == k].groupby("policy"):
            g = g.sort_values("n_skus")
            ref = g.reference_profit
            ax.errorbar(g.n_skus, g.pct_of_reference, marker="o", capsize=3, label=pol, color=COLORS.get(pol),
                        yerr=[100 * (g.diff_iqm - g.diff_low) / ref, 100 * (g.diff_high - g.diff_iqm) / ref])
        ax.axhline(100, color="grey", lw=1, ls="--")
        ax.set_xticks(sorted(q2.n_skus.unique()))
        ax.set_xlabel("number of SKUs N")
        ax.set_title(f"κ = {k:g}")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("% of best heuristic (paired, IQM over seeds)")
    axes[0][-1].legend()
    fig.suptitle("Test-path profit relative to the best heuristic of each cell, 95% CI")
    fig.tight_layout()
    path = out / "q2_vs_best_heuristic.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def learning_curves(runs: pd.DataFrame, runs_root: Path) -> pd.DataFrame:
    frames = []
    for r in runs[runs.policy.isin(RL)].itertuples():
        f = runs_root / r.run_name / "learning_curve.csv"
        if f.exists():
            env_path = yaml.safe_load((f.parent / "config.yaml").read_text())["env_path"]
            frames.append(pd.read_csv(f).assign(policy=r.policy, n_skus=r.n_skus, kappa=r.kappa, seed=r.seed,
                                                env_path=env_path))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def q3_efficiency(runs: pd.DataFrame, curves: pd.DataFrame) -> pd.DataFrame:
    """CPU cost per run and the env steps a run needs to first reach 95 % of its own final tune-path profit."""
    rows = []
    for (n, k, pol), g in runs[runs.policy.isin(RL)].groupby(["n_skus", "kappa", "policy"]):
        steps95 = []
        for _, c in curves[(curves.n_skus == n) & (curves.kappa == k) & (curves.policy == pol)].groupby("seed"):
            c = c.sort_values("env_steps")
            final, start = c.tune_profit.iloc[-1], c.tune_profit.iloc[0]
            target = start + 0.95 * (final - start)
            steps95.append(int(c.env_steps[c.tune_profit >= target].iloc[0]))
        rows.append({"n_skus": n, "kappa": k, "policy": pol, "cpu_hours_per_run": g.cpu_hours.mean(),
                     "env_steps_per_cpu_second": (g.actual_env_steps / (3600 * g.cpu_hours)).mean(),
                     "median_steps_to_95pct": float(np.median(steps95)) if steps95 else np.nan})
    return pd.DataFrame(rows)


def plot_q3(curves: pd.DataFrame, refs: pd.DataFrame, out: Path) -> list[Path]:
    """Learning curves (mean tune-path profit over seeds, min-max band) per cell; dashed = best heuristic/DP."""
    written = []
    for n, gn in curves.groupby("n_skus"):
        kappas = sorted(gn.kappa.unique())
        fig, axes = plt.subplots(1, len(kappas), figsize=(5 * len(kappas), 3.6), squeeze=False)
        for ax, k in zip(axes[0], kappas, strict=True):
            for pol, g in gn[gn.kappa == k].groupby("policy"):
                s = g.groupby("env_steps").tune_profit.agg(["mean", "min", "max"])
                ax.plot(s.index, s["mean"], label=pol, color=COLORS.get(pol))
                ax.fill_between(s.index, s["min"], s["max"], alpha=0.15, color=COLORS.get(pol))
            ref = refs[(refs.n_skus == n) & (refs.kappa == k)]
            for v, r in ref.groupby(ref.tune_profit.round(1)):   # equal baselines share one labelled line
                ax.axhline(v, color="black" if "dp" in set(r.policy) else "grey", ls="--", lw=1)
                ax.annotate(" = ".join(r.policy), (1, v), xycoords=("axes fraction", "data"), fontsize=7,
                            ha="right", va="bottom")
            lo = ref.tune_profit.min() if len(ref) else gn.tune_profit.quantile(0.05)
            ax.set_ylim(0.85 * lo, None)
            ax.set_xlabel("env steps")
            ax.set_title(f"N = {n}, κ = {k:g}")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("mean profit on tune paths")
        axes[0][-1].legend(fontsize=8)
        fig.tight_layout()
        path = out / f"q3_curves_N{n}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        written.append(path)
    return written


def reference_tune_profit(curves: pd.DataFrame, crn_root: str) -> pd.DataFrame:
    """DP (N = 1) and the heuristics scored on the tune paths, as reference lines for the learning curves."""
    from perinv.baselines import dp_policy, newsvendor, s_S
    from perinv.data.crn import load_paths
    from perinv.env.config import load_env_config
    from perinv.env.perishable_env import PerishableInventoryEnv
    from perinv.eval.evaluate import evaluate

    rows = []
    for (n, k, env_path), _ in curves.groupby(["n_skus", "kappa", "env_path"]):   # the RL runs' build worlds
        cfg = load_env_config(env_path)
        env, tune = PerishableInventoryEnv(cfg), load_paths(cfg, "tune", crn_root)
        pols = {"newsvendor": newsvendor.NewsvendorPolicy(cfg)}
        if n == 1:
            pols["dp"] = dp_policy.DPPolicy(dp_policy.load_or_solve(cfg)[0])
        for name, pol in pols.items():
            rows.append({"n_skus": n, "kappa": k, "policy": name,
                         "tune_profit": float(evaluate(pol, env, tune).per_path()["profit"].mean())})
        rows.append({"n_skus": n, "kappa": k, "policy": "ss",
                     "tune_profit": s_S.search(cfg, PerishableInventoryEnv, tune)[1]})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--crn-root", default="data/crn")
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

    q1_pct_of_dp(runs).to_csv(out / "q1_pct_of_dp.csv", index=False)
    q2 = q2_vs_best_heuristic(con, summary)
    q2.to_csv(out / "q2_vs_best_heuristic.csv", index=False)
    if not q2.empty:
        figs.append(plot_q2(q2, out))
    curves = learning_curves(runs, Path(args.runs_root))
    if not curves.empty:   # only trained RL runs write learning curves
        q3_efficiency(runs, curves).to_csv(out / "q3_efficiency.csv", index=False)
        figs += plot_q3(curves, reference_tune_profit(curves, args.crn_root), out)
    print(f"{len(runs)} runs -> {out}: runs.csv, summary.csv, q1-q3 tables, {len(figs)} figure(s)")
    return 0


if __name__ == "__main__":
    init_process(0)  # project rule; bootstrap CIs also use their own fixed seed
    sys.exit(main())
