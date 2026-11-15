"""Run an experiment matrix: skip finished runs, run the rest, K at a time.

    python scripts/run_matrix.py configs/experiments/core.yaml --dry-run
    python scripts/run_matrix.py configs/experiments/core.yaml --jobs 6
    python scripts/run_matrix.py configs/experiments/core.yaml --seeds 0,1,2 --groups A --policies ppo,dp
    python scripts/run_matrix.py configs/experiments/core.yaml --profile smoke --runs-root runs_smoke

Safe to rerun after a crash or a kill: finished runs are skipped, unfinished ones start over.
Exit code 1 if any job failed (see runs/<run_name>/failure.json). Contract: docs/WAREHOUSE.md §4.

Owner: Member 3.
"""

import argparse
import logging
import sys

from perinv.runner import expand, plan, run_matrix


def _csv(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("matrix", help="configs/experiments/<name>.yaml")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes (default 1, in-process)")
    ap.add_argument("--seeds", help="comma list, overrides the matrix/budget seeds (e.g. 0,1,2)")
    ap.add_argument("--groups", help="comma list of groups to keep")
    ap.add_argument("--policies", help="comma list of policies to keep")
    ap.add_argument("--profile", help="budget.yaml profile for train_env_steps (tiny, smoke)")
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--crn-root", default="data/crn")
    ap.add_argument("--configs-root", default="configs")
    ap.add_argument("--force", action="store_true", help="rerun even finished runs")
    ap.add_argument("--official", action="store_true", help="refuse to run on a dirty git tree")
    ap.add_argument("--dry-run", action="store_true", help="only print what would run")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    seeds = _csv(args.seeds)
    try:
        return _run(args, seeds)
    except (FileNotFoundError, ValueError) as exc:  # bad matrix or missing CRN paths: say what to fix
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace, seeds: list[str] | None) -> int:
    jobs = expand(args.matrix, seeds=[int(s) for s in seeds] if seeds else None,
                  groups=set(_csv(args.groups) or []) or None, policies=set(_csv(args.policies) or []) or None,
                  profile=args.profile, runs_root=args.runs_root, crn_root=args.crn_root,
                  configs_root=args.configs_root)
    if args.dry_run:
        rows = plan(jobs, args.force)
        width = max((len(j.name) for j, _, _ in rows), default=10)
        for job, status, action in rows:
            print(f"{job.name:<{width}}  {status:<8}  {action}")
        actions = [a if a in ("run", "skip") else "blocked" for _, _, a in rows]
        print(f"\n{len(rows)} jobs: {actions.count('run')} to run, {actions.count('skip')} done, "
              f"{actions.count('blocked')} blocked (no builder yet)")
        return 0
    counts = run_matrix(jobs, n_jobs=args.jobs, force=args.force, official=args.official)
    print(f"done={counts['done']} failed={counts['failed']} skipped/blocked={counts['skipped']}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    # Each job calls init_process(seed) itself (perinv.runner.run_job), in its own worker process.
    sys.exit(main())
