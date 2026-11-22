# Decision log

Newest first. Add an entry for each agreed change to SPEC.md, the budgets or the fairness rules.
Format: date · who agreed · decision · why.

| Date | Agreed by | Decision | Why |
|---|---|---|---|
| 2026-11-22 | Zhengxin (M3) — enforces SPEC §8, **M2 please note** | `evaluate` rejects a policy action that is not N whole, finite, non-negative order quantities (0-d or `(1,)` is fine for N=1). Orders above remaining capacity stay legal (env clips). The run fails with `InvalidActionError` naming the path and day | SPEC §8 already says `act()` returns `int64[N]`; nothing checked it, and an env may silently clamp a negative or NaN order. So the SB3 adapter must apply `CapacityShareAction` and return order quantities, never the raw Box vector |
| 2026-11-22 | Zhengxin (M3) — extension, off by default | Optional `scenario` on EnvConfig: demand shocks (spike, drop, outage, overdispersed) applied **only to CRN eval paths** (`perinv/data/scenarios.py`, `configs/experiments/shocks.yaml`). Configs without it keep their demand/config hashes and paths (pinned in tests). The runner refuses a scenario config under `envs` and a shock config whose base demand differs from its build config. The YAML loader now rejects unknown top-level keys (allowed provenance: `calibration`, `shock_of`) | Robustness question: what does a policy trained on the base world lose under shocks? No change to env dynamics, SPEC or the core matrix. Shocked path k = base path k + shock from a separate rng stream, so `metrics.paired_worlds` isolates the shock cost. DP is not optimal under shocks: there it is just a policy |
| 2026-11-14 | Zhengxin (M3) — **team to confirm** | Add `duckdb==2.0.0` to read the evaluation warehouse (partitioned Parquet). Partition columns must be read with explicit `hive_types` | DuckDB runs SQL straight on Parquet files, with no server. 2.0.0 is the only release with Python 3.15 wheels; every 1.5.x release needs a C++ compiler on 3.15. Without `hive_types`, `kappa=0.6` is read as text, and `WHERE kappa = 1` silently matches the 0.6 rows too |
| 2026-11-13 | Weibo (scaffold) — **team to confirm** | Drop `rliable`; compute IQM (`scipy.stats.trim_mean(x, 0.25)`) and bootstrap CIs (`scipy.stats.bootstrap`) ourselves | rliable pins `arch<8`, and arch 7.x crashes on import with pandas 3. No pandas 2.x exists for Python 3.15, so rliable cannot work on 3.15. The draft only gave rliable as an example. Dropping it also removes 9 transitive packages (statsmodels, formulaic, seaborn, ...) |
| 2026-11-13 | Weibo (scaffold) — **team to confirm** | Python 3.15 for everyone (pins resolved on 3.15.2) | Homebrew 3.12 was broken on Weibo's Mac. Every pin has a macOS arm64 and Linux x86_64 cp315 wheel or a pure-Python sdist (rliable excluded), including `torch==2.15.0+cpu` |
| 2026-11-13 | (scaffold) | pip + pinned requirements.txt (generated from requirements.in), no Docker | Nothing is deployed. Pinned deps plus `scripts/setup_linux_vm.sh` reproduce the Linux runs |
| 2026-11-13 | (scaffold) | Pin `wrapt==2.5.2` (later moot once rliable was dropped) | `formulaic` declared `wrapt>=1.17.0rc1`, which let pip pick a `.dev` pre-release. Rule: never commit pre-release pins |

## Open items for the week-1 meeting
- [ ] Ratify SPEC.md §3 (event order) and §4 (action mappings)
- [ ] Default cost parameters, λ, L, τ, T (SPEC §1)
- [ ] Initial state: empty, or warm start?
- [ ] Measure O_ref per N and fix C for κ ∈ {0.6, 1.0}
- [ ] DummyVecEnv vs SubprocVecEnv and n_envs (`scripts/bench_vecenv.py`)
- [ ] Training env-step budget per run (`configs/budget.yaml`), after timing one run
