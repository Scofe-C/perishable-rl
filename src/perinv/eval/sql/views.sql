-- Analysis views over the evaluation warehouse. Contract: docs/WAREHOUSE.md §5.
-- Base tables `fact_step` and `dim_run` are created by perinv.eval.warehouse.connect() (done runs only).
-- Every metric is defined here once; Python code reads these views instead of recomputing.

-- Per step, with money columns computed from that SKU's unit costs (DuckDB lists are 1-based).
CREATE OR REPLACE VIEW v_step AS
SELECT
    f.*,
    f.sold + f.lost                                AS demand,
    d.price[f.sku + 1]           * f.sold          AS revenue,
    d.order_cost[f.sku + 1]      * f."order"       AS cost_ordering,
    d.holding_cost[f.sku + 1]    * f.held          AS cost_holding,
    d.expiry_cost[f.sku + 1]     * f.expired       AS cost_expiry,
    d.lost_sales_cost[f.sku + 1] * f.lost          AS cost_lost_sales,
    revenue - cost_ordering - cost_holding - cost_expiry - cost_lost_sales AS sku_reward
FROM fact_step f
JOIN dim_run d USING (run_key);

-- Per path. day_reward is set on the sku = 0 row only, so a plain SUM is the path profit.
CREATE OR REPLACE VIEW v_path AS
SELECT
    run_key,
    path_id,
    SUM(day_reward)                       AS profit,
    SUM("order")                          AS ordered,
    SUM(sold)                             AS sold,
    SUM(lost)                             AS lost,
    SUM(sold + lost)                      AS demand,
    SUM(expired)                          AS expired,
    COUNT(*) FILTER (WHERE clipped)       AS clipped_days
FROM fact_step
GROUP BY run_key, path_id;

-- Per day: total occupancy across SKUs at decision time (for capacity utilisation).
CREATE OR REPLACE VIEW v_day AS
SELECT run_key, path_id, t, SUM(occ) AS occ, SUM(occ + "order") AS occ_after_order
FROM fact_step
GROUP BY run_key, path_id, t;

-- Per run: the metrics of docs/EXPERIMENTS.md, pooled as ratio of sums (WAREHOUSE.md §5.1).
CREATE OR REPLACE VIEW v_run AS
WITH p AS (
    SELECT run_key, COUNT(*) AS n_paths, AVG(profit) AS mean_profit, SUM(ordered) AS ordered, SUM(sold) AS sold,
           SUM(demand) AS demand, SUM(expired) AS expired, SUM(clipped_days) AS clipped_days
    FROM v_path GROUP BY run_key
),
u AS (
    SELECT run_key, AVG(occ) AS mean_occ, AVG(occ_after_order) AS mean_occ_after_order
    FROM v_day GROUP BY run_key
)
SELECT
    d.run_key, d.run_name, d."group", d.n_skus, d.kappa, d.policy, d.seed, d.config_hash,
    d.eval_config_hash, d.cross_eval,
    p.n_paths,
    p.mean_profit,
    p.expired::DOUBLE / NULLIF(p.ordered, 0)           AS waste_rate,
    p.sold::DOUBLE / NULLIF(p.demand, 0)               AS fill_rate,
    u.mean_occ / d.capacity                            AS capacity_utilisation,
    u.mean_occ_after_order / d.capacity                AS capacity_utilisation_after_order,
    p.clipped_days::DOUBLE / (p.n_paths * d.horizon)   AS clipped_day_share,
    d.actual_env_steps,
    d.cpu_seconds / 3600.0                             AS cpu_hours
FROM dim_run d
JOIN p USING (run_key)
JOIN u USING (run_key);

-- % of DP (N = 1): paired on the same evaluation world (eval config) and the same test paths.
-- The reference is the DP evaluated natively in that world (not a cross-evaluated DP).
CREATE OR REPLACE VIEW v_vs_dp AS
WITH p AS (
    SELECT v.run_key, v.path_id, v.profit, d.eval_config_hash, d.crn_test_sha256, d.policy, d.cross_eval
    FROM v_path v JOIN dim_run d USING (run_key)
),
dp AS (SELECT eval_config_hash, crn_test_sha256, path_id, AVG(profit) AS dp_profit FROM p
        WHERE policy = 'dp' AND NOT cross_eval GROUP BY ALL)
SELECT
    p.run_key,
    -- a ratio only means something against a positive DP profit; otherwise read mean_gap_to_dp
    CASE WHEN AVG(dp.dp_profit) > 0 THEN AVG(p.profit) / AVG(dp.dp_profit) END AS pct_of_dp,
    AVG(p.profit - dp.dp_profit)                   AS mean_gap_to_dp,
    COUNT(*)                                       AS n_paired_paths
FROM p JOIN dp USING (eval_config_hash, crn_test_sha256, path_id)   -- same world, same path set, same path
GROUP BY p.run_key;
