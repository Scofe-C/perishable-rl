-- Method 2 (hourly extrapolation): recover demand lost to stockouts within open hours.
--   p_out           = expected share of open-hour demand that falls in this day's stockout hours (from the profile)
--   instock_sales   = sales in open hours that were in stock
--   demand_open_hat = instock_sales / (1 - p_out), never below observed open-hour sales
--   demand_m2       = demand_open_hat + sales outside open hours (taken as observed)
-- Days with p_out > max_censored_share are NOT estimable: almost all of the day's demand fell in stockout hours,
-- so there is too little in-stock signal to extrapolate from. demand_m2 is NULL there (flag `capped`).
-- An earlier version used the cap value instead, which biased those days down: 7% of train days are out of stock
-- for all 16 open hours and came out as demand ~= 0. Gold decides how to treat NULLs.
{{ config(location=var('out_dir') ~ '/demand_m2.parquet') }}

with profile as (
    select product_id, list(share order by hour) as prof, any_value(level) as profile_level
    from {{ ref('silver_hour_profile') }} group by all
),

d as (
    select
        s.store_id, s.product_id, s.dt, s.split, s.sales, s.sales_open, s.stockout_hours_open,
        s.hours_sale[7:22] as hs, s.hours_stock_status[7:22] as st, p.prof, p.profile_level
    from {{ ref('silver_daily') }} s
    join profile p using (product_id)
),

parts as (
    select
        *,
        list_sum(list_transform(range(1, 17), i -> st[i] * prof[i]))             as p_out,
        list_sum(list_transform(range(1, 17), i -> (1 - st[i]) * hs[i]))         as instock_sales
    from d
)

select
    store_id,
    product_id,
    dt,
    split,
    sales,
    stockout_hours_open,
    p_out,
    p_out > {{ var('max_censored_share') }}                                       as capped,
    case
        when stockout_hours_open = 0 then sales_open + (sales - sales_open)
        when p_out > {{ var('max_censored_share') }} then null
        else greatest(sales_open, instock_sales / (1 - p_out)) + (sales - sales_open)
    end                                                                           as demand_m2,
    profile_level
from parts
order by store_id, product_id, dt
