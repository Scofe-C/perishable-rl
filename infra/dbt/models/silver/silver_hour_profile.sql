-- Intraday demand profile over open hours 6..21: the share of a day's open-hour sales in each hour.
-- Learned from TRAIN days with no stockout in open hours (fully observed demand). No eval data is used.
-- A product with fewer than min_clean_days clean days falls back to its third category, then to all products.
-- With no clean days at all (possible on small or unusual inputs) the profile is uniform (1/16 per hour).
{{ config(location=var('out_dir') ~ '/hour_profile.parquet') }}

with clean as (
    select product_id, third_category_id, hours_sale[7:22] as hs
    from {{ ref('silver_daily') }}
    where split = 'train' and stockout_hours_open = 0 and sales_open > 0
),

hourly as (  -- zip unnest: hour 6..21 next to its 16 sales values
    select product_id, third_category_id, unnest(range(6, 22)) as hour, unnest(hs) as s
    from clean
),

by_product as (
    select product_id, hour, sum(s) as s from hourly group by all
),
by_category as (
    select third_category_id, hour, sum(s) as s from hourly group by all
),
by_all as (
    select hour, sum(s) as s from hourly group by all
),

n_product as (select product_id, count(*) as n from clean group by all),
n_category as (select third_category_id, count(*) as n from clean group by all),
n_all as (select count(*) as n from clean),

products as (  -- every product seen in either split, with its category
    select product_id, any_value(third_category_id) as third_category_id
    from {{ ref('silver_daily') }} group by all
),

choice as (
    select
        p.product_id,
        p.third_category_id,
        case
            when coalesce(np.n, 0) >= {{ var('min_clean_days') }} then 'product'
            when coalesce(nc.n, 0) >= {{ var('min_clean_days') }} then 'category'
            when na.n > 0 then 'all'
            else 'uniform'
        end as level,
        coalesce(np.n, 0) as n_clean_days_product
    from products p
    left join n_product np using (product_id)
    left join n_category nc using (third_category_id)
    cross join n_all na
),

shares as (
    select
        c.product_id,
        h.hour,
        c.level,
        c.n_clean_days_product,
        case c.level
            when 'product' then bp.s / sum(bp.s) over (partition by c.product_id)
            when 'category' then bc.s / sum(bc.s) over (partition by c.product_id)
            when 'all' then ba.s / sum(ba.s) over (partition by c.product_id)
            else 1.0 / 16
        end as share
    from choice c
    cross join (select unnest(range(6, 22)) as hour) h
    left join by_product bp on bp.product_id = c.product_id and bp.hour = h.hour
    left join by_category bc on bc.third_category_id = c.third_category_id and bc.hour = h.hour
    left join by_all ba on ba.hour = h.hour
)

select product_id, hour, coalesce(share, 0.0) as share, level, n_clean_days_product
from shares
order by product_id, hour
