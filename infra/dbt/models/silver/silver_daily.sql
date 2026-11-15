-- One row per (store, product, day), both splits. Facts only: no estimation happens here.
-- Open hours = 0-based hours 6..21 (16 hours) = DuckDB list slice [7:22]; verified in bronze.
{{ config(location=var('out_dir') ~ '/daily.parquet') }}

with bronze as (
    select 'train' as split, * from {{ source('bronze', 'train') }}
    union all by name
    select 'eval' as split, * from {{ source('bronze', 'eval') }}
)

select
    store_id,
    product_id,
    dt,
    split,
    city_id,
    management_group_id,
    first_category_id,
    second_category_id,
    third_category_id,
    isodow(dt) - 1                         as dow,                 -- 0 = Monday
    sale_amount                            as sales,               -- normalized units (coefficient unknown)
    list_sum(hours_sale[7:22])             as sales_open,
    stock_hour6_22_cnt::INTEGER            as stockout_hours_open,
    stock_hour6_22_cnt > 0                 as stockout_any_open,   -- the threshold is a gold-layer parameter
    list_sum(hours_stock_status)::INTEGER  as stockout_hours_24,
    hours_sale,
    hours_stock_status,
    discount,
    holiday_flag,
    activity_flag,
    precpt,
    avg_temperature
from bronze
order by store_id, product_id, dt
