-- Every bronze row appears exactly once in silver_daily and in silver_demand_m2 (no rows lost or added).
with b as (
    select (select count(*) from {{ source('bronze', 'train') }})
         + (select count(*) from {{ source('bronze', 'eval') }}) as n
)
select b.n as bronze_rows,
       (select count(*) from {{ ref('silver_daily') }}) as daily_rows,
       (select count(*) from {{ ref('silver_demand_m2') }}) as m2_rows
from b
where b.n <> (select count(*) from {{ ref('silver_daily') }})
   or b.n <> (select count(*) from {{ ref('silver_demand_m2') }})
