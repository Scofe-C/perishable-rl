-- Each product's 16 hourly shares form a distribution.
select product_id, count(*) as hours, sum(share) as total
from {{ ref('silver_hour_profile') }}
group by product_id
having count(*) <> 16 or abs(sum(share) - 1) > 1e-9
