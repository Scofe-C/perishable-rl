-- The category fallback assumes a product belongs to exactly one third-level category.
select product_id, count(distinct third_category_id) as n
from {{ ref('silver_daily') }}
group by product_id
having count(distinct third_category_id) > 1
