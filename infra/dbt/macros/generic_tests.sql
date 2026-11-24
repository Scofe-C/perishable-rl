{# Small generic tests, so the project needs no external dbt packages (no network at build time). #}

{% test unique_combination(model, columns) %}
select {{ columns | join(', ') }}, count(*) as n
from {{ model }}
group by {{ columns | join(', ') }}
having count(*) > 1
{% endtest %}

{% test expect_between(model, column_name, min_value=none, max_value=none) %}
select {{ column_name }}
from {{ model }}
where {{ column_name }} is null
{%- if min_value is not none %} or {{ column_name }} < {{ min_value }}{% endif %}
{%- if max_value is not none %} or {{ column_name }} > {{ max_value }}{% endif %}
{% endtest %}

{% test expression_is_true(model, expression) %}
select * from {{ model }} where not ({{ expression }})
{% endtest %}
