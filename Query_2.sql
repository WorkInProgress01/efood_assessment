WITH top_users AS (
  SELECT
    city,
    user_id,
    COUNT(DISTINCT order_id) AS order_count,
    
  FROM
    `watchful-muse-388316.main_assessment.orders`
  GROUP BY
    city, user_id
),
city_orders AS (
  SELECT
    city,
    COUNT(DISTINCT order_id) AS total_orders
  FROM
    `watchful-muse-388316.main_assessment.orders`
  GROUP BY
    city
),
top_10_contributions AS (
  SELECT
    tu.city,
    SUM(tu.order_count) AS top_10_orders
  FROM
    top_users AS tu
  WHERE
    order_count >= 10
  GROUP BY
    tu.city
)
SELECT
  co.city,
  ROUND((tc.top_10_orders / co.total_orders) * 100, 2) AS top10us_cntrb_perc
FROM
  top_10_contributions AS tc
JOIN
  city_orders AS co ON tc.city = co.city
ORDER BY
  city ASC
;

