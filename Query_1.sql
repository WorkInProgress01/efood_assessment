WITH city_orders AS (
  SELECT
    city,
    COUNT(DISTINCT order_id) AS orders,
    COUNT(DISTINCT user_id) AS users
  FROM
    `watchful-muse-388316.main_assessment.orders`
  GROUP BY
    city
  HAVING
    COUNT(DISTINCT order_id) > 1000
),
city_breakfast AS (
  SELECT
    city,
    SUM(amount) AS breakfast_amount,
    COUNT(DISTINCT order_id) AS breakfast_orders,
    COUNT(DISTINCT user_id) AS breakfast_users
  FROM
    `watchful-muse-388316.main_assessment.orders`
  WHERE
    cuisine = 'Breakfast'
  GROUP BY
    city
),
city_efood AS (
  SELECT
    city,
    SUM(amount) AS efood_amount,
    COUNT(DISTINCT order_id) AS efood_orders,
    COUNT(DISTINCT user_id) AS efood_users
  FROM
    `watchful-muse-388316.main_assessment.orders`
  WHERE
    city IN (SELECT city FROM city_orders)
  GROUP BY
    city
),
city_breakfast_users3freq AS (
  SELECT
    city,
    COUNT(DISTINCT user_id) AS breakfast_users3freq
  FROM (
    SELECT
      city,
      user_id,
      COUNT(DISTINCT order_id) AS order_count
    FROM
      `watchful-muse-388316.main_assessment.orders`
    WHERE
      cuisine = 'Breakfast'
    GROUP BY
      city, user_id
    HAVING
      order_count > 3
  )
  GROUP BY
    city
),
city_efood_users3freq AS (
  SELECT
    city,
    COUNT(DISTINCT user_id) AS efood_users3freq
  FROM (
    SELECT
      city,
      user_id,
      COUNT(DISTINCT order_id) AS order_count
    FROM
      `watchful-muse-388316.main_assessment.orders`
    WHERE
      city IN (SELECT city FROM city_orders)
    GROUP BY
      city, user_id
    HAVING
      order_count > 3
  )
  GROUP BY
    city
)
SELECT
  co.city,
  cb.breakfast_amount / cb.breakfast_orders AS breakfast_basket,
  ce.efood_amount / ce.efood_orders AS efood_basket,
  cb.breakfast_orders / cb.breakfast_users AS breakfast_frequency,
  ce.efood_orders / ce.efood_users AS efood_frequency,
  (SELECT  breakfast_users3freq FROM city_breakfast_users3freq WHERE city = co.city AND breakfast_users3freq > 3) / CAST(cb.breakfast_users AS FLOAT64) AS breakfast_users3freq_perc,
  (SELECT  efood_users3freq FROM city_efood_users3freq WHERE city = co.city AND efood_users3freq > 3) / CAST(ce.efood_users AS FLOAT64) AS efood_users3freq_perc
FROM
  city_orders AS co
JOIN
  city_breakfast AS cb ON co.city = cb.city
JOIN
  city_efood AS ce ON co.city = ce.city
ORDER BY
  cb.breakfast_orders DESC
LIMIT 5;