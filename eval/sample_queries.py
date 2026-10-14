pairs = [
    {
        "name": "customer_360_summary",
        "nl": "Give me a 360° customer summary showing PII (name, email, city, state), number of accounts, total balances, number of transactions, and their most recent transaction date. Sort by total balance descending.",
        "sql": """SELECT
    c.customer_id,
    c.full_name,
    c.email,
    c.city,
    c.state,
    COUNT(DISTINCT a.account_id)                                   AS num_accounts,
    SUM(a.current_balance)                                         AS total_current_balance,
    COUNT(DISTINCT t.transaction_id)                               AS num_transactions,
    MAX(t.created_at)                                              AS last_transaction_at
FROM customers c
LEFT JOIN accounts a      ON a.customer_id = c.customer_id
LEFT JOIN transactions t  ON t.account_id = a.account_id
GROUP BY c.customer_id, c.full_name, c.email, c.city, c.state
ORDER BY total_current_balance DESC NULLS LAST;"""
    },
    {
        "name": "vip_crypto_exposure",
        "nl": "For VIP customers only, show their total spend, crypto spend, and percentage of spend that went to crypto merchants.",
        "sql": """WITH vip_spend AS (
    SELECT
        c.customer_id,
        c.full_name,
        c.state,
        SUM(CASE WHEN t.amount < 0 THEN -t.amount ELSE 0 END) AS total_spend,
        SUM(
            CASE WHEN t.amount < 0 AND m.category = 'crypto'
                 THEN -t.amount ELSE 0
            END
        ) AS crypto_spend
    FROM customers c
    JOIN accounts a      ON a.customer_id = c.customer_id
    JOIN transactions t  ON t.account_id = a.account_id
    LEFT JOIN merchants m ON m.merchant_id = t.merchant_id
    WHERE c.is_vip = TRUE
      AND t.status = 'completed'
    GROUP BY c.customer_id, c.full_name, c.state
)
SELECT
    customer_id,
    full_name,
    state,
    total_spend,
    crypto_spend,
    CASE
        WHEN total_spend = 0 THEN 0
        ELSE ROUND(100.0 * crypto_spend / total_spend, 2)
    END AS crypto_pct_of_spend
FROM vip_spend
ORDER BY crypto_pct_of_spend DESC, total_spend DESC;"""
    },
    {
        "name": "running_transaction_balance",
        "nl": "For each account, compute the running sum of all completed transactions in chronological order.",
        "sql": """SELECT
    a.account_id,
    a.account_number,
    t.transaction_id,
    t.created_at,
    t.amount,
    SUM(t.amount) OVER (
        PARTITION BY a.account_id
        ORDER BY t.created_at, t.transaction_id
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_tx_balance_delta
FROM accounts a
JOIN transactions t ON t.account_id = a.account_id
WHERE t.status = 'completed'
ORDER BY a.account_id, t.created_at, t.transaction_id;"""
    },
    {
        "name": "top_merchants_by_state",
        "nl": "For every state, show the top 3 merchants by total completed spend.",
        "sql": """WITH state_merchant_spend AS (
    SELECT
        c.state,
        m.name AS merchant_name,
        SUM(CASE WHEN t.amount < 0 THEN -t.amount ELSE 0 END) AS total_spend
    FROM customers c
    JOIN accounts a      ON a.customer_id = c.customer_id
    JOIN transactions t  ON t.account_id = a.account_id
    JOIN merchants m     ON m.merchant_id = t.merchant_id
    WHERE t.status = 'completed'
    GROUP BY c.state, m.name
),
ranked AS (
    SELECT
        state,
        merchant_name,
        total_spend,
        DENSE_RANK() OVER (
            PARTITION BY state
            ORDER BY total_spend DESC
        ) AS rnk
    FROM state_merchant_spend
)
SELECT
    state,
    merchant_name,
    total_spend
FROM ranked
WHERE rnk <= 3
ORDER BY state, total_spend DESC;"""
    },
    {
        "name": "rapid_fire_charges",
        "nl": "Find high-value card transactions at the same merchant with the same card that occur within 1 hour of each other.",
        "sql": """WITH card_spend AS (
    SELECT
        t.transaction_id,
        t.account_id,
        t.card_id,
        t.merchant_id,
        t.amount,
        t.created_at
    FROM transactions t
    WHERE t.status = 'completed'
      AND t.transaction_type = 'card'
      AND t.amount < -100
)
SELECT
    c.full_name,
    m.name          AS merchant_name,
    ca.card_number_last4,
    t1.transaction_id AS tx_id_1,
    t1.created_at     AS tx_time_1,
    t1.amount         AS amount_1,
    t2.transaction_id AS tx_id_2,
    t2.created_at     AS tx_time_2,
    t2.amount         AS amount_2,
    EXTRACT(EPOCH FROM (t2.created_at - t1.created_at)) / 60.0 AS minutes_between
FROM card_spend t1
JOIN card_spend t2
  ON t1.card_id = t2.card_id
 AND t1.merchant_id = t2.merchant_id
 AND t2.created_at > t1.created_at
 AND t2.created_at <= t1.created_at + INTERVAL '1 hour'
JOIN cards ca       ON ca.card_id = t1.card_id
JOIN accounts a     ON a.account_id = t1.account_id
JOIN customers c    ON c.customer_id = a.customer_id
JOIN merchants m    ON m.merchant_id = t1.merchant_id
ORDER BY minutes_between ASC;"""
    },
    {
        "name": "payroll_vs_restaurant_spend",
        "nl": "Compare average payroll deposits to average restaurant ticket size per customer.",
        "sql": """WITH payroll AS (
    SELECT
        t.account_id,
        AVG(t.amount) AS avg_payroll_deposit
    FROM transactions t
    WHERE t.transaction_type = 'ach'
      AND t.status = 'completed'
      AND t.description ILIKE '%payroll%'
    GROUP BY t.account_id
),
restaurant_spend AS (
    SELECT
        t.account_id,
        AVG(-t.amount) AS avg_restaurant_ticket
    FROM transactions t
    JOIN merchants m ON m.merchant_id = t.merchant_id
    WHERE t.transaction_type = 'card'
      AND t.status = 'completed'
      AND m.category = 'restaurant'
      AND t.amount < 0
    GROUP BY t.account_id
)
SELECT
    c.customer_id,
    c.full_name,
    c.city,
    c.state,
    p.avg_payroll_deposit,
    r.avg_restaurant_ticket
FROM payroll p
JOIN accounts a        ON a.account_id = p.account_id
JOIN customers c       ON c.customer_id = a.customer_id
LEFT JOIN restaurant_spend r ON r.account_id = a.account_id
ORDER BY p.avg_payroll_deposit DESC;"""
    }
]
