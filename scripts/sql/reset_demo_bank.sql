-- =========================================================
-- Reset demo_bank tables: Drop and recreate with all data
-- Tables will be created in the public schema
-- =========================================================

-- Set search path to public schema
SET search_path TO public;

-- Drop tables if they exist (in reverse dependency order to handle foreign keys)
DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS cards CASCADE;
DROP TABLE IF EXISTS accounts CASCADE;
DROP TABLE IF EXISTS merchants CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

-- =========================================================
-- Tables
-- =========================================================

-- 1) Customers (with extra PII)
CREATE TABLE customers (
    customer_id     SERIAL PRIMARY KEY,
    full_name       TEXT NOT NULL,
    email           TEXT UNIQUE NOT NULL,
    phone_number    TEXT,
    date_of_birth   DATE,
    address_line1   TEXT,
    city            TEXT,
    state           TEXT,
    postal_code     TEXT,
    ssn             TEXT,  -- fake 9-digit IDs
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_vip          BOOLEAN NOT NULL DEFAULT FALSE
);

-- 2) Accounts
CREATE TABLE accounts (
    account_id      SERIAL PRIMARY KEY,
    customer_id     INT NOT NULL REFERENCES customers(customer_id),
    account_number  TEXT UNIQUE NOT NULL,
    account_type    TEXT NOT NULL CHECK (account_type IN ('checking', 'savings', 'credit')),
    currency        TEXT NOT NULL DEFAULT 'USD',
    current_balance NUMERIC(12, 2) NOT NULL DEFAULT 0,
    status          TEXT NOT NULL CHECK (status IN ('active', 'frozen', 'closed')),
    opened_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3) Cards
CREATE TABLE cards (
    card_id             SERIAL PRIMARY KEY,
    account_id          INT NOT NULL REFERENCES accounts(account_id),
    card_number_last4   CHAR(4) NOT NULL,
    network             TEXT NOT NULL CHECK (network IN ('Visa', 'Mastercard', 'Amex')),
    card_type           TEXT NOT NULL CHECK (card_type IN ('debit', 'credit')),
    issued_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at          DATE NOT NULL,
    is_active           BOOLEAN NOT NULL DEFAULT TRUE
);

-- 4) Merchants
CREATE TABLE merchants (
    merchant_id     SERIAL PRIMARY KEY,
    name            TEXT NOT NULL,
    category        TEXT NOT NULL,
    country         TEXT NOT NULL,
    risk_level      TEXT NOT NULL CHECK (risk_level IN ('low', 'medium', 'high'))
);

-- 5) Transactions
CREATE TABLE transactions (
    transaction_id   SERIAL PRIMARY KEY,
    account_id       INT NOT NULL REFERENCES accounts(account_id),
    card_id          INT REFERENCES cards(card_id),
    merchant_id      INT REFERENCES merchants(merchant_id),
    amount           NUMERIC(12, 2) NOT NULL,      -- positive = credit, negative = debit
    currency         TEXT NOT NULL DEFAULT 'USD',
    status           TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'reversed')),
    transaction_type TEXT NOT NULL CHECK (transaction_type IN ('card', 'ach', 'wire', 'fee', 'refund')),
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    description      TEXT
);

-- Helpful indexes
CREATE INDEX idx_accounts_customer_id          ON accounts(customer_id);
CREATE INDEX idx_cards_account_id             ON cards(account_id);
CREATE INDEX idx_tx_account_id_created_at     ON transactions(account_id, created_at DESC);
CREATE INDEX idx_tx_merchant_created_at       ON transactions(merchant_id, created_at DESC);

-- =========================================================
-- Seed data
-- =========================================================

-- 1) 50 Customers with simple names + extra PII
INSERT INTO customers (
    full_name,
    email,
    phone_number,
    date_of_birth,
    address_line1,
    city,
    state,
    postal_code,
    ssn,
    is_vip
)
SELECT
    'User ' || LPAD(gs::text, 3, '0')                                      AS full_name,
    'user' || LPAD(gs::text, 3, '0') || '@example.com'                     AS email,
    '+1-555-01' || LPAD(gs::text, 2, '0')                                  AS phone_number,
    DATE '1980-01-01' + (gs * 200)                                         AS date_of_birth,  -- spread across years
    gs || ' Main St'                                                       AS address_line1,
    (ARRAY['San Francisco','New York','Chicago','Austin','Seattle'])
        [(gs - 1) % 5 + 1]                                                 AS city,
    (ARRAY['CA','NY','IL','TX','WA'])
        [(gs - 1) % 5 + 1]                                                 AS state,
    LPAD((90000 + gs)::text, 5, '0')                                       AS postal_code,
    LPAD((100000000 + gs)::text, 9, '0')                                   AS ssn,
    (gs % 10 = 0)                                                          AS is_vip   -- every 10th user is VIP
FROM generate_series(1, 50) AS gs;

-- 2) Accounts
-- One checking account for each of the 50 customers
INSERT INTO accounts (
    customer_id,
    account_number,
    account_type,
    currency,
    current_balance,
    status,
    opened_at
)
SELECT
    gs                                                  AS customer_id,
    'CHK-' || LPAD(gs::text, 4, '0')                    AS account_number,
    'checking'                                          AS account_type,
    'USD'                                               AS currency,
    ROUND(((random() * 5000) + 500)::numeric, 2)        AS current_balance,
    CASE WHEN random() < 0.05 THEN 'frozen' ELSE 'active' END AS status,
    NOW() - ((50 - gs) || ' days')::interval            AS opened_at
FROM generate_series(1, 50) AS gs;

-- Savings accounts for first 20 customers
INSERT INTO accounts (
    customer_id,
    account_number,
    account_type,
    currency,
    current_balance,
    status,
    opened_at
)
SELECT
    gs                                                  AS customer_id,
    'SAV-' || LPAD(gs::text, 4, '0')                    AS account_number,
    'savings',
    'USD',
    ROUND(((random() * 20000) + 1000)::numeric, 2)      AS current_balance,
    'active'                                            AS status,
    NOW() - ((100 - gs) || ' days')::interval           AS opened_at
FROM generate_series(1, 20) AS gs;

-- Credit accounts for first 15 customers
INSERT INTO accounts (
    customer_id,
    account_number,
    account_type,
    currency,
    current_balance,
    status,
    opened_at
)
SELECT
    gs                                                  AS customer_id,
    'CRD-' || LPAD(gs::text, 4, '0')                    AS account_number,
    'credit',
    'USD',
    ROUND(((random() * 3000) + 100)::numeric, 2) * -1   AS current_balance, -- negative = they owe money
    'active',
    NOW() - ((25 - gs) || ' days')::interval
FROM generate_series(1, 15) AS gs;

-- 3) Cards (1 card per checking or credit account)
INSERT INTO cards (
    account_id,
    card_number_last4,
    network,
    card_type,
    issued_at,
    expires_at,
    is_active
)
SELECT
    a.account_id,
    LPAD((1000 + a.account_id)::text, 4, '0')                                    AS card_number_last4,
    (ARRAY['Visa','Mastercard','Amex'])[(a.account_id - 1) % 3 + 1]              AS network,
    CASE WHEN a.account_type = 'credit' THEN 'credit' ELSE 'debit' END           AS card_type,
    a.opened_at + INTERVAL '7 days'                                              AS issued_at,
    (a.opened_at + INTERVAL '3 years')::date                                     AS expires_at,
    TRUE                                                                         AS is_active
FROM accounts a
WHERE a.account_type IN ('checking', 'credit');

-- 4) Merchants
INSERT INTO merchants (name, category, country, risk_level)
VALUES
  ('Fresh Mart',         'groceries',   'US', 'low'),
  ('Skyline Airlines',   'travel',      'US', 'medium'),
  ('Urban Eats',         'restaurant',  'US', 'low'),
  ('CryptoNow Exchange', 'crypto',      'MT', 'high'),
  ('Global Gadgets',     'electronics', 'CN', 'medium');

-- 5) Transactions

-- 5a) Card-based spend transactions (auto-generated)
WITH ac AS (
    SELECT
        a.account_id,
        c.card_id,
        row_number() OVER (ORDER BY a.account_id, c.card_id) AS rn
    FROM accounts a
    JOIN cards   c ON c.account_id = a.account_id
)
INSERT INTO transactions (
    account_id,
    card_id,
    merchant_id,
    amount,
    currency,
    status,
    transaction_type,
    created_at,
    description
)
SELECT
    ac.account_id,
    ac.card_id,
    (SELECT merchant_id FROM merchants ORDER BY random() LIMIT 1)       AS merchant_id,
    ROUND(((random() * 190) + 10)::numeric, 2) * -1                     AS amount,      -- debits from -10 to -200
    'USD'                                                               AS currency,
    'completed'                                                         AS status,
    'card'                                                              AS transaction_type,
    NOW() - ((ac.rn % 60) || ' days')::interval                         AS created_at,
    'Auto-generated card transaction #' || ac.rn                        AS description
FROM ac
WHERE ac.rn <= 120;    -- ~120 card transactions

-- 5b) ACH payroll deposits into some checking accounts
INSERT INTO transactions (
    account_id,
    card_id,
    merchant_id,
    amount,
    currency,
    status,
    transaction_type,
    created_at,
    description
)
SELECT
    a.account_id,
    NULL                                           AS card_id,
    NULL                                           AS merchant_id,
    ROUND(((random() * 4000) + 1000)::numeric, 2)  AS amount, -- +1000 to +5000
    'USD',
    'completed',
    'ach',
    NOW() - ((gs % 90) || ' days')::interval       AS created_at,
    'Auto-generated payroll deposit #' || gs || ' for account ' || a.account_number
FROM accounts a
CROSS JOIN generate_series(1, 3) AS gs
WHERE a.account_type = 'checking'
  AND a.account_id % 7 = 0;  -- every 7th checking account gets payroll

-- 5c) Fees on some checking/credit accounts
INSERT INTO transactions (
    account_id,
    card_id,
    merchant_id,
    amount,
    currency,
    status,
    transaction_type,
    created_at,
    description
)
SELECT
    a.account_id,
    NULL,
    NULL,
    ROUND(((random() * 15) + 5)::numeric, 2) * -1  AS amount, -- -5 to -20
    'USD',
    'completed',
    'fee',
    NOW() - ((gs % 120) || ' days')::interval      AS created_at,
    'Monthly account fee'
FROM accounts a
CROSS JOIN generate_series(1, 2) AS gs
WHERE a.account_type IN ('checking', 'credit')
  AND a.account_id % 10 = 0;

