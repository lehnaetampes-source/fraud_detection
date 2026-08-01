-- Initialisation de la base fraud_db
CREATE DATABASE fraud_db;
CREATE DATABASE mlflow;

\c fraud_db;

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id  TEXT PRIMARY KEY,
    amount          FLOAT,
    merchant        TEXT,
    timestamp       TIMESTAMP,
    fraud_score     FLOAT,
    is_fraud        INTEGER,
    predicted_at    TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_is_fraud   ON transactions(is_fraud);
CREATE INDEX idx_timestamp  ON transactions(timestamp);
