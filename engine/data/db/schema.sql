-- Minerva v3 — PostgreSQL Şema (Transactional veri; market_db parquet'te kalır)
-- Çalıştır: psql -U minerva -d minerva -f engine/data/db/schema.sql
-- Migration: scripts/migrate_parquet_to_pg.py

-- ─── Uzantılar ─────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "pgcrypto";  -- gen_random_uuid()

-- ─── jobs ──────────────────────────────────────────────────────────────────
-- api/jobs.py SQLite tablosunun Postgres karşılığı.
-- Aktif job'lar in-memory tutulur; biten/hata veren job'lar buraya persist edilir.
CREATE TABLE IF NOT EXISTS jobs (
    id          TEXT        PRIMARY KEY,
    status      TEXT        NOT NULL CHECK (status IN ('pending','running','done','error','stale')),
    progress    REAL        NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 1),
    result      JSONB,
    error       TEXT,
    created_at  DOUBLE PRECISION NOT NULL,
    finished_at DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS idx_jobs_created  ON jobs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_status   ON jobs (status);

-- ─── alpha_catalog ─────────────────────────────────────────────────────────
-- data/alpha_catalog.json'ın ilişkisel karşılığı.
-- JSONB metadata kolonu: ic_mean, ic_std, regime_weights, backtest_stats, vb.
CREATE TABLE IF NOT EXISTS alpha_catalog (
    id          TEXT        PRIMARY KEY DEFAULT gen_random_uuid()::TEXT,
    formula     TEXT        NOT NULL,
    ic_mean     REAL,
    ic_std      REAL,
    sharpe      REAL,
    max_dd      REAL,
    n_days      INTEGER,
    regime      TEXT,           -- 'all' ya da 'regime_0'..'regime_K'
    metadata    JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_champion BOOLEAN     NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_alpha_regime     ON alpha_catalog (regime);
CREATE INDEX IF NOT EXISTS idx_alpha_champion   ON alpha_catalog (is_champion) WHERE is_champion;
CREATE INDEX IF NOT EXISTS idx_alpha_ic         ON alpha_catalog (ic_mean DESC NULLS LAST);

-- ─── decisions_log ─────────────────────────────────────────────────────────
-- engine/execution/forensics.py çıktısı (data/decisions_log.parquet'ın yerine).
-- Her trading kararı (hangi alpha, hangi sinyal, hangi miktar) burada kayıtlı.
CREATE TABLE IF NOT EXISTS decisions_log (
    id          BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    trade_date  DATE        NOT NULL,
    ticker      TEXT        NOT NULL,
    alpha_id    TEXT        REFERENCES alpha_catalog(id) ON DELETE SET NULL,
    signal      REAL,
    position    REAL,
    regime      TEXT,
    metadata    JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_decisions_date   ON decisions_log (trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_ticker ON decisions_log (ticker, trade_date DESC);

-- ─── paper_trades ──────────────────────────────────────────────────────────
-- engine/execution/paper_trader.py çıktısı (data/paper_trades.parquet'ın yerine).
CREATE TABLE IF NOT EXISTS paper_trades (
    id          BIGINT      GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    trade_date  DATE        NOT NULL,
    ticker      TEXT        NOT NULL,
    direction   SMALLINT    NOT NULL CHECK (direction IN (-1, 0, 1)),
    size        REAL,
    entry_px    REAL,
    exit_px     REAL,
    pnl         REAL,
    alpha_id    TEXT        REFERENCES alpha_catalog(id) ON DELETE SET NULL,
    metadata    JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_ptrades_date   ON paper_trades (trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_ptrades_ticker ON paper_trades (ticker, trade_date DESC);
