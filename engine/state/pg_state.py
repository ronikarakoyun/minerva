"""PostgreSQL hot state — asyncpg tabanlı (Faz 1.5).

Tablolar:
- rl_agent_state          : Günlük RL state snapshot
- champion_formulas        : PBO sertifikalı şampiyon formüller
- production_parity_log    : Backtest vs shadow fund sapması
- virtual_orders          : Shadow fund decisions (T → T+1)
- virtual_positions        : Açık sanal pozisyonlar

Kullanım:
    from engine.state.pg_state import PGState

    pg = PGState(dsn="postgresql://localhost/minerva")
    await pg.connect()

    await pg.save_rl_state(date=today(), state_dict={...})
    state = await pg.get_latest_rl_state()

    await pg.save_champion(quarter="2017Q1", regime=0, formula_ast={...},
                            in_sample_sharpe=1.5, pbo_score=0.12)
    champions = await pg.list_certified_champions(pbo_max=0.5)
"""
from __future__ import annotations

import json
import logging
from datetime import date as _date, datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rl_agent_state (
    as_of_date DATE PRIMARY KEY,
    portfolio_vol      DOUBLE PRECISION,
    drawdown           DOUBLE PRECISION,
    regime_entropy     DOUBLE PRECISION,
    recent_ic          DOUBLE PRECISION,
    current_scale      DOUBLE PRECISION,
    chosen_action      INTEGER,
    leverage           DOUBLE PRECISION,
    created_at         TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS champion_formulas (
    quarter_id         TEXT,
    regime_id          INTEGER,
    formula_ast        JSONB,
    formula_str        TEXT,
    in_sample_sharpe   DOUBLE PRECISION,
    pbo_score          DOUBLE PRECISION,
    deflated_sharpe    DOUBLE PRECISION,
    deployed_at        TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (quarter_id, regime_id)
);

CREATE TABLE IF NOT EXISTS production_parity_log (
    as_of_date         DATE,
    ticker             TEXT,
    expected_weight    DOUBLE PRECISION,
    actual_weight      DOUBLE PRECISION,
    deviation_bps      DOUBLE PRECISION,
    PRIMARY KEY (as_of_date, ticker)
);

CREATE TABLE IF NOT EXISTS virtual_orders (
    id                 BIGSERIAL PRIMARY KEY,
    decision_date      DATE NOT NULL,
    exec_date          DATE NOT NULL,
    ticker             TEXT NOT NULL,
    target_weight      DOUBLE PRECISION,
    leverage           DOUBLE PRECISION,
    entry_px           DOUBLE PRECISION,
    slippage_pct       DOUBLE PRECISION,
    status             TEXT DEFAULT 'pending',
    created_at         TIMESTAMP DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_virt_orders_exec ON virtual_orders(exec_date, status);

CREATE TABLE IF NOT EXISTS virtual_positions (
    id                 BIGSERIAL PRIMARY KEY,
    open_date          DATE NOT NULL,
    close_date         DATE,
    ticker             TEXT NOT NULL,
    weight             DOUBLE PRECISION,
    entry_px           DOUBLE PRECISION,
    exit_px            DOUBLE PRECISION,
    m2m_pnl            DOUBLE PRECISION,
    realized_pnl       DOUBLE PRECISION,
    status             TEXT DEFAULT 'open'
);
CREATE INDEX IF NOT EXISTS idx_virt_pos_status ON virtual_positions(status, open_date);
"""


class PGState:
    """asyncpg connection pool wrapper."""

    def __init__(self, dsn: str, min_size: int = 2, max_size: int = 10):
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self._pool = None

    async def connect(self) -> None:
        try:
            import asyncpg
        except ImportError as exc:
            raise ImportError(
                "asyncpg gerekli: `pip install asyncpg`"
            ) from exc

        self._pool = await asyncpg.create_pool(
            dsn=self.dsn, min_size=self.min_size, max_size=self.max_size
        )
        async with self._pool.acquire() as conn:
            await conn.execute(SCHEMA_SQL)
        logger.info("PGState bağlandı: %s", self.dsn)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    # ── RL agent state ────────────────────────────────────────────────────────

    async def save_rl_state(self, as_of: _date, **fields) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO rl_agent_state
                  (as_of_date, portfolio_vol, drawdown, regime_entropy,
                   recent_ic, current_scale, chosen_action, leverage)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (as_of_date) DO UPDATE SET
                  portfolio_vol=EXCLUDED.portfolio_vol,
                  drawdown=EXCLUDED.drawdown,
                  regime_entropy=EXCLUDED.regime_entropy,
                  recent_ic=EXCLUDED.recent_ic,
                  current_scale=EXCLUDED.current_scale,
                  chosen_action=EXCLUDED.chosen_action,
                  leverage=EXCLUDED.leverage
            """, as_of,
                fields.get("portfolio_vol"),
                fields.get("drawdown"),
                fields.get("regime_entropy"),
                fields.get("recent_ic"),
                fields.get("current_scale"),
                fields.get("chosen_action"),
                fields.get("leverage"),
            )

    async def get_latest_rl_state(self) -> Optional[dict]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM rl_agent_state ORDER BY as_of_date DESC LIMIT 1"
            )
            return dict(row) if row else None

    # ── Champion formulas ─────────────────────────────────────────────────────

    async def save_champion(self, quarter_id: str, regime_id: int,
                              formula_ast: dict, formula_str: str,
                              in_sample_sharpe: float,
                              pbo_score: Optional[float] = None,
                              deflated_sharpe: Optional[float] = None) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO champion_formulas
                  (quarter_id, regime_id, formula_ast, formula_str,
                   in_sample_sharpe, pbo_score, deflated_sharpe)
                VALUES ($1, $2, $3::jsonb, $4, $5, $6, $7)
                ON CONFLICT (quarter_id, regime_id) DO UPDATE SET
                  formula_ast=EXCLUDED.formula_ast,
                  formula_str=EXCLUDED.formula_str,
                  in_sample_sharpe=EXCLUDED.in_sample_sharpe,
                  pbo_score=EXCLUDED.pbo_score,
                  deflated_sharpe=EXCLUDED.deflated_sharpe
            """, quarter_id, regime_id, json.dumps(formula_ast),
                formula_str, in_sample_sharpe, pbo_score, deflated_sharpe)

    async def list_certified_champions(self, pbo_max: float = 0.5) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM champion_formulas
                WHERE pbo_score IS NULL OR pbo_score <= $1
                ORDER BY quarter_id DESC, regime_id ASC
            """, pbo_max)
            return [dict(r) for r in rows]

    # ── Virtual orders ────────────────────────────────────────────────────────

    async def insert_virtual_orders(self, orders: list[dict]) -> None:
        if not orders:
            return
        async with self._pool.acquire() as conn:
            await conn.executemany("""
                INSERT INTO virtual_orders
                  (decision_date, exec_date, ticker, target_weight, leverage)
                VALUES ($1, $2, $3, $4, $5)
            """, [(o["decision_date"], o["exec_date"], o["ticker"],
                   o["target_weight"], o.get("leverage", 1.0))
                  for o in orders])

    async def fetch_pending_orders(self, exec_date: _date) -> list[dict]:
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM virtual_orders
                WHERE exec_date = $1 AND status = 'pending'
            """, exec_date)
            return [dict(r) for r in rows]

    async def mark_order_filled(self, order_id: int, entry_px: float,
                                  slippage_pct: float) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("""
                UPDATE virtual_orders
                   SET entry_px=$1, slippage_pct=$2, status='filled'
                 WHERE id=$3
            """, entry_px, slippage_pct, order_id)

    # ── Production parity ─────────────────────────────────────────────────────

    async def log_parity(self, as_of: _date, ticker: str,
                          expected: float, actual: float) -> None:
        deviation_bps = abs(expected - actual) * 10000
        async with self._pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO production_parity_log
                  (as_of_date, ticker, expected_weight, actual_weight, deviation_bps)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (as_of_date, ticker) DO UPDATE SET
                  expected_weight=EXCLUDED.expected_weight,
                  actual_weight=EXCLUDED.actual_weight,
                  deviation_bps=EXCLUDED.deviation_bps
            """, as_of, ticker, expected, actual, deviation_bps)

    async def daily_parity_breach_count(self, as_of: _date,
                                          threshold_bps: float = 50.0) -> int:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT COUNT(*) AS n FROM production_parity_log
                 WHERE as_of_date=$1 AND deviation_bps > $2
            """, as_of, threshold_bps)
            return int(row["n"])
