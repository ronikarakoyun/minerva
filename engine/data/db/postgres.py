"""MinervaDB — PgBouncer-arkalı asyncpg connection pool.

Singleton pattern: startup'ta MinervaDB.init() çağrılır,
her yerde MinervaDB.conn() context manager'ı kullanılır.

Kullanım:
    # FastAPI startup event (api/main.py)
    await MinervaDB.init()

    # Route handler
    async with MinervaDB.conn() as c:
        rows = await c.fetch("SELECT id, status FROM jobs ORDER BY created_at DESC LIMIT 20")

    # FastAPI shutdown event
    await MinervaDB.close()
"""
from __future__ import annotations

import contextlib
import logging
from typing import AsyncIterator

import asyncpg

from engine.data.db.config import (
    ASYNCPG_COMMAND_TIMEOUT_S,
    ASYNCPG_MIN_SIZE,
    ASYNCPG_PER_WORKER_MAX,
    ASYNCPG_STATEMENT_CACHE,
    DEFAULT_DSN,
    assert_pool_invariant,
)

logger = logging.getLogger(__name__)


class MinervaDB:
    """asyncpg bağlantı havuzu — modül-seviyesi singleton."""

    _pool: asyncpg.Pool | None = None

    @classmethod
    async def init(
        cls,
        dsn: str | None = None,
        min_size: int = ASYNCPG_MIN_SIZE,
        max_size: int = ASYNCPG_PER_WORKER_MAX,
    ) -> None:
        """Pool'u oluştur. Uygulama startup'ında bir kez çağrılmalı."""
        assert_pool_invariant()
        if cls._pool is not None:
            logger.warning("MinervaDB.init() zaten çağrılmış — atlanıyor.")
            return
        dsn = dsn or DEFAULT_DSN
        cls._pool = await asyncpg.create_pool(
            dsn,
            min_size=min_size,
            max_size=max_size,
            statement_cache_size=ASYNCPG_STATEMENT_CACHE,
            command_timeout=ASYNCPG_COMMAND_TIMEOUT_S,
        )
        logger.info(
            "MinervaDB pool hazır: min=%d max=%d dsn=%s",
            min_size, max_size, _redact_dsn(dsn),
        )

    @classmethod
    async def close(cls) -> None:
        """Pool'u kapat. Uygulama shutdown'ında çağrılmalı."""
        if cls._pool:
            await cls._pool.close()
            cls._pool = None
            logger.info("MinervaDB pool kapatıldı.")

    @classmethod
    def is_ready(cls) -> bool:
        return cls._pool is not None

    @classmethod
    @contextlib.asynccontextmanager
    async def conn(cls) -> AsyncIterator[asyncpg.Connection]:
        """Havuzdan bağlantı al; blok sonunda otomatik iade et."""
        if cls._pool is None:
            raise RuntimeError(
                "MinervaDB henüz başlatılmadı. "
                "Önce await MinervaDB.init() çağır."
            )
        async with cls._pool.acquire() as connection:
            yield connection

    @classmethod
    @contextlib.asynccontextmanager
    async def transaction(cls) -> AsyncIterator[asyncpg.Connection]:
        """Bağlantı + otomatik transaction. Hata olursa ROLLBACK, başarıda COMMIT."""
        async with cls.conn() as c:
            async with c.transaction():
                yield c


def _redact_dsn(dsn: str) -> str:
    """DSN içindeki şifreyi logdan gizle: postgres://user:***@host:port/db."""
    try:
        if "://" in dsn and "@" in dsn:
            prefix, rest = dsn.split("@", 1)
            if ":" in prefix.split("://", 1)[1]:
                parts = prefix.rsplit(":", 1)
                return f"{parts[0]}:***@{rest}"
    except Exception:
        pass
    return dsn
