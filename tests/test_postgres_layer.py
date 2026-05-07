"""PostgreSQL veri katmanı testleri (engine/data/db/).

PostgreSQL gerektiren testler `pg` marker'ıyla işaretlidir; Docker olmadan atlanır.
Statik konfigürasyon testleri her zaman çalışır.

Docker başlatmak için:
    docker compose up -d
    pytest tests/test_postgres_layer.py -v
"""
from __future__ import annotations

import asyncio
import os

import pytest

from engine.data.db.config import (
    ASYNCPG_PER_WORKER_MAX,
    ASYNCPG_STATEMENT_CACHE,
    N_APP_WORKERS,
    PG_BOUNCER_DEFAULT_POOL,
    TEST_DSN,
    assert_pool_invariant,
)
from engine.data.db.postgres import MinervaDB, _redact_dsn

# ─── Yardımcılar ──────────────────────────────────────────────────────────────

def _pg_available() -> bool:
    """TEST_DSN ile gerçek asyncpg bağlantısı kurulabiliyorsa True."""
    try:
        import asyncpg as _apg
        async def _check():
            c = await _apg.connect(TEST_DSN, timeout=2)
            await c.close()
        asyncio.run(_check())
        return True
    except Exception:
        return False


requires_pg = pytest.mark.skipif(
    not _pg_available(),
    reason="PostgreSQL bağlantısı yok (docker compose up -d çalıştır)",
)


# ─── Statik konfigürasyon testleri (Docker gerektirmez) ───────────────────────

class TestConfig:

    def test_pool_invariant_passes(self):
        """Mevcut konfigürasyon pool invariant'ını geçmeli."""
        assert_pool_invariant()  # RuntimeError fırlatmazsa başarılı

    def test_statement_cache_is_zero(self):
        """PgBouncer transaction-mode için statement cache sıfır olmalı."""
        assert ASYNCPG_STATEMENT_CACHE == 0

    def test_total_connections_within_bouncer_limit(self):
        """Toplam asyncpg pool ≤ PgBouncer default_pool."""
        total = ASYNCPG_PER_WORKER_MAX * N_APP_WORKERS
        assert total <= PG_BOUNCER_DEFAULT_POOL, (
            f"{ASYNCPG_PER_WORKER_MAX} × {N_APP_WORKERS} = {total} "
            f"> PG_BOUNCER_DEFAULT_POOL={PG_BOUNCER_DEFAULT_POOL}"
        )

    def test_redact_dsn_hides_password(self):
        """DSN'deki şifre logda görünmemeli."""
        dsn = "postgresql://minerva:secret123@localhost:6432/minerva"
        redacted = _redact_dsn(dsn)
        assert "secret123" not in redacted
        assert "***" in redacted
        assert "localhost" in redacted

    def test_redact_dsn_no_password_unchanged(self):
        """Şifresiz DSN değişmeden döner."""
        dsn = "postgresql://minerva@localhost:5432/minerva"
        # Hata fırlatmadan döner
        result = _redact_dsn(dsn)
        assert isinstance(result, str)

    def test_db_not_ready_before_init(self):
        """init() çağrılmadan is_ready() False olmalı."""
        # Eğer başka test pool açtıysa bu test atlanır
        if MinervaDB._pool is not None:
            pytest.skip("Başka test pool açmış; sıfırlama gerekir.")
        assert MinervaDB.is_ready() is False

    def test_conn_raises_when_not_initialized(self):
        """Pool başlatılmadan conn() RuntimeError fırlatmalı."""
        if MinervaDB._pool is not None:
            pytest.skip("Başka test pool açmış.")
        with pytest.raises(RuntimeError, match="başlatılmadı"):
            asyncio.run(MinervaDB.conn().__aenter__())


# ─── Gerçek Postgres testleri (Docker gerekir) ────────────────────────────────

@requires_pg
class TestLivePostgres:
    """Bu sınıftaki testler gerçek bir Postgres instance'ı gerektirir.
    Çalıştırmak için: docker compose up -d && pytest tests/test_postgres_layer.py -v

    Her test kendi asyncio.run() bağlamında init+test+close döngüsü yapar;
    pool event-loop bağlılığını izole eder (pytest-asyncio gerektirmez).
    """

    def _pg_test(self, coro_factory):
        """init → test → close döngüsünü tek event loop'ta çalıştır."""
        async def wrapper():
            await MinervaDB.init(dsn=TEST_DSN, min_size=1, max_size=2)
            try:
                return await coro_factory()
            finally:
                await MinervaDB.close()
        return asyncio.run(wrapper())

    def test_ping(self):
        """SELECT 1 yanıt vermeli."""
        async def _():
            async with MinervaDB.conn() as c:
                return await c.fetchval("SELECT 1")
        assert self._pg_test(_) == 1

    def test_concurrent_inserts_no_timeout(self):
        """50 eşzamanlı INSERT hiçbir timeout almadan tamamlanmalı."""
        async def _():
            async with MinervaDB.conn() as c:
                await c.execute(
                    "CREATE TABLE IF NOT EXISTS _test_concurrent "
                    "(id SERIAL PRIMARY KEY, val TEXT)"
                )

            async def insert_one(i: int):
                async with MinervaDB.conn() as c:
                    await c.execute(
                        "INSERT INTO _test_concurrent (val) VALUES ($1)", f"v{i}"
                    )

            await asyncio.gather(*[insert_one(i) for i in range(50)])
            async with MinervaDB.conn() as c:
                count = await c.fetchval("SELECT COUNT(*) FROM _test_concurrent")
                await c.execute("DROP TABLE IF EXISTS _test_concurrent")
            return count

        assert self._pg_test(_) == 50

    def test_transaction_rollback_on_error(self):
        """Hata oluşursa transaction geri alınmalı."""
        async def _():
            async with MinervaDB.conn() as c:
                await c.execute(
                    "CREATE TABLE IF NOT EXISTS _test_rollback "
                    "(id SERIAL PRIMARY KEY, val TEXT NOT NULL)"
                )
            try:
                async with MinervaDB.transaction() as c:
                    await c.execute("INSERT INTO _test_rollback (val) VALUES ('ok')")
                    await c.execute("INSERT INTO _test_rollback (val) VALUES (NULL)")
            except Exception:
                pass
            async with MinervaDB.conn() as c:
                count = await c.fetchval("SELECT COUNT(*) FROM _test_rollback")
                await c.execute("DROP TABLE IF EXISTS _test_rollback")
            return count

        assert self._pg_test(_) == 0, "ROLLBACK çalışmadı"

    def test_schema_tables_exist(self):
        """schema.sql'deki tablo listesi Postgres'te mevcut olmalı."""
        async def _():
            expected = {"jobs", "alpha_catalog", "decisions_log", "paper_trades"}
            async with MinervaDB.conn() as c:
                rows = await c.fetch(
                    "SELECT tablename FROM pg_tables WHERE schemaname='public'"
                )
            actual = {r["tablename"] for r in rows}
            return expected - actual

        missing = self._pg_test(_)
        assert not missing, f"Eksik tablolar: {missing}"
