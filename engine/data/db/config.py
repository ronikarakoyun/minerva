"""PostgreSQL + PgBouncer pool boyutları — tek kaynak (TEK YER DEĞİŞTİR).

docker-compose.yml ve MinervaDB.init() bu modülden import eder.
Matematik garantisi:
  ASYNCPG_PER_WORKER_MAX × N_APP_WORKERS ≤ PG_BOUNCER_DEFAULT_POOL ≤ PG_MAX_CONNECTIONS
"""
from __future__ import annotations

import os

# ─── PgBouncer ────────────────────────────────────────────────────────────────
PG_BOUNCER_MAX_CLIENT_CONN: int = 100   # uygulama tarafından gelen toplam max bağlantı
PG_BOUNCER_DEFAULT_POOL: int = 20       # her DB için Postgres'e açık max bağlantı
# max_db_connections: bir veritabanı için PgBouncer'ın açabileceği toplam Postgres bağlantısı.
# default_pool_size + reserve_pool_size = 20 + 5 = 25 → max_db_connections eşleşmeli.
PG_BOUNCER_MAX_DB_CONNECTIONS: int = PG_BOUNCER_DEFAULT_POOL + 5  # = 25

# ─── Postgres ─────────────────────────────────────────────────────────────────
PG_MAX_CONNECTIONS: int = 50            # = pool*2 + reserve(10); postgresql.conf'u eşler

# ─── asyncpg per-process pool ─────────────────────────────────────────────────
# 5 worker × 4 = 20 ≤ PG_BOUNCER_DEFAULT_POOL  ✓
N_APP_WORKERS: int = int(os.getenv("MINERVA_WORKERS", "5"))
ASYNCPG_PER_WORKER_MAX: int = max(1, PG_BOUNCER_DEFAULT_POOL // N_APP_WORKERS)
ASYNCPG_MIN_SIZE: int = 1
ASYNCPG_STATEMENT_CACHE: int = 0        # ZORUNLU: PgBouncer transaction-mode'da hazırlı ifade güvensiz
ASYNCPG_COMMAND_TIMEOUT_S: float = 10.0

# ─── DSN ──────────────────────────────────────────────────────────────────────
# Üretim: PgBouncer portu (6432).  Test: doğrudan Postgres portu (5432).
DEFAULT_DSN: str = os.getenv(
    "MINERVA_PG_DSN",
    "postgresql://minerva:minerva@localhost:6432/minerva",
)
TEST_DSN: str = os.getenv(
    "MINERVA_PG_TEST_DSN",
    "postgresql://minerva:minerva@localhost:5432/minerva_test",
)


def assert_pool_invariant() -> None:
    """Başlangıçta çağrılır; pool boyutu sınır ihlalini erken yakalar."""
    total = ASYNCPG_PER_WORKER_MAX * N_APP_WORKERS
    if total > PG_BOUNCER_DEFAULT_POOL:
        raise RuntimeError(
            f"asyncpg pool invariant ihlali: "
            f"{ASYNCPG_PER_WORKER_MAX} × {N_APP_WORKERS} workers = {total} "
            f"> PG_BOUNCER_DEFAULT_POOL={PG_BOUNCER_DEFAULT_POOL}. "
            f"N_APP_WORKERS veya ASYNCPG_PER_WORKER_MAX'ı düşür."
        )
    if ASYNCPG_STATEMENT_CACHE != 0:
        raise RuntimeError(
            "asyncpg statement_cache_size sıfır olmak zorunda "
            "(PgBouncer transaction-mode ile prepared statement güvensiz)."
        )
