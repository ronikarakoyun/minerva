"""engine/data/db — PostgreSQL veri katmanı (hibrit mimari).

market_db parquet'te kalır (analytic, RAM-dostu, memory-mapped).
Transactional veri (jobs, alpha_catalog, decisions_log, paper_trades)
buradan Postgres'e yazılır/okunur.

Kullanım:
    from engine.data.db import MinervaDB
    await MinervaDB.init()                    # startup'ta bir kez
    async with MinervaDB.conn() as c:
        await c.execute("SELECT 1")
    await MinervaDB.close()                   # shutdown'da
"""
from engine.data.db.postgres import MinervaDB

__all__ = ["MinervaDB"]
