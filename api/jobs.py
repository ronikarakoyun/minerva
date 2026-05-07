"""Job registry — uzun süren işlemler için.

Her job için:
  - status: pending | running | done | error | stale
  - progress: 0..1
  - log_lines: deque (son N satır)
  - subscribers: set[asyncio.Queue] — WebSocket dinleyicileri

Aktif job'lar in-memory tutulur (WebSocket subscription için zorunlu).
Tamamlanan/başarısız job'lar PostgreSQL'e yazılır; restart sonrası sorgulanabilir.

Hibrit göç: SQLite kaldırıldı; engine/data/db/MinervaDB üzerinden asyncpg kullanılır.
MinervaDB.init() uygulama startup'ında (api/main.py lifespan) çağrılmalıdır.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

_RETENTION_LIMIT = 500   # Postgres'te saklanacak max job sayısı
_MEMORY_LIMIT = 50       # In-memory tutulacak max job sayısı


# ─── MinervaDB lazy import ─────────────────────────────────────────────────────
# Circular import ve test izolasyonu için import'u geç yap.
def _db():
    from engine.data.db.postgres import MinervaDB
    return MinervaDB


# ─── Event / Job ───────────────────────────────────────────────────────────────

@dataclass
class JobEvent:
    type: str          # "progress" | "log" | "result" | "error" | "status"
    data: Any = None


@dataclass
class Job:
    id: str
    status: str = "pending"
    progress: float = 0.0
    result: Any = None
    error: Optional[str] = None
    log_lines: deque = field(default_factory=lambda: deque(maxlen=500))
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    cancelled: bool = False
    _created_at: float = field(default_factory=time.time)

    # N31: last_heartbeat timestamp — stale job tespiti için
    last_heartbeat: float = field(default_factory=time.time)

    def touch(self) -> None:
        """Heartbeat güncelle — en az 30 sn'de bir çağrılmalı."""
        self.last_heartbeat = time.time()

    @property
    def is_stale(self) -> bool:
        """5 dakikadan uzun süre heartbeat yoksa stale kabul et."""
        return self.status == "running" and (time.time() - self.last_heartbeat) > 300

    async def publish(self, event: JobEvent) -> None:
        self.touch()
        dead: list[asyncio.Queue] = []
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
                logger.warning("Subscriber kuyruk taştı — bağlantı koparıldı")
        for q in dead:
            try:
                self.subscribers.remove(q)
            except ValueError:
                pass

    async def emit_progress(self, value: float) -> None:
        self.progress = max(0.0, min(1.0, float(value)))
        await self.publish(JobEvent(type="progress", data=self.progress))

    async def emit_log(self, line: str) -> None:
        self.log_lines.append(line)
        await self.publish(JobEvent(type="log", data=line))

    async def finish(self, result: Any) -> None:
        self.status = "done"
        self.result = result
        await self.publish(JobEvent(type="result", data=result))
        await _persist_job(self)

    async def fail(self, error: str) -> None:
        self.status = "error"
        self.error = error
        await self.publish(JobEvent(type="error", data=error))
        await _persist_job(self)

    async def cancel(self) -> None:
        self.cancelled = True
        await self.fail("İptal edildi")


# ─── Persistence (asyncpg) ────────────────────────────────────────────────────

async def _persist_job(job: Job) -> None:
    """Job'ı PostgreSQL'e kaydet. MinervaDB başlatılmamışsa sessizce atla."""
    db = _db()
    if not db.is_ready():
        logger.debug("MinervaDB hazır değil — job persist atlanıyor (job_id=%s)", job.id)
        return
    result_json = json.dumps(job.result, ensure_ascii=False) if job.result is not None else None
    try:
        async with db.conn() as c:
            await c.execute(
                """
                INSERT INTO jobs (id, status, progress, result, error, created_at, finished_at)
                VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
                ON CONFLICT (id) DO UPDATE
                    SET status      = EXCLUDED.status,
                        progress    = EXCLUDED.progress,
                        result      = EXCLUDED.result,
                        error       = EXCLUDED.error,
                        finished_at = EXCLUDED.finished_at
                """,
                job.id, job.status, job.progress,
                result_json, job.error, job._created_at, time.time(),
            )
    except Exception as exc:
        logger.warning("Job persist başarısız (job_id=%s): %s", job.id, exc)


async def _load_job_from_pg(jid: str) -> Optional[Job]:
    """PostgreSQL'den tamamlanmış bir job'ı yükle."""
    db = _db()
    if not db.is_ready():
        return None
    try:
        async with db.conn() as c:
            row = await c.fetchrow("SELECT * FROM jobs WHERE id = $1", jid)
        if row is None:
            return None
        job = Job(
            id=row["id"],
            status=row["status"],
            progress=row["progress"],
            result=json.loads(row["result"]) if row["result"] else None,
            error=row["error"],
        )
        job._created_at = row["created_at"]
        return job
    except Exception as exc:
        logger.warning("Job PG'den yüklenemedi (job_id=%s): %s", jid, exc)
        return None


async def _cleanup_old_pg(keep: int = _RETENTION_LIMIT) -> None:
    """Postgres'te _RETENTION_LIMIT'i aşan eski job kayıtlarını temizle."""
    db = _db()
    if not db.is_ready():
        return
    try:
        async with db.conn() as c:
            deleted = await c.fetchval(
                """
                WITH oldest AS (
                    SELECT id FROM jobs
                    ORDER BY created_at DESC
                    OFFSET $1
                )
                DELETE FROM jobs WHERE id IN (SELECT id FROM oldest)
                RETURNING 1
                """,
                keep,
            )
        if deleted:
            logger.info("Eski PG job kayıtları temizlendi: %s satır silindi.", deleted)
    except Exception as exc:
        logger.warning("PG job temizleme başarısız: %s", exc)


# ─── JobRegistry ──────────────────────────────────────────────────────────────

class JobRegistry:
    """Aktif job'lar in-memory; biten job'lar PostgreSQL'de."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self) -> Job:
        jid = uuid.uuid4().hex[:12]
        job = Job(id=jid)
        self._jobs[jid] = job
        # Fire-and-forget: startup'ta PG hazır olmayabilir; persist finish/fail'de olur.
        if len(self._jobs) % 20 == 0:
            asyncio.ensure_future(self._cleanup_old())
        return job

    def get_sync(self, jid: str) -> Optional[Job]:
        """In-memory arama (sync). Biten job için get() kullan."""
        return self._jobs.get(jid)

    async def get(self, jid: str) -> Optional[Job]:
        """In-memory varsa döner; yoksa PG'den yükler."""
        if jid in self._jobs:
            return self._jobs[jid]
        return await _load_job_from_pg(jid)

    def all(self) -> list[Job]:
        return list(self._jobs.values())

    async def _cleanup_old(self, keep: int = _MEMORY_LIMIT) -> None:
        # N31: Stale running job'ları işaretle
        for job in list(self._jobs.values()):
            if job.is_stale:
                job.status = "stale"
                await _persist_job(job)
                logger.warning(
                    "Stale job tespit edildi (job_id=%s, last_heartbeat=%.0fs önce)",
                    job.id, time.time() - job.last_heartbeat,
                )
        if len(self._jobs) > keep:
            sorted_ids = list(self._jobs.keys())
            for jid in sorted_ids[:-keep]:
                self._jobs.pop(jid, None)
        await _cleanup_old_pg()


# Modül-seviyesi singleton
registry = JobRegistry()
