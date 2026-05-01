"""Job registry — uzun süren işlemler için.

Her job için:
  - status: pending | running | done | error
  - progress: 0..1
  - log_lines: deque (son N satır)
  - subscribers: set[asyncio.Queue] — WebSocket dinleyicileri

Aktif job'lar in-memory tutulur (WebSocket subscription için zorunlu).
Tamamlanan/başarısız job'lar SQLite'a yazılır; restart sonrası sorgulanabilir.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_DB_PATH = Path("data/jobs.db")
_RETENTION_LIMIT = 500   # SQLite'ta saklanacak max job sayısı
_MEMORY_LIMIT = 50       # In-memory tutulacak max job sayısı

logger = logging.getLogger(__name__)


def _init_db() -> None:
    """DB'yi oluştur ve WAL modunu etkinleştir."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_DB_PATH, timeout=10) as conn:
        # WAL: okuyucular yazarı, yazarlar okuyucuyu bloklamaz
        conn.execute("PRAGMA journal_mode=WAL")
        # WAL ile NORMAL güvenli ve FULL'dan hızlı
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                result TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                finished_at REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)")
        conn.commit()


@contextmanager
def _db():
    # timeout=5: kilit beklenirken 5 saniyede SQLITE_BUSY yerine graceful wait
    conn = sqlite3.connect(_DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


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
        self.touch()  # N31: yayın = heartbeat
        dead: list[asyncio.Queue] = []
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # N29: Taşan kuyruk → o subscriber'ı ölü say, listeden çıkar
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
        _persist_job(self)

    async def fail(self, error: str) -> None:
        self.status = "error"
        self.error = error
        await self.publish(JobEvent(type="error", data=error))
        _persist_job(self)

    async def cancel(self) -> None:
        self.cancelled = True
        await self.fail("İptal edildi")


def _persist_job(job: Job) -> None:
    try:
        result_json = json.dumps(job.result, ensure_ascii=False) if job.result is not None else None
        with _db() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO jobs (id, status, progress, result, error, created_at, finished_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (job.id, job.status, job.progress, result_json,
                  job.error, job._created_at, time.time()))
    except sqlite3.OperationalError as exc:
        # WAL modu ile nadir ama olası: disk dolu, izin hatası, vb.
        logger.warning("Job persist başarısız (job_id=%s): %s", job.id, exc)
    except Exception as exc:
        logger.error("Job persist beklenmedik hata (job_id=%s): %s", job.id, exc)


class JobRegistry:
    """Aktif job'lar in-memory; biten job'lar SQLite'ta."""

    def __init__(self) -> None:
        _init_db()
        self._jobs: dict[str, Job] = {}

    def create(self) -> Job:
        jid = uuid.uuid4().hex[:12]
        job = Job(id=jid)
        self._jobs[jid] = job
        try:
            with _db() as conn:
                conn.execute(
                    "INSERT INTO jobs (id, status, progress, created_at) VALUES (?, ?, ?, ?)",
                    (jid, "pending", 0.0, job._created_at),
                )
        except Exception as exc:
            logger.warning("Job kayıt oluşturulamadı (job_id=%s): %s", jid, exc)
        # Bellek ve disk temizliği — her 20 job'da bir otomatik çalışır
        if len(self._jobs) % 20 == 0:
            self.cleanup_old()
        return job

    def get(self, jid: str) -> Optional[Job]:
        if jid in self._jobs:
            return self._jobs[jid]
        # Bellek'te yoksa SQLite'tan yükle (tamamlanmış job sorgusu)
        with _db() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE id = ?", (jid,)).fetchone()
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

    def all(self) -> list[Job]:
        return list(self._jobs.values())

    def cleanup_old(self, keep: int = _MEMORY_LIMIT) -> None:
        """Bellekte en son N job'ı tut; SQLite'ta _RETENTION_LIMIT'i aşanları sil."""
        # N31: Stale running job'ları "stale" statüsüne al
        for job in list(self._jobs.values()):
            if job.is_stale:
                job.status = "stale"
                _persist_job(job)
                logger.warning(
                    "Stale job tespit edildi (job_id=%s, last_heartbeat=%.0fs önce)",
                    job.id, time.time() - job.last_heartbeat,
                )

        if len(self._jobs) > keep:
            sorted_ids = list(self._jobs.keys())
            for jid in sorted_ids[:-keep]:
                self._jobs.pop(jid, None)
        try:
            with _db() as conn:
                deleted = conn.execute(f"""
                    DELETE FROM jobs WHERE id NOT IN (
                        SELECT id FROM jobs ORDER BY created_at DESC LIMIT {_RETENTION_LIMIT}
                    )
                """).rowcount
            if deleted:
                logger.info("Eski job kayıtları temizlendi: %d satır silindi.", deleted)
        except Exception as exc:
            logger.warning("Job temizleme başarısız: %s", exc)


# Modül-seviyesi singleton
registry = JobRegistry()
