"""In-memory job registry — uzun süren işlemler için.

Her job için:
  - status: pending | running | done | error
  - progress: 0..1
  - log_lines: deque (son N satır)
  - subscribers: set[asyncio.Queue] — WebSocket dinleyicileri

Job restart'ta kaybolur (kabul edilen tradeoff — plan §Riskler).
"""
from __future__ import annotations

import asyncio
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional


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

    async def publish(self, event: JobEvent) -> None:
        """Subscriber kuyruklarına event yay."""
        for q in list(self.subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
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

    async def fail(self, error: str) -> None:
        self.status = "error"
        self.error = error
        await self.publish(JobEvent(type="error", data=error))


class JobRegistry:
    """Thread-safe değil — FastAPI tek event loop'ta çalışır, asyncio yeterli."""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self) -> Job:
        jid = uuid.uuid4().hex[:12]
        job = Job(id=jid)
        self._jobs[jid] = job
        return job

    def get(self, jid: str) -> Optional[Job]:
        return self._jobs.get(jid)

    def all(self) -> list[Job]:
        return list(self._jobs.values())

    def cleanup_old(self, keep: int = 50) -> None:
        """En son N job'ı tut, eskileri at."""
        if len(self._jobs) <= keep:
            return
        sorted_ids = list(self._jobs.keys())
        for jid in sorted_ids[:-keep]:
            self._jobs.pop(jid, None)


# Modül-seviyesi singleton
registry = JobRegistry()
