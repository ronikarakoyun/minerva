"""
GET /api/jobs/{id}    → Job durum snapshot
WS  /ws/jobs/{id}     → progress + log + result event stream

Test endpoint: POST /api/jobs/test/echo  (5 saniyelik fake job)
"""
from __future__ import annotations

import asyncio
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from api.jobs import JobEvent, registry
from api.schemas import JobStatus

router = APIRouter()
ws_router = APIRouter()


@router.get("/{job_id}", response_model=JobStatus)
def get_status(job_id: str) -> JobStatus:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job bulunamadı")
    return JobStatus(
        id=job.id, status=job.status, progress=job.progress,
        error=job.error, result=job.result
    )


@router.post("/test/echo")
async def test_echo() -> dict[str, str]:
    """Smoke test — 5 saniye ilerleyen progress'le biten fake job."""
    job = registry.create()

    async def _run() -> None:
        job.status = "running"
        for i in range(1, 6):
            await asyncio.sleep(1)
            await job.emit_progress(i / 5)
            await job.emit_log(f"step {i}/5")
        await job.finish({"ok": True, "n_steps": 5})

    asyncio.create_task(_run())
    return {"job_id": job.id}


@ws_router.websocket("/ws/jobs/{job_id}")
async def ws_job(ws: WebSocket, job_id: str) -> None:
    """
    Job event stream. Kullanım:
      const ws = new WebSocket(`ws://localhost:8000/ws/jobs/${id}`)
      ws.onmessage = (e) => JSON.parse(e.data)  // {type, data}
    """
    job = registry.get(job_id)
    if job is None:
        await ws.close(code=4404, reason="Job bulunamadı")
        return

    await ws.accept()
    queue: asyncio.Queue[JobEvent] = asyncio.Queue(maxsize=256)
    job.subscribers.append(queue)

    # İlk olarak mevcut durumu yolla (geç bağlananlar için)
    try:
        await ws.send_json({"type": "status", "data": {
            "status": job.status,
            "progress": job.progress,
        }})
        for line in list(job.log_lines):
            await ws.send_json({"type": "log", "data": line})

        while True:
            event = await queue.get()
            await ws.send_json({"type": event.type, "data": event.data})
            if event.type in ("result", "error"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        if queue in job.subscribers:
            job.subscribers.remove(queue)
        try:
            await ws.close()
        except Exception:
            pass
