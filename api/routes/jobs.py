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


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str) -> dict[str, str]:
    """Çalışan bir job'ı iptal et."""
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job bulunamadı")
    if job.status not in ("running", "pending"):
        return {"status": job.status, "message": "Job zaten tamamlandı"}
    await job.cancel()
    return {"status": "cancelled", "job_id": job_id}


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
async def ws_job(ws: WebSocket, job_id: str, api_key: str = "") -> None:
    """
    Job event stream.

    N30: API key query param ile basit auth koruması.
          ?api_key=<key> — eksik veya yanlış → 4401 close.
    Kullanım:
      const ws = new WebSocket(`ws://localhost:8000/ws/jobs/${id}?api_key=<key>`)
      ws.onmessage = (e) => JSON.parse(e.data)  // {type, data}
    """
    import os as _os
    _expected_key = _os.getenv("MINERVA_API_KEY", "")
    if _expected_key and api_key != _expected_key:
        await ws.close(code=4401, reason="Yetkisiz — api_key gerekli")
        return

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
            # Heartbeat desteği: event'i 30 sn timeout ile bekle.
            # Süre dolunca ping mesajı kontrol et; gerçek event yoksa döngüye devam et.
            try:
                event = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # Herhangi bir bekleyen client mesajı var mı kontrol et (ping)
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                    import json as _json
                    msg = _json.loads(raw)
                    if msg.get("type") == "ping":
                        await ws.send_json({"type": "pong"})
                except (asyncio.TimeoutError, Exception):
                    pass
                continue

            # Önce gelen client mesajlarını tüket (ping'ler)
            try:
                raw = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                import json as _json
                msg = _json.loads(raw)
                if msg.get("type") == "ping":
                    await ws.send_json({"type": "pong"})
            except (asyncio.TimeoutError, Exception):
                pass

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
