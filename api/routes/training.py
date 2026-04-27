"""Tree-LSTM eğitim endpoint'leri.

GET  /api/training/buffer   → replay buffer durumu
POST /api/training/run      → async job: train_epochs, progress stream
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter
from pydantic import BaseModel

from api.deps import get_brain
from api.jobs import registry

router = APIRouter()


@router.get("/buffer")
def buffer_status() -> dict:
    """Replay buffer boyutu + kapasite."""
    brain = get_brain()
    buf = brain["buffer"]
    return {"size": len(buf), "capacity": getattr(buf, "capacity", 10_000)}


class TrainRequest(BaseModel):
    epochs: int = 5
    batch_size: int = 32
    use_policy: bool = False


class TrainResponse(BaseModel):
    job_id: str


@router.post("/run", response_model=TrainResponse)
async def run_training(req: TrainRequest) -> TrainResponse:
    """Tree-LSTM eğitim job'ı başlat."""
    job = registry.create()
    job.status = "running"

    loop = asyncio.get_event_loop()

    async def _run():
        try:
            await job.emit_log("Tree-LSTM eğitimi başlatılıyor…")
            await job.emit_progress(0.02)

            brain = get_brain()
            buf = brain["buffer"]
            trainer = brain["trainer"]

            if len(buf) < 2:
                await job.emit_log(f"Buffer çok küçük ({len(buf)} formül) — önce formül parse edin")
                await job.fail("Buffer yeterli değil")
                return

            await job.emit_log(f"Buffer: {len(buf)} formül · {req.epochs} epoch başlıyor")
            epochs_done = [0]

            def progress_cb(epoch: int, total: int, stats: dict) -> None:
                epochs_done[0] = epoch
                pct = epoch / max(total, 1)
                asyncio.run_coroutine_threadsafe(
                    job.emit_progress(0.05 + pct * 0.90), loop
                )
                asyncio.run_coroutine_threadsafe(
                    job.emit_log(
                        f"epoch {epoch}/{total} · loss={stats['total']:.4f} · val={stats['value']:.4f}"
                    ),
                    loop,
                )

            history = await asyncio.to_thread(
                trainer.train_epochs,
                buf,
                req.epochs,
                req.batch_size,
                req.use_policy,
                progress_cb,
            )

            await asyncio.to_thread(trainer.save)
            await job.emit_log(f"Model kaydedildi — {len(history)} epoch tamamlandı")

            loss_curve = [h["total"] for h in history]
            last = history[-1] if history else {}
            await job.finish({
                "epochs_done": len(history),
                "last_loss": last.get("total"),
                "last_val_loss": last.get("value"),
                "buffer_size": len(buf),
                "loss_curve": loss_curve,
            })

        except Exception as exc:
            await job.emit_log(f"HATA: {exc}")
            await job.fail(str(exc))

    asyncio.create_task(_run())
    return TrainResponse(job_id=job.id)
