"""POST /api/mining/start — Mining job endpoint.

Mining, CPU-bound'dur; asyncio.to_thread ile thread pool'da koşar.
Progress callback → job.emit_progress (asyncio loop safe).
Sonuçlar data/alpha_catalog.json'a kaydedilir.

N32/N60: asyncio.wait_for sadece coroutine'i iptal eder; altındaki thread
çalışmaya devam eder. Bunu önlemek için her job'a bir threading.Event
(cancel_event) oluşturup mining döngüsüne geçiriyoruz. Döngü her iterasyonda
event'i kontrol eder; set edilmişse CancelledError fırlatır.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.deps import get_cfg, get_market_db, get_split_date, verify_api_key
from api.jobs import registry
from engine.core.alpha_catalog import save_alpha
from engine.strategies.mining_runner import MiningConfig, run_mining_window

logger = logging.getLogger(__name__)
router = APIRouter(dependencies=[Depends(verify_api_key)])

# Mining timeout: 30 dakika hard cap (hung process koruması)
MINING_TIMEOUT_SECONDS = 1800

# Eş zamanlı en fazla 2 mining job
_MAX_CONCURRENT_MINING = 2
_mining_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_MINING)

# job_id → threading.Event: job iptal edildiğinde thread'e sinyal gönderir
_cancel_events: dict[str, threading.Event] = {}


class MiningRequest(BaseModel):
    window: str = Field(default="train", pattern="^(train|all)$")
    num_gen: int = Field(default=200, ge=1, le=2_000)
    max_K: int = Field(default=15, ge=4, le=30)
    use_wf_fitness: bool = True
    wf_n_folds: int = Field(default=5, ge=2, le=20)
    wf_embargo: int = Field(default=5, ge=0, le=30)
    wf_purge: int = Field(default=10, ge=0, le=60)
    lambda_std: float = Field(default=2.0, ge=0.0, le=20.0)
    lambda_cx: float = Field(default=0.003, ge=0.0, le=1.0)
    lambda_size: float = Field(default=0.5, ge=0.0, le=10.0)
    size_corr_hard_limit: float = Field(default=0.7, ge=0.1, le=1.0)
    neutralize: bool = True
    seed: int = Field(default=42, ge=0)
    min_mean_ric: float = Field(default=0.003, ge=0.0, le=1.0)
    min_pos_ratio: float = Field(default=0.4, ge=0.0, le=1.0)
    save_to_catalog: bool = True


class MiningStartResponse(BaseModel):
    job_id: str


@router.post("/start", response_model=MiningStartResponse)
async def start_mining(req: MiningRequest) -> MiningStartResponse:
    if not _mining_semaphore._value:  # type: ignore[attr-defined]
        raise HTTPException(
            status_code=429,
            detail=f"Eş zamanlı en fazla {_MAX_CONCURRENT_MINING} mining job çalışabilir.",
        )
    job = registry.create()
    job.status = "running"

    # N32/N60: Her job için ayrı iptal event'i oluştur.
    # Thread bu event'i periyodik olarak kontrol eder; set edilince durur.
    cancel_event = threading.Event()
    _cancel_events[job.id] = cancel_event

    loop = asyncio.get_event_loop()

    async def _run():
        async with _mining_semaphore:
            try:
                await job.emit_log("Mining başlatılıyor…")
                await job.emit_progress(0.01)

                db = get_market_db()
                split = get_split_date()
                cfg = get_cfg()

                if req.window == "train":
                    db_slice = db[db["Date"] < split].copy()
                elif req.window == "test":
                    db_slice = db[db["Date"] >= split].copy()
                else:
                    db_slice = db.copy()

                await job.emit_log(
                    f"Veri: {len(db_slice):,} satır — {req.num_gen} nesil koşulacak"
                )

                mcfg = MiningConfig(
                    num_gen=req.num_gen,
                    max_K=req.max_K,
                    use_wf_fitness=req.use_wf_fitness,
                    wf_n_folds=req.wf_n_folds,
                    wf_embargo=req.wf_embargo,
                    wf_purge=req.wf_purge,
                    lambda_std=req.lambda_std,
                    lambda_cx=req.lambda_cx,
                    lambda_size=req.lambda_size,
                    size_corr_hard_limit=req.size_corr_hard_limit,
                    neutralize=req.neutralize,
                    seed=req.seed,
                    min_mean_ric=req.min_mean_ric,
                    min_pos_ratio=req.min_pos_ratio,
                )

                last_pct = [0.0]
                _cb_lock = threading.Lock()

                def progress_cb(done: int, total: int) -> None:
                    # N32/N60: İptal event'i kontrol et — set edilmişse dur.
                    if cancel_event.is_set():
                        raise InterruptedError("mining_cancelled")
                    pct = done / max(total, 1)
                    with _cb_lock:
                        if pct - last_pct[0] < 0.02 and done != total:
                            return
                        last_pct[0] = pct
                    asyncio.run_coroutine_threadsafe(
                        job.emit_progress(0.05 + pct * 0.90),
                        loop,
                    )

                def _mining_with_cancel() -> list:
                    """cancel_event set edilirse InterruptedError fırlatır."""
                    return run_mining_window(
                        db_slice, cfg, mcfg,
                        None, None, None,
                        progress_cb,
                    )

                # Timeout ile çalıştır: 30 dk hard cap (hung process koruması).
                # asyncio.wait_for timeout'ta task'ı iptal eder; cancel_event.set()
                # sayesinde alttaki thread de bir sonraki progress_cb çağrısında durur.
                try:
                    results = await asyncio.wait_for(
                        asyncio.to_thread(_mining_with_cancel),
                        timeout=MINING_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    cancel_event.set()  # Thread'e dur sinyali gönder
                    await job.emit_log(
                        f"TIMEOUT: Mining {MINING_TIMEOUT_SECONDS // 60} dakika "
                        "içinde tamamlanamadı — iptal edildi"
                    )
                    await job.fail("mining_timeout")
                    return
                except InterruptedError:
                    await job.emit_log("Mining kullanıcı tarafından iptal edildi.")
                    await job.fail("mining_cancelled")
                    return

                await job.emit_log(
                    f"Mining tamamlandı — {len(results)} formül kabul edildi"
                )

                if req.save_to_catalog and results:
                    saved = 0
                    failed = 0
                    for r in results:
                        try:
                            save_alpha(
                                formula=r.formula,
                                tree=r.tree,
                                ic=r.mean_ric,
                                rank_ic=r.mean_ric,
                                adj_ic=r.mean_ric - r.std_ric,
                                source="evolution",
                                wf_mean_ric=r.mean_ric,
                                wf_std_ric=r.std_ric,
                                wf_pos_folds=r.pos_folds,
                                wf_n_folds=r.n_folds,
                                wf_fitness=r.fitness,
                            )
                            saved += 1
                        except Exception as save_exc:
                            failed += 1
                            logger.warning(
                                "save_alpha failed for %s: %s",
                                r.formula[:40], save_exc,
                            )
                    await job.emit_log(
                        f"{saved} formül kataloğa kaydedildi"
                        + (f" · {failed} kayıt hatalı" if failed else "")
                    )

                summary = [
                    {
                        "formula": r.formula,
                        "fitness": r.fitness,
                        "mean_ric": r.mean_ric,
                        "std_ric": r.std_ric,
                        "pos_folds": r.pos_folds,
                        "n_folds": r.n_folds,
                        "size_corr": r.size_corr,
                    }
                    for r in results[:50]  # İlk 50 WS üzerinden dön
                ]

                await job.finish({"results": summary, "accepted": len(results)})

            except Exception as exc:
                await job.emit_log(f"HATA: {exc}")
                await job.fail(str(exc))
            finally:
                # Temizlik: cancel_event'i kayıt defterinden kaldır
                _cancel_events.pop(job.id, None)

    asyncio.create_task(_run())
    return MiningStartResponse(job_id=job.id)


@router.post("/cancel/{job_id}")
async def cancel_mining(job_id: str) -> dict:
    """
    N32/N60: Çalışan mining job'ını iptal et.

    cancel_event set edilir → thread bir sonraki progress_cb çağrısında
    InterruptedError fırlatır → mining döngüsü temiz biçimde durur.
    asyncio.wait_for ile sarılmış task da iptal edilir.
    """
    job = registry.get(job_id)
    if not job:
        raise HTTPException(404, "Job bulunamadı")
    if job.status != "running":
        return {"job_id": job_id, "status": job.status, "cancelled": False}

    event = _cancel_events.get(job_id)
    if event:
        event.set()
        logger.info("cancel_event set edildi: job_id=%s", job_id)
    else:
        logger.warning("cancel_event bulunamadı: job_id=%s (zaten bitmiş olabilir)", job_id)

    return {"job_id": job_id, "cancelled": True}


@router.get("/status/{job_id}")
async def mining_status(job_id: str) -> dict:
    job = registry.get(job_id)
    if not job:
        raise HTTPException(404, "Job bulunamadı")
    return {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "log_tail": list(job.log_lines)[-20:],
        "result": job.result,
        "error": job.error,
    }
