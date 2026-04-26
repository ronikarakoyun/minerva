"""POST /api/mining/start — Mining job endpoint.

Mining, CPU-bound'dur; asyncio.to_thread ile thread pool'da koşar.
Progress callback → job.emit_progress (asyncio loop safe).
Sonuçlar data/alpha_catalog.json'a kaydedilir.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from api.deps import get_cfg, get_market_db, get_split_date
from api.jobs import registry
from engine.alpha_catalog import save_alpha
from engine.mining_runner import MiningConfig, run_mining_window

logger = logging.getLogger(__name__)
router = APIRouter()


class MiningRequest(BaseModel):
    window: str = "train"          # "train" | "all"
    num_gen: int = 200
    max_K: int = 15
    use_wf_fitness: bool = True
    wf_n_folds: int = 5
    wf_embargo: int = 5
    wf_purge: int = 10
    lambda_std: float = 2.0
    lambda_cx: float = 0.003
    lambda_size: float = 0.5
    size_corr_hard_limit: float = 0.7
    neutralize: bool = True
    seed: int = 42
    min_mean_ric: float = 0.003
    min_pos_ratio: float = 0.4
    save_to_catalog: bool = True


class MiningStartResponse(BaseModel):
    job_id: str


@router.post("/start", response_model=MiningStartResponse)
async def start_mining(req: MiningRequest) -> MiningStartResponse:
    job = registry.create()
    job.status = "running"

    loop = asyncio.get_event_loop()

    async def _run():
        try:
            await job.emit_log("Mining başlatılıyor…")
            await job.emit_progress(0.01)

            db = get_market_db()
            split = get_split_date()
            cfg = get_cfg()

            if req.window == "train":
                db_slice = db[db["Date"] < split].copy()
            else:
                db_slice = db.copy()

            await job.emit_log(f"Veri: {len(db_slice):,} satır — {req.num_gen} nesil koşulacak")

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

            total_steps = req.num_gen
            last_pct = [0]

            def progress_cb(done: int, total: int) -> None:
                pct = done / max(total, 1)
                # Sadece %2'lik sıçramalarda yay — çok fazla event olmasın
                if pct - last_pct[0] >= 0.02 or done == total:
                    last_pct[0] = pct
                    asyncio.run_coroutine_threadsafe(
                        job.emit_progress(0.05 + pct * 0.90),
                        loop,
                    )

            results = await asyncio.to_thread(
                run_mining_window,
                db_slice, cfg, mcfg,
                None, None, None,
                progress_cb,
            )

            await job.emit_log(f"Mining tamamlandı — {len(results)} formül kabul edildi")

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
                        logger.warning("save_alpha failed for %s: %s", r.formula[:40], save_exc)
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

    asyncio.create_task(_run())
    return MiningStartResponse(job_id=job.id)


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
