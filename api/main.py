"""
api/main.py — FastAPI entry point.

Kullanım:
  uvicorn api.main:app --reload --port 8000

Frontend: http://localhost:5173 (Vite dev server) — CORS izinli.
"""
from __future__ import annotations

import os
import sys

# Engine modüllerini import edebilmek için repo kökünü path'e ekle
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import backtest, catalog, formulas, jobs as jobs_routes, mining, training

app = FastAPI(
    title="Minerva v3 API",
    version="0.1.0",
    description="Variant C SPA backend — engine/* sarmalayıcısı.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "minerva-v3"}


@app.get("/api/meta")
def meta() -> dict:
    """Frontend'in status-bar'ı için veri özeti."""
    from api.deps import get_benchmark, get_market_db, get_split_date

    db = get_market_db()
    bm = get_benchmark()
    split = get_split_date()

    train_n = int((db["Date"] < split).sum())
    test_n = int((db["Date"] >= split).sum())
    return {
        "train_rows": train_n,
        "test_rows": test_n,
        "split_date": split.strftime("%Y-%m-%d"),
        "benchmark_days": int(len(bm)) if bm is not None else 0,
        "date_min": db["Date"].min().strftime("%Y-%m-%d"),
        "date_max": db["Date"].max().strftime("%Y-%m-%d"),
    }


app.include_router(catalog.router, prefix="/api/catalog", tags=["catalog"])
app.include_router(formulas.router, prefix="/api/formulas", tags=["formulas"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["backtest"])
app.include_router(jobs_routes.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(jobs_routes.ws_router)  # WebSocket /ws/jobs/{id}
app.include_router(mining.router, prefix="/api/mining", tags=["mining"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
