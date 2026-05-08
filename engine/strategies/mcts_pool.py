"""Shared-Memory MCTS Worker Pool — zombie-safe, Apple Silicon uyumlu.

Problemin özü:
  meta_optimizer.py her Optuna trial'ında market_db.parquet'i RAM'e yükler.
  50+ trial × ~300 MB = 15 GB → OOM. Paralel çalıştırınca daha da kötü.

Çözüm:
  market_db bir kez multiprocessing.shared_memory.SharedMemory'e kopyalanır.
  N worker process bu bloğu COPY YAPMADAN numpy view olarak okur.
  16 GB RAM'de 4 worker → tek kopya ~300 MB, toplam overhead ~50 MB.

Zombie-RAM koruması (3 katman, Apple Silicon UNIX semantiği):
  1. try/finally  → normal çıkış + Python exception
  2. atexit hook  → interpreter shutdown, sys.exit(), KeyboardInterrupt
  3. SIGINT/SIGTERM handler → Ctrl+C, kill, macOS OOM-killer

Kullanım:
    from engine.strategies.mcts_pool import run_parallel_mining

    results = run_parallel_mining(
        db=market_db_df,
        alpha_cfg=cfg,
        mining_cfg=mcfg,
        prob_df=prob_df,
        n_workers=4,
    )
"""
from __future__ import annotations

import atexit
import contextlib
import logging
import multiprocessing as mp
import os
import signal
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd
from multiprocessing import shared_memory

from engine.util.shm_registry import cleanup_stale, register, unregister

logger = logging.getLogger(__name__)

_SHM_PREFIX = "minerva_mcts_"


# ─── RAII shared-memory context manager ──────────────────────────────────────

@contextlib.contextmanager
def shared_array(arr_np: np.ndarray):
    """RAII wrapper: numpy array'i shared memory'e koy; blok çıkışında temizle.

    Yields (shm_name: str, view: np.ndarray, shape: tuple, dtype: str)

    3 katmanlı zombie koruması:
      1. try/finally
      2. atexit
      3. SIGINT + SIGTERM handler
    """
    name = f"{_SHM_PREFIX}{uuid.uuid4().hex[:8]}"
    shm = shared_memory.SharedMemory(create=True, size=max(arr_np.nbytes, 1))
    cleaned = {"done": False}

    def _cleanup(*_):
        if cleaned["done"]:
            return
        cleaned["done"] = True
        with contextlib.suppress(Exception):
            shm.close()
        with contextlib.suppress(Exception):
            shm.unlink()
        with contextlib.suppress(Exception):
            unregister(name)
        logger.debug("SHM temizlendi: %s", name)

    register(name)
    atexit.register(_cleanup)
    prev_int  = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)

    def _handle_sigint(*_):
        _cleanup()
        # Önceki handler'ı restore et, sonra sinyali yeniden gönder
        # (signal maskı handler çalışırken kaldırılmış olur — CPython garantisi)
        signal.signal(signal.SIGINT, prev_int)
        signal.raise_signal(signal.SIGINT)

    def _handle_sigterm(*_):
        _cleanup()
        os._exit(143)

    signal.signal(signal.SIGINT,  _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigterm)

    try:
        buf = np.ndarray(arr_np.shape, dtype=arr_np.dtype, buffer=shm.buf)
        buf[:] = arr_np
        yield name, buf, arr_np.shape, str(arr_np.dtype)
    finally:
        signal.signal(signal.SIGINT,  prev_int)
        signal.signal(signal.SIGTERM, prev_term)
        _cleanup()


# ─── Worker prosesi — pickle ile veri alır (SHM yerine) ─────────────────────
# macOS + Python 3.14 + spawn context'te SHM race condition var:
# worker shm_open() çağırdığında blok henüz kernel namespace'e yayılmamış.
# Çözüm: DB'yi pickle bytes olarak her worker çağrısına geç.
# train_df (~15 MB) × 4 worker = ~60 MB overhead — 16 GB'da kabul edilebilir.

def _worker_run_trial(
    trial_seed: int,
    db_bytes: bytes,
    alpha_cfg_bytes: bytes,
    mining_cfg_bytes: bytes,
    arrow_path: "str | None" = None,
    arrow_start: "str | None" = None,
    arrow_end:   "str | None" = None,
) -> list:
    """Worker'da tek bir MCTS mining trial çalıştır.

    Faz 1.1: arrow_path verilmişse pickle yerine memory-mapped Arrow okur
    (zero-copy, worker'lar arası paylaşım).
    """
    import pickle
    alpha_cfg  = pickle.loads(alpha_cfg_bytes)
    mining_cfg = pickle.loads(mining_cfg_bytes)
    mining_cfg.seed = trial_seed

    db_window = None
    if arrow_path:
        try:
            from engine.data.arrow_db import MarketDB
            db = MarketDB(arrow_path)
            if arrow_start and arrow_end:
                db_window = db.slice_pandas(arrow_start, arrow_end)
            else:
                db_window = db.to_pandas(cache=False)
        except Exception as exc:
            import sys
            print(f"[worker {trial_seed}] Arrow yolu başarısız ({exc!r}) — pickle fallback",
                  file=sys.stderr, flush=True)
            db_window = None
    if db_window is None:
        if not db_bytes:
            raise RuntimeError(
                f"Worker trial seed={trial_seed}: hem arrow hem pickle yolu boş"
            )
        db_window = pickle.loads(db_bytes)

    from engine.strategies.mining_runner import run_mining_window
    return run_mining_window(db_window, alpha_cfg, mining_cfg)


# ─── Paralel mining orchestrator ─────────────────────────────────────────────

def run_parallel_mining(
    db: pd.DataFrame,
    alpha_cfg,
    mining_cfg,
    prob_df: pd.DataFrame,
    n_workers: int = 4,
    n_trials: int | None = None,
) -> list:
    """N worker'da paralel MCTS mining (pickle tabanlı, SHM yok).

    Parameters
    ----------
    db          : Train penceresi DataFrame (her worker'a pickle ile kopyalanır).
    alpha_cfg   : AlphaCFG instance.
    mining_cfg  : MiningConfig instance.
    prob_df     : Rejim olasılık matrisi (mining_cfg.prob_df olarak set edilir).
    n_workers   : Paralel worker sayısı.
    n_trials    : Kaç paralel trial. None → n_workers.

    Returns
    -------
    list[MiningResult]  — Tüm worker'lardan gelen sonuçların birleşimi.
    """
    import pickle

    n_trials = n_trials or n_workers
    mp_ctx = mp.get_context("spawn")  # macOS: fork + Torch güvensiz

    # prob_df'i mining_cfg'ye bağla (her worker kendi kopyasını alır)
    if prob_df is not None and not prob_df.empty:
        mining_cfg = pickle.loads(pickle.dumps(mining_cfg))  # shallow copy
        mining_cfg.prob_df = prob_df

    # Faz 1.1: Arrow memory-mapped DB (zero-copy worker sharing)
    arrow_path: str | None = None
    arrow_start: str | None = None
    arrow_end:   str | None = None
    try:
        from engine.data.arrow_db import materialize_arrow_db
        import os, tempfile
        # db DataFrame'i geçici parquet'e yaz, Arrow'a dönüştür
        # (tek seferlik; sonraki trial'lar mmap ile zero-copy okur)
        tmp_dir = os.environ.get("MINERVA_ARROW_DIR", tempfile.gettempdir())
        os.makedirs(tmp_dir, exist_ok=True)
        tmp_parquet = os.path.join(tmp_dir, f"mining_db_{os.getpid()}.parquet")
        db.to_parquet(tmp_parquet, index=False)
        arrow_path = materialize_arrow_db(tmp_parquet,
                                           out_path=tmp_parquet.replace(".parquet", ".arrow"))
        if "Date" in db.columns:
            arrow_start = str(db["Date"].min())
            arrow_end   = str(db["Date"].max())
        logger.info("Arrow mmap aktif: %s", arrow_path)
    except Exception as exc:
        logger.debug("Arrow yolu başarısız (%s) — pickle fallback", exc)
        arrow_path = None

    # NOT: Arrow yolu çalışsa bile pickle bytes'i gönder — worker'da
    # Arrow başarısız olursa fallback işlesin (boş bytes ile pickle.loads patlar).
    db_bytes       = pickle.dumps(db)
    alpha_bytes    = pickle.dumps(alpha_cfg)
    mining_bytes   = pickle.dumps(mining_cfg)

    logger.info(
        "Paralel mining başlıyor: %d trial × %d worker  DB=%.1f MB  arrow=%s",
        n_trials, n_workers,
        (len(db_bytes) if db_bytes else 0) / 1e6,
        bool(arrow_path),
    )

    all_results: list = []
    seeds = [mining_cfg.seed + i for i in range(n_trials)]

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as pool:
        futures = {
            pool.submit(
                _worker_run_trial,
                seed,
                db_bytes,
                alpha_bytes,
                mining_bytes,
                arrow_path,
                arrow_start,
                arrow_end,
            ): seed
            for seed in seeds
        }
        for future in as_completed(futures):
            seed = futures[future]
            try:
                results = future.result()
                all_results.extend(results)
                logger.debug("Trial seed=%d tamamlandı: %d formül", seed, len(results))
            except Exception as exc:
                logger.warning("Trial seed=%d başarısız: %s", seed, exc)

    logger.info("Paralel mining bitti: toplam %d formül (%d trial)", len(all_results), n_trials)
    return all_results
