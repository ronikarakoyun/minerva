"""
auto_minerva.py — Faz 6: Kurumsal orkestrasyon (Prefect 3.x).

Tüm Faz 1-5 bileşenlerini sırayla çalıştırır:
    1. fetch_bist_data        → data/market_db.parquet
    2. detect_regime (HMM)    → data/regime_prob_df.parquet
    3. nightly_mining (Cuma)  → data/best_params.json (Optuna)
    4. decay_scan             → aktif şampiyon decay kontrolü
    5. morning_execution      → blend + paper trade + adli log

Kullanım:
    python -m auto_minerva                    # tek seferlik manuel koşu
    prefect deployment build auto_minerva.py:run_daily_cycle \\
        -n minerva-daily --cron "30 18 * * 1-5" --timezone "Europe/Istanbul"
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from prefect import flow, get_run_logger, task

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Task 1 — Veri çekimi
# ──────────────────────────────────────────────────────────────────────
@task(name="fetch_data", retries=3, retry_delay_seconds=300, timeout_seconds=900)
def task_fetch_data() -> Path:
    """yfinance → data/market_db.parquet. 3× retry, 5 dk arayla."""
    log = get_run_logger()
    log.info("Faz 6.1: BIST veri çekimi başlıyor")
    from scripts.fetch_bist_data import main as fetch_main
    fetch_main()
    out = Path("data/market_db.parquet")
    if not out.exists():
        raise RuntimeError(f"Veri dosyası üretilmedi: {out}")
    log.info("Veri hazır: %s", out)
    return out


# ──────────────────────────────────────────────────────────────────────
# Task 2 — HMM rejim tespiti (Forward-only / look-ahead safe)
# ──────────────────────────────────────────────────────────────────────
@task(name="detect_regime", retries=2, retry_delay_seconds=60, timeout_seconds=600)
def task_detect_regime(db_path: Path) -> Path:
    """Faz 1: HMM fit + filtered prob_df. Çıktı: data/regime_prob_df.parquet."""
    log = get_run_logger()
    log.info("Faz 1: HMM rejim tespiti")
    from engine.data.regime_detector import RegimeConfig, run_pipeline
    cfg = RegimeConfig(use_filtered_probs=True)  # backtest güvenli
    prob_df = run_pipeline(cfg)
    out = Path("data/regime_prob_df.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    prob_df.to_parquet(out)
    log.info("prob_df kaydedildi: %s (%d gün, K=%d)",
             out, len(prob_df), prob_df.shape[1])
    return out


# ──────────────────────────────────────────────────────────────────────
# Task 3 — Optuna meta-tuning (yalnız Cuma gecesi)
# ──────────────────────────────────────────────────────────────────────
@task(name="nightly_mining", timeout_seconds=3600)
def task_nightly_mining(db_path: Path, prob_path: Path,
                          only_on_weekday: int = 4,
                          n_trials: int = 50) -> dict:
    """
    Faz 3: MCTS+Optuna. Yalnız Cuma gecesi tam kapasite çalışır
    (only_on_weekday=4); diğer günler skip.
    """
    log = get_run_logger()
    today = datetime.now().weekday()
    if only_on_weekday is not None and today != only_on_weekday:
        log.info("nightly_mining skip — bugün %d, hedef gün %d",
                 today, only_on_weekday)
        return {"skipped": True, "reason": f"weekday={today}"}

    log.info("Faz 3: Optuna meta-tuning (n_trials=%d)", n_trials)
    from engine.strategies.meta_optimizer import (
        MetaOptConfig, run_meta_optimization,
    )
    result = run_meta_optimization(MetaOptConfig(n_trials=n_trials))
    log.info("best_value=%.4f params=%s",
             result.get("best_value", 0.0), result.get("best_params"))
    return result


# ──────────────────────────────────────────────────────────────────────
# Task 4 — Decay scan (aktif şampiyonlar için)
# ──────────────────────────────────────────────────────────────────────
@task(name="decay_scan", retries=1)
def task_decay_scan(prob_path: Path) -> dict:
    """
    Tüm aktif şampiyonları decay_monitor üzerinden geçir; en az bir tetik
    varsa raise → flow Failed → automation tetiklenir.
    """
    log = get_run_logger()
    from engine.core.alpha_catalog import get_active_champions
    from engine.execution.paper_trader import feed_decay_monitor

    champions = get_active_champions()
    log.info("decay_scan: %d aktif şampiyon kontrol edilecek", len(champions))

    triggered = []
    for fid, meta in champions:
        try:
            result = feed_decay_monitor(
                fid, meta["backtest_mean"], meta["backtest_std"],
            )
        except Exception as e:
            log.warning("scan hatası (%s): %s", fid, e)
            continue
        if result.get("triggered"):
            triggered.append((fid, result))
            log.warning("DECAY tetikleyici: %s → %s", fid, result)

    if triggered:
        raise RuntimeError(
            f"DECAY: {len(triggered)} şampiyon emekli edilmeli: "
            + ", ".join(fid for fid, _ in triggered)
        )

    return {"checked": len(champions), "triggered": 0}


# ──────────────────────────────────────────────────────────────────────
# Task 5 — Sabah icrası: blend + paper log + forensic
# ──────────────────────────────────────────────────────────────────────
@task(name="morning_execution", retries=1, timeout_seconds=600)
def task_morning_execution(db_path: Path, prob_path: Path) -> dict:
    """
    Faz 5: Önceki günün exit fill + bugünün blend + paper log + adli log.
    """
    log = get_run_logger()
    from engine.core.alpha_cfg import AlphaCFG
    from engine.execution.blender import (
        BlenderConfig, blend_regime_signals, load_champions_from_catalog,
    )
    from engine.execution.forensics import (
        ForensicConfig, log_decision_forensics,
    )
    from engine.execution.paper_trader import (
        PaperTraderConfig, compute_realized_pnl, log_daily_decisions,
    )
    from engine.execution.slippage import SlippageConfig

    db = pd.read_parquet(db_path)
    prob_df = pd.read_parquet(prob_path)
    today = pd.Timestamp(prob_df.index[-1])

    # 1) Önceki günün exit fill (t+2)
    pt_cfg = PaperTraderConfig()
    compute_realized_pnl(db, pt_cfg)
    log.info("compute_realized_pnl tamam — %s öncesi açık trade'ler kapandı", today.date())

    # 2) Şampiyonlar
    catalog_path = Path("data/alpha_catalog.json")
    if not catalog_path.exists():
        log.warning("alpha_catalog.json yok — morning_execution skip")
        return {"logged": 0, "skipped": True, "reason": "no_catalog"}

    alpha_cfg = AlphaCFG()
    champions = load_champions_from_catalog(catalog_path, alpha_cfg=alpha_cfg)
    if not champions:
        log.warning("Aktif şampiyon yok — morning_execution skip")
        return {"logged": 0, "skipped": True, "reason": "no_champions"}

    # 3) Bugünün target weights
    weights_df = blend_regime_signals(
        champions, prob_df, db,
        BlenderConfig(use_blending=True),
        alpha_cfg=alpha_cfg,
    )
    if today not in weights_df.index:
        log.warning("today=%s blend çıktısında yok", today)
        today_weights = pd.Series(dtype=float)
    else:
        today_weights = weights_df.loc[today].dropna()
    today_weights = today_weights[today_weights > 0]

    # 4) Paper trade log
    formula_id = "blend_v6"
    n_logged = log_daily_decisions(
        today_weights, formula_id, today, db,
        slippage_cfg=SlippageConfig(use_dynamic_slippage=True),
        cfg=pt_cfg,
    )
    log.info("paper_trades.parquet: %d satır", n_logged)

    # 5) Adli log
    if today in prob_df.index:
        prob_row = prob_df.loc[today]
        n_forensic = log_decision_forensics(
            date=today, weights=today_weights, prob_row=prob_row,
            champions=champions, db=db, formula_id=formula_id,
            cfg=ForensicConfig(),
        )
        log.info("decisions_log.parquet: %d satır", n_forensic)
    else:
        n_forensic = 0

    return {
        "logged": int(n_logged),
        "forensic": int(n_forensic),
        "date": str(today.date()),
    }


# ──────────────────────────────────────────────────────────────────────
# FLOW
# ──────────────────────────────────────────────────────────────────────
@flow(name="Minerva_Core_Loop", log_prints=True)
def run_daily_cycle(
    only_mining_on_weekday: Optional[int] = 4,
    mining_n_trials: int = 50,
) -> dict:
    """Akşam: fetch + regime + (Cuma) mining + decay; Sabah: execution."""
    db_path = task_fetch_data()
    prob_path = task_detect_regime(db_path)
    mining_result = task_nightly_mining(
        db_path, prob_path,
        only_on_weekday=only_mining_on_weekday,
        n_trials=mining_n_trials,
    )
    decay_result = task_decay_scan(prob_path)
    exec_result = task_morning_execution(db_path, prob_path)
    return {
        "mining": mining_result,
        "decay": decay_result,
        "execution": exec_result,
    }


if __name__ == "__main__":
    run_daily_cycle()
