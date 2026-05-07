"""10 Yıllık Tarihsel Paper Trading Orchestrator.

3 katmanlı walk-forward döngü:
  - Çeyrek-sonu: MCTS+DML mining → next quarter alpha catalog (en ağır iş)
  - Cuma kapanışı: HMM refit → güncel regime probabilities
  - Her gün: RL Sizer kaldıraç ayarı + paper trade kaydı

Kullanım:
    # Önce veri çek (2012'ye kadar):
    python scripts/fetch_bist_data.py --start 2012-01-01

    # Sonra paper trading başlat:
    python scripts/run_historical_paper_trade.py \\
        --trading-start 2016-01-01 \\
        --trading-end   2025-12-31 \\
        --resume

Çıktılar:
  data/historical_catalogs/{year}_Q{n}.json   — çeyreklik alfa snapshotları
  data/historical_paper_trades.parquet         — paper trade logu
  data/historical_equity.parquet               — günlük equity + leverage
  data/historical_summary.json                  — özet metrikler
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

# Repo kökünden import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from engine.core.alpha_cfg import AlphaCFG
from engine.core.alpha_catalog import (
    CATALOG_PATH, _save_raw, save_regime_champion,
)
from engine.data.fracdiff import (
    find_min_d, load_fracdiff_cache, save_fracdiff_cache,
)
from engine.data.meta_label import MetaModel, apply_meta_filter_to_pool
from engine.data.regime_detector import RegimeConfig
from engine.data.regime_refitter import WalkForwardHMMRefitter
from engine.execution.blender import (
    BlenderConfig, blend_regime_signals, load_champions_from_catalog,
)
from engine.execution.paper_trader import (
    PaperTraderConfig, compute_realized_pnl, log_daily_decisions,
)
from engine.execution.slippage import SlippageConfig
from engine.ml.replay_buffer import _tree_to_dict
from engine.risk.portfolio_allocator import equal_risk_contribution
from engine.risk.rl_sizer import (
    ACTIONS, MinimalPPOAgent, SizingState, train_rl_sizer,
)
from engine.strategies.mining_runner import MiningConfig, run_mining_window
from engine.util.calendar import (
    is_quarter_end_business_day, quarter_of,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("hist_paper_trade")

# ── Sabitler ──────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parent.parent
HIST_CATALOG_DIR = ROOT_DIR / "data" / "historical_catalogs"
EQUITY_PATH      = ROOT_DIR / "data" / "historical_equity.parquet"
SUMMARY_PATH     = ROOT_DIR / "data" / "historical_summary.json"
RL_AGENT_PATH    = ROOT_DIR / "data" / "rl_sizer.pt"
PAPER_TRADES_PATH = ROOT_DIR / "data" / "historical_paper_trades.parquet"
BENCHMARK_PATH    = ROOT_DIR / "data" / "bist100.parquet"
MARKET_DB_PATH    = ROOT_DIR / "data" / "market_db.parquet"

INITIAL_CAPITAL  = 1_000_000.0   # 1M TL paper portföy

K_REGIMES_DEFAULT = 3   # HMM K — pre-trading mining sonrası dinamik güncellenir


# ── Yardımcılar ───────────────────────────────────────────────────────────────

def load_market_db(start: str) -> pd.DataFrame:
    """market_db.parquet'i yükle ve `start` tarihinden itibaren filtrele."""
    if not MARKET_DB_PATH.exists():
        raise FileNotFoundError(
            f"{MARKET_DB_PATH} bulunamadı. Önce çalıştır: "
            f"python scripts/fetch_bist_data.py --start {start}"
        )
    db = pd.read_parquet(MARKET_DB_PATH)
    db["Date"] = pd.to_datetime(db["Date"])
    db = db[db["Date"] >= pd.Timestamp(start)].sort_values(["Ticker", "Date"]).reset_index(drop=True)
    return db


def load_market_index_ohlcv(end_date: pd.Timestamp) -> pd.DataFrame:
    """bist100.parquet'i HMM girdi formatına dönüştür: Date-indexed OHLCV."""
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(f"{BENCHMARK_PATH} yok — fetch_bist_data.py çalıştırıldı mı?")
    bm = pd.read_parquet(BENCHMARK_PATH)
    bm["Date"] = pd.to_datetime(bm["Date"])
    bm = bm[bm["Date"] <= end_date].sort_values("Date").set_index("Date")

    # Kolon isimleri: bist100 long-format → standart OHLCV (Open/High/Low/Close/Volume)
    rename_map = {
        "Popen": "Open", "Phigh": "High", "Plow": "Low",
        "Pclose": "Close", "Vlot": "Volume",
    }
    cols = {src: dst for src, dst in rename_map.items() if src in bm.columns}
    if cols:
        bm = bm.rename(columns=cols)
    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in bm.columns]
    if missing:
        # Graceful fallback: sadece Close varsa synthetic OHLCV üret.
        # High = Low = Close (ATR=0), Volume = forward-fill veya 1.
        # HMM Norm_ATR_14 ve Choppiness özelliği 0 olur → bilgi kaybı var ama crash'ten iyi.
        log.warning(
            "bist100.parquet eksik kolonlar: %s — Close'dan sentetik OHLCV türetiliyor. "
            "Doğru HMM sonuçları için: python scripts/fetch_bist_data.py --start 2012-01-01",
            missing,
        )
        if "Close" not in bm.columns:
            raise ValueError("bist100.parquet'te en azından 'Close' kolonu olmalı")
        bm = bm.copy()
        if "Open"   not in bm.columns: bm["Open"]   = bm["Close"]
        if "High"   not in bm.columns: bm["High"]   = bm["Close"]
        if "Low"    not in bm.columns: bm["Low"]    = bm["Close"]
        if "Volume" not in bm.columns: bm["Volume"] = 1.0
    return bm[needed]


def save_catalog_snapshot(records: list, year: int, q: int) -> Path:
    """alpha_catalog records'ı çeyrek bazında arşivle."""
    HIST_CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    path = HIST_CATALOG_DIR / f"{year}_Q{q}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return path


def restore_catalog_from_snapshot(year: int, q: int) -> bool:
    """Mevcut alpha_catalog.json'a snapshot'ı kopyala."""
    src = HIST_CATALOG_DIR / f"{year}_Q{q}.json"
    if not src.exists():
        return False
    dst = ROOT_DIR / CATALOG_PATH
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def run_quarterly_mining(
    db: pd.DataFrame,
    train_end: pd.Timestamp,
    alpha_cfg: AlphaCFG,
    mining_cfg: MiningConfig,
    n_regimes: int,
    checkpoint_id: str,
    meta_model: "MetaModel | None" = None,
    n_workers: int = 1,
    n_trials: int = 4,
) -> tuple[int, "MetaModel | None"]:
    """Bir çeyrek için MCTS+DML mining çalıştır, meta-veto uygula, şampiyon ata.

    Adım sırası:
      1. run_mining_window → ham formül havuzu
      2. apply_meta_filter_to_pool → düşük güvenilirlik formülleri çıkar
      3. Top-K seç → her rejime şampiyon ata
      4. Bir sonraki çeyrek için yeni MetaModel eğit

    Parameters
    ----------
    meta_model : Önceki çeyrekten gelen MetaModel (ilk çeyrekte None).

    Returns
    -------
    (champion_count, new_meta_model)
    """
    log.info("Mining başlıyor: train_end=%s  ckpt=%s  workers=%d",
             train_end.date(), checkpoint_id, n_workers)
    train_df = db[db["Date"] <= train_end].copy()
    t0 = time.time()

    # ARCH-3 FIX: n_workers > 1 ise SharedMemory pool kullan
    if n_workers > 1:
        try:
            from engine.strategies.mcts_pool import run_parallel_mining
            prob_df_for_pool = mining_cfg.prob_df if mining_cfg.prob_df is not None else pd.DataFrame()
            results = run_parallel_mining(
                db=train_df,
                alpha_cfg=alpha_cfg,
                mining_cfg=mining_cfg,
                prob_df=prob_df_for_pool,
                n_workers=n_workers,
                n_trials=n_trials,
            )
            log.info("Paralel mining (SHM pool): %d worker, %d formül döndü", n_workers, len(results))
        except Exception as exc:
            log.warning("Paralel mining başarısız (%s) — tek-worker'a düşülüyor", exc)
            results = run_mining_window(
                train_df, alpha_cfg, mining_cfg,
                checkpoint_id=checkpoint_id, resume=True,
            )
    else:
        results = run_mining_window(
            train_df,
            alpha_cfg,
            mining_cfg,
            checkpoint_id=checkpoint_id,
            resume=True,
        )
    elapsed = time.time() - t0
    log.info("Mining tamam: %d formül, %.1f dk", len(results), elapsed / 60)

    if not results:
        log.warning("Mining sonucu boş — şampiyon atanamadı.")
        return 0, None

    # ── Meta-Labeling Veto (PR-9) ───────────────────────────────────────
    # Önceki çeyrekten gelen meta_model varsa düşük güven formülleri çıkar.
    if meta_model is not None and not meta_model.fit_failed:
        pre_filter = len(results)
        feature_df = pd.DataFrame([
            {
                "mean_ric": float(r.mean_ric),
                "abs_ric":  abs(float(r.mean_ric)),
                "std_ric":  float(getattr(r, "std_ric", 0.0)),
            }
            for r in results
        ])
        filtered = apply_meta_filter_to_pool(
            results, meta_model, threshold=0.5, feature_df=feature_df
        )
        if len(filtered) >= max(n_regimes, 3):
            results = filtered
            log.info("Meta-veto: %d → %d formül (eşik=0.50)",
                     pre_filter, len(results))
        else:
            log.info("Meta-veto: filtre sonucu çok az formül (%d), orijinal %d korunuyor",
                     len(filtered), pre_filter)

    # ── Bir sonraki çeyrek için MetaModel eğit ─────────────────────────
    new_meta_model = build_meta_model_for_quarter(results)

    # ── Top-K formül seç (mean_ric'e göre azalan) ─────────────────────
    sorted_res = sorted(results, key=lambda r: r.mean_ric, reverse=True)
    top_k = sorted_res[:max(n_regimes, 1)]

    # Mevcut alpha_catalog.json'u temizle (yeni çeyrek başlangıcı)
    catalog_path = ROOT_DIR / CATALOG_PATH
    if catalog_path.exists():
        catalog_path.unlink()

    # Her rejime bir şampiyon ata (RIC sırasına göre)
    for regime_id, mr in enumerate(top_k):
        save_regime_champion(
            regime_id=regime_id,
            formula=mr.formula,
            tree=mr.tree,
            ic=float(mr.mean_ric),
            rank_ic=float(mr.mean_ric),
            adj_ic=float(mr.mean_ric),
        )
        log.info("  Rejim %d şampiyonu: %s (RIC=%.4f)",
                 regime_id, mr.formula[:60], mr.mean_ric)

    # Snapshot al
    with open(catalog_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    snapshot_path = save_catalog_snapshot(records, train_end.year, quarter_of(train_end))
    log.info("Çeyrek snapshot: %s", snapshot_path.relative_to(ROOT_DIR))
    return len(top_k), new_meta_model


def _compute_recent_ic(
    paper_trades_path: Path,
    as_of: pd.Timestamp,
    lookback_days: int = 20,
) -> float:
    """Son `lookback_days` iş günündeki cross-sectional RankIC ortalaması.

    GAP-4 FIX: RL state'e gerçek IC sinyali ver.
    paper_trades'deki signal_value ve net_pnl_pct kolonlarını kullanır.
    """
    if not paper_trades_path.exists():
        return 0.0
    try:
        pt = pd.read_parquet(paper_trades_path)
        if "signal_value" not in pt.columns or "net_pnl_pct" not in pt.columns:
            return 0.0
        filled = pt.dropna(subset=["net_pnl_pct", "signal_value"]).copy()
        if len(filled) < 10:
            return 0.0
        filled["date"] = pd.to_datetime(filled["date"])
        filled = filled[filled["date"] <= as_of]
        recent_dates = sorted(filled["date"].unique())[-lookback_days:]
        if len(recent_dates) < 3:
            return 0.0
        recent = filled[filled["date"].isin(recent_dates)]
        ics = (
            recent.groupby("date")
            .apply(lambda g: (
                float(g["signal_value"].corr(g["net_pnl_pct"], method="spearman"))
                if g["signal_value"].std() > 0 and len(g) >= 3 else np.nan
            ))
            .dropna()
        )
        if len(ics) == 0:
            return 0.0
        return float(np.clip(ics.mean(), -1.0, 1.0))
    except Exception:
        return 0.0


def build_sizing_state(
    equity_history: list[float],
    prob_row: "pd.Series | None",
    current_scale: float,
    max_entropy: float,
    recent_ic: float = 0.0,
) -> SizingState:
    """RL agent için 5-boyutlu state vektörü oluştur."""
    eq_arr = np.asarray(equity_history, dtype=float)

    # Rolling vol (20d annualized)
    if len(eq_arr) > 1:
        rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-10)
        recent = rets[-20:] if len(rets) >= 2 else rets
        port_vol = float(np.std(recent) * np.sqrt(252)) if len(recent) > 1 else 0.15
    else:
        port_vol = 0.15

    # Drawdown
    if len(eq_arr) > 0:
        peak = float(np.maximum.accumulate(eq_arr).max())
        drawdown = max(0.0, 1.0 - eq_arr[-1] / peak)
    else:
        drawdown = 0.0

    # Regime entropy
    if prob_row is not None and len(prob_row) > 1:
        p = np.asarray(prob_row.values, dtype=float).clip(min=1e-9)
        entropy = float(-(p * np.log(p)).sum())
        norm_entropy = entropy / max(max_entropy, 1e-3)
    else:
        norm_entropy = 0.0

    return SizingState(
        portfolio_vol=float(np.clip(port_vol, 0.0, 1.0)),
        drawdown=float(np.clip(drawdown, 0.0, 1.0)),
        regime_entropy=float(np.clip(norm_entropy, 0.0, 1.0)),
        recent_ic=float(np.clip(recent_ic, -1.0, 1.0)),  # GAP-4: gerçek 20g RankIC
        current_scale=float(np.clip(current_scale / 2.0, 0.0, 1.0)),
    )


def setup_rl_agent(db: pd.DataFrame, retrain: bool = False) -> MinimalPPOAgent:
    """RL agent'ı yükle veya eğit."""
    agent = MinimalPPOAgent()
    if RL_AGENT_PATH.exists() and not retrain:
        log.info("RL agent yükleniyor: %s", RL_AGENT_PATH)
        agent.load_state_dict(torch.load(RL_AGENT_PATH, weights_only=True))
        agent.eval()
        return agent

    log.info("RL agent eğitiliyor (200 episode)…")
    eq_curve = (
        1 + db.groupby("Date")["Pclose"].apply(
            lambda s: s.pct_change().mean()
        ).fillna(0.0)
    ).cumprod()
    agent = train_rl_sizer(
        equity_curve=eq_curve,
        n_episodes=200,
        save_path=str(RL_AGENT_PATH),
        seed=42,
    )
    agent.eval()
    return agent


def setup_hmm(db: pd.DataFrame, warmup_end: pd.Timestamp) -> WalkForwardHMMRefitter:
    """HMM'i 2012-2015 verisiyle warmup yap."""
    cfg = RegimeConfig(use_filtered_probs=True)
    refitter = WalkForwardHMMRefitter(
        cfg,
        refit_every_n_days=7,    # haftalık (Cuma)
        min_history_days=500,    # ~2 yıl iş günü
    )
    log.info("HMM warmup: 2012-01-01 → %s", warmup_end.date())
    bm = load_market_index_ohlcv(warmup_end)
    refitter.fit_initial(bm)
    log.info("HMM hazır: K=%d  son fit=%s",
             refitter._best_K, refitter._last_refit_date.date())
    return refitter


def get_prob_row(refitter: WalkForwardHMMRefitter, date_t: pd.Timestamp) -> "pd.Series | None":
    """Refitter'ın önbellekli prob_df'ten date_t için satırı döndür.

    ARCH-4 FIX: HMM forward-filter sonucu yalnızca refit günlerinde (Cuma)
    yeniden hesaplanır. Ara günler _cached_prob_df'i kullanarak O(1) erişimle
    2500x yineleme maliyetinden kaçınır.
    """
    prob_df = refitter.last_prob_df
    if prob_df is None:
        return None
    if date_t in prob_df.index:
        return prob_df.loc[date_t]
    if len(prob_df) > 0:
        return prob_df.iloc[-1]
    return None


def refresh_fracdiff_cache(
    db: pd.DataFrame,
    as_of: pd.Timestamp,
    fracdiff_cache_path: Path,
    lookback_days: int = 500,
) -> None:
    """Her ticker için d_star'ı yeniden hesapla ve cache'i güncelle.

    Her 2 çeyrekte bir (~6 ayda bir) çağrılması önerilir.
    Büyük evren için ~500ms/ticker → paralel değil, sıralı.
    Önbellekte 90 günden eski olan girişler yeniden hesaplanır.

    Parameters
    ----------
    db               : market_db DataFrame (Ticker, Date, Pclose sütunları).
    as_of            : Referans tarihi (bu tarih dahil geçmiş veri kullanılır).
    fracdiff_cache_path: fracdiff_d.json yolu.
    lookback_days    : Her ticker için kaç günlük tarih kullanılır.
    """
    cache = load_fracdiff_cache(fracdiff_cache_path)
    db_t = db[db["Date"] <= as_of]
    tickers = sorted(db_t["Ticker"].unique())

    updated = 0
    for ticker in tickers:
        series = (
            db_t[db_t["Ticker"] == ticker]
            .sort_values("Date")["Pclose"]
            .dropna()
            .tail(lookback_days)
        )
        if len(series) < 30:
            continue
        entry = cache.get(ticker, {})
        computed_at_str = entry.get("computed_at", "")
        # CRIT-3 FIX: simüle edilmiş `as_of` tarihi ile karşılaştır —
        # datetime.now() duvar saatini değil, tarihsel döngünün mevcut tarihini kullan.
        # Böylece Q1 2016 simülasyonunda doldurulan cache Q2 2016'da doğru biçimde
        # stale olarak görünür (6 ay ≥ 90 gün koşulu sağlanır).
        stale = True
        if computed_at_str:
            try:
                ca = pd.Timestamp(computed_at_str)
                stale = (as_of - ca) > pd.Timedelta(days=90)
            except (ValueError, TypeError):
                stale = True
        if not stale:
            continue
        try:
            d_star = find_min_d(series)
            today_close = float(series.iloc[-1]) if len(series) > 0 else None
            cache[ticker] = {
                "d_star": d_star,
                "computed_at": as_of.isoformat(),   # simüle edilmiş tarih
                "ref_close": today_close,
                "n_obs": int(len(series.dropna())),
            }
            updated += 1
        except Exception as exc:
            log.debug("FracDiff hesaplama hatası %s: %s", ticker, exc)

    if updated > 0:
        save_fracdiff_cache(cache, fracdiff_cache_path)
        log.info("FracDiff cache güncellendi: %d ticker (toplam %d), as_of=%s",
                 updated, len(tickers), as_of.date())
    else:
        log.debug("FracDiff cache güncel, güncelleme gerekmedi (%s)", as_of.date())


def build_meta_model_for_quarter(results: list) -> "MetaModel | None":
    """Mining sonuçlarından basit bir meta-label modeli eğit.

    Feature'lar: [mean_ric, abs_ric, std_ric] — her formül için tek satır.
    Label: mean_ric > median (üst yarı = "kârlı", alt yarı = "kârsız").
    Çeyrek bazlı cross-filter sağlar: düşük RIC formüller bir sonraki
    çeyreğe taşınmaz.

    Returns None eğer yetersiz veri veya sklearn yüklü değilse.
    """
    if not results or len(results) < 10:
        return None
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None

    try:
        rows = [
            {
                "mean_ric": float(r.mean_ric),
                "abs_ric":  abs(float(r.mean_ric)),
                "std_ric":  float(getattr(r, "std_ric", 0.0)),
            }
            for r in results
        ]
        df_feat = pd.DataFrame(rows)
        feature_cols = ["mean_ric", "abs_ric", "std_ric"]
        median_ric = float(df_feat["mean_ric"].median())
        labels = (df_feat["mean_ric"] > median_ric).astype(int).values

        if labels.sum() == 0 or labels.sum() == len(labels):
            return None  # tek sınıf → anlamsız

        clf = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")
        clf.fit(df_feat[feature_cols].values, labels)
        model = MetaModel(model=clf, feature_cols=feature_cols, fit_failed=False)
        log.debug("Meta-model eğitildi: %d formül, median_ric=%.4f",
                  len(results), median_ric)
        return model
    except Exception as exc:
        log.debug("Meta-model eğitim hatası: %s", exc)
        return None


# ── Ana Walk-Forward Döngü ───────────────────────────────────────────────────

def run_walk_forward(
    trading_start: str,
    trading_end: str,
    *,
    data_start: str = "2012-01-01",
    pre_trading_mining: bool = True,
    use_rl: bool = True,
    n_workers: int = 4,
    n_trials: int = 50,
    resume: bool = False,
) -> dict:
    """Tarihsel paper trading walk-forward orchestratörü."""
    HIST_CATALOG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Veri yükle
    log.info("Veri yükleniyor: %s → %s", data_start, trading_end)
    db = load_market_db(data_start)
    if len(db) == 0:
        raise RuntimeError("market_db.parquet boş — fetch_bist_data.py çalıştır.")

    db_dates = pd.DatetimeIndex(sorted(db["Date"].unique()))
    log.info("Veri penceresi: %s → %s  (%d gün, %d ticker)",
             db_dates.min().date(), db_dates.max().date(),
             len(db_dates), db["Ticker"].nunique())

    trading_start_ts = pd.Timestamp(trading_start)
    trading_end_ts   = pd.Timestamp(trading_end)
    trading_dates    = db_dates[(db_dates >= trading_start_ts) & (db_dates <= trading_end_ts)]
    if len(trading_dates) == 0:
        raise RuntimeError(f"Trading penceresinde tarih yok: {trading_start} → {trading_end}")

    # 2. RL agent
    rl_agent = setup_rl_agent(db, retrain=False) if use_rl else None

    # 3. HMM warmup (trading başlangıcına kadar)
    warmup_end = trading_dates[0] - pd.offsets.BDay(1)
    hmm = setup_hmm(db, warmup_end)
    max_entropy = float(np.log(max(hmm._best_K, 2)))

    # 4. Pre-trading initial mining (Q1 2016 boşluğunu kapatır)
    alpha_cfg  = AlphaCFG()
    # ── Uyuyan Devleri Uyandır ────────────────────────────────────────────────
    # Geriye dönük uyumluluk için varsayılan False olan tüm opt-in flagler
    # 10 yıllık maraton için burada aktive edilir.
    mining_cfg = MiningConfig(
        num_gen=200,
        search_mode="mcts",            # MCTS arama motoru (Faz 3)
        use_wf_fitness=True,           # Walk-forward fold fitness
        neutralize=True,               # Faktör nötralizasyonu
        use_dml_neutralize=True,       # PR-8: DML (Neyman-ortogonal) — AÇIK
        use_attention=True,            # PR-10: Tree-LSTM Bahdanau attention — AÇIK
        dropout_p=0.1,                 # PR-10: MC Dropout (%10) — AÇIK
        use_regime_weighting=False,    # prob_df yokken False kalır (per-day set edilir)
    )
    log.info("MiningConfig — DML=%s  attention=%s  dropout=%.2f  search=%s",
             mining_cfg.use_dml_neutralize, mining_cfg.use_attention,
             mining_cfg.dropout_p, mining_cfg.search_mode)

    # FracDiff cache yolu
    fracdiff_cache_path = ROOT_DIR / "data" / "fracdiff_d.json"
    fracdiff_quarter_counter = 0   # her 2 çeyrekte bir refresh tetiklenir

    # MetaModel: çeyrekten çeyreğe aktarılır (ilk çeyrekte None)
    current_meta_model: "MetaModel | None" = None

    pre_year, pre_q = warmup_end.year, quarter_of(warmup_end)
    pre_snapshot = HIST_CATALOG_DIR / f"{pre_year}_Q{pre_q}.json"
    if pre_trading_mining and not pre_snapshot.exists():
        log.info("=" * 60)
        log.info("PRE-TRADING MINING (Q1 2016 boşluğu için)")
        log.info("=" * 60)
        _cnt, current_meta_model = run_quarterly_mining(
            db, warmup_end, alpha_cfg, mining_cfg,
            n_regimes=hmm._best_K,
            checkpoint_id=f"hist_q_{pre_year}_{pre_q}",
            meta_model=None,          # ilk mining → meta-model yok
            n_workers=n_workers,
            n_trials=n_trials,
        )
        fracdiff_quarter_counter += 1
    elif pre_snapshot.exists():
        log.info("Pre-trading snapshot mevcut, mining atlanıyor: %s", pre_snapshot.name)
        restore_catalog_from_snapshot(pre_year, pre_q)

    # 5. Walk-forward döngü
    paper_cfg = PaperTraderConfig(
        output_path=PAPER_TRADES_PATH,
        portfolio_capital_TL=INITIAL_CAPITAL,
    )
    slip_cfg = SlippageConfig(use_dynamic_slippage=False)  # tarihsel basit; dynamic çok yavaş

    equity_history: list[float] = [INITIAL_CAPITAL]
    leverage_history: list[float] = []
    daily_records: list[dict] = []
    current_scale: float = 1.0

    # Mevcut paper trades'i resume için yükle
    if resume and PAPER_TRADES_PATH.exists():
        prev = pd.read_parquet(PAPER_TRADES_PATH)
        if len(prev) > 0:
            # Kolon adı: paper_trader çıktısında "date" (entry_date değil)
            date_col = "date" if "date" in prev.columns else "entry_date"
            last_date = pd.Timestamp(prev[date_col].max())
            log.info("RESUME: Önceki son trade %s → o tarihten sonraki günler işlenecek",
                     last_date.date())
            trading_dates = trading_dates[trading_dates > last_date]

    log.info("=" * 60)
    log.info("WALK-FORWARD DÖNGÜ: %d gün (%s → %s)",
             len(trading_dates),
             trading_dates[0].date(), trading_dates[-1].date())
    log.info("=" * 60)

    n_mining_runs = 0
    n_hmm_refits  = 0
    n_paper_logs  = 0

    for i, date_t in enumerate(trading_dates):
        # A. Çeyrek-sonu kontrolü → MINING + FracDiff cache refresh
        if is_quarter_end_business_day(date_t, db_dates):
            year, q = date_t.year, quarter_of(date_t)
            snapshot = HIST_CATALOG_DIR / f"{year}_Q{q}.json"
            if snapshot.exists():
                log.info("Çeyrek snapshot mevcut, mining atlanıyor: %s", snapshot.name)
                restore_catalog_from_snapshot(year, q)
            else:
                log.info("=" * 60)
                log.info("ÇEYREK-SONU MINING: %s (Y%d Q%d)", date_t.date(), year, q)
                log.info("=" * 60)
                # ── FracDiff Cache Refresh (her 2 çeyrekte bir) ──────────
                fracdiff_quarter_counter += 1
                if fracdiff_quarter_counter % 2 == 0:
                    log.info("FracDiff cache yenileniyor (çeyrek=%d)…", fracdiff_quarter_counter)
                    try:
                        refresh_fracdiff_cache(db, date_t, fracdiff_cache_path)
                    except Exception as exc:
                        log.warning("FracDiff refresh başarısız: %s", exc)
                # ── Çeyreklik Mining (Meta-Veto dahil) ───────────────────
                _cnt, current_meta_model = run_quarterly_mining(
                    db, date_t, alpha_cfg, mining_cfg,
                    n_regimes=hmm._best_K,
                    checkpoint_id=f"hist_q_{year}_{q}",
                    meta_model=current_meta_model,
                    n_workers=n_workers,
                    n_trials=n_trials,
                )
                n_mining_runs += 1

        # B. Cuma → HMM refit (refitter._should_refit otomatik 7 günde bir)
        if date_t.weekday() == 4:
            try:
                bm = load_market_index_ohlcv(date_t)
                hmm.update(bm, force=False)
                n_hmm_refits += 1
            except Exception as exc:
                log.warning("HMM refit başarısız %s: %s", date_t.date(), exc)

        # C. Bugünün regime probabilities (ARCH-4: önbellekten, O(1))
        prob_row = get_prob_row(hmm, date_t)
        if prob_row is None:
            log.debug("prob_row yok %s — gün atlanıyor", date_t.date())
            continue

        # prob_df DataFrame'i blender için hazırla (tek satır)
        prob_df_blender = pd.DataFrame([prob_row.values],
                                       index=[date_t],
                                       columns=prob_row.index)

        # D. Şampiyonları yükle (alpha_catalog.json güncel snapshot)
        catalog_path = ROOT_DIR / CATALOG_PATH
        if not catalog_path.exists():
            continue
        champions = load_champions_from_catalog(catalog_path, alpha_cfg=alpha_cfg)
        if not champions:
            continue

        # E. Sinyal üret (rejim-blend)
        try:
            db_through_t = db[db["Date"] <= date_t]
            weights_df = blend_regime_signals(
                champions, prob_df_blender, db_through_t,
                BlenderConfig(use_blending=True),
                alpha_cfg=alpha_cfg,
            )
        except Exception as exc:
            log.warning("Blend başarısız %s: %s", date_t.date(), exc)
            continue

        if len(weights_df) == 0 or date_t not in weights_df.index:
            continue
        today_weights = weights_df.loc[date_t].dropna()
        today_weights = today_weights[today_weights > 0]
        if len(today_weights) == 0:
            continue

        # E2. Risk Parity ağırlık düzeltmesi (PR-11: Equal Risk Contribution)
        # Blender'dan gelen düz ağırlıkları ERC ile yeniden dengele.
        # En az 2 pozisyon ve yeterli tarihsel getiri verisi gerekli.
        if len(today_weights) >= 2:
            try:
                tickers_rp = today_weights.index.tolist()
                hist_prices = (
                    db_through_t[db_through_t["Ticker"].isin(tickers_rp)]
                    .pivot_table(index="Date", columns="Ticker", values="Pclose")
                    .sort_index()
                    .tail(120)         # 6 ay kovaryans penceresi
                )
                returns_rp = hist_prices.pct_change().dropna(how="all")
                if len(returns_rp) >= 20 and returns_rp.shape[1] >= 2:
                    erc_weights = equal_risk_contribution(
                        returns_rp,
                        lookback=min(60, len(returns_rp)),
                        min_weight=0.005,
                        max_weight=0.35,
                    )
                    # ERC weight'leri mevcut pozisyon setine normalize et
                    common = today_weights.index.intersection(erc_weights.index)
                    if len(common) >= 2:
                        today_weights = erc_weights.reindex(common).fillna(0)
                        today_weights = today_weights / today_weights.sum()
            except Exception as exc:
                log.debug("Risk Parity başarısız %s: %s — eşit ağırlık korunuyor",
                          date_t.date(), exc)

        # F. RL leverage
        leverage = 1.0
        if rl_agent is not None:
            _recent_ic = _compute_recent_ic(PAPER_TRADES_PATH, date_t, lookback_days=20)
            state = build_sizing_state(equity_history, prob_row, current_scale, max_entropy,
                                       recent_ic=_recent_ic)
            with torch.no_grad():
                action, _ = rl_agent.act(state.to_array())
            leverage = ACTIONS[action]
            current_scale = leverage
        today_weights = today_weights * leverage

        # G. Paper trader: önceki günün exit fill, bugünün entry log
        try:
            compute_realized_pnl(db_through_t, paper_cfg)
            n_logged = log_daily_decisions(
                today_weights, "hist_blend_v1", date_t, db_through_t,
                slippage_cfg=slip_cfg, cfg=paper_cfg,
            )
            n_paper_logs += int(n_logged)
        except Exception as exc:
            log.warning("Paper trade başarısız %s: %s", date_t.date(), exc)
            continue

        # H. Equity güncelle (paper_trades.parquet'ten)
        # Bileşik getiri (compounding): her günün ağırlıklı PnL'i ayrı dönem olarak hesaplanır.
        # (1 + r1) * (1 + r2) * ... — basit toplam değil.
        try:
            if PAPER_TRADES_PATH.exists():
                pt = pd.read_parquet(PAPER_TRADES_PATH)
                filled = pt.dropna(subset=["net_pnl_pct"])
                if len(filled) > 0:
                    filled = filled.copy()
                    filled["date"] = pd.to_datetime(filled["date"])
                    daily_ret = (
                        filled.groupby("date")
                        .apply(
                            lambda g: float((g["net_pnl_pct"] * g["weight"]).sum()),
                            include_groups=False,
                        )
                        .sort_index()
                    )
                    equity_val = float(INITIAL_CAPITAL * (1 + daily_ret).cumprod().iloc[-1])
                    equity_history.append(equity_val)
                else:
                    equity_history.append(equity_history[-1])
            else:
                equity_history.append(equity_history[-1])
        except Exception:
            equity_history.append(equity_history[-1])

        leverage_history.append(leverage)
        daily_records.append({
            "date": date_t,
            "equity": equity_history[-1],
            "leverage": leverage,
            "n_positions": int(len(today_weights)),
            "regime_dominant": int(prob_row.idxmax().split("_")[-1]) if hasattr(prob_row, "idxmax") else -1,
        })

        # Progress log her 50 günde
        if (i + 1) % 50 == 0:
            log.info("Progress: %d/%d gün, equity=%.0f TL, lev=%.2fx",
                     i + 1, len(trading_dates),
                     equity_history[-1], leverage)

    # 6. Çıktı dosyaları
    eq_df = pd.DataFrame(daily_records)
    if len(eq_df) > 0:
        eq_df.to_parquet(EQUITY_PATH, index=False)
        log.info("Equity yazıldı: %s (%d satır)", EQUITY_PATH.name, len(eq_df))

    # ── Kurumsal tearsheet metrikleri ────────────────────────────────────────
    def _compute_tearsheet(
        eq_hist: list[float],
        lev_hist: list[float],
        rf_annual: float = 0.12,   # BIST proxy: ~%12 yıllık risk-free (TCMB ortalama)
    ) -> dict:
        if len(eq_hist) < 2:
            return {}
        eq_arr = np.array(eq_hist, dtype=float)
        daily_rets = np.diff(eq_arr) / np.maximum(eq_arr[:-1], 1e-10)

        rf_daily = rf_annual / 252
        excess   = daily_rets - rf_daily
        n_days   = len(daily_rets)
        n_years  = n_days / 252.0

        # Yıllık getiri (CAGR)
        total_ret = eq_arr[-1] / eq_arr[0] - 1
        cagr = (1 + total_ret) ** (1 / max(n_years, 1e-6)) - 1

        # Sharpe (yıllıklaştırılmış)
        ex_mean = float(np.mean(excess))
        ex_std  = float(np.std(excess, ddof=1)) if n_days > 1 else 1e-6
        sharpe  = (ex_mean / max(ex_std, 1e-10)) * np.sqrt(252)

        # Sortino (downside std)
        neg_excess = excess[excess < 0]
        down_std   = float(np.std(neg_excess, ddof=1)) if len(neg_excess) > 1 else 1e-6
        sortino    = (ex_mean / max(down_std, 1e-10)) * np.sqrt(252)

        # Maximum Drawdown
        running_max = np.maximum.accumulate(eq_arr)
        drawdowns   = (eq_arr - running_max) / np.maximum(running_max, 1e-10)
        mdd         = float(np.min(drawdowns))

        # Calmar = CAGR / |MDD|
        calmar = float(cagr / max(abs(mdd), 1e-6))

        # Win rate (günlük pozitif getiri)
        win_rate = float(np.mean(daily_rets > 0))

        # Yıllık volatilite
        annual_vol = float(ex_std * np.sqrt(252))

        # Information Ratio (excess over zero benchmark = Sharpe, over rf = above)
        ir = sharpe  # IR ≈ Sharpe için rf-adjusted excess

        return {
            "cagr_pct":        round(cagr * 100, 2),
            "annual_vol_pct":  round(annual_vol * 100, 2),
            "sharpe":          round(sharpe, 3),
            "sortino":         round(sortino, 3),
            "calmar":          round(calmar, 3),
            "ir":              round(ir, 3),
            "mdd_pct":         round(mdd * 100, 2),
            "win_rate_pct":    round(win_rate * 100, 1),
            "n_years":         round(n_years, 2),
            "rf_annual_pct":   round(rf_annual * 100, 1),
        }

    tearsheet = _compute_tearsheet(equity_history, leverage_history)

    summary = {
        "trading_start": trading_start,
        "trading_end":   trading_end,
        "n_days_processed": len(daily_records),
        "n_mining_runs": n_mining_runs,
        "n_hmm_refits":  n_hmm_refits,
        "n_paper_logs":  n_paper_logs,
        "initial_capital": INITIAL_CAPITAL,
        "final_equity": equity_history[-1] if equity_history else INITIAL_CAPITAL,
        "total_return_pct": (
            (equity_history[-1] / INITIAL_CAPITAL - 1) * 100
            if equity_history else 0.0
        ),
        "avg_leverage": float(np.mean(leverage_history)) if leverage_history else 1.0,
        **tearsheet,
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("=" * 60)
    log.info("ÖZET: %s", json.dumps(summary, indent=2))
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="10-yıllık tarihsel paper trading")
    parser.add_argument("--trading-start", default="2016-01-01",
                        help="Trading penceresi başlangıcı (varsayılan: 2016-01-01)")
    parser.add_argument("--trading-end",   default=None,
                        help="Trading penceresi bitişi (varsayılan: bugün)")
    parser.add_argument("--data-start",    default="2012-01-01",
                        help="Veri penceresi başlangıcı (HMM warmup için)")
    parser.add_argument("--no-rl", action="store_true", help="RL leverage devre dışı")
    parser.add_argument("--no-pre-mining", action="store_true",
                        help="Pre-trading mining'i atla")
    parser.add_argument("--workers", type=int, default=4, help="Mining worker sayısı")
    parser.add_argument("--n-trials", type=int, default=50, help="Mining trial sayısı")
    parser.add_argument("--resume", action="store_true",
                        help="Önceki paper_trades'in son tarihinden devam et")
    parser.add_argument("--no-kill-switch", action="store_true",
                        help="Kill-switch eşiklerini tarihsel simülasyon için gevşet "
                             "(-50%% kümülatif, -20%% günlük)")
    args = parser.parse_args()

    # Tarihsel simülasyonda kill-switch eşikleri çok sıkı → gevşet.
    # Gerçek trading'de bu flag kullanılmaz; default değerler korunur.
    if args.no_kill_switch:
        import os as _os
        _os.environ["CUMULATIVE_DD_LIMIT"]  = "-0.50"
        _os.environ["DAILY_LOSS_LIMIT"]     = "-0.20"
        _os.environ["DISABLE_KILL_SWITCH"]  = "1"   # activate_kill_switch() dosya yazmaz
        # Aktif kill switch dosyasını da temizle (bir önceki çalıştırmadan kalmış olabilir)
        _ks_path = Path("data/.kill_switch")
        if _ks_path.exists():
            _ks_path.unlink()
            log.info("Kill-switch dosyası temizlendi (--no-kill-switch aktif)")

    trading_end = args.trading_end or pd.Timestamp.today().strftime("%Y-%m-%d")

    run_walk_forward(
        trading_start=args.trading_start,
        trading_end=trading_end,
        data_start=args.data_start,
        pre_trading_mining=not args.no_pre_mining,
        use_rl=not args.no_rl,
        n_workers=args.workers,
        n_trials=args.n_trials,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
