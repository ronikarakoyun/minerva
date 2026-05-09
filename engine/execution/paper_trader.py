
import os
import logging
from dataclasses import dataclass, field
from datetime import timezone, datetime as _dt
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from engine.execution.slippage import SlippageConfig, compute_slippage_bps
from engine.risk.capacity import CapacityConfig, compute_adv
from engine.risk.position_sizer import compute_asset_vol

_log = logging.getLogger(__name__)
_TZ_IST = ZoneInfo("Europe/Istanbul")

# N51: Pre-trade slippage cap
SLIPPAGE_CAP_BPS: float = float(os.getenv("SLIPPAGE_CAP_BPS", "200.0"))

# --- RISK PARAMETERS ---
_KILL_SWITCH_PATH: Path = Path("data/.kill_switch")
DAILY_LOSS_LIMIT: float = -0.03   
CUMULATIVE_DD_LIMIT: float = -0.10  
_KILL_SWITCH_TTL_HOURS: float = float(os.getenv("KILL_SWITCH_TTL_HOURS", "24"))

PAPER_TRADE_COLUMNS = [
    "date", "formula_id", "ticker", "weight", "signal_value",
    "entry_px", "exit_px", "gross_pnl_pct", "slippage_bps", "net_pnl_pct",
]

@dataclass
class PaperTraderConfig:
    output_path: Path = field(default_factory=lambda: Path("data/paper_trades.parquet"))
    log_time: str = "09:45"
    portfolio_capital_TL: float = 1_000_000
    top_k: int = 13
    n_drop: int = 3
    trailing_stop_pct: float = 0.05

def _load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=PAPER_TRADE_COLUMNS)

def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def is_kill_switch_active() -> bool:
    if not _KILL_SWITCH_PATH.exists(): return False
    try:
        import json as _json
        with open(_KILL_SWITCH_PATH) as f:
            data = _json.load(f)
        activated_at = _dt.fromisoformat(data.get("activated_at"))
        if activated_at.tzinfo is None: activated_at = activated_at.replace(tzinfo=timezone.utc)
        age = (_dt.now(timezone.utc) - activated_at).total_seconds() / 3600
        if age > _KILL_SWITCH_TTL_HOURS:
            _KILL_SWITCH_PATH.unlink(missing_ok=True)
            return False
    except: pass
    return True

def manage_portfolio_and_log(
    target_weights: pd.Series, # Top-50 candidates
    formula_id: str,
    date: pd.Timestamp,        # T günü (Karar günü)
    db: pd.DataFrame,
    cfg: Optional[PaperTraderConfig] = None,
) -> dict:
    """
    TopK-Dropout Portföy Yönetimi (Gerçekçi T+1 Açılış).
    1. Trailing Stop kontrolü yap ve kapat.
    2. En zayıf n_drop'u ele ve kapat.
    3. Eksik kadar yeni (tavan olmayan) ekle.
    """
    cfg = cfg or PaperTraderConfig()
    if is_kill_switch_active(): return {"status": "kill_switch_active"}

    df = _load_existing(cfg.output_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # 1. Mevcut Açık Pozisyonları Bul (exit_px == NaN)
    open_mask = df["exit_px"].isna()
    open_indices = df.index[open_mask].tolist()
    
    all_dates = sorted(db["Date"].unique())
    try:
        next_date = all_dates[all_dates.index(date) + 1]
    except (ValueError, IndexError):
        return {"status": "next_date_missing"}

    db_curr = db[db["Date"] == date].set_index("Ticker")
    db_next = db[db["Date"] == next_date].set_index("Ticker")
    
    # --- STEP 1: TRAILING STOP & SIGNAL CHECK ---
    closed_count = 0
    locked_count = 0
    for idx in open_indices:
        ticker = df.at[idx, "ticker"]
        if ticker not in db_curr.index: continue
        
        current_px = db_curr.at[ticker, "Pclose"]
        entry_date = df.at[idx, "date"]
        hist_prices = db[(db["Ticker"] == ticker) & (db["Date"] >= entry_date) & (db["Date"] <= date)]["Pclose"]
        high_wm = hist_prices.max()
        
        # STOP KONTROLÜ
        if current_px < high_wm * (1 - cfg.trailing_stop_pct):
            # TABAN KONTROLÜ: Satmak istiyoruz ama satabiliyor muyuz?
            if ticker in db_next.index:
                t_plus_1_open = db_next.at[ticker, "Popen"]
                if t_plus_1_open <= current_px * 0.901:
                    _log.warning(f"TABAN KİLİDİ (STOP): {ticker} satılamadı, kilitli kaldı.")
                    locked_count += 1
                    continue # Satışı yapma, portföyde kalsın
                
                _log.info(f"STOP: {ticker} zirveden düştü. Kapatılıyor.")
                exit_px = t_plus_1_open
                df.at[idx, "exit_px"] = exit_px
                df.at[idx, "gross_pnl_pct"] = exit_px / df.at[idx, "entry_px"] - 1
                df.at[idx, "net_pnl_pct"] = df.at[idx, "gross_pnl_pct"] - (df.at[idx, "slippage_bps"] / 1e4)
                closed_count += 1
            
    # --- STEP 2: SLACK EXIT (Sadece 20. sıranın altına düşerse sat) ---
    open_indices = df.index[df["exit_px"].isna()].tolist()
    if len(open_indices) > 0:
        # Tüm adayların rank'lerini bul (target_weights zaten puanlara göre sıralı olmalı)
        # target_weights (Date x Ticker) Series, değerler puanlar.
        rank_series = target_weights.sort_values(ascending=False).rank(ascending=False, method="first")
        
        for idx in open_indices:
            ticker = df.at[idx, "ticker"]
            # Eğer hisse listede yoksa veya rank'ı 20'den büyükse (kötüyse) sat
            current_rank = rank_series.get(ticker, 999) 
            
            if current_rank > 20:
                if ticker in db_curr.index:
                    current_px = db_curr.at[ticker, "Pclose"]
                    if ticker in db_next.index:
                        t_plus_1_open = db_next.at[ticker, "Popen"]
                        # TABAN KONTROLÜ
                        if t_plus_1_open <= current_px * 0.901:
                            _log.warning(f"TABAN KİLİDİ (EXIT): {ticker} satılamadı, kilitli kaldı.")
                            locked_count += 1
                            continue

                        _log.info(f"SLACK EXIT: {ticker} rankı {current_rank} değerine düştü. Kapatılıyor.")
                        exit_px = t_plus_1_open
                        df.at[idx, "exit_px"] = exit_px
                        df.at[idx, "gross_pnl_pct"] = exit_px / df.at[idx, "entry_px"] - 1
                        df.at[idx, "net_pnl_pct"] = df.at[idx, "gross_pnl_pct"] - (df.at[idx, "slippage_bps"] / 1e4)
                        closed_count += 1

    # --- STEP 3: YENİ ALIMLAR (T+1 Açılış) ---
    open_tickers = set(df[df["exit_px"].isna()]["ticker"])
    needed = cfg.top_k - len(open_tickers)
    
    new_rows = []
    if needed > 0:
        # Sinyal listesinde olup portföyde olmayanlar
        candidates = target_weights[~target_weights.index.isin(open_tickers)].nlargest(needed * 2)
        
        added = 0
        for ticker, weight in candidates.items():
            if added >= needed: break
            if ticker not in db_next.index or ticker not in db_curr.index: continue
            
            t_plus_1_open = db_next.at[ticker, "Popen"]
            t_curr_close = db_curr.at[ticker, "Pclose"]
            
            # TAVAN KONTROLÜ
            if t_plus_1_open >= t_curr_close * 1.099:
                _log.warning(f"TAVAN ENGELİ: {ticker} alım iptal.")
                continue
                
            new_rows.append({
                "date":          pd.Timestamp(next_date),
                "formula_id":    formula_id,
                "ticker":        ticker,
                "weight":        float(weight),
                "signal_value":  float(weight),
                "entry_px":      t_plus_1_open,
                "exit_px":       np.nan,
                "gross_pnl_pct": np.nan,
                "slippage_bps":  10.0, # Sabit slipaj
                "net_pnl_pct":   np.nan,
            })
            added += 1

    final_df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    _save(final_df, cfg.output_path)
    
    return {
        "status": "ok",
        "closed": closed_count,
        "opened": len(new_rows),
        "total_active": len(open_tickers) + len(new_rows)
    }

def compute_realized_pnl(db: pd.DataFrame, cfg: Optional[PaperTraderConfig] = None):
    # Bu fonksiyon artık manage_portfolio_and_log içinde yapılıyor.
    # Geriye uyumluluk için boş bırakılabilir veya MtM güncelleyebilir.
    pass
