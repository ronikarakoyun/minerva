"""
engine/execution/paper_trader.py — Faz 5.3: Paper trading karar logları.

Bir formül "production"a verilmeden önce out-of-sample paper trade ile test
edilir. Her gün 09:45'te (cron) sistem paper_trades.parquet'e o günkü
kararları (formula_id, ticker, weight, entry_px) yazar; t+2 sonra exit_px
gerçekleştiğinde net_pnl_pct doldurulur.

Faz 4.2 decay_monitor.scan_decay buradan beslenir: bir formülün paper PnL
serisi yeterli uzunluğa (paper_window_days) ulaşınca decay testine sokulur.

Şema (data/paper_trades.parquet):
    date, formula_id, ticker, weight, signal_value,
    entry_px, exit_px, gross_pnl_pct, slippage_bps, net_pnl_pct
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timezone, datetime as _dt
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

import logging

from engine.execution.slippage import SlippageConfig, compute_slippage_bps
from engine.risk.capacity import CapacityConfig, compute_adv
from engine.risk.position_sizer import compute_asset_vol

_log = logging.getLogger(__name__)

_TZ_IST = ZoneInfo("Europe/Istanbul")

# N51: Pre-trade slippage cap env'den okunabilir (varsayılan 200 bps).
SLIPPAGE_CAP_BPS: float = float(os.getenv("SLIPPAGE_CAP_BPS", "200.0"))

# ── Kill-switch dosyası ────────────────────────────────────────────────
# Eğer bu dosya varsa tüm yeni paper-trade kararları engellenir.
_KILL_SWITCH_PATH: Path = Path("data/.kill_switch")
# Günlük maksimum kayıp eşiği (portföy ağırlıklı PnL) — N26
DAILY_LOSS_LIMIT: float = -0.03   # -%3
# N27: Kümülatif drawdown eşiği — bu değeri aşınca kill-switch aktif edilir
CUMULATIVE_DD_LIMIT: float = -0.10  # -%10


# N25: Kill-switch dosyası bu süre sonra otomatik süresi dolar (24h TTL).
_KILL_SWITCH_TTL_HOURS: float = float(os.getenv("KILL_SWITCH_TTL_HOURS", "24"))


def is_kill_switch_active() -> bool:
    """
    Kill-switch dosyası varsa True döner.

    N25: activated_at alanındaki zaman damgası TTL saatinden eski ise dosyayı
    siler ve False döner (otomatik TTL süresi dolması).
    """
    if not _KILL_SWITCH_PATH.exists():
        return False
    try:
        import json as _json
        with open(_KILL_SWITCH_PATH) as f:
            data = _json.load(f)
        activated_at_str = data.get("activated_at")
        if activated_at_str:
            activated_at = _dt.fromisoformat(activated_at_str)
            if activated_at.tzinfo is None:
                activated_at = activated_at.replace(tzinfo=timezone.utc)
            age_hours = (_dt.now(timezone.utc) - activated_at).total_seconds() / 3600
            if age_hours > _KILL_SWITCH_TTL_HOURS:
                _KILL_SWITCH_PATH.unlink(missing_ok=True)
                _log.warning(
                    "Kill-switch TTL (%.0fh) sona erdi — otomatik devre dışı bırakıldı",
                    _KILL_SWITCH_TTL_HOURS,
                )
                return False
    except Exception:
        pass  # Okuma hatası → güvenli taraf → aktif say
    return True


def activate_kill_switch(reason: str = "") -> None:
    """Kill-switch'i etkinleştir — yeni trade'leri engelle."""
    import json as _json
    _KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
    # N54: pd.Timestamp.utcnow() deprecated → datetime.now(timezone.utc)
    with open(_KILL_SWITCH_PATH, "w") as f:
        _json.dump({"reason": reason, "activated_at": _dt.now(timezone.utc).isoformat()}, f)
    _log.critical("KILL_SWITCH AKTİF: %s", reason)


def deactivate_kill_switch() -> None:
    """Kill-switch'i manuel olarak devre dışı bırak."""
    if _KILL_SWITCH_PATH.exists():
        _KILL_SWITCH_PATH.unlink()
        _log.info("Kill-switch devre dışı bırakıldı")


def check_daily_loss_limit(cfg: Optional[PaperTraderConfig] = None) -> bool:
    """
    Günlük ağırlıklı PnL toplamı DAILY_LOSS_LIMIT'in altındaysa kill-switch etkinleştirir.

    N26: TZ-aware — İstanbul saatiyle "bugün" belirlenir.

    Returns
    -------
    bool
        True → limit aşılmadı (trade devam edebilir).
        False → limit aşıldı, kill-switch etkinleştirildi.
    """
    cfg = cfg or PaperTraderConfig()
    df = _load_existing(cfg.output_path)
    if len(df) == 0:
        return True

    df["date"] = pd.to_datetime(df["date"])
    # N26: İstanbul saat dilimiyle bugünün tarihini belirle (UTC kayması önlenir)
    today_ist = _dt.now(_TZ_IST).date()
    today = pd.Timestamp(today_ist)
    today_df = df[df["date"] == today].dropna(subset=["net_pnl_pct"])
    if len(today_df) == 0:
        return True

    daily_pnl = float((today_df["net_pnl_pct"] * today_df["weight"]).sum())
    if daily_pnl < DAILY_LOSS_LIMIT:
        reason = f"Günlük kayıp eşiği aşıldı: {daily_pnl:.3%} < {DAILY_LOSS_LIMIT:.1%}"
        activate_kill_switch(reason)
        return False
    return True


def check_cumulative_drawdown(cfg: Optional[PaperTraderConfig] = None) -> bool:
    """
    N27: Kümülatif drawdown kontrolü.

    Tüm geçmiş paper trade PnL serisindeki maksimum drawdown CUMULATIVE_DD_LIMIT'i
    aşarsa kill-switch etkinleştirilir.

    Drawdown: running_max_equity - current_equity

    Returns
    -------
    bool
        True → limit aşılmadı.
        False → drawdown aşıldı, kill-switch etkinleştirildi.
    """
    cfg = cfg or PaperTraderConfig()
    df = _load_existing(cfg.output_path)
    if len(df) == 0:
        return True

    df = df.dropna(subset=["net_pnl_pct"])
    if len(df) == 0:
        return True

    df["date"] = pd.to_datetime(df["date"])
    # Günlük portföy ağırlıklı PnL → kümülatif equity eğrisi
    daily = (
        df.assign(wpnl=df["net_pnl_pct"] * df["weight"])
        .groupby("date")["wpnl"].sum()
        .sort_index()
    )
    equity = (1 + daily).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = float(drawdown.min())

    if max_dd < CUMULATIVE_DD_LIMIT:
        reason = f"Kümülatif drawdown eşiği aşıldı: {max_dd:.2%} < {CUMULATIVE_DD_LIMIT:.0%}"
        activate_kill_switch(reason)
        _log.critical("DRAWDOWN LIMIT: %s", reason)
        return False
    return True


PAPER_TRADE_COLUMNS = [
    "date", "formula_id", "ticker", "weight", "signal_value",
    "entry_px", "exit_px", "gross_pnl_pct", "slippage_bps", "net_pnl_pct",
]


@dataclass
class PaperTraderConfig:
    output_path: Path = field(default_factory=lambda: Path("data/paper_trades.parquet"))
    log_time: str = "09:45"                  # cron çağrı saati
    min_sharpe_for_promotion: float = 1.0    # paper → live eşiği
    paper_window_days: int = 60              # min "out-of-sample" süre
    hold_days: int = 2                       # exit_px = Pclose_{t+hold_days}
    portfolio_capital_TL: float = 1_000_000  # paper portföy büyüklüğü


def _load_existing(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=PAPER_TRADE_COLUMNS)


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def log_daily_decisions(
    target_weights: pd.Series,
    formula_id: str,
    date: pd.Timestamp,
    db: pd.DataFrame,
    slippage_cfg: Optional[SlippageConfig] = None,
    cfg: Optional[PaperTraderConfig] = None,
) -> int:
    """
    O günkü kararları paper_trades.parquet'e append et.

    Parameters
    ----------
    target_weights : pd.Series
        Index=Ticker, value=portföy ağırlığı. Σ ≤ 1.
    formula_id : str
        Catalog hash veya formül string'i.
    date : pd.Timestamp
        İşlem günü (entry_px = o günün Popen'i veya Pclose'u).
    db : pd.DataFrame
        Flat price df.
    slippage_cfg : SlippageConfig
    cfg : PaperTraderConfig

    Returns
    -------
    int
        Yazılan satır sayısı.
    """
    cfg = cfg or PaperTraderConfig()
    slip_cfg = slippage_cfg or SlippageConfig()

    # Kill-switch kontrolü
    if is_kill_switch_active():
        _log.critical("KILL_SWITCH aktif — paper trade kararları engellendi")
        return 0

    # Günlük kayıp limiti kontrolü
    if not check_daily_loss_limit(cfg):
        _log.critical("Günlük kayıp limiti aşıldı — kill-switch etkinleştirildi, trade durdu")
        return 0

    # N27: Kümülatif drawdown kontrolü
    if not check_cumulative_drawdown(cfg):
        _log.critical("Kümülatif drawdown limiti aşıldı — kill-switch etkinleştirildi, trade durdu")
        return 0

    # Sadece pozitif ağırlıklı pozisyonlar
    active = target_weights[target_weights > 0].dropna()
    if len(active) == 0:
        return 0

    # O günkü fiyatlar
    db_today = db[db["Date"] == date]
    if len(db_today) == 0:
        return 0

    px_lookup = db_today.set_index("Ticker")["Pclose"].to_dict()

    # σ ve ADV — slipaj hesabı için
    db_window = db[db["Date"] <= date].copy()
    adv_df = compute_adv(db_window, CapacityConfig(adv_window=slip_cfg.adv_window))

    # Günlük getiri serisi (her ticker için)
    db_window["Date"] = pd.to_datetime(db_window["Date"])
    ret_wide = (
        db_window.pivot_table(index="Date", columns="Ticker", values="Pclose")
        .pct_change()
    )

    rows = []
    for ticker, weight in active.items():
        entry_px = float(px_lookup.get(ticker, np.nan))
        if not np.isfinite(entry_px) or entry_px <= 0:
            continue

        v_traded = cfg.portfolio_capital_TL * float(weight)

        # σ, ADV
        if ticker in ret_wide.columns:
            vol_series = compute_asset_vol(ret_wide[ticker], window=slip_cfg.sigma_window)
            asset_vol = float(vol_series.loc[:date].iloc[-1]) if len(vol_series) else np.nan
        else:
            asset_vol = np.nan

        try:
            adv_TL = float(adv_df.loc[(ticker, date), "ADV_TL"])
        except (KeyError, ValueError):
            adv_TL = np.nan

        slip_bps = compute_slippage_bps(v_traded, asset_vol, adv_TL, slip_cfg)

        # Pre-trade slippage cap: aşırı maliyetli işlemleri atla
        if np.isfinite(slip_bps) and slip_bps > SLIPPAGE_CAP_BPS:
            _log.warning(
                "SLIP_CAP: %s beklenen slipaj %.1f bps > %.0f bps eşiği — atlanıyor",
                ticker, slip_bps, SLIPPAGE_CAP_BPS,
            )
            continue

        rows.append({
            "date":          pd.Timestamp(date),
            "formula_id":    formula_id,
            "ticker":        ticker,
            "weight":        float(weight),
            "signal_value":  float(weight),  # weight zaten signal'in normalized hali
            "entry_px":      entry_px,
            "exit_px":       np.nan,
            "gross_pnl_pct": np.nan,
            "slippage_bps":  slip_bps,
            "net_pnl_pct":   np.nan,
        })

    if not rows:
        return 0

    new_df = pd.DataFrame(rows, columns=PAPER_TRADE_COLUMNS)
    existing = _load_existing(cfg.output_path)
    combined = pd.concat([existing, new_df], ignore_index=True)
    _save(combined, cfg.output_path)
    return len(rows)


def compute_realized_pnl(
    db: pd.DataFrame,
    cfg: Optional[PaperTraderConfig] = None,
) -> pd.DataFrame:
    """
    exit_px boş satırlar için fiyat geldikçe doldur, net_pnl_pct hesapla.

    exit_px = Pclose_{date + hold_days}; gross = exit/entry - 1; net = gross - slip/1e4.

    Returns
    -------
    pd.DataFrame
        Güncellenmiş tüm paper_trades tablosu.
    """
    cfg = cfg or PaperTraderConfig()
    df = _load_existing(cfg.output_path)
    if len(df) == 0:
        return df

    db = db.copy()
    db["Date"] = pd.to_datetime(db["Date"])
    px_pivot = db.pivot_table(index="Date", columns="Ticker", values="Pclose")

    df["date"] = pd.to_datetime(df["date"])

    pending_mask = df["exit_px"].isna()
    n_dates = len(px_pivot.index)
    for i in df.index[pending_mask]:
        row_date = df.at[i, "date"]
        ticker = df.at[i, "ticker"]

        # entry pozisyonunu bul: row_date dizide olmayabilir (tatil, weekend).
        # searchsorted(side='left') → row_date'in ekleneceği konumu verir,
        # bu konum row_date > önceki ve row_date <= sonraki arasındaki konumdur.
        # row_date tam olarak dizide varsa doğru pozisyonu verir.
        # Yoksa bir sonraki tarihinin pozisyonunu verir → bir geri git.
        pos = px_pivot.index.searchsorted(row_date, side="left")
        if pos >= n_dates:
            continue  # row_date tüm dizinin ötesinde
        if px_pivot.index[pos] != row_date:
            # row_date dizide yok; entry sonraki iş gününde gerçekleşmiş
            # kabul et — exit için o pozisyonu kullan (hold_days buradan)
            pass  # pos zaten doğru (bir sonraki iş günü)

        exit_date_idx = pos + cfg.hold_days
        if exit_date_idx >= n_dates:
            continue  # exit fiyatı henüz mevcut değil — Cuma+2 = Salı gelmedi
        exit_date = px_pivot.index[exit_date_idx]
        if ticker not in px_pivot.columns:
            continue
        exit_px = px_pivot.at[exit_date, ticker]
        if not np.isfinite(exit_px):
            continue

        entry_px = df.at[i, "entry_px"]
        gross = exit_px / entry_px - 1.0
        slip_pct = df.at[i, "slippage_bps"] / 1e4

        df.at[i, "exit_px"] = exit_px
        df.at[i, "gross_pnl_pct"] = gross
        df.at[i, "net_pnl_pct"] = gross - slip_pct

    _save(df, cfg.output_path)
    return df


def feed_decay_monitor(
    formula_id: str,
    backtest_mean: float,
    backtest_std: float,
    cfg: Optional[PaperTraderConfig] = None,
    decay_cfg=None,
) -> dict:
    """
    formula_id'nin paper PnL serisini scan_decay'e ver.

    Parameters
    ----------
    formula_id : str
    backtest_mean, backtest_std : float
        Formül backtest'inde gözlenen günlük PnL μ ve σ.
    cfg : PaperTraderConfig
    decay_cfg : DecayConfig

    Returns
    -------
    dict
        scan_decay çıktısı (triggered, triggered_at, ...).
    """
    from engine.risk.decay_monitor import DecayConfig, scan_decay

    cfg = cfg or PaperTraderConfig()
    decay_cfg = decay_cfg or DecayConfig()

    df = _load_existing(cfg.output_path)
    if len(df) == 0:
        return {"triggered": False, "n_observations": 0, "reason": "no_paper_trades"}

    df = df[df["formula_id"] == formula_id].copy()
    # Pencere sonu kayıtları hariç tut: hold_days kadar yakın olan girişler
    # hiçbir zaman exit_px alamaz (veri bitmeden önce açılmış pozisyonlar).
    # Bu kayıtlar decay monitor'a beslenirse false alarm tetikleyebilir.
    if len(df) > 0:
        df["date"] = pd.to_datetime(df["date"])
        max_date = df["date"].max()
        cutoff = max_date - pd.Timedelta(days=cfg.hold_days)
        df = df[df["date"] <= cutoff]
    df = df.dropna(subset=["net_pnl_pct"])
    if len(df) < decay_cfg.consecutive_days:
        return {
            "triggered": False,
            "n_observations": len(df),
            "reason": "insufficient_history",
        }

    # Günlük portföy ağırlıklı PnL serisi
    df["date"] = pd.to_datetime(df["date"])
    daily = (
        df.assign(weighted_pnl=df["net_pnl_pct"] * df["weight"])
        .groupby("date")["weighted_pnl"].sum()
        .sort_index()
    )

    return scan_decay(daily, backtest_mean, backtest_std, decay_cfg)
