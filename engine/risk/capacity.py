"""
engine/risk/capacity.py — Faz 4.3: ADV tabanlı formül kapasite tahmini.

Her formülün portföy kapasitesini hesaplar:
    ADV_TL_t  = mean(Vlot_t · Pclose_t)_{t-W..t-1}
    max_pos_TL = adv_pct_limit · ADV_TL_t
    formula_capacity_AUM = min(max_pos_TL) over signal tickers

Look-ahead koruması: ADV hesabında shift(1) uygulanır — t gününde
yalnızca t-1 sonuna kadar olan hacim kullanılır.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class CapacityConfig:
    adv_window: int = 20            # ADV rolling pencere (iş günü)
    adv_pct_limit: float = 0.05     # Max pozisyon: ADV'nin %5'i
    min_advs_TL: float = 1_000_000  # Tradable minimum ADV (1M TL)
    portfolio_size: int = 20        # Eşit-ağırlık varsayım: top-K


def compute_adv(
    db: pd.DataFrame,
    cfg: Optional[CapacityConfig] = None,
) -> pd.DataFrame:
    """
    Her (Ticker, Date) için rolling ADV_TL (TL cinsinden ortalama günlük hacim).

    Parameters
    ----------
    db : pd.DataFrame
        Flat format — en az Ticker, Date, Pclose, Vlot kolonları.
    cfg : CapacityConfig

    Returns
    -------
    pd.DataFrame
        MultiIndex (Ticker, Date), tek kolon ADV_TL. Look-ahead güvenli
        (shift(1) uygulanmış: t gününün ADV'si t-1'e kadar olan veriyle hesaplanır).
    """
    cfg = cfg or CapacityConfig()

    db = db.copy()
    db["Date"] = pd.to_datetime(db["Date"])
    db = db.sort_values(["Ticker", "Date"])

    # Günlük TL hacim: Vlot × Pclose
    db["DailyVolTL"] = db["Vlot"] * db["Pclose"]

    # Rolling ortalama — her ticker için ayrı (groupby → transform)
    db["ADV_TL"] = (
        db.groupby("Ticker")["DailyVolTL"]
        .transform(
            lambda s: s.rolling(cfg.adv_window, min_periods=max(2, cfg.adv_window // 4)).mean()
        )
    )

    # Look-ahead koruması: t gününün ADV'si t-1 verisiyle hesaplanmalı
    db["ADV_TL"] = db.groupby("Ticker")["ADV_TL"].shift(1)

    result = db.set_index(["Ticker", "Date"])[["ADV_TL"]].sort_index()
    return result


def estimate_formula_capacity(
    signal: pd.Series,
    db: pd.DataFrame,
    cfg: Optional[CapacityConfig] = None,
) -> dict:
    """
    Bir formülün AUM kapasitesini tahmin eder.

    Sinyal, portfolio_size kadar hisse içerdiğini varsayar (eşit ağırlık).
    Her tarih için en sığ hissenin ADV sınırı kapasiteyi belirler.

    Parameters
    ----------
    signal : pd.Series
        MultiIndex (Ticker, Date) — formül sinyali (değerlerin mutlak
        büyüklüğü değil, NaN=pozisyon yok mantığı).
    db : pd.DataFrame
        Flat price DataFrame — Ticker, Date, Pclose, Vlot.
    cfg : CapacityConfig

    Returns
    -------
    dict
        {
          "max_aum_TL": float,                  # tahmini maksimum AUM (TL)
          "binding_ticker": str | None,          # kapasiteyi kısıtlayan hisse
          "n_tradable_days": int,                # tradable (ADV>=min) gün sayısı
          "median_daily_capacity_TL": float,     # günlük medyan kapasite (TL)
        }
    """
    cfg = cfg or CapacityConfig()

    if not isinstance(signal, pd.Series):
        signal = pd.Series(signal)

    # Sinyal MultiIndex (Ticker, Date) bekleniyor
    if not isinstance(signal.index, pd.MultiIndex):
        raise TypeError("signal: MultiIndex (Ticker, Date) gerekli")

    # ADV hesapla
    adv_df = compute_adv(db, cfg)

    # Sinyalde aktif olan (NaN olmayan) pozisyonları filtrele
    active_signal = signal.dropna()
    if len(active_signal) == 0:
        return {
            "max_aum_TL": 0.0,
            "binding_ticker": None,
            "n_tradable_days": 0,
            "median_daily_capacity_TL": 0.0,
        }

    # Sinyal tarihlerini ve tickerlarını ADV ile eşleştir
    active_idx = active_signal.index  # MultiIndex (Ticker, Date)

    # ADV'yi signal index'e göre reindex
    adv_aligned = adv_df["ADV_TL"].reindex(active_idx)

    # Tradable filtre: ADV >= min_advs_TL
    tradable_mask = adv_aligned >= cfg.min_advs_TL
    n_tradable_days = int(tradable_mask.sum())

    if n_tradable_days == 0:
        return {
            "max_aum_TL": 0.0,
            "binding_ticker": None,
            "n_tradable_days": 0,
            "median_daily_capacity_TL": 0.0,
        }

    tradable_adv = adv_aligned[tradable_mask]

    # Her tarih için minimum ADV → kapasiteyi belirleyen hisse
    # Günlük kapasite: adv_pct_limit × ADV_TL × portfolio_size (eşit ağırlık)
    # max_pos_per_ticker = adv_pct * ADV
    # AUM = portfolio_size × max_pos = portfolio_size × adv_pct × min(ADV_t)

    # Tarih bazlı: her günkü tüm sinyalli hisselerin ADV'sini topla
    # En kısıtlayıcı: pozisyon başına adv_pct × ADV sınırı, portfolio_size pozisyon
    # AUM = min(ADV_TL_ticker) × adv_pct × portfolio_size (binding ticker)

    # Daha pratik: günlük toplam kapasite = Σ(adv_pct × ADV_i) over active tickers
    # Ama plan binding ticker metodunu seçiyor
    per_ticker_cap = tradable_adv * cfg.adv_pct_limit * cfg.portfolio_size

    max_aum_TL = float(per_ticker_cap.min())
    binding_ticker = str(per_ticker_cap.idxmin()[0]) if len(per_ticker_cap) > 0 else None
    median_daily_capacity_TL = float(per_ticker_cap.median())

    return {
        "max_aum_TL": max_aum_TL,
        "binding_ticker": binding_ticker,
        "n_tradable_days": n_tradable_days,
        "median_daily_capacity_TL": median_daily_capacity_TL,
    }
