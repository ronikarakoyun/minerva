"""
engine/execution/slippage.py — Faz 5.1: Almgren-Chriss dinamik slipaj.

Sabit komisyon yanılsaması: backtest_engine sabit %0.05/%0.15 fee uyguluyor.
Sığ tahta hissede 5M TL emir kendi fiyatını yukarı çeker; backtest hayalet
kârları görür, production'da kaybolur.

Almgren-Chriss karekök modeli (per-ticker, per-day):

    σ_t           = compute_asset_vol(daily_returns, 20)[t]   # annualized
    adv_TL_t      = compute_adv(db)[ticker, t]                # rolling
    participation = v_traded_TL / adv_TL_t                    # 0.05 = %5 ADV
    slip_bps      = γ · σ_t · sqrt(participation) · 1e4       # bps

ADV (engine.risk.capacity) ve σ (engine.risk.position_sizer) Faz 4'te zaten
hesaplanıyor — burada sadece Almgren-Chriss formülünü uyguluyoruz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from engine.risk.capacity import CapacityConfig, compute_adv
from engine.risk.position_sizer import compute_asset_vol


@dataclass
class SlippageConfig:
    use_dynamic_slippage: bool = False    # default kapalı — geriye dönük uyumluluk
    gamma: float = 0.10                   # BIST piyasa likidite katsayısı
    sigma_window: int = 20                # vol pencere (gün)
    adv_window: int = 20                  # ADV pencere (gün)
    min_participation: float = 1e-4       # numerik taban
    fallback_bps: float = 10.0            # ADV yok → 10 bps yedek


def compute_slippage_bps(
    v_traded_TL: float,
    asset_vol: float,
    adv_TL: float,
    cfg: Optional[SlippageConfig] = None,
) -> float:
    """
    Tek işlem için Almgren-Chriss slipajı (bps cinsinden).

    Parameters
    ----------
    v_traded_TL : float
        İşlem hacmi (TL).
    asset_vol : float
        Annualized volatility (compute_asset_vol çıktısı).
    adv_TL : float
        Average Daily Volume (TL). NaN/0 → fallback_bps.
    cfg : SlippageConfig

    Returns
    -------
    float
        Slipaj bps (1bps = 0.0001 = %0.01).
    """
    cfg = cfg or SlippageConfig()

    # ADV yok ya da σ yok → tutucu fallback
    if not np.isfinite(adv_TL) or adv_TL <= 0 or not np.isfinite(asset_vol):
        return float(cfg.fallback_bps)

    participation = max(v_traded_TL / adv_TL, cfg.min_participation)
    slip_bps = cfg.gamma * asset_vol * np.sqrt(participation) * 1e4
    return float(slip_bps)


def build_slippage_matrix(
    db: pd.DataFrame,
    daily_returns: pd.DataFrame,
    cfg: Optional[SlippageConfig] = None,
) -> pd.DataFrame:
    """
    (Date × Ticker) birim hacim başına slipaj bps matrisi.

    Backtest scaling için: actual_slip_bps[t, T] = matrix[t, T] * sqrt(traded_pct).

    Burada matrix[t, T] = γ · σ_T,t · sqrt(1/ADV_T,t) · 1e4 — yani v_traded=1 TL
    için bps. Gerçek işlem hacmi için sqrt(v_traded) ile çarpılır:
        slip_bps_actual = matrix[t,T] * sqrt(v_traded_TL)

    Parameters
    ----------
    db : pd.DataFrame
        Flat (Ticker, Date, Pclose, Vlot).
    daily_returns : pd.DataFrame
        Wide (Date × Ticker) günlük getiri.
    cfg : SlippageConfig

    Returns
    -------
    pd.DataFrame
        Wide (Date × Ticker) — her hücre birim-hacim başına slipaj bps.
        ADV NaN olan hücreler `fallback_bps` ile doldurulur.
    """
    cfg = cfg or SlippageConfig()

    # ADV (Ticker, Date) MultiIndex → wide pivot
    adv_df = compute_adv(db, CapacityConfig(adv_window=cfg.adv_window))
    adv_wide = adv_df["ADV_TL"].unstack("Ticker")  # (Date × Ticker)
    adv_wide = adv_wide.reindex(index=daily_returns.index, columns=daily_returns.columns)

    # σ_t,T — her ticker için rolling annualized vol
    vol_wide = pd.DataFrame(
        index=daily_returns.index, columns=daily_returns.columns, dtype=float
    )
    for ticker in daily_returns.columns:
        vol_wide[ticker] = compute_asset_vol(daily_returns[ticker], window=cfg.sigma_window)

    # Birim-hacim slipajı: γ · σ · sqrt(1/ADV) · 1e4
    safe_adv = adv_wide.where(adv_wide > 0, np.nan)
    unit_slip = cfg.gamma * vol_wide * np.sqrt(1.0 / safe_adv) * 1e4

    # NaN → fallback
    unit_slip = unit_slip.fillna(cfg.fallback_bps)
    return unit_slip
