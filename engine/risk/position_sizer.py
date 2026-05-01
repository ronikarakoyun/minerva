"""
engine/risk/position_sizer.py — Faz 4.1: Volatility Targeting.

Backtest_engine'in eşit-ağırlık (1/N) dağıtımı, kriz günlerinde portföy riskini
patlamaya açık bırakır. Bu modül her ticker için 20-gün rolling realized vol'ü
ölçer ve hedef yıllık fon vol'üne göre pozisyonu ölçeklendirir:

    asset_vol_t = std(daily_ret_{t-W..t-1}) · √252
    scale_t     = clip(target_annual_vol / asset_vol_t, min_scale, max_scale)
    position_t  = scale_t · base_position (1/N)

Kullanım:
    cfg = RiskConfig(use_vol_target=True, target_annual_vol=0.15)
    scaled = apply_vol_target(daily_returns, cfg)

Backtest entegrasyonu opsiyoneldir (`risk_cfg=None` → mevcut 1/N pipeline aynen).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ANNUALIZATION = 252


@dataclass
class RiskConfig:
    """Volatilite hedeflemesi parametreleri."""
    use_vol_target: bool = False        # default kapalı — geriye dönük
    target_annual_vol: float = 0.15     # %15 yıllık fon vol hedefi
    vol_window: int = 20                # rolling std window (iş günü)
    min_scale: float = 0.1              # leverage tabanı
    max_scale: float = 3.0              # leverage tavanı


def compute_asset_vol(
    returns: pd.Series,
    window: int = 20,
    fast_span: int = 5,
    blend: bool = True,
) -> pd.Series:
    """
    Günlük getiri serisinden rolling realized volatility (annualized).

    N21: EWMA fast/slow blend — yüksek vol döneminde pozisyon uzun süre
    düşük kalmasını önler. fast_span (5g) EWMA ile rolling std (window=20g)
    ortalaması alınır: blend = (fast + slow) / 2.

    Parameters
    ----------
    returns : pd.Series
        Günlük log/simple getiriler. NaN'lara karşı dayanıklı.
    window : int
        Yavaş rolling std window (default 20 iş günü).
    fast_span : int
        Hızlı EWMA span (default 5 iş günü).
    blend : bool
        True (N21) → fast/slow blend; False → sadece rolling std.

    Returns
    -------
    pd.Series
        Annualized vol. Aynı index, NaN bilenmemiş.
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    slow_std = returns.rolling(window, min_periods=max(2, window // 4)).std()
    if blend:
        fast_ewm = returns.ewm(span=fast_span, min_periods=max(2, fast_span // 2)).std()
        # N21: blend = max(fast, slow) — vol spike'larını anında yakala
        blended_std = pd.concat([slow_std, fast_ewm], axis=1).max(axis=1)
        return blended_std * np.sqrt(ANNUALIZATION)
    return slow_std * np.sqrt(ANNUALIZATION)


def compute_position_scale(
    asset_vol: pd.Series, cfg: RiskConfig
) -> pd.Series:
    """
    target/asset_vol → [min_scale, max_scale] aralığında clip.

    NaN vol → scale=1.0 (tarafsız davranış: pozisyonu değiştirme).
    """
    target = float(cfg.target_annual_vol)
    safe_vol = asset_vol.replace(0.0, np.nan)
    scale = target / safe_vol
    scale = scale.fillna(1.0)
    return scale.clip(lower=cfg.min_scale, upper=cfg.max_scale)


def apply_vol_target(
    daily_returns: pd.DataFrame,
    cfg: RiskConfig,
) -> pd.DataFrame:
    """
    Wide-form (Date × Ticker) günlük getirileri her ticker için vol-target ile ölçekle.

    Returns
    -------
    pd.DataFrame
        Aynı şekilde, her hücre = scale_t * ret_t. cfg.use_vol_target=False ise
        girdi aynen döner (no-op).
    """
    if not cfg.use_vol_target:
        return daily_returns

    if not isinstance(daily_returns, pd.DataFrame):
        raise TypeError("daily_returns: pd.DataFrame (index=Date, cols=Ticker) gerekli")

    scaled_cols = {}
    for ticker in daily_returns.columns:
        rets = daily_returns[ticker]
        vol = compute_asset_vol(rets, window=cfg.vol_window)
        # Look-ahead koruma: t-günündeki vol'ü t-1 sonuna kadar olan veriyle hesaplandığını
        # garantile → scale'i bir gün shift et.
        scale = compute_position_scale(vol, cfg).shift(1).fillna(1.0)
        scaled_cols[ticker] = rets * scale

    return pd.DataFrame(scaled_cols, index=daily_returns.index)
