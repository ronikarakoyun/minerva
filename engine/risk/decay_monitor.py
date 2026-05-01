"""
engine/risk/decay_monitor.py — Faz 4.2: Alpha decay tespiti (Page-Hinkley).

Bir formül canlıya verildiğinde backtest dağılımının dışına çıkarsa
(ör. arka arkaya 5 gün -2σ altında PnL), sistem "kill-switch" çekmeli ve
formülü emekli etmelidir.

Page-Hinkley (negatif drift) algoritması:

    m_0 = 0
    m_t = max(0, m_{t-1} + (μ_backtest - r_t - δ))
    trigger if m_t > λ AND consecutive_alarms ≥ N

- δ : tespit edilecek minimum drift (gürültü payı)
- λ : kümülatif sapma alarm eşiği
- N : ardışık gün eşiği

Bu modül stateful incremental update + tek-shot full scan modlarını destekler.
Persistence Faz 5'te (engine/core/alpha_catalog.py'de "live" anahtarı).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DecayConfig:
    delta: float = 5e-4              # minimum drift threshold (günlük)
    lambda_threshold: float = 0.01   # kümülatif sapma alarm eşiği
    consecutive_days: int = 5        # ardışık alarm gün eşiği
    sigma_floor: float = 2.0         # |live - μ| > sigma_floor·σ pre-filter (günlük)
    adaptive_delta: bool = True      # True → δ = 0.5 × backtest_std (vol-adaptif)
    # N20: Tek-gün uç şok eşiği — bu eşiği aşan (≥ extreme_sigma_cap × σ)
    # tek günlük kayıplar ardışık alarm sayacını sıfırlamaz ama artırmaz da;
    # piyasa kaynaklı şok olarak muamele görür.
    extreme_sigma_cap: float = 3.5   # |live - μ| > cap → outlier, sayaç dondur


@dataclass
class DecayState:
    m: float = 0.0
    consecutive_alarms: int = 0
    triggered: bool = False
    triggered_at: Optional[pd.Timestamp] = None
    n_observations: int = 0


def update_decay_state(
    state: DecayState,
    live_return: float,
    backtest_mean: float,
    backtest_std: float,
    cfg: DecayConfig,
    today: Optional[pd.Timestamp] = None,
) -> DecayState:
    """
    Tek günlük adım: Page-Hinkley istatistiğini güncelle, tetikleme kontrolü.

    Returns
    -------
    DecayState — yeni state (mutable in-place + return).
    """
    if state.triggered:
        return state  # zaten tetiklendi → değişiklik yok

    state.n_observations += 1

    if not np.isfinite(live_return):
        return state

    # Page-Hinkley negatif drift: μ - r - δ ne kadar büyük → daha çok kayıp
    # Vol-adaptif δ: yüksek volatilite dönemlerinde false alarm azaltır.
    effective_delta = (
        0.5 * max(backtest_std, 1e-12) if cfg.adaptive_delta else cfg.delta
    )
    increment = backtest_mean - live_return - effective_delta
    state.m = max(0.0, state.m + increment)

    # Pre-filter: bu gün de daily live_return σ-floor'dan kötü mü?
    effective_std = max(backtest_std, 1e-12)
    sigma_band = cfg.sigma_floor * effective_std
    extreme_band = cfg.extreme_sigma_cap * effective_std
    deviation = backtest_mean - live_return
    is_outlier = deviation > sigma_band

    # N20: Uç şok tespiti — extreme_sigma_cap'i aşan tek-gün şoklar
    # piyasa kaynaklı kabul edilir; sayaç ne artar ne sıfırlanır (dondurulur).
    is_extreme_shock = deviation > extreme_band

    # Asıl tetikleme koşulu: cumulative m_t threshold'ı geçti VE bugün de outlier
    if is_extreme_shock:
        # N20: Uç şok → sayacı dondur (ne artır ne sıfırla)
        pass
    elif state.m > cfg.lambda_threshold and is_outlier:
        state.consecutive_alarms += 1
    else:
        state.consecutive_alarms = 0

    if state.consecutive_alarms >= cfg.consecutive_days:
        state.triggered = True
        state.triggered_at = today

    return state


def scan_decay(
    live_returns: pd.Series,
    backtest_mean: float,
    backtest_std: float,
    cfg: Optional[DecayConfig] = None,
) -> dict:
    """
    Tüm canlı seriyi tara → ilk tetikleme noktasını bul.

    Returns
    -------
    dict
        {
          "triggered": bool,
          "triggered_at": pd.Timestamp | None,
          "n_observations": int,
          "final_m": float,
          "max_m": float,
        }
    """
    cfg = cfg or DecayConfig()
    state = DecayState()
    max_m = 0.0

    if not isinstance(live_returns, pd.Series):
        live_returns = pd.Series(live_returns)

    for ts, r in live_returns.items():
        update_decay_state(
            state, float(r), backtest_mean, backtest_std, cfg,
            today=ts if hasattr(ts, "to_pydatetime") or isinstance(ts, pd.Timestamp) else None,
        )
        if state.m > max_m:
            max_m = state.m
        if state.triggered:
            break

    return {
        "triggered": state.triggered,
        "triggered_at": state.triggered_at,
        "n_observations": state.n_observations,
        "final_m": state.m,
        "max_m": max_m,
    }
