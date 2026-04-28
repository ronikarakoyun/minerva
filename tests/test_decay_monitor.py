"""
tests/test_decay_monitor.py — Faz 4.2 Page-Hinkley alpha decay testleri.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.risk.decay_monitor import (
    DecayConfig,
    DecayState,
    update_decay_state,
    scan_decay,
)


# ──────────────────────────────────────────────────────────────────────
# 1. Canlı ≈ backtest → decay tetiklenmez
# ──────────────────────────────────────────────────────────────────────
def test_no_decay_when_live_matches_backtest():
    """μ_live ≈ μ_bt → Page-Hinkley m_t ≈ 0 → triggered=False."""
    cfg = DecayConfig(delta=5e-4, lambda_threshold=0.01, consecutive_days=5)
    backtest_mean = 0.001
    backtest_std  = 0.005

    # 60 gün gerçekçi live (μ civarında gürültü)
    rng = np.random.default_rng(7)
    live = pd.Series(rng.normal(backtest_mean, backtest_std * 0.5, 60))

    result = scan_decay(live, backtest_mean, backtest_std, cfg)
    assert result["triggered"] is False, f"Beklenmeyen tetik: {result}"
    assert result["max_m"] < cfg.lambda_threshold * 3


# ──────────────────────────────────────────────────────────────────────
# 2. Kalıcı negatif drift → decay tetiklenir
# ──────────────────────────────────────────────────────────────────────
def test_page_hinkley_triggers_on_persistent_negative_drift():
    """
    Sürekli -3σ live return → 5 gün art arda kümülatif sapma → triggered=True.
    """
    cfg = DecayConfig(delta=5e-4, lambda_threshold=0.005, consecutive_days=5,
                      sigma_floor=2.0)
    backtest_mean = 0.001
    backtest_std  = 0.005

    # 20 gün normal, ardından 30 gün -3σ sapma
    normal_days = [backtest_mean] * 20
    bad_days    = [backtest_mean - 3 * backtest_std] * 30
    live = pd.Series(normal_days + bad_days)

    result = scan_decay(live, backtest_mean, backtest_std, cfg)
    assert result["triggered"] is True, f"Trigger beklendi ama gelmedi: {result}"
    # Tetiklenme noktası bad_days başladıktan sonra olmalı
    assert result["n_observations"] > 20


# ──────────────────────────────────────────────────────────────────────
# 3. Geçici dip → m_t sıfırlanır, alarm tetiklenmez
# ──────────────────────────────────────────────────────────────────────
def test_state_resets_after_recovery():
    """
    Kısa bir dip ardından iyileşme → m_t=0'a iner (max(0,...)), alarm yok.
    """
    cfg = DecayConfig(delta=5e-4, lambda_threshold=0.05, consecutive_days=10,
                      sigma_floor=2.0)
    backtest_mean = 0.001
    backtest_std  = 0.005

    state = DecayState()
    # 4 gün kötü
    for _ in range(4):
        update_decay_state(state, backtest_mean - 4 * backtest_std,
                           backtest_mean, backtest_std, cfg)

    # 10 gün iyi (m_t sıfırlanır)
    for _ in range(10):
        update_decay_state(state, backtest_mean + backtest_std,
                           backtest_mean, backtest_std, cfg)

    assert state.consecutive_alarms == 0, f"consecutive_alarms={state.consecutive_alarms}"
    assert state.triggered is False


# ──────────────────────────────────────────────────────────────────────
# 4. scan_decay — ilk breach tarihi doğru tespit edilir
# ──────────────────────────────────────────────────────────────────────
def test_scan_decay_finds_first_breach_date():
    """scan_decay bilinen breach noktasının tarihini doğru bulur."""
    cfg = DecayConfig(delta=1e-5, lambda_threshold=0.002, consecutive_days=3,
                      sigma_floor=1.0)
    backtest_mean = 0.001
    backtest_std  = 0.005

    # Tarih indexed Series
    dates_normal = pd.date_range("2024-01-01", periods=15, freq="B")
    dates_bad    = pd.date_range(dates_normal[-1] + pd.offsets.BDay(1), periods=20, freq="B")
    all_dates    = dates_normal.append(dates_bad)

    live = pd.Series(
        [backtest_mean] * 15 + [backtest_mean - 4 * backtest_std] * 20,
        index=all_dates,
    )

    result = scan_decay(live, backtest_mean, backtest_std, cfg)
    assert result["triggered"] is True
    # triggered_at, normal dönemin bitmesinden sonra olmalı
    assert result["triggered_at"] > dates_normal[-1]
    assert isinstance(result["triggered_at"], pd.Timestamp)
