"""
tests/test_position_sizer.py — Faz 4.1 Volatility Targeting unit testleri.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.risk.position_sizer import (
    RiskConfig,
    compute_asset_vol,
    compute_position_scale,
    apply_vol_target,
)


# ──────────────────────────────────────────────────────────────────────
# 1. compute_asset_vol — yıllıklandırma
# ──────────────────────────────────────────────────────────────────────
def test_compute_asset_vol_annualized():
    """N(0, 0.01) günlük getiri → yıllık vol ≈ 0.01·√252 ≈ 0.1587."""
    rng = np.random.default_rng(42)
    n = 300
    daily_ret = pd.Series(rng.normal(0, 0.01, n))
    vol = compute_asset_vol(daily_ret, window=20)

    # Son kısmın ortalaması 0.01·√252 civarında olmalı
    last_vol = vol.iloc[20:].mean()
    expected = 0.01 * np.sqrt(252)
    assert abs(last_vol - expected) < 0.03, f"vol={last_vol:.4f} expected≈{expected:.4f}"


# ──────────────────────────────────────────────────────────────────────
# 2. compute_position_scale — min/max clip
# ──────────────────────────────────────────────────────────────────────
def test_position_scale_clipped():
    """Scale min_scale ve max_scale sınırları dışına çıkmamalı."""
    cfg = RiskConfig(target_annual_vol=0.15, min_scale=0.1, max_scale=3.0)

    # Çok yüksek vol (50% yıllık) → scale = 0.15/0.50 = 0.30 — normal range
    # Çok düşük vol (1% yıllık) → scale = 0.15/0.01 = 15.0 → clip to 3.0
    # Çok yüksek vol (300% yıllık) → scale = 0.15/3.0 = 0.05 → clip to 0.1
    test_vols = pd.Series([0.50, 0.01, 3.00, np.nan, 0.0])
    scale = compute_position_scale(test_vols, cfg)

    assert scale.min() >= cfg.min_scale - 1e-9
    assert scale.max() <= cfg.max_scale + 1e-9
    # NaN vol → scale=1.0 (nötr)
    assert abs(scale.iloc[3] - 1.0) < 1e-9
    # Sıfır vol → scale=1.0 (nötr)
    assert abs(scale.iloc[4] - 1.0) < 1e-9


# ──────────────────────────────────────────────────────────────────────
# 3. apply_vol_target — sabit volatilite → scale ≈ 1.0
# ──────────────────────────────────────────────────────────────────────
def test_apply_vol_target_neutral_period():
    """
    target_annual_vol = 0.15, asset vol da ≈ 0.15 → scale ≈ 1.0
    → scaled_ret ≈ original_ret
    """
    rng = np.random.default_rng(99)
    n = 200
    # Günlük std=0.15/√252 ≈ 0.00945 olacak şekilde oluştur
    target_daily_std = 0.15 / np.sqrt(252)
    ret = pd.DataFrame({
        "A": rng.normal(0, target_daily_std, n),
        "B": rng.normal(0, target_daily_std, n),
    })

    cfg = RiskConfig(use_vol_target=True, target_annual_vol=0.15, vol_window=20)
    scaled = apply_vol_target(ret, cfg)

    # Warmup period (ilk 20 gün) hariç, scaled ≈ original (scale ≈ 1.0)
    tail_orig   = ret.iloc[25:].values
    tail_scaled = scaled.iloc[25:].values
    ratio = tail_scaled / np.where(tail_orig == 0, np.nan, tail_orig)
    ratio = ratio[~np.isnan(ratio)]
    assert abs(ratio.mean() - 1.0) < 0.3, f"ortalama scale={ratio.mean():.3f} (beklenen ≈1.0)"


# ──────────────────────────────────────────────────────────────────────
# 4. Geriye dönük uyumluluk — risk_cfg=None → 1/N pipeline bozulmaz
# ──────────────────────────────────────────────────────────────────────
def test_backtest_integration_default_off(syn_db, cfg):
    """
    run_pro_backtest(risk_cfg=None) → mevcut 1/N davranışı değişmez.
    Aynı db + sinyal için iki çağrı birebir aynı metrikleri döndürmeli.
    """
    from engine.core.backtest_engine import run_pro_backtest

    # Sabit sinyal (tüm hisseler eşit — 1/N mantığı)
    signal = pd.Series(
        np.ones(len(syn_db)),
        index=syn_db.index if isinstance(syn_db.index, pd.RangeIndex) else range(len(syn_db))
    )

    _, met1 = run_pro_backtest(syn_db, signal, top_k=5, n_drop=1, risk_cfg=None)
    _, met2 = run_pro_backtest(syn_db, signal, top_k=5, n_drop=1)  # default

    assert abs(met1["IR"] - met2["IR"]) < 1e-9
    assert abs(met1["MDD"] - met2["MDD"]) < 1e-9
