"""
tests/test_slippage.py — Faz 5.1 Almgren-Chriss slipaj testleri.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.execution.slippage import (
    SlippageConfig,
    compute_slippage_bps,
    build_slippage_matrix,
)


# ──────────────────────────────────────────────────────────────────────
# 1. Disabled → backtest_engine sabit fee davranışını korur
# ──────────────────────────────────────────────────────────────────────
def test_slippage_zero_when_disabled(syn_db, cfg):
    """slippage_cfg=None → klasik backtest birebir aynı sonucu vermeli."""
    from engine.core.backtest_engine import run_pro_backtest

    rng = np.random.default_rng(0)
    signal = pd.Series(rng.standard_normal(len(syn_db)))

    _, met1 = run_pro_backtest(syn_db, signal, top_k=10, n_drop=2, slippage_cfg=None)
    _, met2 = run_pro_backtest(syn_db, signal, top_k=10, n_drop=2)

    assert abs(met1["IR"] - met2["IR"]) < 1e-9
    assert abs(met1["MDD"] - met2["MDD"]) < 1e-9


# ──────────────────────────────────────────────────────────────────────
# 2. Slipaj participation ile karekök monotonik artar
# ──────────────────────────────────────────────────────────────────────
def test_slippage_grows_with_participation():
    """
    V_traded küçük → slip küçük; büyük → slip büyük.
    σ ve ADV sabit tutulur, yalnızca v_traded değişir.
    """
    cfg = SlippageConfig(use_dynamic_slippage=True, gamma=0.10)
    asset_vol = 0.30   # %30 yıllık
    adv_TL    = 1e7    # 10M TL

    slip_small = compute_slippage_bps(v_traded_TL=1e4, asset_vol=asset_vol,
                                       adv_TL=adv_TL, cfg=cfg)
    slip_med   = compute_slippage_bps(v_traded_TL=1e5, asset_vol=asset_vol,
                                       adv_TL=adv_TL, cfg=cfg)
    slip_big   = compute_slippage_bps(v_traded_TL=1e6, asset_vol=asset_vol,
                                       adv_TL=adv_TL, cfg=cfg)

    assert slip_small < slip_med < slip_big

    # Karekök ilişkisi: 100x v_traded → ~10x slipaj
    ratio = slip_big / slip_small
    assert 8 < ratio < 12, f"karekök oranı kırıldı: {ratio:.2f}"


# ──────────────────────────────────────────────────────────────────────
# 3. ADV NaN → fallback_bps
# ──────────────────────────────────────────────────────────────────────
def test_slippage_fallback_when_adv_missing():
    """ADV None/NaN/0 → fallback_bps döner."""
    cfg = SlippageConfig(use_dynamic_slippage=True, fallback_bps=15.0)

    assert compute_slippage_bps(1e5, 0.20, np.nan, cfg) == 15.0
    assert compute_slippage_bps(1e5, 0.20, 0.0, cfg) == 15.0
    assert compute_slippage_bps(1e5, np.nan, 1e7, cfg) == 15.0


# ──────────────────────────────────────────────────────────────────────
# 4. Backtest dinamik slipaj → daha düşük net getiri
# ──────────────────────────────────────────────────────────────────────
def test_backtest_dynamic_slip_higher_cost(syn_db):
    """
    Aynı sinyal flat fee vs dynamic slip → dynamic slip daha fazla cost yer,
    böylece net getiri (genelde) daha düşük olur.
    """
    from engine.core.backtest_engine import run_pro_backtest

    rng = np.random.default_rng(7)
    signal = pd.Series(rng.standard_normal(len(syn_db)))

    _, met_flat = run_pro_backtest(syn_db, signal, top_k=10, n_drop=2)
    _, met_slip = run_pro_backtest(
        syn_db, signal, top_k=10, n_drop=2,
        slippage_cfg=SlippageConfig(use_dynamic_slippage=True, gamma=0.50,
                                     fallback_bps=50.0),
    )

    # Slip dahil net getiri ≤ flat (sentetik yüksek vol + agresif gamma)
    assert met_slip["Net Getiri (%)"] <= met_flat["Net Getiri (%)"] + 1e-6


# ──────────────────────────────────────────────────────────────────────
# Bonus: build_slippage_matrix smoke
# ──────────────────────────────────────────────────────────────────────
def test_build_slippage_matrix_shape(syn_db):
    """Matrix shape = (Date × Ticker), pozitif değerler veya fallback."""
    cfg = SlippageConfig(use_dynamic_slippage=True)
    ret_wide = syn_db.pivot_table(index="Date", columns="Ticker", values="Pclose").pct_change()
    slip = build_slippage_matrix(syn_db, ret_wide, cfg)
    assert slip.shape == ret_wide.shape
    assert (slip >= 0).all().all()
