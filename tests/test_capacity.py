"""
tests/test_capacity.py — Faz 4.3 ADV kapasite tahmini testleri.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.risk.capacity import (
    CapacityConfig,
    compute_adv,
    estimate_formula_capacity,
)


def _make_db(tickers=("AAA", "BBB", "CCC"), n_days=60, seed=42) -> pd.DataFrame:
    """Sentetik flat price DataFrame: Ticker, Date, Pclose, Vlot."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.bdate_range("2023-01-01", periods=n_days)
    for t in tickers:
        pclose = 10.0 + rng.normal(0, 0.5, n_days).cumsum()
        pclose = np.abs(pclose) + 1.0  # pozitif tut
        vlot   = rng.integers(50_000, 500_000, n_days).astype(float)
        for i, d in enumerate(dates):
            rows.append({"Ticker": t, "Date": d, "Pclose": pclose[i], "Vlot": vlot[i]})
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
# 1. compute_adv — rolling ADV_TL doğru hesaplanır
# ──────────────────────────────────────────────────────────────────────
def test_compute_adv_rolling():
    """ADV_TL = rolling(20).mean(Vlot × Pclose), shift(1) uygulanmış."""
    cfg = CapacityConfig(adv_window=20)
    db  = _make_db(tickers=["X"], n_days=50)
    adv = compute_adv(db, cfg)

    assert "ADV_TL" in adv.columns
    assert isinstance(adv.index, pd.MultiIndex)

    # Look-ahead: shift(1) → ilk 20+1 gün NaN olmalı (warmup + shift)
    x_adv = adv.loc["X", "ADV_TL"]
    n_nan = x_adv.isna().sum()
    # En az window//4 kadar NaN (min_periods) sonra başlar, ama shift(1) +1 NaN
    assert n_nan >= 1, f"shift(1) sonrası en az 1 NaN bekleniyor, {n_nan} bulundu"

    # NaN olmayan değerler pozitif olmalı
    valid = x_adv.dropna()
    assert (valid > 0).all()


# ──────────────────────────────────────────────────────────────────────
# 2. estimate_formula_capacity — en düşük ADV bağlayıcı ticker'ı belirler
# ──────────────────────────────────────────────────────────────────────
def test_estimate_capacity_min_binding_ticker():
    """
    İki ticker: biri yüksek hacimli, biri düşük hacimli.
    Kapasite düşük hacimli ticker tarafından sınırlanmalı.
    """
    rng = np.random.default_rng(0)
    n = 60
    dates = pd.bdate_range("2023-01-01", periods=n)

    rows = []
    # HIGH ticker: yüksek hacim (1M lot × 10 TL = 10M TL/gün ADV)
    for d in dates:
        rows.append({"Ticker": "HIGH", "Date": d,
                     "Pclose": 10.0, "Vlot": 1_000_000.0})
    # LOW ticker: düşük hacim (10K lot × 10 TL = 100K TL/gün ADV)
    for d in dates:
        rows.append({"Ticker": "LOW", "Date": d,
                     "Pclose": 10.0, "Vlot": 10_000.0})

    db = pd.DataFrame(rows)

    # Sinyal: her iki ticker için son 30 gün
    sig_dates = dates[30:]
    tickers_sig = ["HIGH", "LOW"]
    idx = pd.MultiIndex.from_product([tickers_sig, sig_dates], names=["Ticker", "Date"])
    signal = pd.Series(1.0, index=idx)

    cfg = CapacityConfig(adv_window=10, adv_pct_limit=0.05,
                         min_advs_TL=50_000, portfolio_size=2)
    result = estimate_formula_capacity(signal, db, cfg)

    assert result["binding_ticker"] == "LOW", f"Beklenen LOW, bulundu {result['binding_ticker']}"
    # HIGH'ın ADV ≈ 10M, LOW'un ADV ≈ 100K
    # max_aum = LOW × 0.05 × 2 = 100K × 0.05 × 2 = 10K
    assert result["max_aum_TL"] < 1_000_000


# ──────────────────────────────────────────────────────────────────────
# 3. min_advs_TL filtresi — likit dışı hisseler n_tradable_days'e dahil olmaz
# ──────────────────────────────────────────────────────────────────────
def test_min_advs_TL_filters_illiquid():
    """
    Tüm hisseler min_advs_TL altındaysa n_tradable_days=0 ve max_aum_TL=0.
    """
    rng = np.random.default_rng(5)
    n = 50
    dates = pd.bdate_range("2023-01-01", periods=n)

    rows = []
    # Çok düşük hacim: 100 lot × 1 TL = 100 TL/gün ADV
    for d in dates:
        rows.append({"Ticker": "ILLIQ", "Date": d,
                     "Pclose": 1.0, "Vlot": 100.0})
    db = pd.DataFrame(rows)

    sig_dates = dates[20:]
    idx = pd.MultiIndex.from_tuples(
        [("ILLIQ", d) for d in sig_dates], names=["Ticker", "Date"]
    )
    signal = pd.Series(1.0, index=idx)

    cfg = CapacityConfig(adv_window=10, adv_pct_limit=0.05,
                         min_advs_TL=1_000_000)  # 1M TL eşiği
    result = estimate_formula_capacity(signal, db, cfg)

    assert result["n_tradable_days"] == 0
    assert result["max_aum_TL"] == 0.0
