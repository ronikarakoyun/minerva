"""Birim testler: engine/data/meta_label.py — PR-9 pipeline eklentileri."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.data.meta_label import (
    MetaModel,
    apply_meta_filter_to_pool,
    build_meta_dataset,
)


# ── Yardımcı ─────────────────────────────────────────────────────────────────

def _make_prob_df(n_dates: int = 60, K: int = 3, seed: int = 0) -> pd.DataFrame:
    """K rejimli HMM olasılık matrisi (her satır toplamı = 1)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    raw = rng.dirichlet(alpha=[1.0] * K, size=n_dates)
    return pd.DataFrame(
        raw,
        index=dates,
        columns=[f"regime_{k}" for k in range(K)],
    )


def _make_signal_idx(n_dates: int = 60, n_tickers: int = 30, seed: int = 7):
    """Minimal (Ticker, Date) MultiIndex sinyal + idx."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    index = pd.MultiIndex.from_product([tickers, dates], names=["Ticker", "Date"])
    n = len(index)
    signal = pd.Series(rng.standard_normal(n), index=index, name="signal")
    next_ret = pd.Series(rng.standard_normal(n) * 0.01, index=index, name="Next_Ret")
    idx = pd.DataFrame({"Next_Ret": next_ret, "Pclose": 100.0}, index=index)
    return signal, idx


# ── Testler ───────────────────────────────────────────────────────────────────

def test_regime_entropy_always_nonnegative():
    """build_meta_dataset'e prob_df geçilince regime_entropy ≥ 0 olmalı."""
    signal, idx = _make_signal_idx()
    prob_df = _make_prob_df()

    ds = build_meta_dataset(signal, idx, prob_df=prob_df)

    assert "regime_entropy" in ds.columns, "regime_entropy kolonu oluşturulmadı"
    assert (ds["regime_entropy"] >= 0).all(), (
        f"Negatif entropy değerleri: {ds['regime_entropy'].min()}"
    )


def test_regime_entropy_bounded_by_log_K():
    """K rejimli uniform dağılım için entropy ≤ log(K)."""
    K = 3
    n_dates = 40
    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    # Uniform dağılım → maksimum entropi = log(K)
    uniform_data = np.full((n_dates, K), 1.0 / K)
    prob_df = pd.DataFrame(
        uniform_data,
        index=dates,
        columns=[f"regime_{k}" for k in range(K)],
    )
    signal, idx = _make_signal_idx(n_dates=n_dates)

    ds = build_meta_dataset(signal, idx, prob_df=prob_df)

    assert "regime_entropy" in ds.columns
    max_entropy = np.log(K)
    assert (ds["regime_entropy"] <= max_entropy + 1e-9).all(), (
        f"Entropy log(K)={max_entropy:.4f}'ü aşıyor: {ds['regime_entropy'].max():.4f}"
    )


def test_apply_meta_filter_to_pool_threshold_one_returns_empty():
    """threshold=1.0 → tüm pool filtrelenmeli (proba asla 1.0'a ulaşamaz)."""
    pool = list(range(10))  # dummy items
    meta = MetaModel(fit_failed=True)  # feature_df yok, fallback proba=0.5

    result = apply_meta_filter_to_pool(pool, meta, threshold=1.0)

    assert result == [], f"threshold=1.0 ile boş liste beklendi, {len(result)} döndü"


def test_apply_meta_filter_to_pool_threshold_zero_returns_all():
    """threshold=0.0 → tüm pool geçmeli."""
    pool = ["alpha_A", "alpha_B", "alpha_C"]
    meta = MetaModel(fit_failed=True)  # fallback proba=0.5

    result = apply_meta_filter_to_pool(pool, meta, threshold=0.0)

    assert result == pool, f"threshold=0.0 ile tüm havuz beklendi, {len(result)} döndü"
