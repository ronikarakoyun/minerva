"""Birim testler: engine/data/fracdiff.py.

López de Prado AFML §5 referansı: d=0 kimlik, d=1 birinci fark, ara değerler
hafıza-koruyan kesirli fark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.data.fracdiff import _ffd_weights, find_min_d, frac_diff_ffd


def test_d0_returns_input_minus_constant():
    """d=0 → tek ağırlık [1.0], çıktı = girdi (sabit kayma yok)."""
    rng = np.random.default_rng(0)
    s = pd.Series(np.cumsum(rng.standard_normal(500)))
    out = frac_diff_ffd(s, d=0.0)
    assert len(out) == len(s)
    np.testing.assert_allclose(out.dropna().values, s.values, rtol=1e-12)


def test_d1_equals_first_difference():
    """d=1 ağırlıkları [1, -1] → birinci fark."""
    rng = np.random.default_rng(1)
    s = pd.Series(np.cumsum(rng.standard_normal(500)))
    out = frac_diff_ffd(s, d=1.0, thresh=1e-6).dropna()
    expected = s.diff().dropna()
    n = min(len(out), len(expected))
    np.testing.assert_allclose(
        out.values[-n:], expected.values[-n:], atol=1e-9
    )


def test_random_walk_becomes_stationary():
    """Random walk d_star ile durağan olmalı (ADF p < 0.05)."""
    pytest.importorskip("statsmodels")
    from statsmodels.tsa.stattools import adfuller

    rng = np.random.default_rng(42)
    rw = pd.Series(np.cumsum(rng.standard_normal(2000)))
    d_star = find_min_d(rw, p_value_target=0.05)
    assert 0.0 < d_star <= 1.0

    diffed = frac_diff_ffd(rw, d=d_star).dropna()
    p_val = adfuller(diffed.values, maxlag=1, regression="c", autolag=None)[1]
    assert p_val < 0.05, f"Beklenen p<0.05, gelen={p_val:.4f} (d_star={d_star})"


def test_d_out_of_range_raises():
    """d ∉ [0, 1] → ValueError."""
    s = pd.Series([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        frac_diff_ffd(s, d=1.5)
    with pytest.raises(ValueError):
        frac_diff_ffd(s, d=-0.1)


def test_weights_decay_monotonically():
    """FFD ağırlıkları |w_k| monoton azalmalı; thresh altında kesilmeli."""
    w = _ffd_weights(d=0.5, thresh=1e-4)
    abs_w = np.abs(w[::-1])  # k=0..K-1 sırası
    # İlk birkaç ağırlığın mutlak değeri azalan trendde
    assert abs_w[0] == 1.0
    assert all(abs_w[i] >= abs_w[i + 1] - 1e-12 for i in range(min(20, len(abs_w) - 1)))
    # Son ağırlık kesim eşiğinin üstünde, bir sonrası altında olmalı
    assert abs_w[-1] >= 1e-4


def test_nan_safe_propagates_nan_in_window():
    """Pencere içinde NaN varsa o gözlem NaN çıkar; pencere geçince temiz."""
    rng = np.random.default_rng(7)
    arr = np.cumsum(rng.standard_normal(200))
    arr[10] = np.nan  # tek nokta NaN
    s = pd.Series(arr)
    out = frac_diff_ffd(s, d=0.5, thresh=1e-2)
    # NaN penceresi geçtikten sonraki son nokta temiz olmalı
    assert not np.isnan(out.iloc[-1])
    # NaN'ı içeren konum kesinlikle NaN
    assert np.isnan(out.iloc[10])
