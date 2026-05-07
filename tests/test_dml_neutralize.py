"""Birim testler: engine/data/factor_neutralize.py — DML neutralization."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.data.factor_neutralize import neutralize_signal, _dml_residualize


# ── Yardımcı: Sentetik (Ticker, Date) sinyali + faktörler ────────────────────

def _make_signal_factors(
    n_dates: int = 30,
    n_tickers: int = 80,
    seed: int = 42,
    factor_weight: float = 0.8,
):
    """Faktörlerle kasıtlı korelasyonlu sinyal + precomputed faktörler üret.

    Faktörler doğrudan oluşturulur ve signal = factor_weight * size + noise.
    Bu şekilde neutralize_signal'a `factors=` argümanı geçilerek
    _build_factors() hesaplamasından bağımsız, deterministik bir test sağlanır.
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range("2022-01-01", periods=n_dates)
    tickers = [f"T{i:04d}" for i in range(n_tickers)]
    index = pd.MultiIndex.from_product([tickers, dates], names=["Ticker", "Date"])

    n = len(index)
    # Faktörler: her tarihte cross-sectional olarak random
    size_vals = rng.standard_normal(n)
    vol_vals  = np.abs(rng.standard_normal(n))
    mom_vals  = rng.standard_normal(n)

    factors = pd.DataFrame(
        {"size": size_vals, "vol": vol_vals, "mom": mom_vals},
        index=index,
    )

    # Sinyal = ağırlıklı size faktörü + gürültü → güçlü size korelasyonu
    signal_vals = factor_weight * size_vals + (1 - factor_weight) * rng.standard_normal(n)
    signal = pd.Series(signal_vals, index=index, name="signal")

    # Minimal idx (neutralize_signal için gerekli ama factors= geçilince kullanılmaz)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    idx = pd.DataFrame({"Pclose": close}, index=index)

    return signal, idx, factors


# ── Testler ───────────────────────────────────────────────────────────────────

def test_use_dml_false_matches_ols():
    """use_dml=False varsayılanı mevcut OLS ile özdeş çıktı vermeli (backward compat)."""
    signal, idx, factors = _make_signal_factors()

    ols_resid = neutralize_signal(signal, idx, factors=factors, use_dml=False)
    default_resid = neutralize_signal(signal, idx, factors=factors)  # use_dml default=False

    assert isinstance(ols_resid, pd.Series)
    assert isinstance(default_resid, pd.Series)

    common = ols_resid.dropna().index.intersection(default_resid.dropna().index)
    assert len(common) > 0
    np.testing.assert_allclose(
        ols_resid.loc[common].values,
        default_resid.loc[common].values,
        atol=1e-10,
        err_msg="use_dml=False mevcut OLS ile özdeş olmalı",
    )


def test_dml_reduces_factor_correlation():
    """DML artığının size faktörüyle korelasyonu orijinal sinyalden belirgin az olmalı."""
    signal, idx, factors = _make_signal_factors(n_dates=50, n_tickers=100, seed=7, factor_weight=0.8)

    dml_resid = neutralize_signal(
        signal, idx, factors=factors, use_dml=True, two_stage=False, dml_n_splits=5
    )

    size_series = factors["size"]

    def mean_daily_corr(resid: pd.Series, factor: pd.Series) -> float:
        df = pd.DataFrame({"r": resid, "f": factor}).dropna().reset_index()
        try:
            corrs = (
                df.groupby("Date")
                .apply(lambda g: g["r"].corr(g["f"]), include_groups=False)
                .dropna()
            )
        except TypeError:
            corrs = df.groupby("Date").apply(lambda g: g["r"].corr(g["f"])).dropna()
        return float(corrs.abs().mean()) if len(corrs) > 0 else 0.0

    # Orijinal sinyal size ile ~0.8 korelasyonlu; DML < 0.15'e indirmeli
    raw_corr = mean_daily_corr(signal, size_series)
    dml_corr = mean_daily_corr(dml_resid, size_series)

    assert raw_corr > 0.5, f"Test kurulumu: orijinal korelasyon düşük ({raw_corr:.4f})"
    assert dml_corr < 0.15, (
        f"DML artığı hâlâ fazla korelasyonlu: {dml_corr:.4f} (orijinal: {raw_corr:.4f})"
    )


def test_dml_ols_correlation_difference():
    """Her iki yöntem de faktör korelasyonunu düşürüyor; DML gözle görülür kötü olmamalı."""
    signal, idx, factors = _make_signal_factors(n_dates=40, n_tickers=120, seed=99, factor_weight=0.8)

    ols_resid = neutralize_signal(signal, idx, factors=factors, use_dml=False, two_stage=False)
    dml_resid = neutralize_signal(signal, idx, factors=factors, use_dml=True, two_stage=False)

    size_series = factors["size"]

    def mean_abs_corr(resid: pd.Series) -> float:
        df = pd.DataFrame({"r": resid, "f": size_series}).dropna().reset_index()
        try:
            corrs = (
                df.groupby("Date")
                .apply(lambda g: g["r"].corr(g["f"]), include_groups=False)
                .dropna()
            )
        except TypeError:
            corrs = df.groupby("Date").apply(lambda g: g["r"].corr(g["f"])).dropna()
        return float(corrs.abs().mean()) if len(corrs) > 0 else 0.0

    ols_c = mean_abs_corr(ols_resid)
    dml_c = mean_abs_corr(dml_resid)

    # Her ikisi de 0.15'in altında olmalı; DML en fazla OLS + 0.05 kadar kötü olabilir
    assert ols_c < 0.15, f"OLS artığı temizlenmemiş: {ols_c:.4f}"
    assert dml_c < 0.15, f"DML artığı temizlenmemiş: {dml_c:.4f}"
    assert dml_c <= ols_c + 0.05, (
        f"DML ({dml_c:.4f}) OLS'den ({ols_c:.4f}) belirgin biçimde daha kötü"
    )


def test_dml_n_splits_1_edge_case():
    """n_splits=1 edge case: fonksiyon exception fırlatmamalı, geçerli array döndürmeli."""
    rng = np.random.default_rng(0)
    n = 50
    signal_arr = rng.standard_normal(n)
    factor_arr = rng.standard_normal((n, 3))

    result = _dml_residualize(signal_arr, factor_arr, n_splits=1)

    assert result.shape == (n,)
    assert not np.any(np.isnan(result)), "n_splits=1 NaN üretmemeli"
