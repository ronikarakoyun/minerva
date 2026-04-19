"""
tests/test_deflated_sharpe.py — engine/deflated_sharpe.py için birim testler.

Bailey & López de Prado (2014) Deflated Sharpe Ratio:
  - Havuz büyüklüğüne göre SR deflation
  - Skewness / fat-tail cezası
  - p_value yorumu (Φ(DSR_z))

Çalıştırma:
    python -m pytest tests/test_deflated_sharpe.py -v
"""
import sys
import os
import math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from engine.deflated_sharpe import (
    compute_sharpe_series,
    expected_max_sr_null,
    deflated_sharpe_ratio,
    compute_pool_dsr,
)


def _make_equity(returns: np.ndarray, start: float = 1.0) -> pd.Series:
    """Günlük getiri dizisinden kümülatif equity serisi üret."""
    dates = pd.date_range("2020-01-01", periods=len(returns), freq="B")
    return pd.Series((1.0 + returns).cumprod() * start, index=dates)


class TestComputeSharpeSeries:
    """compute_sharpe_series(equity) → (SR, skew, kurt)"""

    def test_normal_returns_sr(self):
        """Bilinen μ ve σ'dan SR ≈ μ/σ * √252 beklenir."""
        rng = np.random.default_rng(0)
        mu, sigma = 0.001, 0.01
        rets = rng.normal(mu, sigma, 1000)
        eq   = _make_equity(rets)
        sr, _, _ = compute_sharpe_series(eq, freq=252)
        expected = mu / sigma * np.sqrt(252)
        assert abs(sr - expected) < 2.0, (
            f"SR ≈ {expected:.2f} beklenir, alındı {sr:.2f}"
        )

    def test_zero_return_sr_nan(self):
        """Sabit getiri (std=0) → SR = nan."""
        eq = _make_equity(np.zeros(100))
        sr, _, _ = compute_sharpe_series(eq)
        assert np.isnan(sr), "Sabit getiri → SR = nan"

    def test_skew_negative_for_left_tail(self):
        """Sol kuyruklu dağılım → negatif skewness."""
        rng = np.random.default_rng(1)
        rets = rng.normal(0.001, 0.01, 1000)
        rets[rng.choice(1000, 20, replace=False)] = -0.08
        eq = _make_equity(rets)
        _, skw, _ = compute_sharpe_series(eq)
        assert skw < 0, f"Sol kuyruklu dağılım negatif skew vermeli: {skw:.3f}"

    def test_fat_tail_positive_kurt(self):
        """Fat-tail (Student-t) → pozitif excess kurtosis."""
        rng = np.random.default_rng(2)
        rets = rng.standard_t(df=4, size=2000) * 0.01
        eq   = _make_equity(rets)
        _, _, kurt = compute_sharpe_series(eq)
        assert kurt > 0, f"Fat-tail pozitif excess kurtosis vermeli: {kurt:.3f}"

    def test_short_series_returns_nan(self):
        """Çok kısa seri (<5 gün) → SR = nan, crash yok."""
        eq = _make_equity(np.array([0.01, -0.005]))
        sr, skw, kurt = compute_sharpe_series(eq)
        assert np.isnan(sr)


class TestExpectedMaxSrNull:
    """expected_max_sr_null(n_trials, T) → float"""

    def test_single_trial_near_zero(self):
        """Tek deneme → E[max SR] ≈ 0."""
        e = expected_max_sr_null(1, T=500)
        assert abs(e) < 0.5, f"n=1 → E[max SR_null]≈0, alındı {e:.4f}"

    def test_increases_with_n(self):
        """Daha fazla deneme → daha yüksek beklenen maksimum null SR."""
        e10   = expected_max_sr_null(10,   T=500)
        e100  = expected_max_sr_null(100,  T=500)
        e1000 = expected_max_sr_null(1000, T=500)
        assert e10 < e100 < e1000, (
            f"E[max SR_null] N ile artmalı: {e10:.3f} < {e100:.3f} < {e1000:.3f}"
        )

    def test_decreases_with_larger_T(self):
        """Daha büyük T → SE_null = 1/sqrt(T) küçük → E[max SR_null] küçük."""
        e_short = expected_max_sr_null(100, T=100)
        e_long  = expected_max_sr_null(100, T=2000)
        assert e_long < e_short, (
            f"Büyük T → küçük E[max SR_null]: {e_long:.3f} < {e_short:.3f}"
        )


class TestDeflatedSharpeRatio:
    """deflated_sharpe_ratio(sr, T, skew, kurt, n_trials) → (DSR_z, p_value)"""

    def test_larger_n_reduces_dsr_z(self):
        """
        Aynı SR için daha fazla deneme → DSR_z azalmalı.
        (Gürültü barı yükseliyor, mevcut SR nispeten daha az etkileyici.)
        """
        sr, T, skew, kurt = 2.0, 1000, 0.0, 0.0
        z10,   _ = deflated_sharpe_ratio(sr, T, skew, kurt, n_trials=10)
        z10000, _ = deflated_sharpe_ratio(sr, T, skew, kurt, n_trials=10000)
        assert z10 > z10000, (
            f"Küçük N → daha yüksek DSR_z beklenir: {z10:.2f} > {z10000:.2f}"
        )

    def test_single_trial_dsr_high(self):
        """Tek deneme, iyi SR → DSR_z pozitif ve yüksek."""
        z, pv = deflated_sharpe_ratio(sr=2.0, T=1000, skew=0.0, kurt=0.0, n_trials=1)
        assert np.isfinite(z) and z > 1.0, f"Tek trial → DSR_z > 1 beklenir: {z:.3f}"

    def test_negative_skew_reduces_dsr(self):
        """
        Negatif skewness → SE[SR_hat] büyür → DSR_z düşer (AFML yön testi).
        (var_sr = (1 - skew·SR + ...) / T; skew negatif → var_sr artar)
        """
        sr, T, kurt, n = 1.5, 500, 0.0, 50
        z_pos_skew, _ = deflated_sharpe_ratio(sr, T, skew=2.0,  kurt=kurt, n_trials=n)
        z_neg_skew, _ = deflated_sharpe_ratio(sr, T, skew=-2.0, kurt=kurt, n_trials=n)
        # Negatif skew → var_sr büyür → SE büyür → DSR_z küçülür
        assert z_neg_skew < z_pos_skew, (
            f"Negatif skew DSR_z'yi küçültmeli: {z_neg_skew:.3f} < {z_pos_skew:.3f}"
        )

    def test_fat_tail_reduces_dsr(self):
        """Fat-tail (yüksek kurtosis) → SE[SR_hat] büyür → DSR_z düşer."""
        sr, T, skew, n = 1.5, 500, 0.0, 50
        z_normal, _ = deflated_sharpe_ratio(sr, T, skew, kurt=0.0,  n_trials=n)
        z_fatt,   _ = deflated_sharpe_ratio(sr, T, skew, kurt=10.0, n_trials=n)
        assert z_fatt < z_normal, (
            f"Fat-tail DSR_z küçültmeli: {z_fatt:.3f} < {z_normal:.3f}"
        )

    def test_p_value_range(self):
        """p_value ∈ [0, 1]."""
        for sr in [-1.0, 0.0, 1.0, 3.0]:
            z, pv = deflated_sharpe_ratio(sr=sr, T=500, skew=0.0, kurt=0.0, n_trials=100)
            if np.isfinite(pv):
                assert 0.0 <= pv <= 1.0, f"p_value sınır dışı: {pv}"

    def test_very_short_series_returns_nan(self):
        """T < 5 → (nan, nan) döner, crash yok."""
        z, pv = deflated_sharpe_ratio(sr=2.0, T=3, skew=0.0, kurt=0.0, n_trials=10)
        assert np.isnan(z) and np.isnan(pv)

    def test_low_sr_negative_dsr_z(self):
        """
        SR = 0 (düz getiri, gürültü) → DSR_z negatif beklenir.
        (Gürültü barının altında kaldı.)
        """
        z, pv = deflated_sharpe_ratio(
            sr=0.0, T=500, skew=0.0, kurt=0.0, n_trials=100
        )
        if np.isfinite(z):
            assert z < 0.0, f"SR=0 → DSR_z < 0 beklenir: {z:.3f}"

    def test_high_sr_positive_dsr_z(self):
        """
        SR çok yüksek (N küçük) → DSR_z pozitif ve anlamlı (p_value > 0.5).
        """
        z, pv = deflated_sharpe_ratio(
            sr=3.0, T=500, skew=0.0, kurt=0.0, n_trials=5
        )
        if np.isfinite(z):
            assert z > 0.0, f"Yüksek SR, küçük N → DSR_z > 0 beklenir: {z:.3f}"


class TestComputePoolDsr:
    """compute_pool_dsr(equity_curves, n_trials) → pd.DataFrame"""

    def _make_pool(self, n_formulas: int, mu: float = 0.0, seed: int = 0) -> dict:
        rng = np.random.default_rng(seed)
        pool = {}
        for i in range(n_formulas):
            rets = rng.normal(mu, 0.01, 500)
            pool[f"F{i:03d}"] = _make_equity(rets)
        return pool

    def test_returns_dataframe(self):
        """Çıktı pd.DataFrame ve beklenen sütunlar."""
        pool = self._make_pool(10)
        df   = compute_pool_dsr(pool)
        assert isinstance(df, pd.DataFrame)
        for col in ("formula", "SR", "DSR_z", "p_value", "significant"):
            assert col in df.columns, f"Eksik sütun: {col}"

    def test_row_count_matches_pool(self):
        """Satır sayısı = havuz formül sayısı."""
        pool = self._make_pool(15)
        df   = compute_pool_dsr(pool)
        assert len(df) == 15

    def test_significant_flag(self):
        """p_value ≥ 0.95 olanlar significant=True."""
        pool = self._make_pool(5)
        df   = compute_pool_dsr(pool)
        for _, row in df.iterrows():
            if np.isfinite(row["p_value"]):
                expected = row["p_value"] >= 0.95
                assert row["significant"] == expected

    def test_null_equity_handled(self):
        """None / boş equity crash olmadan işlenmeli."""
        pool = {"bad": None, "short": _make_equity(np.array([0.01]))}
        df   = compute_pool_dsr(pool)
        assert len(df) == 2
        assert np.isnan(df[df["formula"] == "bad"]["SR"].values[0])

    def test_larger_n_trials_reduces_dsr(self):
        """
        n_trials arttıkça en yüksek SR formülün DSR_z azalmalı
        (gürültü barı yükseliyor).
        """
        pool   = self._make_pool(5, mu=0.001)
        df5    = compute_pool_dsr(pool, n_trials=5)
        df5000 = compute_pool_dsr(pool, n_trials=5000)
        # Her formül için DSR_z(n=5) ≥ DSR_z(n=5000)
        z5    = df5["DSR_z"].dropna()
        z5000 = df5000["DSR_z"].dropna()
        if len(z5) and len(z5000):
            assert float(z5.mean()) >= float(z5000.mean()), (
                f"n_trials arttıkça DSR_z ortalaması azalmalı: "
                f"z5={z5.mean():.2f}, z5000={z5000.mean():.2f}"
            )
