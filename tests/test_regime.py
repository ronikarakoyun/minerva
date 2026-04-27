"""
tests/test_regime.py — engine/regime.py için birim testler.

Piyasa rejimi tespiti (bull/chop/bear):
  - Monoton artan benchmark → bull dominant
  - Düz + yüksek volatilite → chop
  - Düşüş + yüksek volatilite → bear
  - compute_wf_fitness entegrasyon (regime_breakdown dict)
  - Gap (eksik günler) forward-fill sanity

Çalıştırma:
    python -m pytest tests/test_regime.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from engine.data.regime import compute_regime, attach_regime_to_index, regime_breakdown, REGIMES


def _make_bench(n: int = 300, freq: str = "B") -> pd.Series:
    """Boş iş-günü benchmark serisi iskeleti."""
    dates = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(index=dates, dtype=float)


class TestComputeRegime:
    """compute_regime(bench) → pd.Series[bull/chop/bear]"""

    def test_monoton_bull(self):
        """
        Sürekli artan fiyat + düşük volatilite → dominant 'bull'.
        """
        n = 300
        bench = _make_bench(n)
        # Çok düşük volatilite ile sürekli artış
        bench[:] = np.linspace(1000, 1500, n)  # perfect trend, vol=0
        regime = compute_regime(bench, vol_window=10, trend_window=20, vol_hi=0.05)
        # Warm-up'tan sonra çoğunluk bull olmalı
        valid = regime.dropna()
        valid = valid[regime.notna()]
        bull_ratio = (valid == "bull").mean()
        assert bull_ratio > 0.5, (
            f"Monoton artışta 'bull' dominant beklenir: {bull_ratio:.2f}"
        )

    def test_flat_high_vol_chop(self):
        """
        Yatay fiyat + yüksek volatilite → 'chop' veya 'bear'.
        Trendsiz olduğu için bull olmamalı.
        """
        rng = np.random.default_rng(42)
        n = 300
        bench = _make_bench(n)
        # Yatay ama yüksek volatilite
        bench[:] = 1000 + rng.normal(0, 20, n).cumsum() * 0  # flat
        rets = pd.Series(
            rng.normal(0, 0.03, n),  # yüksek günlük volatilite
            index=bench.index,
        )
        bench = (1 + rets).cumprod() * 1000
        # Benchmark'ı yatay tut (cumsum = 0 net drift)
        bench = bench / bench.iloc[0] * 1000  # normalize
        regime = compute_regime(bench, vol_window=10, trend_window=20, vol_hi=0.022)
        valid = regime.dropna()
        bull_ratio = (valid == "bull").mean()
        # Yatay + yüksek vol → bull olmamalı (az olabilir)
        assert bull_ratio < 0.5, (
            f"Yatay+yüksek-vol senaryoda bull dominant olmamalı: {bull_ratio:.2f}"
        )

    def test_decline_high_vol_bear(self):
        """
        Sürekli düşen fiyat + yüksek volatilite → dominant 'bear'.
        """
        n = 300
        bench = _make_bench(n)
        bench[:] = np.linspace(1500, 500, n)  # düşüş
        # Yüksek volatilite ekle
        rng = np.random.default_rng(7)
        noise = 1 + rng.normal(0, 0.025, n)  # %2.5 günlük std
        price = bench.values * noise.cumprod()
        bench[:] = price
        regime = compute_regime(bench, vol_window=10, trend_window=20, vol_hi=0.022)
        valid = regime.dropna()
        bear_ratio = (valid == "bear").mean()
        assert bear_ratio > 0.3, (
            f"Düşüş + yüksek volatilite → 'bear' bekleniyordu: {bear_ratio:.2f}"
        )

    def test_output_only_valid_labels(self):
        """Çıktıda sadece 'bull', 'chop', 'bear' değerleri olmalı."""
        n = 200
        bench = _make_bench(n)
        rng = np.random.default_rng(0)
        bench[:] = (1 + rng.normal(0.0005, 0.01, n)).cumprod() * 1000
        regime = compute_regime(bench)
        valid_labels = set(REGIMES)
        for val in regime.unique():
            assert val in valid_labels, f"Geçersiz rejim etiketi: {val}"

    def test_gap_forward_fill(self):
        """
        Eksik günler (NaN) chop'a dönüştürülmeli, crash olmamalı.
        """
        n = 200
        bench = _make_bench(n)
        rng = np.random.default_rng(1)
        bench[:] = (1 + rng.normal(0, 0.01, n)).cumprod() * 1000
        # 10 tarih sil (gap)
        bench = bench.drop(bench.index[50:60])
        # Crash olmamalı
        regime = compute_regime(bench)
        assert isinstance(regime, pd.Series)
        assert len(regime) == n - 10

    def test_all_three_regimes_present(self):
        """
        Gerçekçi piyasa senaryosunda 3 rejim de non-zero olmalı.
        """
        rng = np.random.default_rng(42)
        n = 1000
        # Bull: 0-300, Bear: 300-600 (strong downtrend), Chop: 600-1000
        prices = np.empty(n)
        prices[0] = 1000
        for i in range(1, 300):
            prices[i] = prices[i-1] * (1 + rng.normal(0.002, 0.005))  # bull
        for i in range(300, 600):
            prices[i] = prices[i-1] * (1 + rng.normal(-0.003, 0.025))  # bear
        for i in range(600, 1000):
            prices[i] = prices[i-1] * (1 + rng.normal(0.0001, 0.025))  # chop
        bench = pd.Series(prices, index=pd.bdate_range("2018-01-01", periods=n))
        regime = compute_regime(bench, vol_window=20, trend_window=60, vol_hi=0.022)
        counts = regime.value_counts()
        for r in REGIMES:
            assert r in counts.index and counts[r] > 0, f"'{r}' rejimi tespit edilmedi"


class TestAttachRegimeToIndex:
    """attach_regime_to_index(idx, regime) → idx + 'regime' kolonu"""

    def test_regime_column_added(self):
        """Çıktıda 'regime' kolonu bulunmalı."""
        idx = pd.DataFrame(
            {"val": [1, 2, 3]},
            index=pd.MultiIndex.from_tuples(
                [("T1", pd.Timestamp("2020-01-02")),
                 ("T1", pd.Timestamp("2020-01-03")),
                 ("T2", pd.Timestamp("2020-01-02"))],
                names=["Ticker", "Date"],
            ),
        )
        regime = pd.Series(
            {"2020-01-02": "bull", "2020-01-03": "bear"},
        )
        regime.index = pd.to_datetime(regime.index)
        result = attach_regime_to_index(idx, regime)
        assert "regime" in result.columns

    def test_unknown_date_defaults_to_chop(self):
        """Bilinmeyen tarih → 'chop' varsayılan."""
        idx = pd.DataFrame(
            {"val": [1]},
            index=pd.MultiIndex.from_tuples(
                [("T1", pd.Timestamp("2025-01-01"))],
                names=["Ticker", "Date"],
            ),
        )
        regime = pd.Series({"2020-01-02": "bull"})
        regime.index = pd.to_datetime(regime.index)
        result = attach_regime_to_index(idx, regime)
        assert result["regime"].iloc[0] == "chop"


class TestRegimeBreakdown:
    """regime_breakdown(signal, target, regime) → {"bull":ric, "chop":ric, "bear":ric}"""

    def _make_signal_target(self, n_dates: int = 100, n_tickers: int = 10, seed: int = 0):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2020-01-01", periods=n_dates)
        tickers = [f"T{i}" for i in range(n_tickers)]
        idx_tuples = [(t, d) for d in dates for t in tickers]
        mi = pd.MultiIndex.from_tuples(idx_tuples, names=["Ticker", "Date"])
        signal = pd.Series(rng.normal(0, 1, len(mi)), index=mi)
        target = pd.Series(rng.normal(0, 0.01, len(mi)), index=mi)
        return signal, target

    def test_returns_three_regimes(self):
        """Çıktı dict 3 anahtar içermeli: bull, chop, bear."""
        signal, target = self._make_signal_target()
        dates = pd.bdate_range("2020-01-01", periods=100)
        regime = pd.Series(["bull"] * 40 + ["chop"] * 30 + ["bear"] * 30, index=dates)
        result = regime_breakdown(signal, target, regime)
        for r in REGIMES:
            assert r in result, f"'{r}' anahtarı eksik"

    def test_empty_regime_returns_nan(self):
        """Rejim verisinde hiç 'bear' yoksa bear IC = NaN döner."""
        signal, target = self._make_signal_target(n_dates=50)
        dates = pd.bdate_range("2020-01-01", periods=50)
        regime = pd.Series(["bull"] * 25 + ["chop"] * 25, index=dates)
        result = regime_breakdown(signal, target, regime)
        assert np.isnan(result["bear"]), "Veri olmayan rejim NaN döndürmeli"


class TestWfFitnessRegimeIntegration:
    """compute_wf_fitness regime= argümanıyla 'regime_breakdown' döndürmeli."""

    def test_regime_breakdown_in_result(self):
        """
        regime= verildiğinde sonuç dict'inde 'regime_breakdown' anahtarı bulunmalı
        ve 3 rejim içermeli.
        """
        from tests.conftest import make_synthetic_idx
        from engine.core.alpha_cfg import AlphaCFG
        from engine.validation.wf_fitness import compute_wf_fitness, make_date_folds
        from engine.core.formula_parser import parse_formula

        cfg = AlphaCFG()
        idx = make_synthetic_idx(n_tickers=20, n_days=300, seed=55)
        dates_arr = idx.index.get_level_values("Date").values
        folds = make_date_folds(dates_arr, n_folds=4, embargo_days=3)
        if len(folds) < 3:
            pytest.skip("Yeterli fold yok")

        # Basit bir rejim serisi: tüm tarihlere chop
        unique_dates = pd.to_datetime(np.unique(dates_arr))
        regime = pd.Series("chop", index=unique_dates)

        tree = parse_formula("Delta(Pclose, 5)", cfg)
        result = compute_wf_fitness(
            tree, cfg.evaluate, idx, folds,
            size_corr_hard_limit=0.95, neutralize=False,
            regime=regime,
        )
        assert "regime_breakdown" in result, "regime_breakdown anahtarı eksik"
        if result["regime_breakdown"] is not None:
            for r in REGIMES:
                assert r in result["regime_breakdown"], f"'{r}' anahtarı eksik"

    def test_no_regime_none_breakdown(self):
        """regime=None verildiğinde regime_breakdown=None döner."""
        from tests.conftest import make_synthetic_idx
        from engine.core.alpha_cfg import AlphaCFG
        from engine.validation.wf_fitness import compute_wf_fitness, make_date_folds
        from engine.core.formula_parser import parse_formula

        cfg = AlphaCFG()
        idx = make_synthetic_idx(n_tickers=20, n_days=250, seed=66)
        dates_arr = idx.index.get_level_values("Date").values
        folds = make_date_folds(dates_arr, n_folds=4, embargo_days=3)
        if len(folds) < 3:
            pytest.skip("Yeterli fold yok")

        tree = parse_formula("Delta(Pclose, 5)", cfg)
        result = compute_wf_fitness(
            tree, cfg.evaluate, idx, folds,
            size_corr_hard_limit=0.95, neutralize=False,
            regime=None,
        )
        assert result.get("regime_breakdown") is None
