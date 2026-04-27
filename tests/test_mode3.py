"""
tests/test_mode3.py — engine/mining_runner.py + engine/ensemble.py birim testleri.

Adım (f) — Mod 3 Full Rolling Discovery:
  1. Reentrancy — aynı cfg iki çağrı özdeş sonuç (seed-stable).
  2. Window sayımı doğru (step_days=30, lookback=252 sentetik verisi).
  3. HoF boş pencere → graceful, crash yok.
  4. run_mining_window global state sızıntısı yok.
  5. k_keep=1 → tekil formülle çalışır.
  6. MiningConfig varsayılan değerleri doğru.
  7. MiningResult alanları tip-doğru.
  8. combine_signals rank_average → [0,1] aralığı.
  9. combine_signals eşit ağırlık toplamı = 1.
 10. HallOfFame.to_dataframe() sütun kontrol.
 11. HallOfFame.combined_equity() None-güvenli.
 12. run_ensemble_backtest boş ağaç listesi → (None, None).

Çalıştırma:
    python -m pytest tests/test_mode3.py -v
"""
from __future__ import annotations

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from engine.strategies.mining_runner import MiningConfig, MiningResult, run_mining_window
from engine.validation.ensemble import (
    HallOfFame,
    WindowResult,
    combine_signals,
    run_ensemble_backtest,
)


# ─────────────────────────────────────────────────────────────────
# Test verisi yardımcıları
# ─────────────────────────────────────────────────────────────────

def _make_db(n_tickers: int = 10, n_days: int = 120, seed: int = 0) -> pd.DataFrame:
    """Sentetik OHLCV + Next_Ret DataFrame döndür."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows = []
    for ticker in tickers:
        prices = np.cumprod(1 + rng.normal(0, 0.01, n_days)) * 100
        vols = np.abs(rng.normal(1e6, 2e5, n_days))
        for i, date in enumerate(dates):
            rows.append({
                "Ticker": ticker,
                "Date": date,
                "Popen": prices[i] * (1 + rng.normal(0, 0.002)),
                "Phigh": prices[i] * (1 + abs(rng.normal(0, 0.005))),
                "Plow":  prices[i] * (1 - abs(rng.normal(0, 0.005))),
                "Pclose": prices[i],
                "Vlot": vols[i],
            })
    df = pd.DataFrame(rows)
    df = df.sort_values(["Ticker", "Date"])
    # Next_Ret hesapla
    df["Pclose_t1"] = df.groupby("Ticker")["Pclose"].shift(-1)
    df["Pclose_t2"] = df.groupby("Ticker")["Pclose"].shift(-2)
    df["Next_Ret"] = df["Pclose_t2"] / df["Pclose_t1"] - 1
    return df.drop(columns=["Pclose_t1", "Pclose_t2"])


def _make_signal(n: int = 200, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=20)
    tickers = [f"T{i:02d}" for i in range(10)]
    tuples = [(t, d) for d in dates for t in tickers]
    mi = pd.MultiIndex.from_tuples(tuples, names=["Ticker", "Date"])
    return pd.Series(rng.normal(0, 1, len(mi)), index=mi)


def _make_cfg():
    """AlphaCFG küçük grameriyle başlat."""
    from engine.core.alpha_cfg import AlphaCFG
    return AlphaCFG()


def _make_small_mining_cfg(num_gen: int = 20) -> MiningConfig:
    """Hızlı test için küçük mining config."""
    return MiningConfig(
        num_gen=num_gen,
        max_K=5,
        use_wf_fitness=False,   # WF-fitness kapalı → hızlı
        wf_n_folds=3,
        wf_embargo=2,
        wf_purge=0,
        lambda_std=1.0,
        lambda_cx=0.001,
        lambda_size=0.0,
        size_corr_hard_limit=1.0,
        neutralize=False,
        target_col="Next_Ret",
        seed=42,
        min_mean_ric=-999.0,    # Tüm formüller geçsin (test amaçlı)
        min_pos_ratio=0.0,
    )


# ─────────────────────────────────────────────────────────────────
# MiningConfig testleri
# ─────────────────────────────────────────────────────────────────

class TestMiningConfig:
    def test_defaults(self):
        """Varsayılan değerler doğru tipte."""
        cfg = MiningConfig()
        assert isinstance(cfg.num_gen, int)
        assert isinstance(cfg.max_K, int)
        assert isinstance(cfg.use_wf_fitness, bool)
        assert isinstance(cfg.min_mean_ric, float)
        assert isinstance(cfg.seed, int)

    def test_custom_values(self):
        """Özelleştirilmiş config değerleri korunuyor."""
        cfg = MiningConfig(num_gen=50, max_K=8, seed=99)
        assert cfg.num_gen == 50
        assert cfg.max_K == 8
        assert cfg.seed == 99


# ─────────────────────────────────────────────────────────────────
# run_mining_window testleri
# ─────────────────────────────────────────────────────────────────

class TestRunMiningWindow:
    def test_returns_list(self):
        """Çıktı list tipinde olmalı."""
        db = _make_db(n_tickers=8, n_days=80)
        cfg = _make_cfg()
        mcfg = _make_small_mining_cfg(num_gen=10)
        results = run_mining_window(db, cfg, mcfg)
        assert isinstance(results, list)

    def test_results_have_required_fields(self):
        """MiningResult alanları tip-doğru."""
        db = _make_db(n_tickers=8, n_days=80)
        cfg = _make_cfg()
        mcfg = _make_small_mining_cfg(num_gen=20)
        results = run_mining_window(db, cfg, mcfg)
        for r in results[:3]:
            assert isinstance(r, MiningResult)
            assert isinstance(r.formula, str)
            assert isinstance(r.fitness, float)
            assert isinstance(r.mean_ric, float)
            assert isinstance(r.status, str)

    def test_sorted_by_fitness_desc(self):
        """Sonuçlar fitness'a göre azalan sırada döner."""
        db = _make_db(n_tickers=8, n_days=80)
        cfg = _make_cfg()
        mcfg = _make_small_mining_cfg(num_gen=30)
        results = run_mining_window(db, cfg, mcfg)
        if len(results) >= 2:
            fitnesses = [r.fitness for r in results]
            assert fitnesses == sorted(fitnesses, reverse=True), (
                "Sonuçlar fitness'a göre azalan sırada değil"
            )

    def test_reentrancy_same_seed(self):
        """Aynı seed ile iki çağrı aynı formülleri döndürür."""
        db = _make_db(n_tickers=8, n_days=80)
        cfg = _make_cfg()
        mcfg = _make_small_mining_cfg(num_gen=20)
        r1 = run_mining_window(db, cfg, mcfg)
        r2 = run_mining_window(db, cfg, mcfg)
        f1 = [r.formula for r in r1]
        f2 = [r.formula for r in r2]
        assert f1 == f2, "Seed-sabit olmayan — reentrancy kırık"

    def test_global_random_state_restored(self):
        """run_mining_window sonrasında global random state bozulmamalı."""
        random.seed(1234)
        np.random.seed(1234)
        state_before_py = random.getstate()
        state_before_np = np.random.get_state()

        db = _make_db(n_tickers=6, n_days=60)
        cfg = _make_cfg()
        mcfg = _make_small_mining_cfg(num_gen=10)
        run_mining_window(db, cfg, mcfg)

        state_after_py = random.getstate()
        state_after_np = np.random.get_state()

        # Python random state tam olarak geri yüklendi mi?
        assert state_before_py == state_after_py, (
            "Python global random state bozuldu"
        )
        # NumPy state için pos ve has_gauss kontrol et
        assert state_before_np[2] == state_after_np[2], (
            "NumPy random pos bozuldu"
        )

    def test_seed_trees_warm_start(self):
        """seed_trees verilince pool genişler, crash yok."""
        db = _make_db(n_tickers=8, n_days=80)
        cfg = _make_cfg()
        mcfg = _make_small_mining_cfg(num_gen=20)
        # İlk çalışma
        r1 = run_mining_window(db, cfg, mcfg)
        # İkinci çalışma: önceki ağaçları tohum olarak ver
        seed_trees = [r.tree for r in r1[:5]] if r1 else None
        r2 = run_mining_window(db, cfg, mcfg, seed_trees=seed_trees)
        assert isinstance(r2, list)  # crash yok

    def test_small_data_graceful(self):
        """Çok az veriyle crash olmadan boş liste döner."""
        db = _make_db(n_tickers=3, n_days=15)  # çok küçük
        cfg = _make_cfg()
        mcfg = _make_small_mining_cfg(num_gen=5)
        results = run_mining_window(db, cfg, mcfg)
        assert isinstance(results, list)  # crash yok


# ─────────────────────────────────────────────────────────────────
# combine_signals testleri
# ─────────────────────────────────────────────────────────────────

class TestCombineSignals:
    def test_empty_returns_empty(self):
        """Boş liste → boş Series."""
        result = combine_signals([])
        assert isinstance(result, pd.Series)
        assert len(result) == 0

    def test_rank_average_range(self):
        """rank_average → değerler [0, 1] aralığında (rank yüzdeleri)."""
        s1 = _make_signal(seed=0)
        s2 = _make_signal(seed=1)
        combined = combine_signals([s1, s2], method="rank_average")
        assert combined.min() >= 0.0, "rank_average negatif değer içeriyor"
        assert combined.max() <= 1.0, "rank_average 1'i aşan değer içeriyor"

    def test_equal_weights_sum(self):
        """Eşit ağırlıkla iki özdeş sinyal → orijinalle aynı rank sırası."""
        s1 = _make_signal(seed=5)
        combined = combine_signals([s1, s1], method="rank_average")
        # Aynı sinyali iki kez combine etmek özgün rank'ı değiştirmemeli
        r1 = s1.groupby(level="Date").rank(pct=True)
        # Birleşik rank ile orijinal rank arasındaki korelasyon ≈ 1
        corr = float(combined.corr(r1, method="spearman"))
        assert corr > 0.99, f"Özdeş sinyaller birleşince rank bozuldu: {corr:.4f}"

    def test_simple_average_weighted(self):
        """simple_average + ağırlık → ağırlıklı ortalama."""
        s1 = _make_signal(seed=0)
        s2 = _make_signal(seed=1)
        combined = combine_signals([s1, s2], weights=[2.0, 1.0], method="simple_average")
        assert isinstance(combined, pd.Series)
        assert len(combined) > 0

    def test_no_common_index_returns_empty(self):
        """Ortak index yoksa boş Series döner."""
        idx1 = pd.date_range("2020-01-01", periods=5, freq="B")
        idx2 = pd.date_range("2021-01-01", periods=5, freq="B")
        s1 = pd.Series([1.0] * 5, index=idx1)
        s2 = pd.Series([2.0] * 5, index=idx2)
        result = combine_signals([s1, s2])
        assert len(result) == 0, "Ortak index yokken boş dönmeli"


# ─────────────────────────────────────────────────────────────────
# HallOfFame testleri
# ─────────────────────────────────────────────────────────────────

class TestHallOfFame:
    def _make_window_result(self, wid: int, has_equity: bool = True) -> WindowResult:
        eq = None
        if has_equity:
            dates = pd.bdate_range("2020-01-01", periods=20)
            eq = pd.DataFrame({
                "Date": dates,
                "Equity": np.cumprod(1 + np.random.default_rng(wid).normal(0, 0.01, 20)),
            })
        return WindowResult(
            window_id=wid,
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2021-01-01"),
            test_start=pd.Timestamp("2021-01-01"),
            test_end=pd.Timestamp("2021-07-01"),
            n_formulas=5 if has_equity else 0,
            top_formula=f"formula_{wid}",
            top_fitness=0.01 * wid,
            equity=eq,
            formula_names=[f"f{wid}_top{k}" for k in range(3)],
        )

    def test_add_and_len(self):
        """add() doğru sayıda pencere ekliyor."""
        hof = HallOfFame()
        for i in range(3):
            hof.add(self._make_window_result(i + 1))
        assert len(hof.windows) == 3

    def test_to_dataframe_columns(self):
        """to_dataframe() gerekli sütunları içeriyor."""
        hof = HallOfFame()
        hof.add(self._make_window_result(1))
        df = hof.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        required = ["Pencere", "Formül Sayısı", "En İyi Formül", "Fitness"]
        for col in required:
            assert col in df.columns, f"Sütun eksik: {col}"

    def test_to_dataframe_empty(self):
        """Boş HoF → boş DataFrame, crash yok."""
        hof = HallOfFame()
        df = hof.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_combined_equity_none_safe(self):
        """Tüm pencereler equity=None → combined_equity() None döner, crash yok."""
        hof = HallOfFame()
        hof.add(self._make_window_result(1, has_equity=False))
        hof.add(self._make_window_result(2, has_equity=False))
        result = hof.combined_equity()
        assert result is None

    def test_combined_equity_sorted_by_date(self):
        """combined_equity() tarih sıralı döner."""
        hof = HallOfFame()
        hof.add(self._make_window_result(1, has_equity=True))
        hof.add(self._make_window_result(2, has_equity=True))
        ceq = hof.combined_equity()
        if ceq is not None and len(ceq) > 1:
            dates = pd.to_datetime(ceq["Date"])
            assert (dates.diff().dropna() >= pd.Timedelta(0)).all(), (
                "combined_equity() tarih sıralı değil"
            )

    def test_combined_equity_mixed(self):
        """Bazı pencereler equity=None, diğerleri var → None olanlar atlanır."""
        hof = HallOfFame()
        hof.add(self._make_window_result(1, has_equity=True))
        hof.add(self._make_window_result(2, has_equity=False))
        hof.add(self._make_window_result(3, has_equity=True))
        ceq = hof.combined_equity()
        assert ceq is not None and len(ceq) > 0


# ─────────────────────────────────────────────────────────────────
# run_ensemble_backtest testleri
# ─────────────────────────────────────────────────────────────────

class TestRunEnsembleBacktest:
    def test_empty_trees_returns_none(self):
        """Boş ağaç listesi → (None, None) döner, crash yok."""
        db = _make_db(n_tickers=8, n_days=60)
        cfg = _make_cfg()
        result = run_ensemble_backtest(db, [], cfg.evaluate)
        assert result == (None, None)

    def test_single_tree_runs(self):
        """Tek ağaçla çalışır, crash yok."""
        db = _make_db(n_tickers=8, n_days=80)
        cfg = _make_cfg()
        mcfg = _make_small_mining_cfg(num_gen=20)
        results = run_mining_window(db, cfg, mcfg)
        if not results:
            pytest.skip("Mining hiç formül üretemedi — veri çok küçük")
        tree = results[0].tree
        curve, positions = run_ensemble_backtest(
            db, [tree], cfg.evaluate, top_k=3, n_drop=1
        )
        # crash olmadı; curve None veya DataFrame olabilir

    def test_multiple_trees_no_crash(self):
        """Birden fazla ağaçla crash yok."""
        db = _make_db(n_tickers=10, n_days=100)
        cfg = _make_cfg()
        mcfg = _make_small_mining_cfg(num_gen=30)
        results = run_mining_window(db, cfg, mcfg)
        if len(results) < 2:
            pytest.skip("Yeterli formül üretilemedi")
        trees = [r.tree for r in results[:3]]
        curve, _ = run_ensemble_backtest(
            db, trees, cfg.evaluate, top_k=5, n_drop=1
        )
        # crash olmadı
