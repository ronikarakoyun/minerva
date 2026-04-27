"""
tests/test_factor_neutralize.py — engine/factor_neutralize.py birim testleri.

Belgelenmiş gereksinimler (DOKUMAN_ONERILER.md §11.1):
  - test_neutralize_removes_size : Pclose sinyali → size_corr ≈ 1.0; nötralize → < 0.2
  - test_rank_norm_idempotent    : _rank_norm(_rank_norm(x)) ≈ _rank_norm(x)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from engine.data.factor_neutralize import (
    _rank_norm,
    _bin_demean,
    _build_factors,
    neutralize_signal,
    compute_size_corr,
    build_factors_cache,
)
from tests.conftest import make_synthetic_idx


# ─── _rank_norm testleri ──────────────────────────────────────────────────────

class TestRankNorm:

    def test_range_is_minus_half_to_half(self):
        """|_rank_norm çıktısı [-0.5, 0.5] aralığında olmalı."""
        rng = np.random.default_rng(0)
        for _ in range(50):
            x = rng.standard_normal(rng.integers(5, 100))
            r = _rank_norm(x)
            assert r.min() >= -0.5 - 1e-12, f"min={r.min()} < -0.5"
            assert r.max() <=  0.5 + 1e-12, f"max={r.max()} > +0.5"

    def test_idempotent(self):
        """
        _rank_norm(_rank_norm(x)) == _rank_norm(x)

        Rank-normalize işlemi monoton dönüşüm olduğundan tekrar uygulandığında
        sıralama değişmez → aynı sonuç.
        """
        rng = np.random.default_rng(1)
        for _ in range(30):
            x  = rng.standard_normal(rng.integers(5, 60))
            r1 = _rank_norm(x)
            r2 = _rank_norm(r1)
            np.testing.assert_allclose(
                r1, r2, atol=1e-10,
                err_msg=f"_rank_norm idempotency failed for x={x}"
            )

    def test_known_example(self):
        """Elle hesaplanmış örnek: [1,5,3,8,2] → [-0.5, 0.25, 0.0, 0.5, -0.25]."""
        x        = np.array([1.0, 5.0, 3.0, 8.0, 2.0])
        expected = np.array([-0.5, 0.25, 0.0, 0.5, -0.25])
        np.testing.assert_allclose(_rank_norm(x), expected, atol=1e-10)

    def test_monotone_preserving(self):
        """Rank sıralaması monoton: büyük değer → büyük rank-normalized değer."""
        x = np.array([10.0, 1.0, 5.0, 3.0, 8.0])
        r = _rank_norm(x)
        # r'nın sıralaması x'inkiyle aynı olmalı
        assert list(np.argsort(x)) == list(np.argsort(r))

    def test_single_element(self):
        """n=1: 0 döndürmeli (mean subtraction)."""
        r = _rank_norm(np.array([42.0]))
        assert r[0] == 0.0

    def test_two_elements(self):
        """n=2: [-0.5, 0.5] döndürmeli."""
        r = _rank_norm(np.array([3.0, 7.0]))
        np.testing.assert_allclose(r, [-0.5, 0.5], atol=1e-10)

    def test_ties_are_handled(self):
        """Eşit değerler (ties) hata vermemeli."""
        x = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        r = _rank_norm(x)
        assert not np.any(np.isnan(r))
        assert len(r) == 5


# ─── _bin_demean testleri ─────────────────────────────────────────────────────

class TestBinDemean:

    def test_each_bin_mean_near_zero(self):
        """
        Bin-demean sonrası her bin'in ortalaması ≈ 0 olmalı.
        Stage 2 nötralizasyonun temel özelliği.
        """
        rng = np.random.default_rng(42)
        n   = 300
        signal   = rng.standard_normal(n)
        size_arr = rng.standard_normal(n)
        size_rank = _rank_norm(size_arr)

        result = _bin_demean(signal, size_rank, n_bins=10)

        # Her bin içinde ortalama 0 olmalı
        bin_labels = pd.qcut(
            pd.Series(size_rank), q=10, labels=False, duplicates="drop"
        ).values
        for b in np.unique(bin_labels[~pd.isna(bin_labels)]):
            mask     = bin_labels == b
            bin_mean = result[mask].mean()
            assert abs(bin_mean) < 1e-10, f"bin {b}: mean={bin_mean:.2e} ≠ 0"

    def test_overall_mean_near_zero(self):
        """Toplam ortalama da ≈ 0 olmalı (tüm binlerin demean'i kümülatif 0 verir)."""
        rng  = np.random.default_rng(7)
        sig  = rng.standard_normal(200)
        sz   = _rank_norm(rng.standard_normal(200))
        res  = _bin_demean(sig, sz, n_bins=10)
        # Her bin kendi içinde demean → global mean da ~ 0 (eşit bin boyutunda)
        assert abs(res.mean()) < 0.05   # küçük tolerans (eşitsiz bin boyutları)

    def test_output_same_length(self):
        """Çıktı giriş uzunluğunda olmalı."""
        n   = 150
        sig = np.random.randn(n)
        sz  = _rank_norm(np.random.randn(n))
        res = _bin_demean(sig, sz, n_bins=5)
        assert len(res) == n

    def test_few_bins_fallback(self):
        """Az veri (n<10) olduğunda global demean ile fallback yapmalı."""
        sig = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        sz  = _rank_norm(np.array([5.0, 4.0, 3.0, 2.0, 1.0]))
        res = _bin_demean(sig, sz, n_bins=10)
        # Crash olmamalı, çıktı var
        assert len(res) == 5
        assert not np.any(np.isnan(res))


# ─── _build_factors testleri ──────────────────────────────────────────────────

class TestBuildFactors:

    def test_columns_present(self, syn_idx):
        """Dönen DataFrame'de size, vol, mom sütunları olmalı."""
        factors = _build_factors(syn_idx)
        assert "size" in factors.columns
        assert "vol"  in factors.columns
        assert "mom"  in factors.columns

    def test_index_matches(self, syn_idx):
        """Faktör matrisinin index'i syn_idx ile aynı olmalı."""
        factors = _build_factors(syn_idx)
        assert factors.index.names == ["Ticker", "Date"]
        # En az %90'ı ortak olmalı
        overlap = len(factors.index.intersection(syn_idx.index))
        assert overlap / len(syn_idx) > 0.9

    def test_size_is_log_price(self, syn_idx):
        """size = log(Pclose) olmalı — negatif infinity içermemeli."""
        factors = _build_factors(syn_idx)
        size    = factors["size"].dropna()
        assert len(size) > 0
        assert not np.any(np.isinf(size))
        # log(Pclose) ~ log(fiyat) — tüm fiyatlar > 0 ise sonlu olmalı
        assert size.isna().mean() < 0.05   # %95'i dolu

    def test_vol_nonnegative(self, syn_idx):
        """Volatilite ≥ 0 olmalı."""
        factors = _build_factors(syn_idx)
        vol = factors["vol"].dropna()
        assert (vol >= 0).all(), "Negatif volatilite değeri var"

    def test_no_pclose_column(self, syn_idx):
        """Pclose olmayan idx → size/vol/mom NaN olmalı, hata vermemeli."""
        idx_no_price = syn_idx[["Next_Ret"]].copy()
        factors = _build_factors(idx_no_price)
        assert factors["size"].isna().all()


# ─── compute_size_corr testleri ───────────────────────────────────────────────

class TestComputeSizeCorr:

    def test_pure_size_signal_near_one(self, syn_idx):
        """
        Signal = log(Pclose) = size faktörü.
        Spearman korelasyonu ≈ 1.0 olmalı (aynı şey!).
        """
        signal = np.log(syn_idx["Pclose"].replace(0, np.nan).clip(lower=1e-6))
        corr   = compute_size_corr(signal, syn_idx)
        assert corr > 0.9, f"Saf size sinyali için corr={corr:.3f} < 0.9"

    def test_random_signal_near_zero(self, syn_idx):
        """Bağımsız rastgele sinyal için size_corr ≈ 0 (küçük) olmalı."""
        rng    = np.random.default_rng(99)
        signal = pd.Series(rng.standard_normal(len(syn_idx)), index=syn_idx.index)
        corr   = compute_size_corr(signal, syn_idx)
        # Bağımsız randomda ±0.1 içinde bekleniyor
        assert abs(corr) < 0.15, f"Rastgele sinyalde |corr|={abs(corr):.3f} > 0.15"

    def test_returns_float(self, syn_idx):
        """Dönüş değeri float olmalı."""
        signal = pd.Series(
            np.random.randn(len(syn_idx)), index=syn_idx.index
        )
        c = compute_size_corr(signal, syn_idx)
        assert isinstance(c, float)
        assert not np.isnan(c)


# ─── neutralize_signal testleri ───────────────────────────────────────────────

class TestNeutralizeSignal:

    def test_neutralize_removes_size(self, syn_idx):
        """
        §11.1 Kritik Test:
        Pclose sinyali → size_corr ≈ 1.0 (ham)
        Nötralize sonrası → size_corr < 0.2
        """
        # Ham sinyal = log(Pclose) = tam size faktörü
        signal_raw = np.log(
            syn_idx["Pclose"].replace(0, np.nan).clip(lower=1e-6)
        )

        # Nötralizasyon öncesi size_corr yüksek olmalı
        corr_before = compute_size_corr(signal_raw, syn_idx)
        assert corr_before > 0.9, (
            f"Test data sorunu: ham size_corr={corr_before:.3f} < 0.9"
        )

        # İki-aşamalı nötralizasyon
        signal_neut = neutralize_signal(signal_raw, syn_idx, two_stage=True)
        signal_neut = signal_neut.dropna()

        # Nötralizasyon sonrası size_corr düşmeli
        corr_after = compute_size_corr(signal_neut, syn_idx)
        assert corr_after < 0.2, (
            f"Nötralizasyon yetersiz: size_corr={corr_after:.3f} ≥ 0.2  "
            f"(öncesi: {corr_before:.3f})"
        )

    def test_neutralize_stage1_reduces_size_corr(self, syn_idx):
        """
        Sadece Stage 1 (OLS) bile size_corr'u belirgin şekilde düşürmeli.
        """
        signal_raw  = np.log(syn_idx["Pclose"].replace(0, np.nan).clip(lower=1e-6))
        corr_before = compute_size_corr(signal_raw, syn_idx)

        signal_s1   = neutralize_signal(signal_raw, syn_idx, two_stage=False)
        corr_s1     = compute_size_corr(signal_s1.dropna(), syn_idx)

        assert corr_s1 < corr_before * 0.5, (
            f"Stage 1 size_corr yeterince düşürmedi: "
            f"before={corr_before:.3f}, after_s1={corr_s1:.3f}"
        )

    def test_two_stage_better_than_one_stage(self, syn_idx):
        """
        İki-aşamalı (Stage 1 + 2), tek aşamalıdan daha iyi olmalı.
        (stage2_corr ≤ stage1_corr)
        """
        signal_raw = np.log(syn_idx["Pclose"].replace(0, np.nan).clip(lower=1e-6))

        s1 = neutralize_signal(signal_raw, syn_idx, two_stage=False).dropna()
        s2 = neutralize_signal(signal_raw, syn_idx, two_stage=True).dropna()

        corr_s1 = compute_size_corr(s1, syn_idx)
        corr_s2 = compute_size_corr(s2, syn_idx)

        assert corr_s2 <= corr_s1 + 0.03, (
            f"Stage 2 daha kötü: s1={corr_s1:.3f}, s2={corr_s2:.3f}"
        )

    def test_neutralize_preserves_index(self, syn_idx):
        """Nötralizasyon sonrası index değişmemeli."""
        signal = pd.Series(np.random.randn(len(syn_idx)), index=syn_idx.index)
        result = neutralize_signal(signal, syn_idx)
        # Ortak index %90+ korunmalı (NaN'lar düşürülmüş olabilir)
        overlap = len(result.index.intersection(signal.index))
        assert overlap / len(signal) > 0.85

    def test_neutralize_no_crash_on_random_signal(self, syn_idx):
        """Rastgele sinyalde hata vermemeli."""
        rng    = np.random.default_rng(55)
        signal = pd.Series(rng.standard_normal(len(syn_idx)), index=syn_idx.index)
        result = neutralize_signal(signal, syn_idx)
        assert result is not None
        assert len(result) > 0


# ─── build_factors_cache testleri ────────────────────────────────────────────

class TestBuildFactorsCache:

    def test_precomputed_rank_columns_exist(self, syn_idx):
        """
        build_factors_cache, {factor}_rank kolonlarını içermeli (8.3 optimizasyonu).
        """
        cache = build_factors_cache(syn_idx)
        for col in ["size", "vol", "mom"]:
            rank_col = f"{col}_rank"
            assert rank_col in cache.columns, f"'{rank_col}' kolonu eksik"

    def test_rank_columns_in_range(self, syn_idx):
        """Pre-computed rank kolonları [-0.5, 0.5] aralığında olmalı."""
        cache = build_factors_cache(syn_idx)
        for col in ["size_rank", "vol_rank", "mom_rank"]:
            vals = cache[col].dropna().values
            assert vals.min() >= -0.5 - 1e-6, f"{col}: min={vals.min()}"
            assert vals.max() <=  0.5 + 1e-6, f"{col}: max={vals.max()}"

    def test_cache_faster_than_rebuild(self, syn_idx):
        """
        Pre-built cache, neutralize_signal içinde kullanıldığında
        cache olmadan kullanımdan daha hızlı (ya da en azından yavaş değil) olmalı.

        Bu test kesin bir süre ölçümü yapmak yerine sonuçların tutarlılığını kontrol eder.
        """
        import time
        signal = pd.Series(np.random.randn(len(syn_idx)), index=syn_idx.index)

        # Cache ile
        cache   = build_factors_cache(syn_idx)
        t0      = time.time()
        r_cache = neutralize_signal(signal, syn_idx, factors=cache)
        t_cache = time.time() - t0

        # Cache olmadan (factors=None → yeniden hesaplanır)
        t1       = time.time()
        r_nocache = neutralize_signal(signal, syn_idx, factors=None)
        t_nocache = time.time() - t1

        # Sonuçlar aynı (veya çok yakın) olmalı
        common = r_cache.index.intersection(r_nocache.index)
        if len(common) > 10:
            # Aynı hesaplamayı yapıyorlar, büyük sapma olmamalı
            diff = (r_cache[common] - r_nocache[common]).abs().median()
            assert diff < 0.01, f"Cache ve no-cache sonuçları farklı: median_diff={diff:.4f}"
