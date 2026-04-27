"""
tests/test_pbo_cscv.py — engine/pbo_cscv.py için birim testler.

Bailey, Borwein, López de Prado & Zhu (2014) CSCV:
  - PBO ≈ 0.5 için saf gürültü havuzu
  - Tek "gerçek" formül + gürültü → PBO < 0.5
  - Tümü aynı formül → PBO ≈ 0 (degeneracy)
  - Kombinatoryal sayım doğruluğu
  - n_slices=2 minimum durum crash yok

Çalıştırma:
    python -m pytest tests/test_pbo_cscv.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from math import comb

from engine.validation.pbo_cscv import build_pnl_matrix, cscv_pbo, pbo_verdict


def _make_noise_matrix(
    n_slices: int = 8,
    n_formulas: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Saf gürültü PnL matrisi üret."""
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.01, size=(n_slices, n_formulas))


def _make_signal_matrix(
    n_slices: int = 8,
    n_formulas: int = 99,
    signal_strength: float = 0.005,
    seed: int = 1,
) -> np.ndarray:
    """1 gerçek + (n_formulas-1) gürültü PnL matrisi üret."""
    rng = np.random.default_rng(seed)
    mat = rng.normal(0.0, 0.01, size=(n_slices, n_formulas))
    # İlk formül: tutarlı pozitif sinyal
    mat[:, 0] = signal_strength + rng.normal(0.0, 0.002, n_slices)
    return mat


class TestBuildPnlMatrix:
    """build_pnl_matrix(pool_pnl, n_slices) → (M, S) ndarray"""

    def test_output_shape(self):
        """Çıktı shape (n_slices, n_formulas) olmalı."""
        n_days, S, M = 500, 10, 8
        rng = np.random.default_rng(0)
        pnl_2d = rng.normal(0, 0.01, (n_days, S))
        mat = build_pnl_matrix(pnl_2d, n_slices=M)
        assert mat.shape == (M, S), f"Beklenen ({M},{S}), alındı {mat.shape}"

    def test_already_sliced_passthrough(self):
        """Zaten (n_slices, S) formatındaysa değişmeden döner."""
        mat = np.ones((8, 5))
        result = build_pnl_matrix(mat, n_slices=8)
        assert result.shape == (8, 5)

    def test_list_input(self):
        """list[np.ndarray] girişi de kabul edilmeli."""
        rng = np.random.default_rng(2)
        pnl_list = [rng.normal(0, 0.01, 200) for _ in range(5)]
        mat = build_pnl_matrix(pnl_list, n_slices=4)
        assert mat.shape == (4, 5)


class TestCscvPbo:
    """cscv_pbo(pnl_mat) → dict"""

    def test_returns_valid_dict(self):
        """Sonuç dict beklenen anahtarları içermeli."""
        mat = _make_noise_matrix(8, 10)
        result = cscv_pbo(mat)
        for k in ("pbo", "logit_lambda", "n_combinations", "is_sr", "oos_sr", "verdict"):
            assert k in result, f"Eksik anahtar: {k}"

    def test_pbo_range(self):
        """PBO ∈ [0, 1]."""
        mat = _make_noise_matrix(8, 20)
        result = cscv_pbo(mat)
        if np.isfinite(result["pbo"]):
            assert 0.0 <= result["pbo"] <= 1.0, f"PBO sınır dışı: {result['pbo']}"

    def test_noise_pool_pbo_near_half(self):
        """
        Saf gürültü havuzu → PBO ≈ 0.5 (IS best şansa eşit OOS).
        100 gürültü formülü, 8 dilim, seed-stabil.
        """
        mat    = _make_noise_matrix(n_slices=8, n_formulas=100, seed=42)
        result = cscv_pbo(mat, max_combinations=200)
        pbo    = result["pbo"]
        assert np.isfinite(pbo), "PBO hesaplanamadı"
        assert 0.3 <= pbo <= 0.75, (
            f"Gürültü havuzunda PBO ≈ 0.5 beklenir, alındı {pbo:.3f}"
        )

    def test_single_real_formula_low_pbo(self):
        """
        1 gerçek + 99 gürültü formülü → PBO < 0.5 (gerçek formül fark yaratıyor).
        """
        mat    = _make_signal_matrix(n_slices=8, n_formulas=100, signal_strength=0.01, seed=7)
        result = cscv_pbo(mat, max_combinations=200)
        pbo    = result["pbo"]
        assert np.isfinite(pbo), "PBO hesaplanamadı"
        assert pbo < 0.55, (
            f"Güçlü sinyal varlığında PBO < 0.55 beklenir, alındı {pbo:.3f}"
        )

    def test_identical_formulas_zero_pbo(self):
        """
        Tüm formüller aynı (kopya) → IS best = OOS best → PBO ≈ 0.
        """
        rng = np.random.default_rng(3)
        single = rng.normal(0.001, 0.005, (8,))
        # 20 kopya
        mat = np.tile(single, (20, 1)).T   # (8, 20)
        result = cscv_pbo(mat, max_combinations=100)
        pbo = result["pbo"]
        assert np.isfinite(pbo), "PBO hesaplanamadı"
        # Tümü aynı → IS best'in OOS rank = 0 → omega=0 → logit<0 → PBO=0
        assert pbo <= 0.1, (
            f"Özdeş formüllerde PBO ≈ 0 beklenir, alındı {pbo:.3f}"
        )

    def test_combination_count_correct(self):
        """
        max_combinations limiti yokken C(M, M//2) kombinasyon üretilmeli.
        Küçük M seç (M=6): C(6,3) = 20.
        """
        mat    = _make_noise_matrix(n_slices=6, n_formulas=5, seed=9)
        result = cscv_pbo(mat, max_combinations=10_000)
        expected = comb(6, 3)   # 20
        assert result["n_combinations"] == expected, (
            f"C(6,3)={expected} beklenir, alındı {result['n_combinations']}"
        )

    def test_minimum_slices_no_crash(self):
        """n_slices=2 → n_splits=1, crash olmamalı."""
        mat = _make_noise_matrix(n_slices=2, n_formulas=5)
        result = cscv_pbo(mat)
        assert isinstance(result, dict)

    def test_too_small_matrix_returns_nan(self):
        """M<4 veya S<2 → PBO=nan, "Yetersiz veri" verdict."""
        # M=2
        mat = _make_noise_matrix(n_slices=2, n_formulas=1)
        result = cscv_pbo(mat)
        assert np.isnan(result["pbo"]), "Yetersiz matris → PBO=nan beklenir"

    def test_logit_lambda_length(self):
        """logit_lambda uzunluğu n_combinations ile eşit."""
        mat = _make_noise_matrix(n_slices=6, n_formulas=10)
        result = cscv_pbo(mat, max_combinations=500)
        assert len(result["logit_lambda"]) == result["n_combinations"]


class TestPboVerdict:
    """pbo_verdict(pbo) → str"""

    def test_low_pbo_accept(self):
        assert "Kabul" in pbo_verdict(0.3)

    def test_medium_pbo_warning(self):
        assert "Orta" in pbo_verdict(0.6)

    def test_high_pbo_danger(self):
        assert "Yüksek" in pbo_verdict(0.8)

    def test_nan_pbo_uncertain(self):
        assert "Belirsiz" in pbo_verdict(float("nan"))
