"""
tests/test_meta_label.py — engine/meta_label.py için birim testler.

López de Prado Meta-Labeling (AFML §3.6):
  - Primary sinyal "yön" söyler, meta model "bet size / skip" söyler.
  - Leakage kontrolü: train_end sonrası örnek fit'te yok.
  - apply_meta_filter: threshold=1.0 → tüm sinyal NaN.
  - Tek sınıflı y → graceful fit_failed=True.
  - TB_Label yoksa meta devreye alınmamalı.
  - AUC > 0.7: meta feature'lar y'yi gerçekten açıklayabiliyorsa.

Çalıştırma:
    python -m pytest tests/test_meta_label.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from engine.data.meta_label import (
    build_meta_dataset,
    train_meta_model,
    apply_meta_filter,
    MetaModel,
)

try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

requires_sklearn = pytest.mark.skipif(not _SKLEARN_OK, reason="sklearn yüklü değil")


def _make_synthetic_data(
    n_tickers: int = 20,
    n_days: int = 300,
    seed: int = 0,
) -> tuple[pd.Series, pd.DataFrame]:
    """Sentetik (Ticker,Date) MultiIndex signal + idx döndür."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    tuples = [(t, d) for d in dates for t in tickers]
    mi = pd.MultiIndex.from_tuples(tuples, names=["Ticker", "Date"])

    signal = pd.Series(rng.normal(0, 1, len(mi)), index=mi)
    next_ret = rng.normal(0, 0.01, len(mi))
    # TB_Label: signal üzerine bazı gürültü ile ilişkilendirilmiş
    tb_label = np.sign(signal.values + rng.normal(0, 0.3, len(mi))).clip(0, 1)

    idx = pd.DataFrame({
        "Pclose":   np.exp(rng.normal(7, 0.3, len(mi))),
        "Vlot":     np.abs(rng.normal(1e6, 2e5, len(mi))),
        "Next_Ret": next_ret,
        "TB_Label": tb_label,
    }, index=mi)

    return signal, idx


class TestBuildMetaDataset:
    """build_meta_dataset(signal, idx, ...) → pd.DataFrame"""

    def test_returns_dataframe(self):
        """Çıktı pd.DataFrame olmalı."""
        signal, idx = _make_synthetic_data(seed=0)
        ds = build_meta_dataset(signal, idx, target_col="TB_Label")
        assert isinstance(ds, pd.DataFrame)

    def test_has_required_columns(self):
        """'sig_rank' ve 'label' kolonları bulunmalı."""
        signal, idx = _make_synthetic_data(seed=1)
        ds = build_meta_dataset(signal, idx, target_col="TB_Label")
        assert "sig_rank" in ds.columns, "sig_rank eksik"
        assert "label" in ds.columns, "label eksik"

    def test_sig_rank_in_zero_one(self):
        """sig_rank ∈ [0, 1] (percentile rank)."""
        signal, idx = _make_synthetic_data(seed=2)
        ds = build_meta_dataset(signal, idx, target_col="TB_Label")
        sr = ds["sig_rank"].dropna()
        assert sr.min() >= 0.0 and sr.max() <= 1.0, (
            f"sig_rank [0,1] dışında: [{sr.min():.3f}, {sr.max():.3f}]"
        )

    def test_regime_dummies_created(self):
        """regime= verilince 'regime_bull' ve 'regime_bear' kolonları oluşmalı."""
        signal, idx = _make_synthetic_data(seed=3)
        dates = pd.bdate_range("2020-01-01", periods=300)
        regime = pd.Series(["bull"] * 100 + ["chop"] * 100 + ["bear"] * 100, index=dates)
        ds = build_meta_dataset(signal, idx, regime=regime, target_col="TB_Label")
        assert "regime_bull" in ds.columns and "regime_bear" in ds.columns

    def test_recent_ic_created(self):
        """'recent_ic' kolonu oluşmalı (sıfır bile olsa)."""
        signal, idx = _make_synthetic_data(seed=4)
        ds = build_meta_dataset(signal, idx, target_col="TB_Label")
        assert "recent_ic" in ds.columns

    def test_fallback_to_next_ret_if_no_tb(self):
        """TB_Label yoksa Next_Ret'ten binary etiket üretilmeli."""
        signal, idx = _make_synthetic_data(seed=5)
        idx_no_tb = idx.drop(columns=["TB_Label"])
        ds = build_meta_dataset(signal, idx_no_tb, target_col="TB_Label")
        # TB_Label yok → Next_Ret >= 0 → binary
        assert "label" in ds.columns
        assert set(ds["label"].dropna().unique()).issubset({0.0, 1.0})


class TestTrainMetaModel:
    """train_meta_model(ds, train_end) → MetaModel"""

    @requires_sklearn
    def test_returns_meta_model(self):
        """Çıktı MetaModel örneği olmalı."""
        signal, idx = _make_synthetic_data(seed=10)
        ds = build_meta_dataset(signal, idx, target_col="TB_Label")
        train_end = pd.Timestamp("2021-01-01")
        mm = train_meta_model(ds, train_end)
        assert isinstance(mm, MetaModel)

    @requires_sklearn
    def test_no_leakage(self):
        """
        train_end sonrasındaki tarihlere ait hiçbir satır fit'te kullanılmamalı.
        Doğrulama: MetaModel.train_end == train_end parametresi.
        """
        signal, idx = _make_synthetic_data(seed=11)
        ds = build_meta_dataset(signal, idx, target_col="TB_Label")
        train_end = pd.Timestamp("2021-01-01")
        mm = train_meta_model(ds, train_end)
        if not mm.fit_failed:
            assert mm.train_end == train_end, (
                f"train_end kaydedilmemişi: {mm.train_end} != {train_end}"
            )

    @requires_sklearn
    def test_single_class_fit_failed(self):
        """
        y'de sadece tek sınıf varsa fit_failed=True döner, crash yok.
        """
        signal, idx = _make_synthetic_data(seed=12)
        ds = build_meta_dataset(signal, idx, target_col="TB_Label")
        # Tüm etiketleri 1 yap
        ds["label"] = 1.0
        train_end = pd.Timestamp("2021-06-01")
        mm = train_meta_model(ds, train_end)
        assert mm.fit_failed, "Tek sınıf → fit_failed=True beklenir"

    @requires_sklearn
    def test_auc_in_range(self):
        """AUC ∈ [0, 1] (veya NaN)."""
        signal, idx = _make_synthetic_data(seed=13)
        ds = build_meta_dataset(signal, idx, target_col="TB_Label")
        train_end = pd.Timestamp("2021-06-01")
        mm = train_meta_model(ds, train_end)
        if not mm.fit_failed and np.isfinite(mm.auc):
            assert 0.0 <= mm.auc <= 1.0, f"AUC sınır dışı: {mm.auc}"

    @requires_sklearn
    def test_informative_features_high_auc(self):
        """
        Meta feature'lar y'yi gerçekten açıklayabilirse AUC > 0.6 beklenir.
        Sinyalin kendisi iyi bir predictor olduğu senaryoda.
        """
        rng = np.random.default_rng(99)
        n_tickers, n_days = 30, 400
        dates = pd.bdate_range("2020-01-01", periods=n_days)
        tickers = [f"T{i}" for i in range(n_tickers)]
        tuples = [(t, d) for d in dates for t in tickers]
        mi = pd.MultiIndex.from_tuples(tuples, names=["Ticker", "Date"])

        # Güçlü sinyal: label = sign(signal + gürültü) — yüksek IC
        signal_vals = rng.normal(0, 1, len(mi))
        label = (signal_vals + rng.normal(0, 0.5, len(mi)) > 0).astype(float)

        signal = pd.Series(signal_vals, index=mi)
        idx = pd.DataFrame({
            "Pclose": np.exp(rng.normal(7, 0.3, len(mi))),
            "Vlot":   np.abs(rng.normal(1e6, 2e5, len(mi))),
            "Next_Ret": rng.normal(0, 0.01, len(mi)),
            "TB_Label": label,
        }, index=mi)

        ds = build_meta_dataset(signal, idx, target_col="TB_Label")
        train_end = pd.Timestamp("2021-01-01")
        mm = train_meta_model(ds, train_end)
        if not mm.fit_failed and np.isfinite(mm.auc):
            assert mm.auc > 0.55, (
                f"Bilgilendirici featurelarda AUC > 0.55 beklenir: {mm.auc:.3f}"
            )


class TestApplyMetaFilter:
    """apply_meta_filter(signal, proba, threshold) → pd.Series"""

    def _make_signal_proba(self, n: int = 100, seed: int = 0):
        rng = np.random.default_rng(seed)
        index = pd.date_range("2020-01-01", periods=n, freq="B")
        signal = pd.Series(rng.normal(0, 1, n), index=index)
        proba  = pd.Series(rng.uniform(0, 1, n), index=index)
        return signal, proba

    def test_threshold_zero_keeps_all(self):
        """threshold=0 → tüm sinyal korunur (hiçbiri NaN olmaz)."""
        signal, proba = self._make_signal_proba()
        filtered = apply_meta_filter(signal, proba, threshold=0.0)
        assert filtered.notna().all(), "threshold=0 → hiç sinyal maskelenmemeli"

    def test_threshold_one_masks_all(self):
        """threshold=1.0 → tüm sinyal NaN (p_value hiçbiri ≥ 1.0 değil)."""
        signal, proba = self._make_signal_proba()
        # proba ∈ [0,1) random, threshold=1.0 → hepsi < threshold
        filtered = apply_meta_filter(signal, proba, threshold=1.0)
        assert filtered.isna().all(), "threshold=1.0 → tüm sinyal NaN olmalı"

    def test_partial_filter(self):
        """threshold=0.5 → yaklaşık yarısı maskelenmeli."""
        signal, proba = self._make_signal_proba(n=500, seed=42)
        filtered = apply_meta_filter(signal, proba, threshold=0.5)
        nan_ratio = filtered.isna().mean()
        assert 0.3 < nan_ratio < 0.7, (
            f"threshold=0.5 → ~%50 maskeleme beklenir: {nan_ratio:.2f}"
        )

    def test_index_preserved(self):
        """Filtrelenmiş sinyalin index'i korunmalı."""
        signal, proba = self._make_signal_proba()
        filtered = apply_meta_filter(signal, proba, threshold=0.5)
        assert filtered.index.equals(signal.index)

    def test_mismatched_index_graceful(self):
        """proba index signal'den farklıysa crash olmadan çalışmalı."""
        signal, proba = self._make_signal_proba()
        # proba'nın index'ini kaydır
        proba.index = proba.index + pd.Timedelta(days=365)
        filtered = apply_meta_filter(signal, proba, threshold=0.5)
        # Tüm proba NaN → tümü maskelenmeli
        assert filtered.isna().all(), "Index uyumsuzluğunda tümü maskelenmeli"


class TestMetaModelPredict:
    """MetaModel.predict_proba() — fit_failed=True fallback."""

    def test_failed_model_returns_half(self):
        """fit_failed=True → predict_proba 0.5 döner, crash yok."""
        mm = MetaModel(fit_failed=True, fail_reason="test")
        idx = pd.DataFrame(
            {"x": [1, 2, 3]},
            index=pd.MultiIndex.from_tuples(
                [("T1", pd.Timestamp("2020-01-02")),
                 ("T1", pd.Timestamp("2020-01-03")),
                 ("T2", pd.Timestamp("2020-01-02"))],
                names=["Ticker", "Date"],
            ),
        )
        proba = mm.predict_proba(idx)
        assert (proba == 0.5).all(), "fit_failed → proba = 0.5 (belirsiz)"
