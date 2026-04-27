"""
tests/test_triple_barrier.py — engine/triple_barrier.py birim testleri.

Belgelenmiş gereksinimler (DOKUMAN_ONERILER.md §11.1):
  - test_long_only_mapping : long_only=True → hiç -1 etiketi olmamalı
  - Ek: etiket değerleri geçerli, horizon sınırı, label_stats doğruluğu
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from engine.data.triple_barrier import (
    compute_triple_barrier_labels,
    add_triple_barrier_to_idx,
    label_stats,
)
from tests.conftest import make_synthetic_db, make_synthetic_idx


# ─── Yardımcı ─────────────────────────────────────────────────────────────────

def _make_volatile_db(n_tickers: int = 10, n_days: int = 300, seed: int = 0) -> pd.DataFrame:
    """
    Yüksek volatiliteli veri üretir: bariyer çarpılma olasılığı artar.
    Hem üst hem alt bariyerin çarpılmasını gözlemlemek için yeterli varyasyon.
    """
    rng  = np.random.default_rng(seed)
    dates   = pd.date_range("2020-01-01", periods=n_days, freq="B")
    tickers = [f"V{i:02d}" for i in range(n_tickers)]
    rows = []
    for ticker in tickers:
        # Yüksek volatilite: bazı hisseler güçlü yükselir, bazıları düşer
        trend   = rng.uniform(-0.002, 0.002)
        vol_lvl = 0.035   # %3.5 günlük — bariyer çarpmayı artırır
        prices  = 100.0 * np.exp(np.cumsum(rng.normal(trend, vol_lvl, n_days)))
        for j, date in enumerate(dates):
            rows.append({"Ticker": ticker, "Date": date, "Pclose": float(prices[j])})
    return pd.DataFrame(rows)


# ─── compute_triple_barrier_labels testleri ──────────────────────────────────

class TestComputeTripleBarrierLabels:

    def test_long_only_no_minus_one(self):
        """
        §11.1 Kritik Test:
        long_only=True olduğunda hiçbir etiket -1 olmamalı.
        BIST gerçeği: short işlem yapılamadığından alt bariyer → 0.
        """
        db     = _make_volatile_db(n_tickers=15, n_days=400, seed=1)
        labels = compute_triple_barrier_labels(
            db, horizon=10, multiplier=1.0, long_only=True
        )
        labels_clean = labels.dropna()
        assert len(labels_clean) > 0, "Etiket üretilemedi"
        assert (labels_clean == -1).sum() == 0, (
            f"long_only=True'da -1 etiket var: "
            f"{(labels_clean == -1).sum()} adet"
        )

    def test_long_short_has_minus_one(self):
        """
        long_only=False + yüksek volatilite → bazı -1 etiketler üretilmeli.
        """
        db     = _make_volatile_db(n_tickers=15, n_days=400, seed=2)
        labels = compute_triple_barrier_labels(
            db, horizon=10, multiplier=0.5,  # düşük çarpan → daha kolay çarpılma
            long_only=False
        )
        labels_clean = labels.dropna()
        assert len(labels_clean) > 0
        # Yüksek vol + düşük çarpanla bazı -1 görmeli
        n_neg = (labels_clean == -1).sum()
        assert n_neg > 0, (
            "long_only=False ve yüksek vol'da -1 etiketi görülmedi. "
            "Test verisi veya parametreler gözden geçirilmeli."
        )

    def test_label_values_valid_long_only(self):
        """long_only=True: etiketler sadece {0.0, 1.0} içermeli."""
        db     = _make_volatile_db(n_tickers=10, n_days=300, seed=3)
        labels = compute_triple_barrier_labels(db, long_only=True)
        valid  = {0.0, 1.0}
        actual = set(labels.dropna().unique())
        assert actual.issubset(valid), f"Geçersiz etiketler: {actual - valid}"

    def test_label_values_valid_long_short(self):
        """long_only=False: etiketler {-1.0, 0.0, 1.0} alt kümesi içermeli."""
        db    = _make_volatile_db(n_tickers=10, n_days=300, seed=4)
        labels = compute_triple_barrier_labels(db, long_only=False)
        valid  = {-1.0, 0.0, 1.0}
        actual = set(labels.dropna().unique())
        assert actual.issubset(valid), f"Geçersiz etiketler: {actual - valid}"

    def test_returns_series_with_multiindex(self):
        """Dönüş değeri (Ticker, Date) MultiIndex'li pd.Series olmalı."""
        db     = make_synthetic_db(n_tickers=5, n_days=200, seed=5)
        labels = compute_triple_barrier_labels(db, horizon=5)
        assert isinstance(labels, pd.Series)
        assert labels.index.names == ["Ticker", "Date"]

    def test_horizon_limits_lookahead(self):
        """
        Son `horizon` gün için etiket üretilmemeli (gelecek fiyat yok).
        Şeffaf veri sızıntısı yokluğu testi.
        """
        db      = make_synthetic_db(n_tickers=5, n_days=200, seed=6)
        horizon = 10
        labels  = compute_triple_barrier_labels(db, horizon=horizon)
        all_dates = sorted(db["Date"].unique())
        last_h_dates = set(all_dates[-horizon:])
        # Son horizon tarihinde üretilmiş etiket olmamalı
        if len(labels) > 0:
            labeled_dates = set(labels.index.get_level_values("Date"))
            overlap = labeled_dates & last_h_dates
            assert len(overlap) == 0, (
                f"Son {horizon} günde etiket var (lookahead!): {overlap}"
            )

    def test_no_nan_in_valid_window(self):
        """
        vol_window + horizon sonrasındaki tarihler için NaN olmamalı
        (en azından bazı değerler üretilmeli).
        """
        db     = make_synthetic_db(n_tickers=5, n_days=250, seed=7)
        labels = compute_triple_barrier_labels(db, horizon=5, vol_window=20)
        labels_clean = labels.dropna()
        assert len(labels_clean) > 0, "Hiç geçerli etiket üretilemedi"

    def test_multiplier_effect(self):
        """
        Çok büyük çarpan (10×σ) → neredeyse hiç bariyer çarpmaz → çoğu 0.
        Çok küçük çarpan (0.1×σ) → çoğu zaman bariyer çarpar → az 0.
        """
        db = _make_volatile_db(n_tickers=10, n_days=300, seed=8)

        labels_wide   = compute_triple_barrier_labels(db, multiplier=10.0, long_only=True)
        labels_narrow = compute_triple_barrier_labels(db, multiplier=0.1,  long_only=True)

        flat_wide   = labels_wide.dropna()
        flat_narrow = labels_narrow.dropna()

        if len(flat_wide) > 10 and len(flat_narrow) > 10:
            # Geniş bariyer: çoğu 0 (timeout)
            zero_pct_wide   = (flat_wide   == 0).mean()
            # Dar bariyer: çoğu 1 (anında çarpar)
            nonzero_pct_narrow = (flat_narrow != 0).mean()

            assert zero_pct_wide > 0.5, (
                f"Geniş bariyer (×10σ): beklenen çoğunlukla 0, "
                f"zero_pct={zero_pct_wide:.2f}"
            )
            assert nonzero_pct_narrow > 0.3, (
                f"Dar bariyer (×0.1σ): beklenen sık çarpma, "
                f"nonzero_pct={nonzero_pct_narrow:.2f}"
            )


# ─── add_triple_barrier_to_idx testleri ──────────────────────────────────────

class TestAddTripleBarrierToIdx:

    def test_adds_tb_label_column(self, syn_idx):
        """TB_Label kolonu idx'e eklenmeli."""
        result = add_triple_barrier_to_idx(syn_idx, horizon=5, long_only=True)
        assert "TB_Label" in result.columns

    def test_preserves_original_columns(self, syn_idx):
        """Orijinal kolonlar korunmalı."""
        orig_cols = set(syn_idx.columns)
        result    = add_triple_barrier_to_idx(syn_idx, horizon=5)
        assert orig_cols.issubset(set(result.columns))

    def test_long_only_no_minus_one_in_idx(self, syn_idx):
        """add_triple_barrier_to_idx + long_only=True → TB_Label'da -1 yok."""
        result = add_triple_barrier_to_idx(syn_idx, horizon=5, long_only=True)
        labels = result["TB_Label"].dropna()
        assert (labels == -1).sum() == 0

    def test_raises_without_pclose(self):
        """Pclose kolonu olmayan idx → ValueError fırlatmalı."""
        idx_no_price = pd.DataFrame(
            {"Next_Ret": [0.01, 0.02]},
            index=pd.MultiIndex.from_tuples(
                [("T001", pd.Timestamp("2020-01-01")),
                 ("T001", pd.Timestamp("2020-01-02"))],
                names=["Ticker", "Date"]
            )
        )
        with pytest.raises(ValueError, match="Pclose"):
            add_triple_barrier_to_idx(idx_no_price)


# ─── label_stats testleri ────────────────────────────────────────────────────

class TestLabelStats:

    def test_keys_present(self, syn_idx):
        """label_stats dönüşü beklenen anahtarları içermeli."""
        result  = add_triple_barrier_to_idx(syn_idx, horizon=5, long_only=True)
        labels  = result["TB_Label"].dropna()
        stats   = label_stats(labels)
        assert "buy    (+1)" in stats
        assert "flat    (0)"  in stats
        assert "n_total"      in stats

    def test_proportions_sum_to_one(self, syn_idx):
        """buy + flat (+ short eğer varsa) ≈ 1.0 olmalı."""
        result = add_triple_barrier_to_idx(syn_idx, horizon=5, long_only=False)
        labels = result["TB_Label"].dropna()
        stats  = label_stats(labels)

        total_pct = (
            stats.get("buy    (+1)", 0.0) +
            stats.get("flat    (0)", 0.0) +
            stats.get("short  (-1)", 0.0)
        )
        assert abs(total_pct - 1.0) < 0.01, (
            f"Oranlar toplamı ≈ 1.0 değil: {total_pct:.4f}"
        )

    def test_n_total_positive(self, syn_idx):
        """n_total > 0 olmalı."""
        result = add_triple_barrier_to_idx(syn_idx, horizon=5, long_only=True)
        stats  = label_stats(result["TB_Label"].dropna())
        assert stats["n_total"] > 0

    def test_long_only_no_short_key(self):
        """
        long_only=True ile üretilen etiketlerde short_(-1) satırı sıfır olduğundan
        label_stats bunu göstermemeli.
        """
        db     = make_synthetic_db(n_tickers=10, n_days=200, seed=10)
        idx    = db.set_index(["Ticker", "Date"]).sort_index()
        result = add_triple_barrier_to_idx(idx, horizon=5, long_only=True)
        stats  = label_stats(result["TB_Label"].dropna())
        # "short (-1)" anahtarı ya yok ya da 0
        assert stats.get("short  (-1)", 0.0) == 0.0
