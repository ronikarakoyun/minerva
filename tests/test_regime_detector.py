"""
tests/test_regime_detector.py — Faz 1 HMM rejim modülü için unit testler.

yfinance.download monkeypatch ile mock'lanır; HMM gerçekten fit edilir.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.data import regime_detector as rd
from engine.data.regime_detector import (
    FEATURES,
    RegimeConfig,
    compute_features,
    compute_probability_vector,
    fit_constrained_hmm,
    run_pipeline,
    save_model,
)


# ──────────────────────────────────────────────────────────────────────
# Synthetic data fixtures
# ──────────────────────────────────────────────────────────────────────
def _make_synthetic_ohlcv(n_days: int = 1500, seed: int = 7) -> pd.DataFrame:
    """
    İki-rejim sentetik OHLCV: ilk yarı düşük volatilite + uptrend, ikinci yarı yüksek vol.
    HMM en az 2 rejim bulabilmeli.
    """
    rng = np.random.default_rng(seed)
    half = n_days // 2

    # İlk yarı: ortalama +0.05% getiri, σ=0.7%
    rets1 = rng.normal(0.0005, 0.007, size=half)
    # İkinci yarı: ortalama -0.05% getiri, σ=2.5%
    rets2 = rng.normal(-0.0005, 0.025, size=n_days - half)
    rets = np.concatenate([rets1, rets2])

    close = 100.0 * np.exp(np.cumsum(rets))
    # Open ≈ önceki close, High/Low rastgele aralık
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    spread = np.abs(rng.normal(0, 0.005, n_days)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.integers(1_000_000, 50_000_000, size=n_days)

    idx = pd.date_range("2018-01-01", periods=n_days, freq="B")
    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=pd.DatetimeIndex(idx, name="Date"),
    )


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    return _make_synthetic_ohlcv()


@pytest.fixture
def cfg_short() -> RegimeConfig:
    """Test için ufak konfigürasyon (sentetik veriyle min_samples gevşetilmiş)."""
    return RegimeConfig(
        min_K=2,
        max_K=3,
        min_samples_per_regime=200,
        n_iter=200,  # daha hızlı
    )


# ──────────────────────────────────────────────────────────────────────
# 1. Feature engineering
# ──────────────────────────────────────────────────────────────────────
def test_compute_features_no_nan(synthetic_ohlcv: pd.DataFrame, cfg_short: RegimeConfig):
    feats = compute_features(synthetic_ohlcv, cfg_short)
    assert not feats.isna().any().any(), "Feature DataFrame'inde NaN olmamalı"
    assert list(feats.columns) == FEATURES
    # RobustScaler sonrası medyan ≈ 0
    assert feats.median().abs().max() < 0.5, "Scaled feature medyanı sıfıra yakın olmalı"
    # Boyut: ~ASLanlardaki rolling kayıp kadar daha az
    assert len(feats) >= len(synthetic_ohlcv) - 30


# ──────────────────────────────────────────────────────────────────────
# 2. Constrained HMM (min-samples disqualification)
# ──────────────────────────────────────────────────────────────────────
def test_constrained_hmm_disqualifies_too_small_regimes(
    synthetic_ohlcv: pd.DataFrame,
):
    """
    1500 günlük veri, min_samples_per_regime=600 → K=3 zorla diskalifiye olmalı
    (3 rejimden en az biri kesinlikle 600 günden az çıkar).
    Geçerli K aralığı bu durumda K=2 olmalı.
    """
    cfg = RegimeConfig(min_K=2, max_K=3, min_samples_per_regime=600, n_iter=200)
    feats = compute_features(synthetic_ohlcv, cfg)
    model, best_K, candidates = fit_constrained_hmm(feats, cfg)

    assert best_K == 2, f"K=2 seçilmeli (K=3 disqualified), seçilen={best_K}"
    assert 3 not in candidates, "K=3 candidates'ta olmamalı (disqualified)"
    assert candidates[2]["counts"][0] >= 600
    assert candidates[2]["counts"][1] >= 600


def test_no_valid_K_raises(synthetic_ohlcv: pd.DataFrame):
    """min_samples imkansız yüksekse RuntimeError fırlatmalı."""
    cfg = RegimeConfig(
        min_K=2, max_K=3,
        min_samples_per_regime=10_000,  # 1500 günden fazla → hiç geçemez
        n_iter=100,
    )
    feats = compute_features(synthetic_ohlcv, cfg)
    with pytest.raises(RuntimeError, match="min_samples_per_regime"):
        fit_constrained_hmm(feats, cfg)


# ──────────────────────────────────────────────────────────────────────
# 3. Probability vector
# ──────────────────────────────────────────────────────────────────────
def test_probability_vector_shape_and_sum(
    synthetic_ohlcv: pd.DataFrame, cfg_short: RegimeConfig
):
    feats = compute_features(synthetic_ohlcv, cfg_short)
    model, best_K, _ = fit_constrained_hmm(feats, cfg_short)
    prob_df = compute_probability_vector(model, feats)

    assert prob_df.shape == (len(feats), best_K)
    assert list(prob_df.columns) == [f"regime_{i}" for i in range(best_K)]
    # Her satırın olasılık toplamı 1.0
    row_sums = prob_df.sum(axis=1)
    assert (row_sums - 1.0).abs().max() < 1e-6, "Olasılıklar 1.0'a toplanmıyor"
    # Date index korunmuş
    assert prob_df.index.equals(feats.index)


# ──────────────────────────────────────────────────────────────────────
# 4. Save / load roundtrip
# ──────────────────────────────────────────────────────────────────────
def test_save_load_roundtrip(
    synthetic_ohlcv: pd.DataFrame, cfg_short: RegimeConfig, tmp_path: Path
):
    cfg = RegimeConfig(
        min_K=cfg_short.min_K,
        max_K=cfg_short.max_K,
        min_samples_per_regime=cfg_short.min_samples_per_regime,
        n_iter=cfg_short.n_iter,
        model_path=tmp_path / "hmm.pkl",
        metadata_path=tmp_path / "metadata.json",
        plot_path=tmp_path / "plot.png",
    )
    feats = compute_features(synthetic_ohlcv, cfg)
    model, best_K, candidates = fit_constrained_hmm(feats, cfg)
    prob_df = compute_probability_vector(model, feats)
    raw_returns = np.log(synthetic_ohlcv["Close"] / synthetic_ohlcv["Close"].shift(1)).dropna()

    save_model(model, best_K, candidates, prob_df, raw_returns, cfg)

    assert cfg.model_path.exists()
    assert cfg.metadata_path.exists()

    loaded = rd.load_model(cfg)
    probs_loaded = compute_probability_vector(loaded, feats)
    np.testing.assert_allclose(probs_loaded.values, prob_df.values, atol=1e-10)


# ──────────────────────────────────────────────────────────────────────
# 5. Metadata anahtarları
# ──────────────────────────────────────────────────────────────────────
def test_metadata_keys_and_values(
    synthetic_ohlcv: pd.DataFrame, cfg_short: RegimeConfig, tmp_path: Path
):
    cfg = RegimeConfig(
        min_K=cfg_short.min_K,
        max_K=cfg_short.max_K,
        min_samples_per_regime=cfg_short.min_samples_per_regime,
        n_iter=cfg_short.n_iter,
        model_path=tmp_path / "hmm.pkl",
        metadata_path=tmp_path / "metadata.json",
        plot_path=tmp_path / "plot.png",
    )
    feats = compute_features(synthetic_ohlcv, cfg)
    model, best_K, candidates = fit_constrained_hmm(feats, cfg)
    prob_df = compute_probability_vector(model, feats)
    raw_returns = np.log(synthetic_ohlcv["Close"] / synthetic_ohlcv["Close"].shift(1)).dropna()
    save_model(model, best_K, candidates, prob_df, raw_returns, cfg)

    meta = json.loads(cfg.metadata_path.read_text())
    required = {"trained_at", "ticker", "period", "optimal_K", "scores",
                "regime_stats", "last_day", "last_day_probs"}
    assert required.issubset(meta.keys()), f"Eksik anahtar: {required - meta.keys()}"
    assert meta["optimal_K"] == best_K
    assert len(meta["last_day_probs"]) == best_K
    # Her rejim için n_days, mean_daily_return, annualized_vol
    for k in range(best_K):
        rs = meta["regime_stats"][f"regime_{k}"]
        assert {"n_days", "mean_daily_return", "annualized_vol"} == set(rs.keys())


# ──────────────────────────────────────────────────────────────────────
# 6. End-to-end pipeline (yfinance mocked)
# ──────────────────────────────────────────────────────────────────────
def test_run_pipeline_mocked_yfinance(monkeypatch, tmp_path: Path):
    """Tam pipeline'ı yfinance'i mock ederek çalıştır."""
    fake_ohlcv = _make_synthetic_ohlcv(n_days=1500, seed=11)

    def fake_download(*args, **kwargs):
        return fake_ohlcv

    monkeypatch.setattr(rd.yf, "download", fake_download)

    cfg = RegimeConfig(
        min_K=2, max_K=3,
        min_samples_per_regime=200,
        n_iter=200,
        model_path=tmp_path / "hmm.pkl",
        metadata_path=tmp_path / "metadata.json",
        plot_path=tmp_path / "plot.png",
    )
    prob_df = run_pipeline(cfg)

    assert 2 <= prob_df.shape[1] <= 3
    assert (prob_df.sum(axis=1) - 1.0).abs().max() < 1e-6
    assert cfg.plot_path.exists() and cfg.plot_path.stat().st_size > 1000
    assert cfg.model_path.exists()
    assert cfg.metadata_path.exists()
