"""Birim testler: engine/data/regime_refitter.py — WalkForwardHMMRefitter."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.data.regime_detector import RegimeConfig
from engine.data.regime_refitter import WalkForwardHMMRefitter


# ── Yardımcı: Sentetik OHLCV ──────────────────────────────────────────────────

def _make_ohlcv(n_days: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    half = n_days // 2
    rets = np.concatenate([
        rng.normal(0.0005, 0.007, half),
        rng.normal(-0.0005, 0.025, n_days - half),
    ])
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.roll(close, 1); open_[0] = close[0]
    spread = np.abs(rng.normal(0, 0.005, n_days)) * close
    high = np.maximum(open_, close) + spread
    low  = np.minimum(open_, close) - spread
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": rng.integers(1_000_000, 5_000_000, n_days),
    }, index=dates)


_cfg = RegimeConfig(
    min_K=2, max_K=3,
    min_samples_per_regime=50,
    n_iter=20,
    random_state=0,
)


# ── Testler ───────────────────────────────────────────────────────────────────

def test_fit_initial_returns_prob_df():
    """fit_initial() geçerli olasılık DataFrame döndürmeli."""
    df = _make_ohlcv(1000)
    refitter = WalkForwardHMMRefitter(_cfg, refit_every_n_days=30)
    prob_df = refitter.fit_initial(df)

    assert isinstance(prob_df, pd.DataFrame)
    assert prob_df.shape[1] >= 2
    # Her satır bir olasılık dağılımı: toplamı ≈ 1
    row_sums = prob_df.sum(axis=1)
    np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)


def test_k_stable_after_fit_initial():
    """İlk fit'ten sonra _best_K belirlenmeli ve değişmemeli."""
    df = _make_ohlcv(1000)
    refitter = WalkForwardHMMRefitter(_cfg, refit_every_n_days=30)
    refitter.fit_initial(df)

    k_after_init = refitter._best_K
    assert k_after_init is not None
    assert _cfg.min_K <= k_after_init <= _cfg.max_K


def test_refit_triggers_after_n_days():
    """refit_every_n_days geçince needs_refit() True döndürmeli."""
    # İş günü bazlı takvim: 700 iş günü fit edildi
    df_base = _make_ohlcv(700)
    refitter = WalkForwardHMMRefitter(_cfg, refit_every_n_days=30)
    refitter.fit_initial(df_base)

    # Aynı son tarihle yeni veri yok — refit gerekmemeli
    assert not refitter.needs_refit(df_base)

    # Son tarihten 29 gün sonrası — eşik altında
    last = df_base.index[-1]
    df_29 = _make_ohlcv(700)
    df_29.index = df_29.index + pd.DateOffset(days=29)
    assert not refitter.needs_refit(df_29)

    # Son tarihten 31 gün sonrası — eşik aşıldı, refit gerekmeli
    df_31 = _make_ohlcv(700)
    df_31.index = df_31.index + pd.DateOffset(days=31)
    assert refitter.needs_refit(df_31)


def test_hungarian_alignment_preserves_column_names():
    """update() sonrası prob_df kolonları ['regime_0', ...] formatında kalmalı."""
    df = _make_ohlcv(1000)
    refitter = WalkForwardHMMRefitter(_cfg, refit_every_n_days=0)  # her update'te refit
    refitter.fit_initial(df)

    df_extended = _make_ohlcv(1100)
    prob_df = refitter.update(df_extended, force=True)

    # Kolon isimleri regex: regime_\d+
    for col in prob_df.columns:
        assert col.startswith("regime_"), f"Beklenmeyen kolon: {col}"

    # Toplamlar hâlâ 1
    row_sums = prob_df.sum(axis=1)
    np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)


def test_refit_history_grows():
    """Her refit sonrası refit_history uzuyor olmalı."""
    df = _make_ohlcv(1000)
    refitter = WalkForwardHMMRefitter(_cfg, refit_every_n_days=0)
    refitter.fit_initial(df)
    assert len(refitter.refit_history) == 1

    df2 = _make_ohlcv(1100)
    refitter.update(df2, force=True)
    assert len(refitter.refit_history) == 2


def test_update_without_initial_calls_fit_initial():
    """update() ilk fit yapılmamışsa sessizce fit_initial() çağırmalı."""
    df = _make_ohlcv(1000)
    refitter = WalkForwardHMMRefitter(_cfg)
    prob_df = refitter.update(df)   # fit_initial() yapılmadan çağrılıyor

    assert prob_df is not None
    assert refitter._model is not None


def test_min_history_guard():
    """Veri min_history_days'ten azsa refit yapılmamalı, model değişmemeli."""
    df_init = _make_ohlcv(700)
    refitter = WalkForwardHMMRefitter(_cfg, refit_every_n_days=0,
                                      min_history_days=800)
    refitter.fit_initial(df_init)
    model_before = refitter._model

    # 700 < 800 → refit atlanmalı, model değişmemeli
    df_small = _make_ohlcv(700)
    refitter.update(df_small, force=True)
    assert refitter._model is model_before  # aynı nesne
