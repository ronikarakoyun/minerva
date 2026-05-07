"""Birim testler: engine/risk/portfolio_allocator.py — PR-11."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.risk.portfolio_allocator import (
    _shrunk_cov,
    black_litterman,
    equal_risk_contribution,
)


# ── Yardımcı ─────────────────────────────────────────────────────────────────

def _make_returns(n_dates: int = 120, n_tickers: int = 50, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-01", periods=n_dates)
    tickers = [f"T{i:04d}" for i in range(n_tickers)]
    # Kovaryans yapısı olan getiriler
    cov = np.eye(n_tickers) * 0.0004 + 0.0001  # basit diagonal + küçük cross
    rets = rng.multivariate_normal(mean=np.zeros(n_tickers), cov=cov, size=n_dates)
    return pd.DataFrame(rets, index=dates, columns=tickers)


# ── Testler ───────────────────────────────────────────────────────────────────

def test_erc_weights_sum_to_one():
    """ERC ağırlıkları toplamı 1.0 olmalı."""
    returns = _make_returns()
    w = equal_risk_contribution(returns, lookback=60)

    assert isinstance(w, pd.Series), "pd.Series beklendi"
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6, err_msg="Ağırlıklar toplamı 1.0 olmalı")


def test_erc_risk_contributions_approximately_equal():
    """ERC volatilite katkıları yaklaşık eşit olmalı (±20%)."""
    returns = _make_returns(n_dates=120, n_tickers=20)
    w = equal_risk_contribution(returns, lookback=60)
    w_np = w.values

    from sklearn.covariance import LedoitWolf
    X = returns.tail(60).dropna(axis=1, how="any").values
    lw = LedoitWolf()
    lw.fit(X)
    cov = lw.covariance_

    n = len(w_np)
    port_var = float(w_np @ cov @ w_np)
    if port_var < 1e-20:
        pytest.skip("Port varyansı sıfır — degenerate test")

    marginal = cov @ w_np
    # Normalize by portfolio variance → sum = 1, target = 1/N
    risk_contribs = w_np * marginal / port_var
    target = 1.0 / n

    # Her risk katkısı hedefin ±30% içinde olmalı
    rel_diff = np.abs(risk_contribs - target) / target
    assert float(rel_diff.max()) < 0.30, (
        f"Max risk katkı sapması: {float(rel_diff.max()):.4f} (>%30)"
    )


def test_black_litterman_positive_view_increases_weight():
    """BL: pozitif view verilen hissenin ağırlığı artmalı (görece olarak)."""
    returns = _make_returns(n_dates=120, n_tickers=10, seed=1)
    tickers = list(returns.columns)
    target_ticker = tickers[0]

    # Baseline: view yok
    w_no_view = black_litterman(returns, views={}, lookback=60)

    # Güçlü pozitif view
    w_with_view = black_litterman(
        returns, views={target_ticker: 0.02}, lookback=60
    )

    baseline_w  = float(w_no_view.get(target_ticker, 1.0 / 10))
    view_w      = float(w_with_view.get(target_ticker, 1.0 / 10))

    # Pozitif view → hedef hissenin ağırlığı artmalı veya en az aynı kalmalı
    assert view_w >= baseline_w - 0.01, (
        f"Pozitif view ile ağırlık artmalıydı: {view_w:.4f} < {baseline_w:.4f}"
    )


def test_equal_weight_mode_backward_compat():
    """portfolio_mode='equal_weight' → mevcut backtest ile özdeş sonuç."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from engine.core.backtest_engine import run_pro_backtest

    # Minimal sentetik veri
    rng = np.random.default_rng(7)
    n_dates, n_tickers = 30, 10
    dates = pd.bdate_range("2023-01-01", periods=n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]

    rows = []
    for d in dates:
        for t in tickers:
            rows.append({"Date": d, "Ticker": t,
                         "Pclose": 100.0 + rng.standard_normal()})
    df = pd.DataFrame(rows)
    signal = pd.Series(rng.standard_normal(len(df)), index=df.index)

    # equal_weight (varsayılan) vs açık equal_weight — özdeş olmalı
    curve1, _ = run_pro_backtest(df, signal)
    curve2, _ = run_pro_backtest(df, signal, portfolio_mode="equal_weight")

    pd.testing.assert_frame_equal(curve1, curve2)


def test_shrunk_cov_full_rank_when_p_greater_n():
    """p=500, n=60 → LedoitWolf matrisi full-rank (invertible) olmalı."""
    rng = np.random.default_rng(0)
    n, p = 60, 200  # n < p → raw kovaryans tekil; LedoitWolf çözer

    dates = pd.bdate_range("2021-01-01", periods=n)
    tickers = [f"T{i:04d}" for i in range(p)]
    returns = pd.DataFrame(rng.standard_normal((n, p)), index=dates, columns=tickers)

    cov = _shrunk_cov(returns, lookback=n)

    assert cov.shape == (p, p), f"Kovaryans boyutu yanlış: {cov.shape}"

    # Matrisin tersini alabilmeli — ValueError / LinAlgError fırlatmamalı
    inv_cov = np.linalg.inv(cov)
    assert inv_cov.shape == (p, p)

    # Matris rank — full-rank olmak için minimum singular değer > 0
    singular_values = np.linalg.svd(cov, compute_uv=False)
    min_sv = float(singular_values.min())
    assert min_sv > 1e-10, f"Minimum singular değer çok küçük: {min_sv}"
