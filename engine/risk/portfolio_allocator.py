"""Portfolio Allocator — Risk Parity (ERC) + Black-Litterman.

Ledoit-Wolf shrinkage garantisi:
  BIST 500 hisse, lookback=60 gün → p=500 > n=60 → raw kovaryans tekil.
  Her hesaplamada sklearn.covariance.LedoitWolf kullanılır → her zaman invertible.

Kullanım:
    from engine.risk.portfolio_allocator import equal_risk_contribution, black_litterman

    weights_erc = equal_risk_contribution(returns_df)
    weights_bl  = black_litterman(returns_df, views={"THYAO": 0.015})
"""
from __future__ import annotations

import logging
import warnings
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


# ── Kovaryans (Ledoit-Wolf shrinkage — p > n'de asla tekil değil) ────────────

def _shrunk_cov(returns: pd.DataFrame, lookback: int) -> np.ndarray:
    """Ledoit-Wolf büzülmeli kovaryans matrisi.

    p > n durumunda raw kovaryans matrisinin tersini almak sayısal olarak
    kararsız olur. LedoitWolf garantili şekilde full-rank (invertible) matris üretir.
    """
    from sklearn.covariance import LedoitWolf

    X = returns.tail(lookback).dropna(axis=1, how="any").values
    if X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError(f"Yeterli veri yok: shape={X.shape}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lw = LedoitWolf(assume_centered=False)
        lw.fit(X)

    return lw.covariance_  # (n_tickers, n_tickers) — her zaman full-rank


# ── Equal Risk Contribution (ERC) ────────────────────────────────────────────

def equal_risk_contribution(
    returns: pd.DataFrame,
    lookback: int = 60,
    min_weight: float = 0.01,
    max_weight: float = 0.30,
) -> pd.Series:
    """Equal Risk Contribution portföy ağırlıkları.

    Hedef: w_i * (Σw)_i = 1/N for all i (eşit risk katkısı).
    Kısıtlar: Σw = 1, min_weight ≤ w_i ≤ max_weight.
    Kovaryans: LedoitWolf shrinkage — p > n durumunda asla tekil olmaz.

    Parameters
    ----------
    returns  : (Date × Ticker) günlük getiri matrisi.
    lookback : Kovaryans penceresi (iş günü).
    min_weight, max_weight : Bireysel ağırlık sınırları.

    Returns
    -------
    pd.Series : Ticker → ağırlık (toplamı 1.0).
    """
    available = returns.tail(lookback).dropna(axis=1, how="any").columns.tolist()
    if not available:
        raise ValueError("Yeterli getiri verisi yok.")

    sub_returns = returns[available]
    n = len(available)
    target_rc = 1.0 / n  # eşit risk katkısı hedefi

    try:
        cov = _shrunk_cov(sub_returns, lookback)
    except Exception as exc:
        logger.warning("ERC kovaryans hatası (%s) — 1/N fallback.", exc)
        return pd.Series(1.0 / n, index=available)

    def _risk_contributions(w: np.ndarray) -> np.ndarray:
        # Normalize by portfolio variance → sum = 1, target = 1/N
        portfolio_var = float(w @ cov @ w)
        if portfolio_var < 1e-20:
            return np.zeros(n)
        marginal = cov @ w
        return w * marginal / portfolio_var

    def _objective(w: np.ndarray) -> float:
        rc = _risk_contributions(w)
        return float(np.sum((rc - target_rc) ** 2))

    w0 = np.full(n, 1.0 / n)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(min_weight, max_weight)] * n

    try:
        result = minimize(
            _objective, w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-10},
        )
        if result.success:
            w_opt = result.x / result.x.sum()
        else:
            logger.warning("ERC optimize edilemedi (%s) — 1/N fallback.", result.message)
            w_opt = w0
    except Exception as exc:
        logger.warning("ERC optimize hatası (%s) — 1/N fallback.", exc)
        w_opt = w0

    return pd.Series(w_opt, index=available)


# ── Black-Litterman ───────────────────────────────────────────────────────────

def black_litterman(
    returns: pd.DataFrame,
    views: Dict[str, float],
    tau: float = 0.05,
    lookback: int = 60,
    min_weight: float = 0.01,
    max_weight: float = 0.30,
) -> pd.Series:
    """Bayesian Black-Litterman portföy ağırlıkları.

    Formül (klasik BL):
        π = piyasa dengesi getirisi (eşit ağırlıklı implied return)
        μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹q]
        Ω = tau * P × Σ × P' (view belirsizliği)
        P = kimlik matrisi (her view için ayrı)

    Parameters
    ----------
    returns  : (Date × Ticker) günlük getiri matrisi.
    views    : {ticker: beklenen günlük getiri} (formül sinyalinden türetilir).
    tau      : Prior belirsizlik ölçeği (varsayılan: 0.05).
    lookback : Kovaryans penceresi (iş günü).
    min_weight, max_weight : Ağırlık sınırları.

    Returns
    -------
    pd.Series : Ticker → ağırlık (toplamı 1.0).
    """
    available = returns.tail(lookback).dropna(axis=1, how="any").columns.tolist()
    view_tickers = [t for t in views if t in available]

    if not view_tickers:
        logger.warning("Hiçbir view ticker mevcut kolonlarla eşleşmedi — ERC fallback.")
        return equal_risk_contribution(returns, lookback, min_weight, max_weight)

    n = len(available)

    try:
        cov = _shrunk_cov(returns[available], lookback)
    except Exception as exc:
        logger.warning("BL kovaryans hatası (%s) — 1/N fallback.", exc)
        return pd.Series(1.0 / n, index=available)

    # Piyasa dengesi getirisi (eşit ağırlıklı portföy implied return)
    w_eq = np.full(n, 1.0 / n)
    lam = 2.5  # risk aversion (standart BL)
    pi = lam * cov @ w_eq  # (n,) — prior expected returns

    ticker_idx = {t: i for i, t in enumerate(available)}
    k = len(view_tickers)
    P = np.zeros((k, n))
    q = np.zeros(k)
    for j, t in enumerate(view_tickers):
        P[j, ticker_idx[t]] = 1.0
        q[j] = views[t]

    Sigma = cov
    tauSigma = tau * Sigma
    tauSigma_inv = np.linalg.inv(tauSigma)

    Omega = tau * P @ Sigma @ P.T  # (k, k)
    Omega_inv = np.linalg.inv(Omega + np.eye(k) * 1e-8)

    # Posterior expected returns
    M = tauSigma_inv + P.T @ Omega_inv @ P          # (n, n)
    mu_bl = np.linalg.solve(M, tauSigma_inv @ pi + P.T @ Omega_inv @ q)  # (n,)

    # Markowitz mean-variance → BL ağırlıkları
    Sigma_inv = np.linalg.inv(Sigma + np.eye(n) * 1e-8)
    w_raw = Sigma_inv @ mu_bl
    w_raw = np.clip(w_raw, min_weight, max_weight)
    w_sum = w_raw.sum()
    if w_sum < 1e-9:
        w_opt = np.full(n, 1.0 / n)
    else:
        w_opt = w_raw / w_sum

    return pd.Series(w_opt, index=available)
