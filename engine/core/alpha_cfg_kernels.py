"""Numba JIT-compile rolling operator kernels (Faz 1.4).

Mevcut alpha_cfg.py operatörleri Pandas `groupby + rolling + apply` zinciri
kullanıyor — Python interpreter overhead yüzünden formül evaluation darboğaz.

Bu modül **per-ticker numpy array** üzerinde çalışan numba-jit kernels sağlar:
- `wma_kernel`     — Weighted Moving Average
- `ema_kernel`     — Exponential Moving Average (adjust=True)
- `rolling_rank_pct` — Pencere sonu rank percentile
- `rolling_mad`    — Mean Absolute Deviation
- `rolling_corr`   — Paired rolling Pearson
- `rolling_cov`    — Paired rolling Covariance

`alpha_cfg.py` lambda'larından çağrılır; numba yoksa pandas eski yola döner.

Kullanım:
    from engine.core.alpha_cfg_kernels import wma_groupby

    result = wma_groupby(series, window=20)  # MultiIndex (Ticker, Date) Series
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_NUMBA_AVAILABLE: Optional[bool] = None


def _check_numba():
    global _NUMBA_AVAILABLE
    if _NUMBA_AVAILABLE is None:
        try:
            import numba  # noqa: F401
            _NUMBA_AVAILABLE = True
        except ImportError:
            _NUMBA_AVAILABLE = False
    return _NUMBA_AVAILABLE


# ── Numba kernels (lazy compile) ──────────────────────────────────────────────

if _check_numba():
    from numba import njit, prange

    @njit(cache=True, fastmath=True)
    def _wma_kernel(x: np.ndarray, window: int) -> np.ndarray:
        n = len(x)
        out = np.full(n, np.nan)
        if window <= 0 or n < window:
            return out
        weights = np.empty(window, dtype=np.float64)
        for i in range(window):
            weights[i] = i + 1.0
        wsum = weights.sum()
        for i in range(window - 1, n):
            s = 0.0
            valid = True
            for j in range(window):
                v = x[i - window + 1 + j]
                if np.isnan(v):
                    valid = False
                    break
                s += v * weights[j]
            if valid:
                out[i] = s / wsum
        return out

    @njit(cache=True, fastmath=True)
    def _ema_kernel(x: np.ndarray, span: int) -> np.ndarray:
        """Adjust=True EMA — pandas ewm ile aynı sonuç."""
        n = len(x)
        out = np.full(n, np.nan)
        if span <= 0 or n == 0:
            return out
        alpha = 2.0 / (span + 1.0)
        # min_periods = span (eski kodla uyum)
        # adjust=True: weighted_sum / weight_sum
        if n < span:
            return out
        # Warm-up: ilk span gözlemi tek seferde adjust=True ile hesapla
        for i in range(span - 1, n):
            num = 0.0
            den = 0.0
            valid = True
            # Geçmiş tüm geçerli noktaları topla (truncated to current i)
            # Pandas ewm adjust=True formülü: numerator = Σ (1-α)^k × x_{i-k}
            #                                 denom    = Σ (1-α)^k
            w = 1.0
            for k in range(0, i + 1):
                v = x[i - k]
                if np.isnan(v):
                    valid = False
                    break
                num += w * v
                den += w
                w *= (1.0 - alpha)
                if w < 1e-12:
                    break
            if valid and den > 0:
                out[i] = num / den
        return out

    @njit(cache=True, fastmath=True)
    def _rolling_rank_pct(x: np.ndarray, window: int) -> np.ndarray:
        """Her pencerede son elemanın rank percentile'i (pct=True)."""
        n = len(x)
        out = np.full(n, np.nan)
        if window <= 0 or n < window:
            return out
        for i in range(window - 1, n):
            last = x[i]
            if np.isnan(last):
                continue
            count = 0
            le = 0
            valid = True
            for j in range(window):
                v = x[i - window + 1 + j]
                if np.isnan(v):
                    valid = False
                    break
                if v <= last:
                    le += 1
                count += 1
            if valid and count > 0:
                out[i] = le / count
        return out

    @njit(cache=True, fastmath=True)
    def _rolling_mad(x: np.ndarray, window: int) -> np.ndarray:
        n = len(x)
        out = np.full(n, np.nan)
        if window <= 0 or n < window:
            return out
        for i in range(window - 1, n):
            mean = 0.0
            valid = True
            for j in range(window):
                v = x[i - window + 1 + j]
                if np.isnan(v):
                    valid = False
                    break
                mean += v
            if not valid:
                continue
            mean /= window
            mad = 0.0
            for j in range(window):
                v = x[i - window + 1 + j]
                mad += abs(v - mean)
            out[i] = mad / window
        return out

    @njit(cache=True, fastmath=True)
    def _rolling_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        n = len(x)
        out = np.full(n, np.nan)
        if window <= 1 or n < window:
            return out
        for i in range(window - 1, n):
            sx = 0.0; sy = 0.0
            sxx = 0.0; syy = 0.0; sxy = 0.0
            valid = True
            for j in range(window):
                xv = x[i - window + 1 + j]
                yv = y[i - window + 1 + j]
                if np.isnan(xv) or np.isnan(yv):
                    valid = False
                    break
                sx += xv; sy += yv
                sxx += xv * xv; syy += yv * yv
                sxy += xv * yv
            if not valid:
                continue
            mx = sx / window; my = sy / window
            varx = sxx / window - mx * mx
            vary = syy / window - my * my
            cov  = sxy / window - mx * my
            denom = (varx * vary)
            if denom > 1e-20:
                out[i] = cov / np.sqrt(denom)
        return out

    @njit(cache=True, fastmath=True)
    def _rolling_cov(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
        n = len(x)
        out = np.full(n, np.nan)
        if window <= 1 or n < window:
            return out
        for i in range(window - 1, n):
            sx = 0.0; sy = 0.0; sxy = 0.0
            valid = True
            for j in range(window):
                xv = x[i - window + 1 + j]
                yv = y[i - window + 1 + j]
                if np.isnan(xv) or np.isnan(yv):
                    valid = False
                    break
                sx += xv; sy += yv; sxy += xv * yv
            if valid:
                mx = sx / window; my = sy / window
                out[i] = sxy / window - mx * my
        return out


# ── Public API: per-ticker groupby uygulama ───────────────────────────────────

def _apply_kernel_per_ticker(series: pd.Series, kernel, window: int) -> pd.Series:
    """MultiIndex (Ticker, Date) Series üzerinde her ticker için kernel uygula."""
    # series.values → her ticker bloğu sıralı
    out = np.empty(len(series), dtype=np.float64)
    out.fill(np.nan)
    if "Ticker" not in series.index.names:
        # Düz Series → tek blok
        out = kernel(series.values.astype(np.float64), window)
        return pd.Series(out, index=series.index)

    # Per-ticker iterate
    by_ticker = series.groupby(level="Ticker", group_keys=False)
    out_pieces: list[pd.Series] = []
    for tkr, sub in by_ticker:
        arr = sub.values.astype(np.float64)
        res = kernel(arr, window)
        out_pieces.append(pd.Series(res, index=sub.index))
    return pd.concat(out_pieces).reindex(series.index)


def wma_groupby(series: pd.Series, window: int) -> pd.Series:
    """WMA — JIT varsa hızlı, yoksa pandas rolling fallback."""
    if _check_numba():
        return _apply_kernel_per_ticker(series, _wma_kernel, int(window))
    # Fallback: pandas
    return series.groupby(level="Ticker", group_keys=False).apply(
        lambda g: g.rolling(int(window)).apply(
            lambda a: np.average(a, weights=np.arange(1, len(a) + 1)), raw=True)
    )


def ema_groupby(series: pd.Series, window: int) -> pd.Series:
    """EMA adjust=True, min_periods=window — JIT varsa hızlı."""
    if _check_numba():
        return _apply_kernel_per_ticker(series, _ema_kernel, int(window))
    return series.groupby(level="Ticker", group_keys=False).apply(
        lambda g: g.ewm(span=int(window), adjust=True, min_periods=int(window)).mean()
    )


def rolling_rank_pct_groupby(series: pd.Series, window: int) -> pd.Series:
    """Pencere sonu rank percentile — JIT varsa hızlı."""
    if _check_numba():
        return _apply_kernel_per_ticker(series, _rolling_rank_pct, int(window))
    return series.groupby(level="Ticker", group_keys=False).apply(
        lambda g: g.rolling(int(window)).apply(
            lambda a: pd.Series(a).rank(pct=True).iloc[-1], raw=False)
    )


def rolling_mad_groupby(series: pd.Series, window: int) -> pd.Series:
    """Rolling MAD — JIT varsa hızlı."""
    if _check_numba():
        return _apply_kernel_per_ticker(series, _rolling_mad, int(window))
    return series.groupby(level="Ticker", group_keys=False).apply(
        lambda g: g.rolling(int(window)).apply(
            lambda a: np.mean(np.abs(a - np.mean(a))), raw=True)
    )


def rolling_corr_groupby(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Paired rolling Pearson korelasyonu — JIT varsa hızlı."""
    if _check_numba():
        out_pieces: list[pd.Series] = []
        common_idx = x.index.intersection(y.index)
        x = x.reindex(common_idx); y = y.reindex(common_idx)
        if "Ticker" not in x.index.names:
            arr = _rolling_corr(x.values.astype(np.float64),
                                  y.values.astype(np.float64), int(window))
            return pd.Series(arr, index=x.index)
        for tkr, sub_x in x.groupby(level="Ticker"):
            sub_y = y.loc[tkr] if tkr in y.index.get_level_values("Ticker") else None
            if sub_y is None:
                out_pieces.append(pd.Series(np.nan, index=sub_x.index))
                continue
            xv = sub_x.values.astype(np.float64)
            yv = sub_y.values.astype(np.float64)
            n = min(len(xv), len(yv))
            res = _rolling_corr(xv[:n], yv[:n], int(window))
            out_pieces.append(pd.Series(res, index=sub_x.index[:n]))
        return pd.concat(out_pieces).reindex(x.index)
    # Fallback
    d = pd.DataFrame({"x": x, "y": y})
    return d.groupby(level="Ticker", group_keys=False).apply(
        lambda g: g["x"].rolling(int(window)).corr(g["y"]))


def rolling_cov_groupby(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    """Paired rolling Covariance — JIT varsa hızlı."""
    if _check_numba():
        out_pieces: list[pd.Series] = []
        common_idx = x.index.intersection(y.index)
        x = x.reindex(common_idx); y = y.reindex(common_idx)
        if "Ticker" not in x.index.names:
            arr = _rolling_cov(x.values.astype(np.float64),
                                  y.values.astype(np.float64), int(window))
            return pd.Series(arr, index=x.index)
        for tkr, sub_x in x.groupby(level="Ticker"):
            sub_y = y.loc[tkr] if tkr in y.index.get_level_values("Ticker") else None
            if sub_y is None:
                out_pieces.append(pd.Series(np.nan, index=sub_x.index))
                continue
            xv = sub_x.values.astype(np.float64)
            yv = sub_y.values.astype(np.float64)
            n = min(len(xv), len(yv))
            res = _rolling_cov(xv[:n], yv[:n], int(window))
            out_pieces.append(pd.Series(res, index=sub_x.index[:n]))
        return pd.concat(out_pieces).reindex(x.index)
    d = pd.DataFrame({"x": x, "y": y})
    return d.groupby(level="Ticker", group_keys=False).apply(
        lambda g: g["x"].rolling(int(window)).cov(g["y"]))
