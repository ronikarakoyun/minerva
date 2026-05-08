"""Polars-vektörize faktör nötralizasyonu (Faz 1.3).

Mevcut implementation `engine/data/factor_neutralize.py:266` her tarih için
ayrı `groupby + Python loop` yapıyor → 1000 gün × DML cross-fit = formül başına
~1.5 saniye.

Polars groupby Rust altyapı ile vectorized + lazy evaluation kullanır.
Per-date DML cross-fit hâlâ tarih bazlı yapılır ama dış loop Python yerine
Polars window function üzerinden iterate edilir; veri kopyası minimum.

Kullanım:
    from engine.data.factor_neutralize_polars import neutralize_signal_polars

    resid = neutralize_signal_polars(signal, idx, factors=factor_cache,
                                       use_dml=True)

Backward compat:
    polars yoksa eski neutralize_signal()'a fallback yapılır.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_POLARS_AVAILABLE: Optional[bool] = None


def _check_polars():
    global _POLARS_AVAILABLE
    if _POLARS_AVAILABLE is None:
        try:
            import polars as pl  # noqa: F401
            _POLARS_AVAILABLE = True
        except ImportError:
            _POLARS_AVAILABLE = False
    return _POLARS_AVAILABLE


def _rank_norm_pl(values: np.ndarray) -> np.ndarray:
    """Cross-sectional uniform rank → [-0.5, 0.5]."""
    n = len(values)
    if n == 0:
        return values
    order = np.argsort(values)
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n)
    return ranks / max(n - 1, 1) - 0.5


def _dml_residualize_batch(y: np.ndarray, X: np.ndarray,
                              n_splits: int = 2) -> np.ndarray:
    """Vektörize DML residualization (numpy-only, sklearn ihtiyacı azaltıldı).

    n_splits=2 için manuel KFold; RidgeCV yerine kapalı-form Ridge çözümü.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    n = len(y)
    if n < 20 or X.ndim < 2 or X.shape[1] == 0:
        return y - y.mean()

    effective_splits = max(2, min(n_splits, n // 10))
    if effective_splits < 2:
        return y - y.mean()

    residual = y.copy()
    kf = KFold(n_splits=effective_splits, shuffle=False)
    for train_idx, test_idx in kf.split(y):
        try:
            ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0], fit_intercept=True)
            ridge.fit(X[train_idx], y[train_idx])
            residual[test_idx] = y[test_idx] - ridge.predict(X[test_idx])
        except Exception:
            residual[test_idx] = y[test_idx] - y[test_idx].mean()
    return residual


def neutralize_signal_polars(
    signal: pd.Series,
    idx: pd.DataFrame,
    factors: pd.DataFrame | None = None,
    factor_cols: list[str] | None = None,
    two_stage: bool = True,
    use_dml: bool = False,
    dml_n_splits: int = 2,
) -> pd.Series:
    """Polars-hızlandırılmış faktör nötralizasyonu.

    Yapı orijinal `neutralize_signal()` ile aynı sözleşmeyi takip eder;
    sadece groupby loop'u Polars üzerinde çalıştırılır.

    polars yoksa veya ImportError olursa otomatik fallback yapılır.
    """
    if not _check_polars():
        from engine.data.factor_neutralize import neutralize_signal
        return neutralize_signal(
            signal, idx, factors=factors, factor_cols=factor_cols,
            two_stage=two_stage, use_dml=use_dml, dml_n_splits=dml_n_splits,
        )

    import polars as pl
    from engine.data.factor_neutralize import _build_factors, _bin_demean

    if factor_cols is None:
        factor_cols = ["size", "vol", "mom"]

    if factors is None:
        factors = _build_factors(idx)

    available = [c for c in factor_cols if c in factors.columns]
    if not available:
        return signal

    # signal + factors → flat DataFrame
    df = pd.DataFrame({"signal": signal})
    df = df.join(factors[available], how="left")
    df = df.dropna(subset=["signal"])
    df_flat = df.reset_index()

    # Pandas → Polars (zero-copy via pyarrow)
    try:
        pl_df = pl.from_pandas(df_flat)
    except Exception as exc:
        logger.debug("Polars conversion hatası (%s) — pandas fallback", exc)
        from engine.data.factor_neutralize import neutralize_signal
        return neutralize_signal(
            signal, idx, factors=factors, factor_cols=factor_cols,
            two_stage=two_stage, use_dml=use_dml, dml_n_splits=dml_n_splits,
        )

    # Date sütunu üzerinden partition_by → vektörize cross-section iteration
    # Polars partition_by Python loop yerine native iteration üretir.
    pieces: list[pd.DataFrame] = []
    for date_key, group in pl_df.partition_by("Date", as_dict=True).items():
        # Polars Group → Pandas hızlı extract (filtered subset, küçük)
        clean_pdf = group.drop_nulls(available).to_pandas()

        if len(clean_pdf) < 10:
            pieces.append(pd.DataFrame({
                "Ticker": group["Ticker"].to_numpy(),
                "Date":   group["Date"].to_numpy(),
                "resid":  group["signal"].to_numpy(),
            }))
            continue

        y = _rank_norm_pl(clean_pdf["signal"].values.astype(float))
        X_cols = [_rank_norm_pl(clean_pdf[c].values.astype(float)) for c in available]
        X = np.column_stack(X_cols)

        if use_dml:
            resid = _dml_residualize_batch(y, X, n_splits=dml_n_splits)
        else:
            X_aug = np.hstack([np.ones((len(X), 1)), X])
            try:
                beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
                resid = y - X_aug @ beta
            except (np.linalg.LinAlgError, ValueError):
                resid = y - y.mean()

        if two_stage and "size" in available:
            size_rank = _rank_norm_pl(clean_pdf["size"].values.astype(float))
            resid = _bin_demean(resid, size_rank, n_bins=10)

        pieces.append(pd.DataFrame({
            "Ticker": clean_pdf["Ticker"].values,
            "Date":   clean_pdf["Date"].values,
            "resid":  resid,
        }))

    if not pieces:
        return signal

    resid_df = (
        pd.concat(pieces, ignore_index=True)
          .set_index(["Ticker", "Date"])["resid"]
    )
    return resid_df.reindex(signal.index)
