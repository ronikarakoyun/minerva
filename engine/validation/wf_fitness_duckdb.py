"""DuckDB-vektörize fitness IC hesaplaması.

Faz 1.2 (v4 vision plan): wf_fitness.py içindeki pandas
`groupby('Date').apply(spearman)` Python-level loop'u DuckDB'ye taşı.
DuckDB columnar vectorized engine ~10-30× hızlı.

Spearman rank correlation = ranklerin Pearson korelasyonu.
DuckDB'de rank() pencere fonksiyonu + corr() agregasyonu ile vektörize.

Kullanım:
    from engine.validation.wf_fitness_duckdb import (
        compute_per_date_rank_ic,
        compute_fold_rank_ic,
    )

    rank_ic = compute_per_date_rank_ic(tmp_df)            # toplam ortalama
    fold_ic = compute_fold_rank_ic(tmp_df, fold_dates)    # fold subset
"""
from __future__ import annotations

import logging
import threading
from typing import Iterable, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DUCKDB_AVAILABLE: Optional[bool] = None
_LOCK = threading.Lock()


def _check_duckdb():
    """duckdb varlığını lazy kontrol et (worker forklarında güvenli)."""
    global _DUCKDB_AVAILABLE
    if _DUCKDB_AVAILABLE is None:
        try:
            import duckdb  # noqa: F401
            _DUCKDB_AVAILABLE = True
        except ImportError:
            _DUCKDB_AVAILABLE = False
            logger.debug("duckdb yok — pandas fallback aktif")
    return _DUCKDB_AVAILABLE


def _get_conn():
    """Thread-local DuckDB bağlantısı (her thread kendi in-memory DB'si)."""
    import duckdb
    return duckdb.connect(":memory:")


def compute_per_date_rank_ic(tmp_df: pd.DataFrame,
                               method: str = "spearman") -> float:
    """Date bazlı ortalama RankIC.

    Parameters
    ----------
    tmp_df : pd.DataFrame
        Kolonlar: ["Date", "Signal", "Target"]
    method : "spearman" veya "pearson"

    Returns
    -------
    float : Tarihler arası ortalama IC. NaN olursa 0.0 döner.
    """
    if not _check_duckdb():
        return _pandas_fallback(tmp_df, method)

    if len(tmp_df) == 0:
        return 0.0

    try:
        import duckdb
        with _LOCK:
            con = duckdb.connect(":memory:")
            con.register("t", tmp_df)
            if method == "spearman":
                # Spearman = Pearson on ranks
                # PERCENT_RANK over Date partition normalize eder
                result = con.sql("""
                    SELECT AVG(ic) FROM (
                        SELECT
                            Date,
                            CORR(sig_rank, tgt_rank) AS ic
                        FROM (
                            SELECT
                                Date,
                                PERCENT_RANK() OVER (PARTITION BY Date ORDER BY Signal) AS sig_rank,
                                PERCENT_RANK() OVER (PARTITION BY Date ORDER BY Target) AS tgt_rank
                            FROM t
                        )
                        GROUP BY Date
                        HAVING COUNT(*) >= 3
                    )
                """).fetchone()
            else:  # pearson
                result = con.sql("""
                    SELECT AVG(ic) FROM (
                        SELECT Date, CORR(Signal, Target) AS ic
                        FROM t GROUP BY Date HAVING COUNT(*) >= 3
                    )
                """).fetchone()
            con.close()
        ic = result[0] if result and result[0] is not None else 0.0
        return float(ic) if not np.isnan(ic) else 0.0
    except Exception as exc:
        logger.warning("DuckDB IC hatası (%s) — pandas fallback", exc)
        return _pandas_fallback(tmp_df, method)


def compute_fold_rank_ic(tmp_df: pd.DataFrame,
                          fold_dates: Iterable,
                          method: str = "spearman") -> float:
    """Belirli tarihlerdeki ortalama RankIC (fold filtresi)."""
    if len(tmp_df) == 0:
        return float("nan")

    fold_set = set(pd.to_datetime(list(fold_dates)))
    sub = tmp_df[tmp_df["Date"].isin(fold_set)]
    if len(sub) < 20:
        return float("nan")
    return compute_per_date_rank_ic(sub, method=method)


def _pandas_fallback(tmp_df: pd.DataFrame, method: str) -> float:
    """Pandas fallback (eski yol — DuckDB yoksa)."""
    if len(tmp_df) == 0:
        return 0.0
    try:
        ics = tmp_df.groupby("Date").apply(
            lambda g: g["Signal"].corr(g["Target"], method=method)
            if g["Signal"].std() > 0 else 0.0,
            include_groups=False,
        )
        ic = ics.dropna().mean()
        return float(ic) if not np.isnan(ic) else 0.0
    except Exception:
        return 0.0
