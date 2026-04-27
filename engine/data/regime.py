"""
engine/regime.py — Piyasa rejimi tespiti.

Benchmark (örn. BIST100) fiyat serisinden günlük 3-state rejim etiketi üretir:
  "bull"  : Trend yukarı  + volatilite düşük
  "chop"  : Trend yatay   + volatilite yüksek veya yatay
  "bear"  : Trend aşağı   + volatilite yüksek

Kural tabanlı (parametre-önerilen):
  trend  = bench.pct_change(trend_window)   # trend_window günlük momentum
  vol    = ret.rolling(vol_window).std()    # realized volatilite
  bull   : trend > 0  AND vol < vol_hi
  bear   : trend < 0  AND vol >= vol_hi
  chop   : aksi halde

Entegrasyon:
  - build_factors_cache (factor_neutralize.py): attach_regime= argümanıyla
    factor cache'e "regime" kolonu eklenir.
  - compute_wf_fitness (wf_fitness.py): regime= argümanıyla per-regime IC breakdown.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = ("bull", "chop", "bear")
REGIME_DTYPE = pd.CategoricalDtype(categories=list(REGIMES), ordered=False)


def compute_regime(
    bench: pd.Series,
    vol_window: int = 20,
    trend_window: int = 60,
    vol_hi: float = 0.022,
) -> pd.Series:
    """
    Benchmark fiyat serisinden günlük rejim etiketi hesapla.

    Parametreler
    ------------
    bench : pd.Series
        Benchmark kapanış fiyatları (Date index'li, monoton artmalı tarihlere gerek yok).
    vol_window : int
        Gerçekleşmiş volatilite penceresi (iş günü, varsayılan 20).
    trend_window : int
        Momentum / trend penceresi (iş günü, varsayılan 60).
    vol_hi : float
        Yüksek volatilite eşiği — günlük getiri std (varsayılan 0.022 ≈ %2.2).

    Döner
    ------
    pd.Series
        index = Date (bench.index ile aynı), değer ∈ {"bull","chop","bear"}.
        İlk trend_window+vol_window günü NaN olabilir → "chop" ile doldurulur.
    """
    bench = bench.copy()
    bench.index = pd.to_datetime(bench.index)
    bench = bench.sort_index()

    # Günlük getiri
    ret = bench.pct_change()

    # Trend: trend_window-gün fiyat değişimi
    trend = bench.pct_change(trend_window)

    # Realized volatilite (rolling std)
    vol = ret.rolling(vol_window).std()

    # Rejim etiketleri
    regime = pd.Series("chop", index=bench.index, dtype=str)

    bull_mask = (trend > 0) & (vol < vol_hi)
    bear_mask = (trend < 0) & (vol >= vol_hi)

    regime[bull_mask] = "bull"
    regime[bear_mask] = "bear"

    # İlk warm-up dönemi (eksik trend/vol) → "chop" (varsayılan, zaten set)
    regime = regime.where(trend.notna() & vol.notna(), "chop")

    return regime


def attach_regime_to_index(
    idx: pd.DataFrame,
    regime: pd.Series,
) -> pd.DataFrame:
    """
    (Ticker, Date) MultiIndex idx'e rejim kolonunu ekle.

    Parametreler
    ------------
    idx : pd.DataFrame
        MultiIndex (Ticker, Date) üzerinde WF-fitness veya factor_cache DataFrame'i.
    regime : pd.Series
        index=Date, değer ∈ {"bull","chop","bear"}.

    Döner
    ------
    pd.DataFrame
        idx + "regime" kolonu (sağ taraf birleşim, eksik günler "chop").
    """
    dates = idx.index.get_level_values("Date")
    date_to_regime = regime.to_dict()
    regime_vals = [date_to_regime.get(pd.Timestamp(d), "chop") for d in dates]
    result = idx.copy()
    result["regime"] = regime_vals
    return result


def regime_breakdown(
    signal: pd.Series,
    target: pd.Series,
    regime: pd.Series,
    method: str = "spearman",
) -> dict:
    """
    Sinyal-hedef korelasyonunu (IC) rejim bazında ayrıştır.

    Parametreler
    ------------
    signal : pd.Series
        MultiIndex (Ticker, Date) veya Date index'li sinyal.
    target : pd.Series
        Hedef getiri (signal ile aynı index).
    regime : pd.Series
        index=Date, değer ∈ {"bull","chop","bear"}.
    method : str
        "spearman" veya "pearson".

    Döner
    ------
    dict: {"bull": float, "chop": float, "bear": float}
        Her rejim için ortalama cross-sectional IC (NaN → eksik veri).
    """
    # Date seviyesine indir
    if isinstance(signal.index, pd.MultiIndex):
        dates = signal.index.get_level_values("Date")
    else:
        dates = signal.index

    tmp = pd.DataFrame({
        "Date":   pd.to_datetime(dates),
        "Signal": signal.values,
        "Target": target.values,
    }).dropna()

    # Her güne rejim ata
    date_to_regime = regime.to_dict()
    tmp["regime"] = tmp["Date"].map(lambda d: date_to_regime.get(d, "chop"))

    def _ic_on_group(g: pd.DataFrame) -> float:
        if len(g) < 5 or g["Signal"].std() == 0:
            return np.nan
        return float(g["Signal"].corr(g["Target"], method=method))

    result = {}
    for r in REGIMES:
        sub = tmp[tmp["regime"] == r]
        if len(sub) == 0:
            result[r] = np.nan
            continue
        day_ics = sub.groupby("Date").apply(_ic_on_group).dropna()
        result[r] = float(day_ics.mean()) if len(day_ics) > 0 else np.nan

    return result
