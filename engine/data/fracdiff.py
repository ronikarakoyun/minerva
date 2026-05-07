"""Fractional Differentiation (López de Prado, AFML §5) + Cache Yönetimi.

find_min_d() ADF tabanlı grid arama (500+ hisse için pahalı; ~50ms/seri).
Cache (data/fracdiff_d.json) sayesinde günlük çağrıda hesaplama yapılmaz.
Yeniden hesaplama kuralları (should_recompute_d):
  1. Takvim: computed_at > 90 gün önce
  2. Fiyat şoku: |today_close / ref_close - 1| > 0.50
  3. Yeni hisse: ticker cache'de yoksa
  4. Manuel: force=True

log(Pclose) gibi durağan-olmayan serileri MINIMUM bilgi kaybı ile durağanlaştırır.
Klasik birinci fark (d=1) tüm momentum hafızasını öldürür; kesirli d ∈ (0, 1)
hafızayı kısmen korurken ADF testini geçirebilir.

Kullanım:
    >>> from engine.data.fracdiff import frac_diff_ffd, find_min_d
    >>> d_star = find_min_d(np.log(df["Pclose"]))
    >>> df["LogPx_FFD"] = frac_diff_ffd(np.log(df["Pclose"]), d=d_star)

Performans: tek-pass O(n·k), k ~ 50-200 ağırlık (thresh=1e-4'te).
sliding_window_view ile sıfır-copy view; ek bellek = k * 8 byte.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

logger = logging.getLogger(__name__)

_CACHE_RECOMPUTE_DAYS = 90
_CACHE_PRICE_SHOCK_THRESHOLD = 0.50


def _ffd_weights(d: float, thresh: float) -> np.ndarray:
    """Sabit-pencereli FFD ağırlıkları; |w_k| < thresh olunca durur.

    Çıktı: en eski (k=K-1) → en yeni (k=0) sırasında, dot product için hazır.
    """
    w = [1.0]
    k = 1
    while True:
        w_k = -w[-1] * (d - k + 1) / k
        if abs(w_k) < thresh:
            break
        w.append(w_k)
        k += 1
        if k > 10_000:
            break
    return np.array(w[::-1], dtype=np.float64)


def frac_diff_ffd(series: pd.Series, d: float, thresh: float = 1e-4) -> pd.Series:
    """Sabit-pencere kesirli fark.

    Args:
        series: Girdi (örn. log fiyat). DatetimeIndex önerilir.
        d: Kesirli mertebe ∈ [0, 1]. d=0 → kimlik, d=1 → birinci fark.
        thresh: Ağırlık kesim eşiği. Küçük → daha geniş pencere (daha fazla hafıza).

    Returns:
        Aynı index'te pd.Series. İlk (window-1) satır NaN; pencerede NaN olan
        gözlemler de NaN olarak çıkar.
    """
    if not 0.0 <= d <= 1.0:
        raise ValueError(f"d must be in [0, 1], got {d}")

    w = _ffd_weights(d, thresh)
    width = len(w)
    arr = series.to_numpy(dtype=np.float64)
    n = arr.size
    out_name = f"{series.name}_ffd_d{d:.2f}" if series.name is not None else f"ffd_d{d:.2f}"
    out = np.full(n, np.nan)
    if n < width:
        return pd.Series(out, index=series.index, name=out_name)

    windows = sliding_window_view(arr, window_shape=width)
    valid = ~np.isnan(windows).any(axis=1)
    if valid.any():
        out[width - 1:][valid] = windows[valid] @ w
    return pd.Series(out, index=series.index, name=out_name)


def load_fracdiff_cache(path: Path | str | None = None) -> dict[str, Any]:
    """fracdiff_d.json'ı yükle; yoksa boş dict döner."""
    p = Path(path or "data/fracdiff_d.json")
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("fracdiff cache okunamadı (%s): %s", p, e)
        return {}


def save_fracdiff_cache(cache: dict[str, Any], path: Path | str | None = None) -> None:
    """fracdiff_d.json'ı atomik olarak kaydet."""
    p = Path(path or "data/fracdiff_d.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=2, ensure_ascii=False))
    tmp.replace(p)


def should_recompute_d(
    ticker: str,
    cache: dict[str, Any],
    today_close: float | None = None,
    force: bool = False,
) -> bool:
    """Cache girişine göre d_star'ın yeniden hesaplanması gerekip gerekmediğine karar ver.

    Kurallar (herhangi biri True → recompute):
      1. force=True
      2. ticker cache'de yoksa
      3. computed_at > 90 gün önce (takvim periyodu)
      4. |today_close / ref_close - 1| > 0.50 (fiyat şoku)
    """
    if force:
        return True
    entry = cache.get(ticker)
    if not entry:
        return True
    computed_at_str = entry.get("computed_at")
    if not computed_at_str:
        # computed_at alanı yoksa (eski cache formatı veya bozuk entry) → yeniden hesapla
        return True
    try:
        computed_at = datetime.fromisoformat(computed_at_str)
        if computed_at.tzinfo is None:
            computed_at = computed_at.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - computed_at > timedelta(days=_CACHE_RECOMPUTE_DAYS):
            return True
    except ValueError:
        return True
    ref_close = entry.get("ref_close")
    if today_close is not None and ref_close and ref_close > 0:
        if abs(today_close / ref_close - 1) > _CACHE_PRICE_SHOCK_THRESHOLD:
            return True
    return False


def update_fracdiff_cache_entry(
    cache: dict[str, Any],
    ticker: str,
    series: pd.Series,
    today_close: float | None = None,
    p_value_target: float = 0.05,
) -> dict[str, Any]:
    """Tek ticker için d_star hesapla ve cache'e yaz; cache dict'ini döner."""
    d_star = find_min_d(series, p_value_target=p_value_target)
    cache[ticker] = {
        "d_star": d_star,
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "ref_close": float(today_close) if today_close is not None else None,
        "n_obs": int(len(series.dropna())),
    }
    logger.info("fracdiff cache güncellendi: %s d_star=%.2f", ticker, d_star)
    return cache


def find_min_d(
    series: pd.Series,
    p_value_target: float = 0.05,
    d_grid: np.ndarray | None = None,
    thresh: float = 1e-4,
) -> float:
    """ADF p-value < target sağlayan en küçük d değerini grid arar.

    d ∈ {0.00, 0.05, ..., 1.00}. Hiçbir d hedefi sağlamazsa 1.0 döner
    (klasik birinci fark — durağanlık garantili).

    NOT: ADF testi pahalıdır (~50ms / seri). 500+ hisselik evren için
    günlük çağırma yapma; haftalık/aylık batch'le `data/fracdiff_d.json`'a
    cache'le (bkz. Faz 1 plan, Cache Invalidation Kuralı).
    """
    from statsmodels.tsa.stattools import adfuller

    if d_grid is None:
        d_grid = np.arange(0.0, 1.05, 0.05)

    s_clean = series.dropna()
    for d in d_grid:
        diffed = frac_diff_ffd(s_clean, d=float(d), thresh=thresh).dropna()
        if len(diffed) < 30:
            continue
        try:
            p_val = adfuller(
                diffed.values, maxlag=1, regression="c", autolag=None
            )[1]
        except (ValueError, np.linalg.LinAlgError):
            continue
        if p_val < p_value_target:
            return float(d)
    return 1.0
