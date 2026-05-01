"""
engine/triple_barrier.py — Triple-Barrier Label (López de Prado, AFML Ch.3)

Klasik Next_Ret hedefinin sorunu:
  - Tek gün getirisi çok gürültülü
  - Asimetrik değil (küçük pozitif = büyük negatif kadar ağırlık alır)
  - Risk-adjusted alpha değil, raw return yakalar

Triple-Barrier:
  Bir hisse için t anında 3 bariyer koy:
    - Üst bariyer : +multiplier × σ_t  (long hedef)
    - Alt bariyer : -multiplier × σ_t  (stop-loss)
    - Zaman bariyeri: t + horizon gün (timeout)

  Etiket:
    +1 → üst bariyere önce çarptı (güçlü yükseliş)
    -1 → alt bariyere önce çarptı (güçlü düşüş)
     0 → timeout (zayıf/belirsiz hareket)

Bu etiketle IC hesaplarsak, formül "anlamlı hareketin yönünü" tahmin eder.
Küçük gürültülü günler 0 etiketi alır → signal/noise oranı yükselir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_triple_barrier_labels(
    db: pd.DataFrame,
    horizon: int = 10,
    multiplier: float = 1.5,
    vol_window: int = 20,
    min_vol: float = 0.005,
    long_only: bool = True,
    return_weights: bool = False,
) -> "pd.Series | tuple[pd.Series, pd.Series]":
    """
    Her (Ticker, Date) çifti için triple-barrier etiketi hesapla.

    Parameters
    ----------
    db         : market_db formatında DataFrame (Date, Ticker, Pclose sütunları)
    horizon    : maksimum bekleme süresi (gün)
    multiplier : bariyer çarpanı — σ × multiplier kadar yukarı/aşağı
    vol_window : volatilite hesabı için geriye bakan pencere
    min_vol    : minimum volatilite (çok sakin günlerde sıfır bölmeyi önler)
    long_only  : True → alt bariyere çarpan etiket -1 yerine 0 olur.
                 BIST'te BIST30 dışında short işlem yapılamadığı için
                 varsayılan True. IC artık sadece "üst bariyeri tahmin et"
                 üzerine kurulu olur (binary: 1=al, 0=bekle/kaçın).
    return_weights : N8 — True → (labels, sample_weights) tuple döner.
                 Son horizon gün eksik etiket alır (timeout=0.0) ama
                 sample_weight=0 → IC/model eğitimde bu satırlar yok sayılır.

    Returns
    -------
    pd.Series ya da tuple[pd.Series, pd.Series]
        return_weights=False (default): labels Series, index=(Ticker, Date)
        return_weights=True: (labels, weights) — weights ∈ {0.0, 1.0}
    """
    db = db.sort_values(["Ticker", "Date"]).copy()

    labels = []

    for ticker, grp in db.groupby("Ticker"):
        grp = grp.set_index("Date").sort_index()
        close = grp["Pclose"]
        dates = close.index

        if len(dates) < vol_window + horizon + 5:
            continue

        # Günlük log-return + rolling vol
        ret  = close.pct_change()
        vol  = ret.rolling(vol_window).std().bfill().clip(lower=min_vol)

        for i, t in enumerate(dates):
            if i < vol_window:
                continue
            # N8: Son horizon günleri artık atlanmıyor — sample_weight=0 ile dahil
            is_end_window = (i + horizon >= len(dates))
            if is_end_window:
                # Zaman bariyeri tamamlanamaz → timeout etiketi, sıfır ağırlık
                labels.append({"Ticker": ticker, "Date": t, "TB_Label": 0.0, "TB_Weight": 0.0})
                continue

            p0        = close.iloc[i]
            sigma_t   = vol.iloc[i]
            upper     = p0 * (1 + multiplier * sigma_t)
            lower_b   = p0 * (1 - multiplier * sigma_t)

            # Ufuk penceresi [t+1, t+horizon]
            future_prices = close.iloc[i + 1: i + horizon + 1]

            hit_upper = future_prices[future_prices >= upper]
            hit_lower = future_prices[future_prices <= lower_b]

            first_upper = hit_upper.index[0] if len(hit_upper) else None
            first_lower = hit_lower.index[0] if len(hit_lower) else None

            if first_upper is None and first_lower is None:
                label = 0.0       # timeout — belirsiz hareket
            elif first_upper is not None and first_lower is None:
                label = 1.0       # üst bariyere çarptı
            elif first_lower is not None and first_upper is None:
                label = 0.0 if long_only else -1.0   # alt bariyer
            else:
                # İkisi de çarptı → hangisi önce?
                if first_upper <= first_lower:
                    label = 1.0
                else:
                    label = 0.0 if long_only else -1.0

            labels.append({"Ticker": ticker, "Date": t, "TB_Label": label, "TB_Weight": 1.0})

    if not labels:
        if return_weights:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        return pd.Series(dtype=float)

    df_labels = pd.DataFrame(labels).set_index(["Ticker", "Date"]).sort_index()
    label_series = df_labels["TB_Label"]
    if return_weights:
        weight_series = df_labels["TB_Weight"]
        return label_series, weight_series
    return label_series


def add_triple_barrier_to_idx(
    idx: pd.DataFrame,
    horizon: int = 10,
    multiplier: float = 1.5,
    vol_window: int = 20,
    long_only: bool = True,
) -> pd.DataFrame:
    """
    Mevcut idx (MultiIndex Ticker/Date) DataFrame'ine TB_Label kolonu ekle.
    IC hesabında Next_Ret yerine TB_Label kullanılabilir.

    long_only=True (varsayılan): BIST gerçeği — BIST30 dışında short yok.
      Alt bariyer etiketleri 0 olur. IC binary: 1=güçlü yükseliş, 0=diğer.

    idx üretmek için:
        db_sorted = db.sort_values(["Ticker","Date"])
        idx = db_sorted.set_index(["Ticker","Date"])
    """
    # idx'i flat DataFrame'e çevir
    flat = idx.reset_index()
    if "Pclose" not in flat.columns:
        raise ValueError("idx'te Pclose kolonu bulunamadı.")

    labels = compute_triple_barrier_labels(
        flat[["Date", "Ticker", "Pclose"]],
        horizon=horizon,
        multiplier=multiplier,
        vol_window=vol_window,
        long_only=long_only,
    )

    # idx'e merge
    idx_out = idx.copy()
    idx_out["TB_Label"] = labels.reindex(idx_out.index)
    return idx_out


def label_stats(labels: pd.Series) -> dict:
    """
    Etiket dağılımını özetle (UI'da göstermek için).
    long_only modda -1 etiket yoktur; short satırı 0.0 olarak görünür.
    """
    vc = labels.value_counts(normalize=True)
    has_short = round(vc.get(-1.0, 0.0), 3) > 0
    d = {
        "buy    (+1)": round(vc.get(1.0, 0.0), 3),
        "flat    (0)": round(vc.get(0.0, 0.0), 3),
        "n_total":     int(len(labels.dropna())),
    }
    if has_short:
        d["short  (-1)"] = round(vc.get(-1.0, 0.0), 3)
    return d
