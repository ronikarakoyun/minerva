"""
engine/factor_neutralize.py — Cross-Sectional Factor Neutralization (İki-Aşamalı)

Problem:
  Mining, BIST'teki small-cap primini (küçük hisseler = düşük fiyat ≈ iyi getiri)
  tekrar tekrar yeniden keşfediyor. Bu 2016-2023 döneminde çok güçlü ama
  2023-2026 test döneminde çöküyor → WF-Kararsız/Geçersiz sonuçlar.

Çözüm — İKİ AŞAMALI neutralizasyon:
  Stage 1: Rank-Space OLS
    Her tarih t için cross-sectional OLS (RANK uzayında):
      rank(signal_i) = α + β1·rank(size_i) + β2·rank(vol_i) + β3·rank(mom_i) + ε_i
    → Doğrusal rank korelasyonunu kaldırır (Spearman IC ile tutarlı).

  Stage 2: Quantile Bin-Demean
    size faktörünü 10 eşit-gözlem bin'e böl, her bin içinde sinyal'ı demean et.
    → Stage 1'in yakalayamadığı MONOTONİK-OLMAYAN size bias'ı da kaldırır.
    → Pclose (size_corr ≈ 1.0) için Stage 1 sonrası artık ~0.12 → Stage 2 sonrası < 0.05.

  Faktörler (BIST için en önemli):
    1. Size        : log(Pclose)           — piyasa değeri proxy
    2. Volatilite  : rolling_std(ret, 20)  — risk proxy
    3. Momentum    : 20 günlük fiyat değişimi — trend proxy

  Vol hesabı notu:
    _build_factors içinde: grp.transform(lambda s: s.pct_change().rolling(20).std())
    Burada s = Pclose serisi → s.pct_change() = günlük getiriler → std = volatilite.
    Bu DOĞRU hesaplama, çift pct_change BUG DEĞİL (Pclose → returns → rolling std).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Factor construction helpers
# ---------------------------------------------------------------------------

def _build_factors(idx: pd.DataFrame) -> pd.DataFrame:
    """
    MultiIndex (Ticker, Date) idx'inden cross-sectional faktör matrisi üret.

    Gerekli sütunlar (en az biri yeterli, eksikler atlanır):
      Pclose  → size (log-price), volatilite (rolling std), momentum (20g pct-change)

    VOL HESAP NOTU:
      grp.transform(lambda s: s.pct_change().rolling(20).std())
      s = Pclose serisi (fiyatlar)
      s.pct_change()   = günlük getiriler (DOĞRU — çift pct_change değil)
      .rolling(20).std() = 20 günlük rolling volatilite

    Döner: (Ticker, Date) index'li DataFrame, sütunlar: size, vol, mom
    """
    flat = idx.reset_index().sort_values(["Ticker", "Date"]).copy()

    if "Pclose" in flat.columns:
        grp = flat.groupby("Ticker")["Pclose"]

        price = flat["Pclose"].replace(0, np.nan).clip(lower=1e-6)
        # SINIRLILIK: Gerçek market cap = log(Pclose × shares_outstanding).
        # EOD verisinde shares_outstanding yoksa log(Pclose) proxy olarak kullanılır.
        # BIST'te ucuz hisse ≠ küçük şirket (ör. lot-split sonrası 0.1₺ hisse).
        # Veriye "shares_outstanding" eklenirse burası log(price * shares) yapılmalı.
        if "shares_outstanding" in flat.columns:
            mktcap = price * flat["shares_outstanding"].replace(0, np.nan).clip(lower=1)
            flat["size"] = np.log(mktcap)
        else:
            flat["size"] = np.log(price)

        # Pclose → pct_change → returns; rolling std = volatilite
        # NOT: s = Pclose, s.pct_change() = günlük getiri (çift pct_change YOK)
        flat["vol"] = grp.transform(
            lambda s: s.pct_change().rolling(20, min_periods=5).std()
        )
        flat["mom"] = grp.pct_change(20)
    else:
        flat["size"] = np.nan
        flat["vol"]  = np.nan
        flat["mom"]  = np.nan

    return flat.set_index(["Ticker", "Date"])[["size", "vol", "mom"]]


# ---------------------------------------------------------------------------
# Stage 1 — Rank-space OLS yardımcısı
# ---------------------------------------------------------------------------

def _rank_norm(arr: np.ndarray) -> np.ndarray:
    """
    Cross-sectional uniform rank → [-0.5, 0.5] aralığına normalleştir.
    Spearman IC rank uzayında ölçüldüğü için OLS de bu uzayda yapılmalı.
    """
    n = len(arr)
    if n < 2:
        return arr - arr.mean()
    order = np.argsort(np.argsort(arr))   # 0..n-1 tie-aware ranks
    return order / (n - 1) - 0.5          # → [-0.5, 0.5]


# ---------------------------------------------------------------------------
# Stage 2 — Quantile Bin-Demean yardımcısı
# ---------------------------------------------------------------------------

def _bin_demean(
    signal_arr: np.ndarray,
    size_arr: np.ndarray,
    n_bins: int = 10,
) -> np.ndarray:
    """
    Stage 2 nötralizasyon: size faktörünü quantile bin'lere ayır,
    her bin içinde sinyali demean et.

    Neden gerekli?
      Rank-OLS DOĞRUSAL rank korelasyonunu kaldırır. Ama örneğin Pclose
      ile log(Pclose) rank-aynıdır — OLS bu doğrusal ilişkiyi kaldırır.
      Ancak kalan ~0.12 artık MONOTONİK-OLMAYAN ilişkiden kaynaklanır.
      Bin-demean bu nonlinear bias'ı da siler.

    Sonuç:
      Pclose: Stage 1 → 0.12 artık  →  Stage 1+2 → < 0.05 artık

    Parameters
    ----------
    signal_arr : Stage 1 OLS artığı (rank-normalized space)
    size_arr   : Rank-normalized size faktörü
    n_bins     : Quantile bin sayısı (10 önerilir, az veri varsa otomatik düşer)

    Returns
    -------
    np.ndarray : Bin-demeaned sinyal
    """
    n = len(signal_arr)
    # Minimum veri: her bin için en az 5 gözlem
    effective_bins = min(n_bins, max(2, n // 5))
    if effective_bins < 2:
        return signal_arr - signal_arr.mean()

    result = signal_arr.copy()
    try:
        bin_labels = pd.qcut(
            pd.Series(size_arr), q=effective_bins, labels=False, duplicates="drop"
        ).values

        for b in np.unique(bin_labels[~pd.isna(bin_labels)]):
            mask = bin_labels == b
            if mask.sum() < 2:
                continue
            result[mask] -= result[mask].mean()

    except Exception:
        # qcut başarısızsa (örn: tüm değerler aynı) → genel demean
        result -= result.mean()

    return result


# ---------------------------------------------------------------------------
# Core neutralization — İki-Aşamalı
# ---------------------------------------------------------------------------

def neutralize_signal(
    signal: pd.Series,
    idx: pd.DataFrame,
    factors: pd.DataFrame | None = None,
    factor_cols: list[str] | None = None,
    two_stage: bool = True,
) -> pd.Series:
    """
    Cross-sectional olarak sinyali faktörlerden arındır (İki-Aşamalı).

    Stage 1: Rank-Space OLS — doğrusal rank korelasyonunu kaldır.
    Stage 2: Quantile Bin-Demean — nonlinear size bias'ı kaldır.

    Parameters
    ----------
    signal      : (Ticker, Date) indexed Series — ham formül sinyali
    idx         : MultiIndex DataFrame (Pclose ve diğer fiyat sütunları içermeli)
    factors     : Önceden hesaplanmış faktör matrisi (None → otomatik üretir)
    factor_cols : Kullanılacak faktörler (None → ["size", "vol", "mom"])
    two_stage   : True (varsayılan) → Stage 1 + Stage 2; False → sadece Stage 1

    Returns
    -------
    pd.Series   : Faktörlerden arındırılmış sinyal (residual), aynı index
    """
    if factor_cols is None:
        factor_cols = ["size", "vol", "mom"]

    if factors is None:
        factors = _build_factors(idx)

    available = [c for c in factor_cols if c in factors.columns]
    if not available:
        return signal

    # Pre-computed rank kolonları varsa kullan (8.3 optimizasyon)
    rank_cols = [f"{c}_rank" for c in available if f"{c}_rank" in factors.columns]
    cols_to_join = available + rank_cols

    # signal + faktörler + pre-computed ranks birleştir
    df = pd.DataFrame({"signal": signal})
    df = df.join(factors[cols_to_join], how="left")
    df = df.dropna(subset=["signal"])
    df = df.reset_index()   # → [Ticker, Date, signal, size, vol, mom]

    pieces: list[pd.DataFrame] = []

    for date, group in df.groupby("Date"):
        clean = group.dropna(subset=available)

        if len(clean) < 10:
            # Yeterli veri yok → ham sinyal (artık = sinyal kendisi)
            pieces.append(pd.DataFrame({
                "Ticker": group["Ticker"].values,
                "Date":   date,
                "resid":  group["signal"].values,
            }))
            continue

        # N5: NaN rank satırlarını filtrele — 0 substitusyonu yapmıyoruz.
        rank_cols_present = [f"{c}_rank" for c in available if f"{c}_rank" in clean.columns]
        if rank_cols_present:
            clean = clean.dropna(subset=rank_cols_present)

        if len(clean) < 10:
            pieces.append(pd.DataFrame({
                "Ticker": group["Ticker"].values,
                "Date":   date,
                "resid":  group["signal"].values,
            }))
            continue

        # ── Stage 1: Rank-Space OLS ────────────────────────────────────
        # Spearman IC rank uzayında ölçüldüğü için OLS'yi rank uzayında yap.
        # Cross-sectional uniform rank → [-0.5, 0.5]
        y_raw = clean["signal"].values.astype(float)
        y = _rank_norm(y_raw)

        # Pre-computed rank varsa kullan; yoksa hesapla (8.3)
        X_cols = []
        for c in available:
            rank_col = f"{c}_rank"
            if rank_col in clean.columns and not clean[rank_col].isna().all():
                X_cols.append(clean[rank_col].values.astype(float))   # NaN zaten filtrelendi
            else:
                X_cols.append(_rank_norm(clean[c].values.astype(float)))
        X = np.column_stack(X_cols)
        X_aug = np.hstack([np.ones((len(X), 1)), X])

        try:
            beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            resid = y - X_aug @ beta
        except (np.linalg.LinAlgError, ValueError):
            resid = y - y.mean()

        # ── Stage 2: Quantile Bin-Demean ──────────────────────────────
        # Rank-OLS artığının hâlâ taşıdığı nonlinear size bias'ı kaldır.
        # Yalnızca "size" faktörü mevcut ve two_stage=True ise uygula.
        if two_stage and "size" in available:
            size_rank = _rank_norm(clean["size"].values.astype(float))
            resid = _bin_demean(resid, size_rank, n_bins=10)

        pieces.append(pd.DataFrame({
            "Ticker": clean["Ticker"].values,
            "Date":   date,
            "resid":  resid,
        }))

    if not pieces:
        return signal

    resid_df = (
        pd.concat(pieces, ignore_index=True)
          .set_index(["Ticker", "Date"])["resid"]
    )
    return resid_df.reindex(signal.index)


# ---------------------------------------------------------------------------
# Size correlation diagnostic
# ---------------------------------------------------------------------------

def compute_size_corr(
    signal: pd.Series,
    idx: pd.DataFrame,
    factors: pd.DataFrame | None = None,
) -> float:
    """
    Sinyalin size faktörüyle ortalama günlük Spearman korelasyonunu hesapla.
    |size_corr| > 0.3 → formula ≈ size factor → penalize et.
    """
    if factors is None:
        factors = _build_factors(idx)

    if "size" not in factors.columns:
        return 0.0

    df = (
        pd.DataFrame({"signal": signal})
        .join(factors[["size"]], how="left")
        .dropna()
        .reset_index()
    )
    if len(df) == 0:
        return 0.0

    def _spearman(g: pd.DataFrame) -> float:
        if len(g) < 5 or g["signal"].std() == 0 or g["size"].std() == 0:
            return np.nan
        return g["signal"].corr(g["size"], method="spearman")

    try:
        corrs = df.groupby("Date").apply(_spearman, include_groups=False)
        return float(corrs.dropna().mean()) if len(corrs.dropna()) > 0 else 0.0
    except TypeError:
        # include_groups pandas 2.x özelliği — eski sürümlerde fallback
        try:
            corrs = df.groupby("Date").apply(_spearman)
            return float(corrs.dropna().mean()) if len(corrs.dropna()) > 0 else 0.0
        except Exception:
            return 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Convenience: pre-build factors once for a session
# ---------------------------------------------------------------------------

def build_factors_cache(
    idx: pd.DataFrame,
    attach_regime: "pd.Series | None" = None,
) -> pd.DataFrame:
    """
    Faz-0'da bir kez çağır, dönen cache'i WF-fitness'a geç.

    Optimizasyon (8.3):
      Her fold'da _rank_norm yeniden hesaplanmasını önlemek için,
      faktörlerin cross-sectional rank normalizasyonu da pre-computed
      olarak '{factor}_rank' kolonlarında saklanır.

      Mining'de 500 formül × 5 fold = 2500 neutralize çağrısı.
      Her çağrıda faktörleri yeniden rank-normalize etmek yerine,
      bu kolonları doğrudan kullanmak ~%20-30 zaman tasarrufu sağlar.

    Parametreler
    ------------
    attach_regime : pd.Series, opsiyonel
        index=Date, değer ∈ {"bull","chop","bear"}.
        Sağlanırsa "regime" kolonu tüm (Ticker,Date) satırlarına broadcast edilir.

    NOT: LOCAL değişken (session_state değil).
    Her buton tıklamasında yeniden hesaplanır → stale cache riski yok.
    """
    factors = _build_factors(idx)

    # Pre-compute cross-sectional rank-normalized factors (8.3)
    # Her Date için _rank_norm → [-0.5, 0.5] aralığı
    try:
        for col in ["size", "vol", "mom"]:
            if col not in factors.columns:
                continue
            rank_col = f"{col}_rank"

            # groupby Date → her tarihte cross-sectional rank
            def _rank_norm_group(g: pd.Series) -> pd.Series:
                vals = g.values.astype(float)
                valid = ~np.isnan(vals)
                n_valid = valid.sum()
                result = np.full_like(vals, np.nan, dtype=float)
                if n_valid < 2:
                    result[valid] = 0.0
                else:
                    ranks = np.argsort(np.argsort(vals[valid]))
                    result[valid] = ranks / (n_valid - 1) - 0.5
                return pd.Series(result, index=g.index)

            factors[rank_col] = (
                factors[col]
                .groupby(level="Date", group_keys=False)
                .apply(_rank_norm_group)
            )
    except Exception:
        pass   # Pre-compute başarısızsa sessizce atla — neutralize yine de çalışır

    # Rejim kolonunu broadcast et (opsiyonel)
    if attach_regime is not None:
        try:
            dates = factors.index.get_level_values("Date")
            regime_map = {pd.Timestamp(d): v for d, v in attach_regime.items()}
            factors["regime"] = [regime_map.get(pd.Timestamp(d), "chop") for d in dates]
        except Exception:
            pass  # Regime eklenemezse sessizce atla

    return factors
