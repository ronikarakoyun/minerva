"""
engine/meta_label.py — Meta-Labeling ikincil modeli.

López de Prado, AFML §3.6: Primary model (alfa sinyali) "ne zaman / yön"
söyler; secondary (meta) model "bet size / skip" söyler.

İş akışı:
  1. Primary: mevcut formül sinyali → Top-K adaylar belirlenir.
  2. build_meta_dataset(): her (Ticker, Date) için feature matrisi oluştur.
     Features: sig_rank, size_rank, vol_rank, regime_dummy, recent_ic.
  3. train_meta_model(): logistic regression, TimeSeries aware split.
  4. MetaModel.predict_proba(): proba = P(TB_Label=1 | aday).
  5. apply_meta_filter(): proba < threshold → sinyal NaN (skip).

MVI yaklaşımı (sinyal maskeleme): `run_pro_backtest` imzası değişmez,
sadece düşük güvenli günlerde sinyal NaN olur → Top-K seçimde atlanır.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


@dataclass
class MetaModel:
    """Eğitilmiş meta-label modeli."""
    model: object = None
    scaler: object = None
    feature_cols: list = field(default_factory=list)
    train_end: Optional[pd.Timestamp] = None
    auc: float = float("nan")
    fit_failed: bool = False
    fail_reason: str = ""

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        """
        df: (Ticker, Date) MultiIndex — feature_cols mevcut olmalı.
        Döner: P(label=1) Series, aynı index.
        """
        if self.fit_failed or self.model is None:
            return pd.Series(0.5, index=df.index)  # fallback: belirsiz

        try:
            X = df[self.feature_cols].fillna(0.0).values
            X = self.scaler.transform(X)
            proba = self.model.predict_proba(X)[:, 1]
            return pd.Series(proba, index=df.index)
        except Exception:
            return pd.Series(0.5, index=df.index)


def build_meta_dataset(
    signal: pd.Series,
    idx: pd.DataFrame,
    factors: Optional[pd.DataFrame] = None,
    regime: Optional[pd.Series] = None,
    rolling_ic_window: int = 20,
    target_col: str = "TB_Label",
) -> pd.DataFrame:
    """
    Meta-label feature matrisi oluştur.

    Parametreler
    ------------
    signal : pd.Series
        MultiIndex (Ticker, Date) — formül sinyali.
    idx : pd.DataFrame
        MultiIndex (Ticker, Date) — Next_Ret/TB_Label + market columns.
    factors : pd.DataFrame, opsiyonel
        build_factors_cache() çıktısı (size, vol, mom kolonları).
    regime : pd.Series, opsiyonel
        index=Date, değer ∈ {"bull","chop","bear"}.
    rolling_ic_window : int
        recent_ic için kullanılacak rolling pencere (iş günü).
    target_col : str
        "TB_Label" (önerilir) veya "Next_Ret" sign-based (≥0 → 1).

    Döner
    ------
    pd.DataFrame
        index = (Ticker, Date) MultiIndex
        Kolonlar: sig_rank, size_rank (varsa), vol_rank (varsa),
                  regime_bull, regime_bear (varsa), recent_ic, label (y).
    """
    # Sinyal rank (cross-sectional, per Date)
    sig_df = signal.rename("signal").reset_index()
    sig_df.columns = ["Ticker", "Date", "signal"]
    sig_df["sig_rank"] = sig_df.groupby("Date")["signal"].rank(pct=True)

    # Hedef etiket
    if target_col not in idx.columns:
        target_col = "Next_Ret"
    target_series = idx[target_col].copy()
    if target_col == "Next_Ret":
        # Sürekli getiriden binary etiket: ≥ 0 → 1
        label = (target_series >= 0).astype(float)
    else:
        # TB_Label: {-1, 0, 1} → 1 = pozitif; 0,-1 = negatif
        label = (target_series > 0).astype(float)

    # Feature DataFrame'i başlat
    feat = sig_df.set_index(["Ticker", "Date"])[["sig_rank"]].copy()
    feat = feat.reindex(idx.index)
    feat["label"] = label.values

    # Faktör rankları (size, vol, mom)
    if factors is not None:
        for col in ["size_rank", "vol_rank", "mom_rank"]:
            if col in factors.columns:
                feat[col] = factors[col].reindex(feat.index)
            elif col.replace("_rank", "") in factors.columns:
                # Rank kolonu yoksa ham faktörden cross-sectional rank hesapla
                raw = factors[col.replace("_rank", "")].reindex(feat.index)
                feat[col] = raw.groupby(level="Date").rank(pct=True)

    # Rejim dummy'leri (bull=1, bear=1, chop=reference)
    if regime is not None:
        dates = feat.index.get_level_values("Date")
        reg_map = {pd.Timestamp(d): v for d, v in regime.items()}
        reg_vals = pd.Series([reg_map.get(pd.Timestamp(d), "chop") for d in dates],
                             index=feat.index)
        feat["regime_bull"] = (reg_vals == "bull").astype(float)
        feat["regime_bear"] = (reg_vals == "bear").astype(float)

    # Rolling IC (son rolling_ic_window günün ortalama cross-sectional IC)
    try:
        tmp_ic = sig_df.copy()
        tmp_ic = tmp_ic.merge(
            target_series.reset_index().rename(columns={target_col: "target"}),
            on=["Ticker", "Date"], how="inner",
        )
        # Her günün cross-sectional IC
        daily_ic = (
            tmp_ic.dropna(subset=["signal", "target"])
            .groupby("Date")
            .apply(lambda g: g["signal"].corr(g["target"], method="spearman"))
            .rename("daily_ic")
        )
        rolling_ic = daily_ic.rolling(rolling_ic_window, min_periods=5).mean()
        # Tüm satırlara broadcast
        dates_idx = feat.index.get_level_values("Date")
        feat["recent_ic"] = [
            float(rolling_ic.get(d, 0.0) or 0.0) for d in pd.to_datetime(dates_idx)
        ]
    except Exception:
        feat["recent_ic"] = 0.0

    return feat.dropna(subset=["sig_rank", "label"])


def train_meta_model(
    ds: pd.DataFrame,
    train_end: pd.Timestamp,
    model_type: str = "logit",
    max_iter: int = 500,
) -> MetaModel:
    """
    Meta-label modeli eğit (TimeSeries split — train_end'den sonrası test).

    Parametreler
    ------------
    ds : pd.DataFrame
        build_meta_dataset() çıktısı — "label" kolonu gerekli.
    train_end : pd.Timestamp
        Bu tarihten ÖNCE train, sonrası asla fit'e dahil edilmez.
    model_type : str
        "logit" (LogisticRegression) — genişletilebilir.
    max_iter : int
        Logistic regression max_iter.

    Döner
    ------
    MetaModel
        .fit_failed=True ise model eğitilemedi (tek sınıf, sklearn yok, vb.)
    """
    if not _SKLEARN_AVAILABLE:
        return MetaModel(fit_failed=True, fail_reason="sklearn yüklü değil")

    dates = pd.to_datetime(ds.index.get_level_values("Date"))
    train_mask = dates < pd.Timestamp(train_end)
    train_ds = ds[train_mask]

    if len(train_ds) < 20:
        return MetaModel(fit_failed=True, fail_reason="Yeterli train verisi yok (<20)")

    # Feature kolonları
    feature_cols = [c for c in ds.columns if c != "label"]
    X_train = train_ds[feature_cols].fillna(0.0).values
    y_train = train_ds["label"].values

    # Tek sınıf kontrolü
    unique_classes = np.unique(y_train)
    if len(unique_classes) < 2:
        return MetaModel(
            fit_failed=True,
            fail_reason=f"Train'de tek sınıf: {unique_classes.tolist()}",
            feature_cols=feature_cols,
            train_end=pd.Timestamp(train_end),
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    clf = LogisticRegression(max_iter=max_iter, random_state=42, class_weight="balanced")
    try:
        clf.fit(X_scaled, y_train)
    except Exception as e:
        return MetaModel(fit_failed=True, fail_reason=str(e))

    # AUC — test kümesi (train_end sonrası)
    auc = float("nan")
    test_ds = ds[~train_mask]
    if len(test_ds) >= 10:
        X_test = test_ds[feature_cols].fillna(0.0).values
        y_test = test_ds["label"].values
        if len(np.unique(y_test)) == 2:
            try:
                X_test_s = scaler.transform(X_test)
                proba_test = clf.predict_proba(X_test_s)[:, 1]
                auc = float(roc_auc_score(y_test, proba_test))
            except Exception:
                pass

    return MetaModel(
        model=clf,
        scaler=scaler,
        feature_cols=feature_cols,
        train_end=pd.Timestamp(train_end),
        auc=auc,
        fit_failed=False,
    )


def apply_meta_filter(
    signal: pd.Series,
    proba: pd.Series,
    threshold: float = 0.55,
) -> pd.Series:
    """
    Meta-label eşiği altındaki sinyal değerlerini NaN ile maskele.

    Parametreler
    ------------
    signal : pd.Series
        Formül sinyali (MultiIndex veya Date index).
    proba : pd.Series
        P(label=1) — signal ile aynı index.
    threshold : float
        Bu değerin altındaki günler skip edilir (NaN).

    Döner
    ------
    pd.Series — signal ile aynı index, düşük güvenli pozisyonlar NaN.
    """
    filtered = signal.copy()
    aligned_proba = proba.reindex(filtered.index)
    # proba < threshold veya NaN → skip
    low_confidence = aligned_proba.isna() | (aligned_proba < threshold)
    filtered[low_confidence] = np.nan
    return filtered
