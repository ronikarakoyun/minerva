"""Walk-Forward HMM Refitter — online rejim modeli güncelleme.

Mevcut HMM tek seferlik fit ediliyordu; piyasa dinamikleri değişince körleşiyordu.
Bu modül her N günde bir modeli yeniden eğitir, Hungarian algoritması ile
label switching'i düzeltir — böylece 'regime_0' her refit'te aynı ekonomik
anlamı korur.

Kullanım:
    refitter = WalkForwardHMMRefitter(cfg, refit_every_n_days=30)
    prob_df = refitter.fit_initial(df)           # ilk eğitim

    # Yeni günlük veri geldiğinde:
    prob_df_updated = refitter.update(df_extended)  # gerekirse refit
    is_stale = refitter.needs_refit(df_extended)    # kontrol

Hungarian (label switching) implementasyonu:
    align_regime_labels() + reorder_prob_df() — engine/data/regime_detector.py
    zaten mevcut (lines 419-513), burada doğrudan kullanılıyor.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RefitRecord:
    """Tek bir refit olayının kaydı."""
    refit_date: pd.Timestamp
    best_K: int
    bic: float
    perm: list[int]   # Hungarian permütasyonu — label switching düzeltmesi


class WalkForwardHMMRefitter:
    """Her N günde bir GaussianHMM'i yeniden eğiten walk-forward sarmalayıcı.

    Label Switching Koruması:
        Her refit'te `align_regime_labels()` (Hungarian) çağrılır.
        Yeni model'in state indeksleri eski model'in means'lerine göre hizalanır.
        Böylece 'regime_0' her refit'te en düşük volatilite / trend devamı
        anlamını koruyor (RegimeConfig'e bağlı yorumlanabilir özellikler).

    Parametreler
    ----------
    cfg : RegimeConfig
        HMM eğitim konfigürasyonu (K aralığı, covariance_type, vb.)
    refit_every_n_days : int
        Kaç takvim gününde bir yeniden eğitim yapılacağı (varsayılan: 30).
    min_history_days : int
        Refit için gereken minimum gözlem sayısı (varsayılan: 500).
        Daha az veri varsa refit atlanır, mevcut model kullanılır.
    """

    def __init__(
        self,
        cfg: "RegimeConfig",
        refit_every_n_days: int = 30,
        min_history_days: int = 500,
    ):
        from .regime_detector import RegimeConfig
        if not isinstance(cfg, RegimeConfig):
            raise TypeError(f"cfg bir RegimeConfig olmalı, gelen: {type(cfg)}")

        self.cfg = cfg
        self.refit_every_n_days = refit_every_n_days
        self.min_history_days = min_history_days

        self._model: Optional["GaussianHMM"] = None
        self._ref_means: Optional[np.ndarray] = None   # Hungarian referansı
        self._last_refit_date: Optional[pd.Timestamp] = None
        self._best_K: Optional[int] = None
        self.refit_history: list[RefitRecord] = []
        # ARCH-4 FIX: forward-filter sonucunu önbellekle — her gün yeniden
        # hesaplanmasın; yalnızca refit günü (Cuma update) yenilensin.
        self._cached_prob_df: Optional[pd.DataFrame] = None

    # ── Genel API ─────────────────────────────────────────────────────────

    def fit_initial(self, df: pd.DataFrame) -> pd.DataFrame:
        """İlk eğitim.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV verisi (Date index, Close/High/Low/Volume sütunları).

        Returns
        -------
        pd.DataFrame
            Forward-filtered olasılık vektörleri (Date × regime_K).
        """
        from .regime_detector import (
            compute_features,
            compute_filtered_probability_vector,
            fit_constrained_hmm,
        )

        features, _ = compute_features(df, self.cfg)
        model, best_K, _ = fit_constrained_hmm(features, self.cfg)

        self._model = model
        self._best_K = best_K
        self._ref_means = model.means_.copy()
        self._last_refit_date = df.index[-1] if hasattr(df.index[-1], 'date') else pd.Timestamp(df.index[-1])

        self.refit_history.append(RefitRecord(
            refit_date=self._last_refit_date,
            best_K=best_K,
            bic=float("nan"),
            perm=list(range(best_K)),
        ))
        logger.info(
            "WalkForwardHMMRefitter — ilk fit: K=%d  tarih=%s",
            best_K, self._last_refit_date.date(),
        )
        self._cached_prob_df = compute_filtered_probability_vector(self._model, features)
        return self._cached_prob_df

    def update(self, df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        """Yeni veriyle modeli güncelle; gerekirse yeniden eğit.

        Parameters
        ----------
        df : pd.DataFrame
            Tam (tarihsel + yeni) OHLCV verisi.
        force : bool
            True → refit periyodundan bağımsız olarak zorla yeniden eğit.

        Returns
        -------
        pd.DataFrame
            Güncel forward-filtered olasılık vektörleri.
        """
        if self._model is None:
            logger.info("İlk fit henüz yapılmamış — fit_initial() çağrılıyor.")
            return self.fit_initial(df)

        from .regime_detector import (
            compute_features,
            compute_filtered_probability_vector,
            fit_constrained_hmm,
            align_regime_labels,
            reorder_prob_df,
        )

        features, _ = compute_features(df, self.cfg)

        if force or self._should_refit(df):
            if len(features) < self.min_history_days:
                logger.warning(
                    "Refit atlandı: %d gözlem < min_history_days=%d",
                    len(features), self.min_history_days,
                )
            else:
                try:
                    new_model, best_K, candidates = fit_constrained_hmm(features, self.cfg)

                    # Hungarian: yeni state ID'lerini eski means'e hizala
                    perm = align_regime_labels(
                        new_model,
                        old_means=self._ref_means,
                    )

                    # Model ve referans means güncelle
                    self._model = new_model
                    self._best_K = best_K
                    self._ref_means = new_model.means_.copy()
                    last_date = df.index[-1]
                    self._last_refit_date = (
                        last_date if isinstance(last_date, pd.Timestamp)
                        else pd.Timestamp(last_date)
                    )

                    bic = candidates[best_K]["bic"] if best_K in candidates else float("nan")
                    self.refit_history.append(RefitRecord(
                        refit_date=self._last_refit_date,
                        best_K=best_K,
                        bic=bic,
                        perm=perm.tolist(),
                    ))
                    logger.info(
                        "HMM refit: K=%d  BIC=%.1f  perm=%s  tarih=%s",
                        best_K, bic, perm.tolist(), self._last_refit_date.date(),
                    )

                    prob_df = compute_filtered_probability_vector(self._model, features)
                    self._cached_prob_df = reorder_prob_df(prob_df, perm)
                    return self._cached_prob_df

                except Exception as exc:
                    logger.warning("Refit başarısız (%s) — mevcut model kullanılıyor.", exc)

        # Refit gerekmedi — önbellekte güncel sonuç varsa döndür.
        # Yoksa (ilk çağrı veya cache henüz dolu değil) yeniden hesapla.
        if self._cached_prob_df is not None:
            return self._cached_prob_df
        self._cached_prob_df = compute_filtered_probability_vector(self._model, features)
        return self._cached_prob_df

    @property
    def last_prob_df(self) -> "Optional[pd.DataFrame]":
        """En son hesaplanan forward-filter prob_df önbelleği.

        ARCH-4: `get_prob_row` bunu kullanarak gereksiz `update()` çağrısından
        kaçınır — forward-filter yalnızca refit günlerinde yeniden çalışır.
        """
        return self._cached_prob_df

    def needs_refit(self, df: pd.DataFrame) -> bool:
        """Bir sonraki `update()` çağrısında refit yapılacak mı?"""
        if self._model is None:
            return True
        return self._should_refit(df)

    # ── İç yardımcılar ───────────────────────────────────────────────────

    def _should_refit(self, df: pd.DataFrame) -> bool:
        if self._last_refit_date is None:
            return True
        last_date = df.index[-1]
        last_ts = last_date if isinstance(last_date, pd.Timestamp) else pd.Timestamp(last_date)
        days_since = (last_ts - self._last_refit_date).days
        return days_since >= self.refit_every_n_days
