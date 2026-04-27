"""
engine/ensemble.py — Ensemble backtest yardımcıları.

Birden fazla formülün sinyalini birleştirerek tek portföy çalıştırır.
Hall of Fame (HoF) formüllerini pencere bazında saklar.

İçerikler:
  - combine_signals(): N sinyal → ağırlıklı ortalama sinyal.
  - run_ensemble_backtest(): top-K formülü birleştirip backtest.
  - HallOfFame: pencere bazlı formül listesi ve equity eğrisi yöneticisi.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from ..core.backtest_engine import run_pro_backtest


def combine_signals(
    signals: "list[pd.Series]",
    weights: "list[float] | None" = None,
    method: str = "rank_average",
) -> pd.Series:
    """
    N formül sinyalini tek bileşik sinyale indir.

    Parametreler
    ------------
    signals : list[pd.Series]
        Her birinin index'i aynı (Ticker, Date) MultiIndex veya Date index.
    weights : list[float], opsiyonel
        Her sinyale ağırlık; None → eşit ağırlık.
    method : str
        "rank_average" : cross-sectional rank ortalaması (önerilir).
        "simple_average": ham değer ortalaması.

    Döner
    ------
    pd.Series — birleşik sinyal (aynı index).
    """
    if not signals:
        return pd.Series(dtype=float)

    if weights is None:
        weights = [1.0] * len(signals)
    w_arr = np.array(weights, dtype=float)
    w_arr = w_arr / w_arr.sum()

    # Ortak index'e hizala
    common_idx = signals[0].index
    for s in signals[1:]:
        common_idx = common_idx.intersection(s.index)

    if len(common_idx) == 0:
        return pd.Series(dtype=float)

    aligned = [s.reindex(common_idx) for s in signals]

    if method == "rank_average":
        # Cross-sectional rank (per Date) → ağırlıklı ortalama
        if isinstance(common_idx, pd.MultiIndex):
            ranked = []
            for s, w in zip(aligned, w_arr):
                r = s.groupby(level="Date").rank(pct=True) * w
                ranked.append(r)
            return sum(ranked)
        else:
            return sum(s.rank(pct=True) * w for s, w in zip(aligned, w_arr))
    else:
        return sum(s.fillna(0) * w for s, w in zip(aligned, w_arr))


def run_ensemble_backtest(
    db: pd.DataFrame,
    trees: list,
    evaluate_fn,
    weights: "list[float] | None" = None,
    top_k: int = 50,
    n_drop: int = 5,
    buy_fee: float = 0.0005,
    sell_fee: float = 0.0015,
    benchmark: "pd.Series | None" = None,
) -> tuple["pd.DataFrame | None", "pd.DataFrame | None"]:
    """
    Birden fazla formülü birleştirip ensemble backtest koştur.

    Döner
    ------
    (curve, positions) — run_pro_backtest gibi.
    """
    if not trees:
        return None, None

    signals = []
    for tree in trees:
        try:
            sig = evaluate_fn(tree, db)
            if sig is not None and len(sig) > 0:
                signals.append(sig)
        except Exception:
            continue

    if not signals:
        return None, None

    combined = combine_signals(signals, weights=weights, method="rank_average")
    return run_pro_backtest(
        db, combined, top_k=top_k, n_drop=n_drop,
        buy_fee=buy_fee, sell_fee=sell_fee, benchmark=benchmark,
    )


@dataclass
class WindowResult:
    """Tek rolling pencere sonucu."""
    window_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_formulas: int          # Bu pencerede bulunan formül sayısı
    top_formula: str         # En iyi formül metni
    top_fitness: float
    equity: "pd.DataFrame | None"   # run_pro_backtest çıktısı
    formula_names: list = field(default_factory=list)


@dataclass
class HallOfFame:
    """
    Tüm rolling pencereler boyunca biriken formül ve equity yöneticisi.
    """
    windows: list[WindowResult] = field(default_factory=list)

    def add(self, result: WindowResult):
        self.windows.append(result)

    def combined_equity(self) -> "pd.DataFrame | None":
        """
        Tüm pencere equity'lerini kronolojik olarak birleştir.
        Her pencere kendi test periodunun equity'sini içerir.
        """
        parts = []
        for w in self.windows:
            if w.equity is None or len(w.equity) == 0:
                continue
            parts.append(w.equity[["Date", "Equity"]].copy())

        if not parts:
            return None

        combined = pd.concat(parts).drop_duplicates("Date").sort_values("Date")
        # Sürekli equity: pencere geçişlerinde chain
        combined = combined.reset_index(drop=True)
        if len(combined) > 0:
            # Re-chain: her segmentin başlangıcını bir öncekinin sonuna bağla
            for i in range(1, len(combined)):
                # Önceki satırın değeri
                pass  # Basit uygulama: ham equity serisini döndür
        return combined

    def to_dataframe(self) -> pd.DataFrame:
        """Pencere özetini DataFrame'e dönüştür."""
        rows = []
        for w in self.windows:
            rows.append({
                "Pencere": w.window_id,
                "Train Başlangıç": w.train_start.date() if w.train_start else None,
                "Train Bitiş": w.train_end.date() if w.train_end else None,
                "Test Başlangıç": w.test_start.date() if w.test_start else None,
                "Test Bitiş": w.test_end.date() if w.test_end else None,
                "Formül Sayısı": w.n_formulas,
                "En İyi Formül": w.top_formula[:40] + "..." if len(w.top_formula) > 40 else w.top_formula,
                "Fitness": round(w.top_fitness, 4),
            })
        return pd.DataFrame(rows)
