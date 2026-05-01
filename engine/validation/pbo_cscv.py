"""
engine/pbo_cscv.py — Probability of Backtest Overfitting (PBO) via CSCV.

Bailey, Borwein, López de Prado & Zhu (2014):
"The Probability of Backtest Overfitting".

Yöntem: Combinatorially Symmetric Cross-Validation (CSCV).

1. Veriyi M eşit zaman dilimine böl → (S × M) PnL matris.
   Satır = zaman dilimi, Sütun = formül.
2. C(M, M/2) kombinasyonu üret: yarısı IS, diğer yarısı OOS.
3. Her kombinasyon için:
   - IS'te en iyi formülü bul (max ortalama PnL).
   - OOS'ta bu formülün rank'ine bak.
4. PBO = P(OOS rank < median) — IS'te en iyinin OOS'ta kaybetme olasılığı.

PBO < 0.5 → düşük overfit riski.
PBO ≥ 0.5 → seçim sürecinde belirgin overfit riski var.
"""
from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def build_pnl_matrix(
    pool_pnl: "np.ndarray | list[np.ndarray]",
    n_slices: int = 8,  # N17: 16→8 (252/8=31g/slice, yeterli istatistik)
) -> np.ndarray:
    """
    Önceden hesaplanmış günlük PnL dizilerini (M × S) matrise böl.

    Parametreler
    ------------
    pool_pnl : array-like, shape (n_days,) veya list[(n_days,)]
        Her formül için kronolojik günlük PnL serisi.
        Eşit uzunlukta olmaları gerekir (kısa olanlar NaN ile doldurulur).
    n_slices : int
        Zaman dilimi sayısı M (genellikle 8).
        N17: 252g/16=15g slice çok kısa → 8 slice önerilir (31g/slice).

    Döner
    ------
    np.ndarray, shape (n_slices, n_formulas)
        Her hücre = o dilimdeki ortalama günlük PnL.
    """
    if isinstance(pool_pnl, list):
        pool_pnl = np.column_stack([np.asarray(p, dtype=float) for p in pool_pnl])
    mat = np.asarray(pool_pnl, dtype=float)  # (n_days, n_formulas) veya (n_slices, n_formulas)

    # Zaten (n_slices, n_formulas) formatındaysa doğrudan döndür
    if mat.ndim == 2 and mat.shape[0] == n_slices:
        return mat

    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)

    n_days, n_formulas = mat.shape
    # Eşit boyutlu dilimler
    edges  = np.linspace(0, n_days, n_slices + 1, dtype=int)
    sliced = np.zeros((n_slices, n_formulas), dtype=float)
    for i in range(n_slices):
        chunk = mat[edges[i]:edges[i + 1], :]
        if len(chunk) > 0:
            sliced[i, :] = np.nanmean(chunk, axis=0)
    return sliced


def cscv_pbo(
    pnl_mat: np.ndarray,
    n_splits: Optional[int] = None,
    max_combinations: int = 1000,
) -> dict:
    """
    Combinatorially Symmetric Cross-Validation (CSCV) ile PBO hesapla.

    Parametreler
    ------------
    pnl_mat : np.ndarray, shape (M, S)
        M = zaman dilimi sayısı, S = formül sayısı.
        Her hücre = o dilimdeki ortalama PnL.
    n_splits : int, opsiyonel
        Kaç eşit IS/OOS kombinasyonu denesin? None → tümü C(M, M//2) (max_combinations ile kısıtlı).
    max_combinations : int
        Maksimum kombinasyon sayısı (çok büyük M için koruma).

    Döner
    ------
    dict içeriği:
        "pbo"            : float — overfit olasılığı ∈ [0, 1].
        "logit_lambda"   : np.ndarray — her kombinasyonun logit(ω*) değerleri.
        "n_combinations" : int — değerlendirilen kombinasyon sayısı.
        "is_sr"          : float — IS en iyinin ortalama IS SR'ı.
        "oos_sr"         : float — IS en iyinin OOS SR'ı (performans düşüşü).
        "perf_degradation": float — oos_sr / is_sr (1'den küçük → düşüş var).
        "verdict"        : str — "Kabul (düşük risk)" | "Overfit riski" | "Belirsiz"
    """
    M, S = pnl_mat.shape
    if M < 4 or S < 2:
        return {
            "pbo": np.nan, "logit_lambda": np.array([]),
            "n_combinations": 0, "is_sr": np.nan,
            "oos_sr": np.nan, "perf_degradation": np.nan, "verdict": "Yetersiz veri",
        }

    half = M // 2
    # Tüm C(M, half) kombinasyonları (IS dilim indeksleri)
    all_is_combos = list(combinations(range(M), half))

    # Çok fazlaysa rastgele örnekle
    if len(all_is_combos) > max_combinations:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_is_combos), size=max_combinations, replace=False)
        all_is_combos = [all_is_combos[i] for i in sorted(idx)]

    # Her kombinasyon için:
    # 1. IS'te en iyi formülü bul (max ortalama PnL)
    # 2. OOS'ta bu formülün rank'ini bul
    # 3. ω* = (rank / (S-1)) → logit(ω*)

    logit_lambdas  = []
    is_sr_vals     = []
    oos_sr_vals    = []

    for is_idx in all_is_combos:
        oos_idx = [i for i in range(M) if i not in is_idx]

        is_pnl  = pnl_mat[list(is_idx), :]   # (half, S)
        oos_pnl = pnl_mat[oos_idx, :]         # (half, S)

        is_mean  = np.nanmean(is_pnl, axis=0)   # (S,)
        oos_mean = np.nanmean(oos_pnl, axis=0)  # (S,)

        best_is  = int(np.argmax(is_mean))
        is_sr    = float(is_mean[best_is])

        # OOS rank: kaç formül IS-best'i geçiyor?
        oos_rank = int(np.sum(oos_mean > oos_mean[best_is]))
        # omega* = normalized rank ∈ [0, 1]
        omega = oos_rank / max(S - 1, 1)

        # Logit(omega*): eğer omega < 0.5 → negatif → OOS'ta başarılı
        logit_val = float(np.log(max(omega, 1e-7) / max(1.0 - omega, 1e-7)))
        logit_lambdas.append(logit_val)
        is_sr_vals.append(is_sr)
        oos_sr_vals.append(float(oos_mean[best_is]))

    logit_arr = np.array(logit_lambdas)
    # PBO = P(logit_lambda > 0) = P(omega* > 0.5)
    # omega* > 0.5 → IS best, OOS'ta median'ın altında → overfit
    # logit > 0 → IS best OOS'ta kaybetti
    pbo = float(np.mean(logit_arr > 0.0))

    is_sr_avg  = float(np.nanmean(is_sr_vals)) if is_sr_vals else np.nan
    oos_sr_avg = float(np.nanmean(oos_sr_vals)) if oos_sr_vals else np.nan
    perf_deg   = (oos_sr_avg / abs(is_sr_avg)) if (is_sr_vals and abs(is_sr_avg) > 1e-10) else np.nan

    verdict = pbo_verdict(pbo)
    return {
        "pbo":              pbo,
        "logit_lambda":     logit_arr,
        "n_combinations":   len(all_is_combos),
        "is_sr":            is_sr_avg,
        "oos_sr":           oos_sr_avg,
        "perf_degradation": perf_deg,
        "verdict":          verdict,
    }


def pbo_verdict(pbo: float) -> str:
    """PBO değerini yorumla."""
    if not np.isfinite(pbo):
        return "Belirsiz"
    if pbo < 0.5:
        return "✅ Kabul (düşük overfit riski)"
    if pbo < 0.7:
        return "⚠️ Orta overfit riski"
    return "🔴 Yüksek overfit riski"


def compute_pool_pnl(
    alphas_df: pd.DataFrame,
    trees: dict,
    db: pd.DataFrame,
    evaluate_fn,
    n_slices: int = 8,  # N17: 16→8 (252/8=31g/slice, yeterli istatistik)
    top_k: int = 50,
    n_drop: int = 5,
    buy_fee: float = 0.0005,
    sell_fee: float = 0.0015,
) -> tuple[np.ndarray, list[str]]:
    """
    Mining havuzundaki formülleri günlük PnL matrisine dönüştür.

    Parametreler
    ------------
    alphas_df : pd.DataFrame
        session_state.alphas — "Formül" sütunlu mining sonucu.
    trees : dict
        session_state.trees — AST haritası.
    db : pd.DataFrame
        Backtest verisi (Date, Ticker, Pclose, Next_Ret sütunları).
    evaluate_fn : callable
        cfg.evaluate(tree, df) → sinyal Serisi.
    n_slices : int
        Zaman dilimi sayısı.
    top_k, n_drop, buy_fee, sell_fee : float
        run_pro_backtest parametreleri.

    Döner
    ------
    (pnl_mat, formula_names) : (np.ndarray shape (n_slices, S), list[str])
    """
    from engine.backtest_engine import run_pro_backtest

    formula_names = []
    pnl_series_list = []

    all_dates = sorted(db["Date"].unique())
    n_days = len(all_dates)
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    for _, row in alphas_df.iterrows():
        formula = row.get("Formül", "")
        tree = trees.get(formula)
        if tree is None:
            continue
        try:
            sig = evaluate_fn(tree, db)
            curve, _ = run_pro_backtest(
                db, sig, top_k=top_k, n_drop=n_drop,
                buy_fee=buy_fee, sell_fee=sell_fee,
            )
            if curve is None or len(curve) < 10:
                continue
            # Günlük PnL = Equity farkı / önceki Equity
            eq = curve.set_index("Date")["Equity"]
            daily_ret = eq.pct_change().fillna(0.0)
            # Tüm veri günlerine hizala (eksik günler 0)
            pnl_arr = np.zeros(n_days, dtype=float)
            for d, v in daily_ret.items():
                if d in date_to_idx:
                    pnl_arr[date_to_idx[d]] = v
            pnl_series_list.append(pnl_arr)
            formula_names.append(formula)
        except Exception:
            continue

    if not pnl_series_list:
        return np.empty((n_slices, 0)), []

    pnl_2d = np.column_stack(pnl_series_list)  # (n_days, S)
    pnl_mat = build_pnl_matrix(pnl_2d, n_slices=n_slices)
    return pnl_mat, formula_names
