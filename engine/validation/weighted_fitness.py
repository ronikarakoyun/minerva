"""
engine/validation/weighted_fitness.py — Faz 2: Rejim-koşullu ağırlıklı fitness.

Faz 1'in `prob_df` (Date × K rejim olasılık vektörü) çıktısını alıp, her tarih için
"bugünün rejim profiline" cosine similarity hesaplar; bu skoru üstel bir transform
ile [w_min, w_max] aralığında bir ağırlığa dönüştürür.

MCTS WF-fitness döngüsü, fold içindeki günlük cross-sectional RankIC zincirini
bu ağırlıklarla harmanlayarak `mean_ric_weighted` üretir. Böylece güncel rejim
profiline benzer geçmiş günler 10x'e kadar, alakasız günler 1x ağırlıkla
fitness'a katkı yapar.

Zaman serisi BOZULMAZ — stitching yok, shift/rolling yerinde durur, look-ahead
sızıntısı oluşmaz.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from ..core.alpha_cfg import Node
from ..data.factor_neutralize import compute_size_corr, neutralize_signal


@dataclass
class WeightConfig:
    """Rejim → tarih ağırlığı dönüşümünün parametreleri."""
    temperature: float = 2.0       # exp keskinliği (yüksek=odaklı, düşük=düz)
    w_min: float = 1.0             # tabandaki ağırlık (asla 0 — temel öğrenme korunur)
    w_max: float = 10.0            # tavandaki ağırlık (en benzer günler)
    ref_date: Optional[pd.Timestamp] = None  # None → prob_df'in son günü


def cosine_similarity_to_ref(
    prob_df: pd.DataFrame, ref_vec: Optional[np.ndarray] = None
) -> pd.Series:
    """
    Her gün için K-boyutlu rejim olasılık vektörü ile referans vektör arasında
    cosine similarity hesapla.

    Parameters
    ----------
    prob_df : pd.DataFrame
        index=Date, columns=regime_0..K-1, satır toplamı 1.0.
    ref_vec : np.ndarray, opsiyonel
        Şekil (K,). Verilmezse prob_df.iloc[-1].values kullanılır.

    Returns
    -------
    pd.Series
        index=prob_df.index, dtype=float, değer ∈ [0, 1] (olasılıklar non-negatif).
    """
    P = prob_df.values.astype(float)
    if ref_vec is None:
        ref_vec = P[-1]
    ref = np.asarray(ref_vec, dtype=float).reshape(-1)
    if ref.shape[0] != P.shape[1]:
        raise ValueError(
            f"ref_vec boyutu {ref.shape[0]} ≠ prob_df kolon sayısı {P.shape[1]}"
        )

    ref_norm = float(np.linalg.norm(ref))
    if ref_norm == 0.0:
        return pd.Series(0.0, index=prob_df.index, name="cosine_sim")

    row_norms = np.linalg.norm(P, axis=1)
    safe_norms = np.where(row_norms == 0.0, 1.0, row_norms)
    sims = (P @ ref) / (safe_norms * ref_norm)
    sims = np.where(row_norms == 0.0, 0.0, sims)
    sims = np.clip(sims, 0.0, 1.0)
    return pd.Series(sims, index=prob_df.index, name="cosine_sim")


def compute_regime_weights(
    prob_df: pd.DataFrame, cfg: Optional[WeightConfig] = None
) -> pd.Series:
    """
    Üstel scaling ile her tarih için ağırlık çarpanı üret.

        w_t = w_min + (w_max - w_min) · exp(T · sim_t) / exp(T)

    sim_t = 1 → w_t = w_max ;  sim_t = 0 → w_t ≈ w_min + (w_max-w_min)·exp(-T).

    Returns
    -------
    pd.Series
        index=prob_df.index, dtype=float, değer ∈ [w_min, w_max].
    """
    cfg = cfg or WeightConfig()

    ref_vec: Optional[np.ndarray] = None
    if cfg.ref_date is not None:
        ref_ts = pd.Timestamp(cfg.ref_date)
        if ref_ts not in prob_df.index:
            # En yakın geçmiş tarih
            valid = prob_df.index[prob_df.index <= ref_ts]
            if len(valid) == 0:
                raise ValueError(
                    f"ref_date={ref_ts} prob_df aralığından önce — referans yok"
                )
            ref_ts = valid[-1]
        ref_vec = prob_df.loc[ref_ts].values.astype(float)

    sims = cosine_similarity_to_ref(prob_df, ref_vec)

    T = float(cfg.temperature)
    span = float(cfg.w_max - cfg.w_min)
    # exp(T·sim) / exp(T) = exp(T·(sim-1))  — sayısal olarak daha stabil
    factor = np.exp(T * (sims.values - 1.0))
    w = cfg.w_min + span * factor
    return pd.Series(w, index=prob_df.index, name="regime_weight")


def _fold_weighted_ric(group_df: pd.DataFrame, weights_by_date: pd.Series) -> float:
    """
    Bir fold için ağırlıklı günlük RankIC ortalaması.

    group_df: ["Date", "Signal", "Target"] kolonları içerir.
    weights_by_date: index=Date, dtype=float.
    Eksik tarihler için ağırlık tabana (w_min=1) düşmesin diye reindex ederken
    fillna(0) yapmıyoruz — burada sadece var olan tarihlerle çalışıyoruz.
    """
    daily = group_df.groupby("Date")
    rics: list[float] = []
    ws: list[float] = []
    for date, g in daily:
        if g["Signal"].std() == 0:
            r = 0.0
        else:
            r = g["Signal"].corr(g["Target"], method="spearman")
        if pd.isna(r):
            continue
        w = float(weights_by_date.get(date, np.nan))
        if pd.isna(w):
            continue
        rics.append(float(r))
        ws.append(w)

    if not rics:
        return float("nan")
    rics_arr = np.asarray(rics, dtype=float)
    ws_arr = np.asarray(ws, dtype=float)
    if ws_arr.sum() <= 0:
        return float("nan")
    return float(np.average(rics_arr, weights=ws_arr))


def _fold_weighted_stats(
    group_df: pd.DataFrame, weights_by_date: pd.Series
) -> tuple[float, float, int, int]:
    """Fold'un (mean_w, std_w, pos_days, n_days) ağırlıklı istatistikleri."""
    rics: list[float] = []
    ws: list[float] = []
    for date, g in group_df.groupby("Date"):
        if g["Signal"].std() == 0:
            r = 0.0
        else:
            r = g["Signal"].corr(g["Target"], method="spearman")
        if pd.isna(r):
            continue
        w = float(weights_by_date.get(date, np.nan))
        if pd.isna(w):
            continue
        rics.append(float(r))
        ws.append(w)

    if not rics:
        return float("nan"), float("nan"), 0, 0
    rics_arr = np.asarray(rics, dtype=float)
    ws_arr = np.asarray(ws, dtype=float)
    if ws_arr.sum() <= 0:
        return float("nan"), float("nan"), 0, len(rics)
    mean_w = float(np.average(rics_arr, weights=ws_arr))
    var_w = float(np.average((rics_arr - mean_w) ** 2, weights=ws_arr))
    std_w = float(np.sqrt(max(var_w, 0.0)))
    pos = int((rics_arr > 0).sum())
    return mean_w, std_w, pos, len(rics)


def compute_weighted_wf_fitness(
    tree: Node,
    evaluate_fn,
    idx: pd.DataFrame,
    folds: "list[np.ndarray] | list[dict]",
    weights: pd.Series,
    lambda_std: float = 2.0,
    lambda_cx: float = 0.003,
    min_valid_folds: int = 3,
    target_col: str = "Next_Ret",
    neutralize: bool = False,
    factor_cache: "pd.DataFrame | None" = None,
    lambda_size: float = 0.5,
    size_corr_hard_limit: float = 0.7,
) -> dict:
    """
    Rejim-koşullu ağırlıklı WF-fitness.

    `compute_wf_fitness` ile aynı semantik ve aynı dönüş şeması — fark:
      - Her fold içindeki günlük cross-sectional RankIC zinciri,
        `weights` (index=Date) serisiyle ağırlıklı ortalanır.
      - Fold'lar arası mean/std da ağırlıklı IC'ler üzerinden hesaplanır.

    Returns
    -------
    dict — `compute_wf_fitness` çıktısıyla aynı anahtarlar + "regime_weighted": True.
    """
    from .wf_fitness import _node_complexity  # type: ignore

    complexity = _node_complexity(tree)
    result = {
        "fold_rics": [],
        "mean_ric": np.nan,
        "std_ric": np.nan,
        "pos_folds": 0,
        "complexity": complexity,
        "fitness": -1e9,
        "rank_ic": np.nan,
        "ic": np.nan,
        "size_corr": np.nan,
        "neutralized": neutralize,
        "status": "invalid",
        "regime_breakdown": None,
        "regime_weighted": True,
    }

    try:
        sig = evaluate_fn(tree, idx)
    except Exception:
        return result

    if sig is None or len(sig) == 0:
        result["status"] = "empty"
        return result

    try:
        size_corr = compute_size_corr(sig, idx, factors=factor_cache)
        result["size_corr"] = float(size_corr)
    except Exception:
        size_corr = 0.0

    if abs(size_corr) > size_corr_hard_limit:
        result["status"] = "size_factor"
        result["fitness"] = -1e9
        return result

    if neutralize:
        try:
            sig = neutralize_signal(sig, idx, factors=factor_cache)
        except Exception:
            pass

    if target_col not in idx.columns:
        target_col = "Next_Ret"

    dates = idx.index.get_level_values("Date")
    tmp = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Signal": sig.values,
        "Target": idx[target_col].values,
    }).dropna()

    if len(tmp) == 0:
        result["status"] = "empty"
        return result

    # weights index'i datetime'a hizala
    w_aligned = weights.copy()
    w_aligned.index = pd.to_datetime(w_aligned.index)

    # Hiç ağırlık-tarihiyle örtüşmeyen veri → no_overlap
    if not tmp["Date"].isin(w_aligned.index).any():
        result["status"] = "no_overlap"
        return result

    # Tüm-train ağırlıklı RankIC (geri uyumluluk amaçlı)
    try:
        result["rank_ic"] = _fold_weighted_ric(tmp, w_aligned)
        result["ic"] = result["rank_ic"]
    except Exception:
        pass

    # Per-fold ağırlıklı RankIC
    fold_rics: list[float] = []
    fold_pos_total = 0
    fold_day_total = 0
    for fold_item in folds:
        fold_dates = (
            fold_item.get("test", np.array([])) if isinstance(fold_item, dict)
            else fold_item
        )
        mask = tmp["Date"].isin(fold_dates)
        sub = tmp[mask]
        if len(sub) < 20:
            continue
        try:
            mean_w, _std_w, pos, n_days = _fold_weighted_stats(sub, w_aligned)
            if not np.isnan(mean_w):
                fold_rics.append(mean_w)
                fold_pos_total += pos
                fold_day_total += n_days
        except Exception:
            continue

    result["fold_rics"] = fold_rics
    if len(fold_rics) < min_valid_folds:
        result["status"] = "invalid"
        return result

    arr = np.array(fold_rics, dtype=float)
    mean_ric = float(arr.mean())
    std_ric = float(arr.std(ddof=0))
    pos_folds = int((arr > 0).sum())

    size_penalty = 0.0
    if not np.isnan(size_corr):
        if neutralize:
            size_penalty = lambda_size * 0.2 * max(0.0, abs(size_corr) - 0.15)
        else:
            size_penalty = lambda_size * max(0.0, abs(size_corr) - 0.3)

    fitness = mean_ric - lambda_std * std_ric - lambda_cx * complexity - size_penalty

    result.update({
        "mean_ric": mean_ric,
        "std_ric": std_ric,
        "pos_folds": pos_folds,
        "fitness": fitness,
        "status": "ok",
    })
    return result
