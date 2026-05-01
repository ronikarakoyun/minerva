"""
engine/execution/blender.py — Faz 5.2: Rejim olasılık vektörü ile harmanlama.

Sert rejim geçişleri whipsaw yaratır: bugün %100 Boğa formülü → yarın %100 Ayı
formülü → tüm pozisyonu sat-al → komisyon erozyonu.

Faz 1 HMM'in soft prob_df çıktısı (Date × K) zaten yumuşak geçiş veriyor.
Bu modül her rejim için "şampiyon formül"ün sinyalini değerlendirir, prob
vektörüyle harmanlar:

    target_weights_t = Σ_k prob_t[k] · champion_signal_k[t]

Çıktı (Date × Ticker) target weights — backtest_engine'e doğrudan signal
olarak geçirilebilir. EMA smoothing turnover'ı bastırır.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from engine.core.alpha_cfg import Node


@dataclass
class BlenderConfig:
    use_blending: bool = False         # default kapalı
    top_k: int = 20                    # nihai portföy boyutu (top-K rank)
    smoothing_alpha: float = 0.3       # EMA: w_t = α·w_new + (1-α)·w_{t-1}; 1.0 → smoothing yok
    min_weight: float = 0.01           # gürültü filtresi
    fallback_to_argmax: bool = True    # şampiyon eksikse rejim argmax


def _evaluate_champion(tree: Node, df: pd.DataFrame, alpha_cfg) -> pd.Series:
    """
    Şampiyon formül AST'sini df üzerinde değerlendir.
    cfg.evaluate(tree, df) → MultiIndex (Ticker, Date) Series döner.
    """
    return alpha_cfg.evaluate(tree, df)


def blend_regime_signals(
    champion_trees: dict[int, Node],
    prob_df: pd.DataFrame,
    df: pd.DataFrame,
    cfg: Optional[BlenderConfig] = None,
    alpha_cfg=None,
) -> pd.DataFrame:
    """
    Rejim olasılıklarına göre şampiyon sinyallerini harmanla.

    Parameters
    ----------
    champion_trees : dict[int, Node]
        {regime_id: AST Node} — Faz 1'in K rejimi için şampiyon formüller.
    prob_df : pd.DataFrame
        (Date × K) — sütunlar "regime_0", "regime_1", ... veya integer K.
        Faz 1: regime_detector.compute_probability_vector çıktısı.
    df : pd.DataFrame
        Flat price df (Ticker, Date, Pclose, ...).
    cfg : BlenderConfig
    alpha_cfg : AlphaCFG
        cfg.evaluate(tree, df) için.

    Returns
    -------
    pd.DataFrame
        Wide (Date × Ticker) — günlük target weights. Top-K hisse pozitif
        ağırlık alır, kalan 0. EMA smoothing uygulanmıştır.
    """
    cfg = cfg or BlenderConfig()
    if alpha_cfg is None:
        raise ValueError("alpha_cfg gerekli (cfg.evaluate için)")

    if not champion_trees:
        raise ValueError("champion_trees boş")

    # prob_df sütun sırası: regime_0, regime_1, ...
    regime_ids = sorted(champion_trees.keys())

    # Her şampiyon formülü değerlendir → MultiIndex Series → wide pivot
    champion_signals_wide: dict[int, pd.DataFrame] = {}
    for k in regime_ids:
        sig_mi = _evaluate_champion(champion_trees[k], df, alpha_cfg)
        # MultiIndex (Ticker, Date) → wide (Date × Ticker)
        if isinstance(sig_mi.index, pd.MultiIndex):
            sig_wide = sig_mi.unstack("Ticker")
        else:
            # Flat — varsay Date index
            sig_wide = sig_mi.to_frame().T
        champion_signals_wide[k] = sig_wide

    # Ortak Date × Ticker boyutu
    common_dates = champion_signals_wide[regime_ids[0]].index
    common_tickers = champion_signals_wide[regime_ids[0]].columns

    # Hiç olmayan rejim için fallback: en yakın argmax kullan (config flag)
    if len(champion_trees) < prob_df.shape[1] and not cfg.fallback_to_argmax:
        raise ValueError(
            f"Şampiyon eksik: {len(champion_trees)} formül, "
            f"{prob_df.shape[1]} rejim. fallback_to_argmax=False."
        )

    # Sinyal stack: (D × T × K)
    sig_stack = np.stack(
        [champion_signals_wide[k].reindex(index=common_dates, columns=common_tickers).values
         for k in regime_ids],
        axis=2,
    )  # shape (D, T, K)
    sig_stack = np.nan_to_num(sig_stack, nan=0.0)

    # Prob vektörü: (D × K) — sadece elimizdeki rejim sütunlarını seç
    prob_cols = [f"regime_{k}" for k in regime_ids]
    available_cols = [c for c in prob_cols if c in prob_df.columns]
    if not available_cols:
        # Kolon yoksa bütün prob_df sütunlarını rejim sırasıyla varsay
        prob_aligned = prob_df.reindex(index=common_dates).iloc[:, :len(regime_ids)].fillna(0.0)
    else:
        prob_aligned = prob_df.reindex(index=common_dates)[available_cols].fillna(0.0)

    prob_arr = prob_aligned.values  # (D, K')

    # Boyut uyuşmazlığı: prob K' ≤ K — eksik rejimler için 0 prob
    K_actual = sig_stack.shape[2]
    if prob_arr.shape[1] < K_actual:
        padded = np.zeros((prob_arr.shape[0], K_actual))
        padded[:, :prob_arr.shape[1]] = prob_arr
        prob_arr = padded

    # blended[t, T] = Σ_k prob[t, k] · sig_stack[t, T, k]
    blended = np.einsum("dtk,dk->dt", sig_stack, prob_arr)
    blended_df = pd.DataFrame(blended, index=common_dates, columns=common_tickers)

    # Top-K rank → ağırlık (eşit ağırlık top-K içinde)
    weights_df = pd.DataFrame(0.0, index=common_dates, columns=common_tickers)
    for d in common_dates:
        row = blended_df.loc[d]
        if row.isna().all():
            continue
        # En yüksek top_k sinyalli hisse
        ranked = row.dropna().nlargest(cfg.top_k)
        if len(ranked) == 0:
            continue
        eq_weight = 1.0 / len(ranked)
        weights_df.loc[d, ranked.index] = eq_weight

    # min_weight filtresi
    weights_df = weights_df.where(weights_df >= cfg.min_weight, 0.0)

    # N22: min_weight filtresi sonrası NaN ve toplam sıfır satırları normalize et.
    # NaN → 0, ardından row_sum > 0 olan satırları yeniden normalize.
    weights_df = weights_df.fillna(0.0)
    row_sums = weights_df.sum(axis=1)
    nonzero = row_sums > 0
    weights_df.loc[nonzero] = weights_df.loc[nonzero].div(row_sums[nonzero], axis=0)

    # EMA smoothing: w_t = α·w_new + (1-α)·w_{t-1}
    if cfg.smoothing_alpha < 1.0:
        smoothed = weights_df.copy()
        for i in range(1, len(smoothed)):
            smoothed.iloc[i] = (
                cfg.smoothing_alpha * weights_df.iloc[i].values
                + (1 - cfg.smoothing_alpha) * smoothed.iloc[i - 1].values
            )
        weights_df = smoothed

        # N22: EMA sonrası yeniden normalize et
        weights_df = weights_df.fillna(0.0)
        row_sums = weights_df.sum(axis=1)
        nonzero = row_sums > 0
        weights_df.loc[nonzero] = weights_df.loc[nonzero].div(row_sums[nonzero], axis=0)

    return weights_df


def load_champions_from_catalog(
    catalog_path: Optional[Path] = None,
    alpha_cfg=None,
) -> dict[int, Node]:
    """
    Alpha katalog'tan rejim şampiyonlarını yükle.

    Catalog'da `regime_champions` kayıtları aranır (her rejim için en iyi
    formül). save_regime_champion ile yazılan formatla uyumludur.

    Returns
    -------
    dict[int, Node]
        {regime_id: AST Node}
    """
    import json
    import os

    from engine.core.alpha_catalog import CATALOG_PATH
    from engine.core.formula_parser import parse_formula

    path = str(catalog_path) if catalog_path else CATALOG_PATH
    if not os.path.exists(path):
        return {}

    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)

    champions: dict[int, Node] = {}
    for r in records:
        regime_id = r.get("regime_champion_for")
        if regime_id is None:
            continue
        formula = r.get("formula", "")
        if not formula or alpha_cfg is None:
            continue
        try:
            tree = parse_formula(formula, alpha_cfg)
            champions[int(regime_id)] = tree
        except Exception:
            continue

    return champions
