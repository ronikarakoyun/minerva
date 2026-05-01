"""
engine/wf_fitness.py — Walk-Forward fitness computation for mining.

Gelenek: fitness = RankIC (tek sayı) → multiple-testing bias'a açık.
Burada: formülü TRAIN verisinde K iç-fold'a ayırır, her fold'da RankIC ölçer,
fitness = mean(ric) - λ_std · std(ric) - λ_cx · complexity.

Bu sayede mining "tüm train dönemlerinde tutarlı" formülleri seçer.
Backtest'te gördüğümüz 800% → 50% çöküşünün temel sebebi olan
"tek dönem lucky formul" problemini kökünden çözer.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..core.alpha_cfg import Node
from ..data.factor_neutralize import neutralize_signal, compute_size_corr


def _node_complexity(node: Node) -> int:
    """AST node sayısını hesapla — Node.size() kullanır."""
    try:
        return node.size()
    except Exception:
        # Fallback: manuel recursive
        count = 1
        for child in getattr(node, "children", []) or []:
            if isinstance(child, Node):
                count += _node_complexity(child)
        return count


def _slice_fold_ranges(
    unique_dates: np.ndarray,
    n_folds: int,
    min_fold_days: int,
    embargo_days: int,
) -> list[tuple[int, int]]:
    """
    İç yardımcı: unique_dates üzerinde fold (start_idx, end_idx) aralıklarını üret.
    embargo_days fold sınırlarına uygulanır.
    """
    n = len(unique_dates)
    if n < n_folds * min_fold_days:
        n_folds = max(2, n // min_fold_days)
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    ranges = []
    for i in range(n_folds):
        start_idx = edges[i] + embargo_days
        end_idx   = edges[i + 1] - embargo_days
        if end_idx > start_idx:
            ranges.append((int(start_idx), int(end_idx)))
    return ranges


def make_date_folds(
    dates: np.ndarray,
    n_folds: int = 5,
    min_fold_days: int = 20,
    embargo_days: int = 5,
) -> list[np.ndarray]:
    """
    Train tarihlerini eşit boyutlu K fold'a böl (non-overlapping).
    Genişleyen-pencere değil; **bağımsız** fold'lar — bu sayede her fold
    birbirinin kopyası olmaz, gerçek stabilite testi olur.

    embargo_days > 0: Her fold sınırının başından ve sonundan bu kadar gün
    çıkarılır. Triple-Barrier label horizon'u kadar boşluk bırakmak,
    fold sınırlarındaki label leakage'ını önler.
    Önerilen: Triple-Barrier horizon değeriyle eşit (varsayılan 5 gün).
    """
    unique_dates = np.sort(np.unique(pd.to_datetime(dates)))
    ranges = _slice_fold_ranges(unique_dates, n_folds, min_fold_days, embargo_days)
    folds = []
    for s, e in ranges:
        fold_dates = unique_dates[s:e]
        if len(fold_dates) >= min_fold_days:
            folds.append(fold_dates)
    return folds


def make_purged_date_folds(
    dates: np.ndarray,
    n_folds: int = 5,
    min_fold_days: int = 20,
    embargo_days: int = 5,
    purge_horizon: int = 10,
    return_window: int = 2,
) -> list[dict]:
    """
    Purged K-Fold (López de Prado, AFML §7).

    Her fold için ayrı test + train tarih dizisi döner:
      [{"test": np.ndarray, "train": np.ndarray}, ...]

    purge_horizon > 0: Her test fold'unun başlangıç tarihinden
    geriye doğru purge_horizon iş günü, train kümesinden çıkarılır.
    Bu, TB horizon'u kadar ileriye bakan label'ların test'e bilgi
    sızdırmasını engeller.

    Geriye uyumluluk: purge_horizon=0 → list[dict] formatında eski
    make_date_folds davranışı (train = tüm veri - test).

    return_window: Hedef değişkenin kaç gün ileriye baktığı (Next_Ret=2,
    TB_Label=tb_horizon). purge_horizon < return_window ise fold
    sınırlarında label leakage oluşur — bu durumda uyarı verilir.
    """
    import warnings
    if purge_horizon < return_window:
        warnings.warn(
            f"purge_horizon={purge_horizon} < return_window={return_window}: "
            "fold sınırlarında label leakage riski var. "
            f"purge_horizon'ı en az {return_window} olarak ayarla.",
            stacklevel=2,
        )
    unique_dates = np.sort(np.unique(pd.to_datetime(dates)))
    n = len(unique_dates)
    ranges = _slice_fold_ranges(unique_dates, n_folds, min_fold_days, embargo_days)

    folds = []
    for s, e in ranges:
        test_dates = unique_dates[s:e]
        if len(test_dates) < min_fold_days:
            continue

        # Test fold başlangıcından purge_horizon iş günü öncesine kadar olan
        # train örnekleri temizlenir (leakage bölgesi)
        test_start = test_dates[0]

        # Purge maskesi: test_start'tan purge_horizon İŞ GÜNÜ önce.
        # np.busday_offset ile takvim günü yerine iş günü sayımı yapılır;
        # bu sayede Next_Ret=t+2 gibi iş günü bazlı hedefler için tam purge sağlanır.
        test_start_day = test_start.astype("datetime64[D]")
        purge_cutoff = np.datetime64(
            np.busday_offset(test_start_day, -purge_horizon, roll="backward"),
            "ns",
        )

        # Train: test dışındaki + purge bölgesi dışındaki tarihler
        not_in_test = ~np.isin(unique_dates, test_dates)
        not_purged  = unique_dates <= purge_cutoff
        # "test'ten sonraki" günler train'e dahil olabilir (expanding window var)
        after_test  = unique_dates >= test_dates[-1] + np.timedelta64(1, "D")
        train_mask  = not_in_test & (not_purged | after_test)
        train_dates = unique_dates[train_mask]

        folds.append({"test": test_dates, "train": train_dates})

    return folds


def compute_wf_fitness(
    tree: Node,
    evaluate_fn,                     # cfg.evaluate gibi (tree, idx) -> Series
    idx: pd.DataFrame,               # MultiIndex (Ticker, Date) + hedef kolonu
    folds: "list[np.ndarray] | list[dict]",
    lambda_std: float = 2.0,
    lambda_cx: float = 0.003,
    min_valid_folds: int = 3,
    target_col: str = "Next_Ret",    # "Next_Ret" veya "TB_Label"
    neutralize: bool = False,        # Faktör nötralizasyonu uygula
    factor_cache: "pd.DataFrame | None" = None,  # Önceden hesaplanmış faktörler
    lambda_size: float = 0.5,        # Size-corr penaltı ağırlığı
    size_corr_hard_limit: float = 0.5,  # N16: 0.7→0.5 — daha sıkı size filtresi
    regime: "pd.Series | None" = None,  # Date→{"bull","chop","bear"} rejim serisi
) -> dict:
    """
    Tek formül için WF-fitness hesapla.

    Döner:
      {
        "fold_rics":       [ric_1, ric_2, ...],
        "mean_ric":        float,
        "std_ric":         float,
        "pos_folds":       int (pozitif IC veren fold sayısı),
        "complexity":      int,
        "fitness":         float,  # seçim skoru
        "rank_ic":         float,  # toplam (eski uyumluluk)
        "ic":              float,
        "size_corr":       float,  # size faktörüyle ortalama korelasyon
        "neutralized":     bool,   # nötralizasyon uygulandı mı
        "status":          "ok" | "invalid" | "empty",
        "regime_breakdown": {"bull": float, "chop": float, "bear": float} | None
      }
    """
    complexity = _node_complexity(tree)
    result = {
        "fold_rics":  [],
        "mean_ric":   np.nan,
        "std_ric":    np.nan,
        "pos_folds":  0,
        "complexity": complexity,
        "fitness":    -1e9,        # Geçersizlik: çok negatif
        "rank_ic":    np.nan,
        "ic":         np.nan,
        "size_corr":  np.nan,
        "neutralized": neutralize,
        "status":     "invalid",
        "regime_breakdown": None,
    }

    try:
        sig = evaluate_fn(tree, idx)
    except Exception:
        return result

    if sig is None or len(sig) == 0:
        result["status"] = "empty"
        return result

    # --- Size korelasyonu ölç (nötralizasyondan ÖNCE, ham sinyalde) ---
    try:
        size_corr = compute_size_corr(sig, idx, factors=factor_cache)
        result["size_corr"] = float(size_corr)
    except Exception:
        size_corr = 0.0

    # --- Hard size filter: |size_corr| > limit → direkt reddet ---
    # Nötralizasyon açıksa bile rank-space OLS sonrası küçük bir artık kalabilir,
    # ama HAM sinyalde size_corr > 0.7 olan formüller zaten price-level proxy.
    # Nötralizasyon kapalıyken bu tamamen gereksiz bir formula demektir.
    if abs(size_corr) > size_corr_hard_limit:
        result["status"] = "size_factor"
        result["fitness"] = -1e9
        return result

    # --- Faktör Nötralizasyonu ---
    if neutralize:
        try:
            sig = neutralize_signal(sig, idx, factors=factor_cache)
        except Exception:
            pass  # nötralizasyon başarısızsa ham sinyale devam et

    # Signal + hedef + Date tablosu
    if target_col not in idx.columns:
        target_col = "Next_Ret"  # fallback
    dates = idx.index.get_level_values("Date")
    tmp = pd.DataFrame({
        "Date":   dates,
        "Signal": sig.values,
        "Target": idx[target_col].values,
    }).dropna()

    if len(tmp) == 0:
        result["status"] = "empty"
        return result

    def _ic_on_group(g: pd.DataFrame, method: str) -> float:
        if g["Signal"].std() == 0:
            return 0.0
        return g["Signal"].corr(g["Target"], method=method)

    # Tüm train IC (geri uyumluluk için)
    try:
        result["ic"]      = float(tmp.groupby("Date").apply(lambda g: _ic_on_group(g, "pearson")).mean())
        result["rank_ic"] = float(tmp.groupby("Date").apply(lambda g: _ic_on_group(g, "spearman")).mean())
    except Exception:
        pass

    # Per-fold RankIC
    # folds: list[np.ndarray] (eski format) veya list[dict] (purged format)
    fold_rics = []
    for fold_item in folds:
        # Purged format: {"test": ..., "train": ...} → test tarihlerinde IC ölç
        if isinstance(fold_item, dict):
            fold_dates = fold_item.get("test", np.array([]))
        else:
            fold_dates = fold_item
        mask = tmp["Date"].isin(fold_dates)
        sub = tmp[mask]
        if len(sub) < 20:        # küçük fold'u atla
            continue
        try:
            fric = float(sub.groupby("Date").apply(lambda g: _ic_on_group(g, "spearman")).mean())
            if not np.isnan(fric):
                fold_rics.append(fric)
        except Exception:
            continue

    result["fold_rics"] = fold_rics
    if len(fold_rics) < min_valid_folds:
        result["status"] = "invalid"
        return result

    arr = np.array(fold_rics, dtype=float)
    mean_ric = float(arr.mean())
    std_ric  = float(arr.std(ddof=0))
    pos_folds = int((arr > 0).sum())

    # Size korelasyon penaltısı:
    #   Nötralizasyon KAPALI: |size_corr| > 0.3 → lambda_size × (|corr| - 0.3)
    #   Nötralizasyon AÇIK : Rank-OLS artığı ~0.10-0.15 bırakabilir;
    #                         küçük penaltı uygula (lambda_size × 0.2 × max(0, |corr| - 0.15))
    #                         Tamamen sıfırlama yapma — nötralizasyon zaten ağır iş yaptı.
    size_penalty = 0.0
    if not np.isnan(size_corr):
        if neutralize:
            # Artık penaltı: nötralizasyon sonrası kalan sızıntı için
            size_penalty = lambda_size * 0.2 * max(0.0, abs(size_corr) - 0.15)
        else:
            size_penalty = lambda_size * max(0.0, abs(size_corr) - 0.3)

    # Fitness — stabilite-cezalı + karmaşıklık-cezalı + size-cezalı
    fitness = mean_ric - lambda_std * std_ric - lambda_cx * complexity - size_penalty

    result.update({
        "mean_ric":   mean_ric,
        "std_ric":    std_ric,
        "pos_folds":  pos_folds,
        "fitness":    fitness,
        "size_corr":  result.get("size_corr", np.nan),
        "status":     "ok",
    })

    # Rejim bazlı IC ayrıştırması (opsiyonel)
    if regime is not None:
        try:
            from .regime import regime_breakdown as _regime_bd
            target_series = tmp["Target"].copy()
            target_series.index = pd.to_datetime(tmp["Date"].values)
            sig_for_regime = sig.copy()
            # tmp ile aynı satır sayısına indir
            tmp_sig = pd.DataFrame({
                "Date":   pd.to_datetime(tmp["Date"].values),
                "Signal": sig.reindex(tmp.index).values if hasattr(sig, "reindex") else sig.values[:len(tmp)],
                "Target": tmp["Target"].values,
            }).dropna()
            sig_rb = pd.Series(
                tmp_sig["Signal"].values,
                index=pd.to_datetime(tmp_sig["Date"].values),
            )
            tgt_rb = pd.Series(
                tmp_sig["Target"].values,
                index=pd.to_datetime(tmp_sig["Date"].values),
            )
            result["regime_breakdown"] = _regime_bd(sig_rb, tgt_rb, regime)
        except Exception:
            result["regime_breakdown"] = None

    return result


def wf_verdict(stats: dict, min_pos_ratio: float = 0.8, min_mean_ric: float = 0.005) -> str:
    """Mining içi hızlı karar — mining UI'da gösterilebilir."""
    if stats.get("status") == "size_factor":
        return "🏷️ Size-Faktör"
    n = len(stats.get("fold_rics", []))
    if n == 0 or stats.get("status") != "ok":
        return "—"
    pos_ratio = stats["pos_folds"] / n
    if pos_ratio >= min_pos_ratio and stats["mean_ric"] >= min_mean_ric:
        return "✅ WF-Stabil"
    if pos_ratio >= 0.6 and stats["mean_ric"] > 0:
        return "⚠️ Kararsız"
    if stats["mean_ric"] <= 0 or pos_ratio < 0.4:
        return "💀 Geçersiz"
    return "❌ Overfit"
