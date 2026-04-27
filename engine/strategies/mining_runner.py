"""
engine/mining_runner.py — Reentrant mining pencere fonksiyonu.

Mod 3 (Full Rolling Discovery) için: her rolling pencerede bağımsız
formül havuzu üretip değerlendiren, session_state'e bağımlı OLMAYAN
saf fonksiyon.

İş akışı:
  1. run_mining_window(): verilen train veri + config ile Faz1+Faz2+Faz3 koştur.
  2. MiningConfig: mining parametrelerini taşır (app.py sidebar'dan geçirilir).
  3. WF-fitness opsiyonel (MiningConfig.use_wf_fitness).

Not: Bu fonksiyon Streamlit'e BAĞIMLI DEĞİL. Progress callback opsiyonel.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..core.alpha_cfg import AlphaCFG, Node
from ..validation.wf_fitness import compute_wf_fitness, make_date_folds, make_purged_date_folds


@dataclass
class MiningConfig:
    """Mining parametrelerini taşıyan yapı."""
    num_gen: int = 200            # Toplam nesil sayısı (Faz1 + Faz2)
    max_K: int = 15               # Maksimum AST derinliği
    use_wf_fitness: bool = True
    wf_n_folds: int = 5
    wf_embargo: int = 5
    wf_purge: int = 10
    lambda_std: float = 2.0
    lambda_cx: float = 0.003
    lambda_size: float = 0.5
    size_corr_hard_limit: float = 0.7
    neutralize: bool = True
    target_col: str = "Next_Ret"
    seed: int = 42
    min_mean_ric: float = 0.003   # Kabul eşiği
    min_pos_ratio: float = 0.4


@dataclass
class MiningResult:
    """Tek pencere mining sonucu."""
    formula: str
    tree: Node
    fitness: float
    mean_ric: float
    std_ric: float
    pos_folds: int
    n_folds: int
    size_corr: float
    status: str
    regime_breakdown: Optional[dict] = None


def run_mining_window(
    db_window: pd.DataFrame,
    cfg: AlphaCFG,
    mining_cfg: MiningConfig,
    seed_trees: "list[Node] | None" = None,
    factor_cache: "pd.DataFrame | None" = None,
    regime: "pd.Series | None" = None,
    progress_cb: "Callable[[int, int], None] | None" = None,
) -> list[MiningResult]:
    """
    Bir train penceresi üzerinde tam mining döngüsü (Faz 1 + 2 + 3).

    Parametreler
    ------------
    db_window : pd.DataFrame
        Train penceresi — (Ticker, Date, Pclose, Vlot, ..., Next_Ret/TB_Label).
    cfg : AlphaCFG
        Formül grameri.
    mining_cfg : MiningConfig
        Mining parametreleri.
    seed_trees : list[Node], opsiyonel
        Warm-start: Faz 2'de tohum olarak kullanılacak önceki formüller.
    factor_cache : pd.DataFrame, opsiyonel
        Önceden hesaplanmış faktör matrisi (pencere-lokal olmalı).
    regime : pd.Series, opsiyonel
        Rejim serisi.
    progress_cb : callable, opsiyonel
        progress_cb(done, total) — UI progress bar güncelleme.

    Döner
    ------
    list[MiningResult] — kabul edilen formüller (fitness > -inf, status=="ok")
    """
    rng_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(mining_cfg.seed)
    np.random.seed(mining_cfg.seed)

    try:
        return _run_mining_window_impl(
            db_window, cfg, mining_cfg,
            seed_trees, factor_cache, regime, progress_cb,
        )
    finally:
        # Global random state'i geri yükle (reentrancy için)
        random.setstate(rng_state)
        np.random.set_state(np_state)


def _run_mining_window_impl(
    db_window: pd.DataFrame,
    cfg: AlphaCFG,
    mcfg: MiningConfig,
    seed_trees: "list[Node] | None",
    factor_cache: "pd.DataFrame | None",
    regime: "pd.Series | None",
    progress_cb: "Callable[[int, int], None] | None",
) -> list[MiningResult]:
    # ── Veri hazırlığı ──────────────────────────────────────────────
    db_w = db_window.copy()
    db_w = db_w.sort_values(["Ticker", "Date"])
    if mcfg.target_col == "Next_Ret" and "Next_Ret" not in db_w.columns:
        db_w["Pclose_t1"] = db_w.groupby("Ticker")["Pclose"].shift(-1)
        db_w["Pclose_t2"] = db_w.groupby("Ticker")["Pclose"].shift(-2)
        db_w["Next_Ret"]  = db_w["Pclose_t2"] / db_w["Pclose_t1"] - 1

    idx = db_w.set_index(["Ticker", "Date"]).sort_index()

    # ── Fold hazırlığı ────────────────────────────────────────────────
    mining_folds = None
    if mcfg.use_wf_fitness:
        dates_arr = idx.index.get_level_values("Date").values
        if mcfg.wf_purge > 0:
            mining_folds = make_purged_date_folds(
                dates_arr,
                n_folds=mcfg.wf_n_folds,
                min_fold_days=20,
                embargo_days=mcfg.wf_embargo,
                purge_horizon=mcfg.wf_purge,
            )
        else:
            mining_folds = make_date_folds(
                dates_arr,
                n_folds=mcfg.wf_n_folds,
                min_fold_days=20,
                embargo_days=mcfg.wf_embargo,
            )

    # ── Faz 1: Başlangıç havuzu ───────────────────────────────────────
    n1 = mcfg.num_gen // 2
    pool: list[Node] = [cfg.generate(mcfg.max_K) for _ in range(n1)]

    # ── Faz 2: Mutation & Crossover ──────────────────────────────────
    seed_pool = seed_trees if (seed_trees and len(seed_trees) >= 5) else pool
    n2 = mcfg.num_gen // 2
    for _ in range(n2):
        if random.random() < 0.7 and len(seed_pool) > 1:
            p1, p2 = random.sample(seed_pool, 2)
            pool.append(cfg.crossover(p1, p2))
        else:
            pool.append(cfg.mutate(random.choice(seed_pool)))

    # ── Faz 3: Değerlendirme ──────────────────────────────────────────
    wf_kwargs = dict(
        lambda_std=mcfg.lambda_std,
        lambda_cx=mcfg.lambda_cx,
        min_valid_folds=3,
        target_col=mcfg.target_col,
        neutralize=mcfg.neutralize,
        factor_cache=factor_cache,
        lambda_size=mcfg.lambda_size,
        size_corr_hard_limit=mcfg.size_corr_hard_limit,
        regime=regime,
    )

    results: list[MiningResult] = []
    n_pool = len(pool)
    for done_i, tree in enumerate(pool):
        if progress_cb:
            progress_cb(done_i, n_pool)

        if mcfg.use_wf_fitness and mining_folds and len(mining_folds) >= 3:
            try:
                stats = compute_wf_fitness(tree, cfg.evaluate, idx, mining_folds, **wf_kwargs)
            except Exception:
                continue
        else:
            # Klasik IC modu
            try:
                sig = cfg.evaluate(tree, idx)
                if sig is None or len(sig) == 0:
                    continue
                tmp = pd.DataFrame({
                    "Signal": sig.values,
                    "Target": idx[mcfg.target_col].values if mcfg.target_col in idx.columns else np.nan,
                }).dropna()
                if len(tmp) < 20:
                    continue
                ic = float(tmp["Signal"].corr(tmp["Target"], method="spearman"))
                stats = {
                    "status": "ok" if not np.isnan(ic) else "invalid",
                    "fitness": ic, "mean_ric": ic, "std_ric": 0.0,
                    "pos_folds": 1 if ic > 0 else 0, "size_corr": 0.0,
                    "regime_breakdown": None,
                }
            except Exception:
                continue

        if stats.get("status") != "ok":
            continue

        n_folds_v = len(stats.get("fold_rics", [1]))
        pos_ratio = stats["pos_folds"] / max(n_folds_v, 1)
        if stats["mean_ric"] < mcfg.min_mean_ric or pos_ratio < mcfg.min_pos_ratio:
            continue

        results.append(MiningResult(
            formula=str(tree),
            tree=tree,
            fitness=float(stats["fitness"]),
            mean_ric=float(stats["mean_ric"]),
            std_ric=float(stats.get("std_ric", 0.0)),
            pos_folds=int(stats["pos_folds"]),
            n_folds=n_folds_v,
            size_corr=float(stats.get("size_corr", 0.0)),
            status=stats["status"],
            regime_breakdown=stats.get("regime_breakdown"),
        ))

    if progress_cb:
        progress_cb(n_pool, n_pool)

    # Fitness'a göre sırala
    results.sort(key=lambda r: r.fitness, reverse=True)
    return results
