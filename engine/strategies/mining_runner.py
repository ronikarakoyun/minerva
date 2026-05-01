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

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from ..core.alpha_cfg import AlphaCFG, Node
from ..validation.deflated_sharpe import deflated_sharpe_ratio
from ..validation.wf_fitness import compute_wf_fitness, make_date_folds, make_purged_date_folds
from ..validation.weighted_fitness import (
    WeightConfig,
    compute_regime_weights,
    compute_weighted_wf_fitness,
)


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
    # DSR gating — True ise top-100 havuzu DSR testi ile filtreler
    use_dsr_filter: bool = False
    dsr_min_p: float = 0.75       # DSR p-value minimum eşiği (0.75 = %75 güven)
    # Faz 2 — Rejim-koşullu ağırlıklı fitness
    use_regime_weighting: bool = False
    weight_cfg: Optional[WeightConfig] = None
    prob_df: Optional[pd.DataFrame] = None  # Faz 1 çıktısı (Date × K)
    # Faz 3 — MCTS arama motoru
    search_mode: str = "gp"               # {"gp", "mcts"} — default geriye dönük
    c_puct: float = 1.4                   # PUCT exploration weight
    mcts_rollouts: int = 16               # value_fn yoksa rollout sayısı
    mcts_iterations_per_root: int = 50    # her formül için MCTS arama bütçesi

    @classmethod
    def from_best_params(
        cls,
        json_path: "str | Path" = "data/best_params.json",
        **overrides,
    ) -> "MiningConfig":
        """Optuna `best_params.json` çıktısından MiningConfig kur."""
        with open(json_path) as f:
            payload = json.load(f)
        bp = payload["best_params"]
        kwargs: dict = {
            "search_mode": "mcts",
            "max_K": int(bp["max_K"]),
            "c_puct": float(bp["c_puct"]),
            "mcts_rollouts": int(bp["mcts_rollouts"]),
            "lambda_std": float(bp["lambda_std"]),
            "lambda_cx": float(bp["lambda_cx"]),
            "use_regime_weighting": True,
            "weight_cfg": WeightConfig(temperature=float(bp["temperature"])),
        }
        kwargs.update(overrides)
        return cls(**kwargs)


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

    # ── Pool üretimi: GP (default) veya MCTS (Faz 3) ──────────────────
    if mcfg.search_mode == "mcts":
        from .mcts import GrammarMCTS
        prior = {}
        if seed_trees:
            for t in seed_trees[:50]:
                prior[str(t)] = 0.1
        # N15: Cascading prior — replay buffer'dan top-K formülleri prior olarak ekle.
        # Her pencere, bir önceki pencerenin en iyi formüllerinden beslenilir.
        try:
            from engine.ml.replay_buffer import ReplayBuffer
            _rb = ReplayBuffer().load()
            if len(_rb) > 0:
                _rb_samples = sorted(_rb.buf, key=lambda s: s[1], reverse=True)
                for _tree, _ic, _vd in _rb_samples[:20]:
                    _key = str(_tree)
                    # IC'yi [0, 1] aralığına normalize et — mevcut prior ile birleştir
                    prior[_key] = max(prior.get(_key, 0.0), min(float(_ic), 1.0))
        except Exception:
            pass  # Replay buffer erişim hatası — prior olmadan devam et
        searcher = GrammarMCTS(
            cfg, max_K=mcfg.max_K, c_puct=mcfg.c_puct,
            rollouts=mcfg.mcts_rollouts, subtree_prior=prior,
        )
        pool: list[Node] = [
            searcher.search(iterations=mcfg.mcts_iterations_per_root)
            for _ in range(mcfg.num_gen)
        ]
    else:
        # ── Faz 1: Başlangıç havuzu ───────────────────────────────────
        n1 = mcfg.num_gen // 2
        pool = [cfg.generate(mcfg.max_K) for _ in range(n1)]

        # ── Faz 2: Mutation & Crossover ──────────────────────────────
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
    )

    # Faz 2: Rejim-koşullu ağırlık serisi (opsiyonel)
    regime_weights: Optional[pd.Series] = None
    use_weighted = mcfg.use_regime_weighting and mcfg.prob_df is not None
    if use_weighted:
        regime_weights = compute_regime_weights(mcfg.prob_df, mcfg.weight_cfg)

    results: list[MiningResult] = []
    n_pool = len(pool)
    for done_i, tree in enumerate(pool):
        if progress_cb:
            progress_cb(done_i, n_pool)

        if mcfg.use_wf_fitness and mining_folds and len(mining_folds) >= 3:
            try:
                if use_weighted:
                    stats = compute_weighted_wf_fitness(
                        tree, cfg.evaluate, idx, mining_folds,
                        weights=regime_weights,
                        prob_df=mcfg.prob_df,
                        weight_cfg=mcfg.weight_cfg,
                        **wf_kwargs,
                    )
                else:
                    stats = compute_wf_fitness(
                        tree, cfg.evaluate, idx, mining_folds,
                        regime=regime, **wf_kwargs,
                    )
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

    # ── DSR Gating ──────────────────────────────────────────────────────────
    # Top-100 aday üzerinde Deflated Sharpe Ratio testi uygula.
    # Selection bias'ı azaltır: havuz büyüklüğü kadar şans düzeltmesi yapılır.
    if mcfg.use_dsr_filter and len(results) > 1:
        n_trials = len(results)
        filtered: list[MiningResult] = []
        for r in results[:100]:  # En fazla top-100'ü test et (hız için)
            try:
                _dsr_z, p_value = deflated_sharpe_ratio(
                    sr=r.mean_ric * 16,  # IC → yaklaşık annualized Sharpe
                    T=max(n_pool, 60),
                    skew=0.0,
                    kurt=0.0,
                    n_trials=n_trials,
                )
                if not np.isfinite(p_value) or p_value >= mcfg.dsr_min_p:
                    filtered.append(r)
            except Exception:
                filtered.append(r)  # DSR hatasında koru
        results = filtered + results[100:]  # 100 sonrasını olduğu gibi tut
    # ────────────────────────────────────────────────────────────────────────

    return results
