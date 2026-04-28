"""
tests/test_mcts_integration.py — Faz 3 MCTS-mining entegrasyon testleri.

GrammarMCTS'in mining_runner'a `search_mode="mcts"` flag'i ile bağlandığını ve
mevcut GP davranışını bozmadığını doğrular.
"""
from __future__ import annotations

import pandas as pd
import pytest

from engine.strategies.mcts import GrammarMCTS
from engine.strategies.mining_runner import MiningConfig, run_mining_window


# ──────────────────────────────────────────────────────────────────────
# 1. MCTS modu pool üretir
# ──────────────────────────────────────────────────────────────────────
def test_search_mode_mcts_produces_pool(syn_db, cfg):
    """search_mode='mcts' ile mining_runner çağrısı bir pool üretir ve fitness değerlendirir."""
    mcfg = MiningConfig(
        search_mode="mcts",
        num_gen=8,                       # küçük havuz, hızlı test
        max_K=10,
        c_puct=1.4,
        mcts_rollouts=4,
        mcts_iterations_per_root=8,      # düşük bütçe
        use_wf_fitness=True,
        wf_n_folds=3,
        wf_purge=2,
        wf_embargo=2,
        neutralize=False,
        min_mean_ric=-1.0,               # filtreyi gevşet — herhangi bir sonuç üretmek yeter
        min_pos_ratio=0.0,
        seed=42,
    )

    results = run_mining_window(syn_db, cfg, mcfg)
    # MCTS pool üretti, fitness pipeline koştu — boş olmayan sonuç bekleniyor
    assert isinstance(results, list)
    # Bazı formüller filtreden geçmiş olmalı
    assert len(results) >= 1


# ──────────────────────────────────────────────────────────────────────
# 2. GP modu (default) bozulmadı
# ──────────────────────────────────────────────────────────────────────
def test_search_mode_gp_default_unchanged(syn_db, cfg):
    """Default search_mode='gp' eski davranışı korur."""
    mcfg = MiningConfig(
        search_mode="gp",   # explicit
        num_gen=12,
        max_K=10,
        use_wf_fitness=True,
        wf_n_folds=3,
        wf_purge=2,
        wf_embargo=2,
        neutralize=False,
        min_mean_ric=-1.0,
        min_pos_ratio=0.0,
        seed=42,
    )
    results = run_mining_window(syn_db, cfg, mcfg)
    assert isinstance(results, list)
    assert len(results) >= 1
    # MCTS özel anahtarları bu modda etkili değil (placeholder fitness aynı semantik)


# ──────────────────────────────────────────────────────────────────────
# 3. GrammarMCTS subtree_prior etkilidir
# ──────────────────────────────────────────────────────────────────────
def test_grammar_mcts_with_subtree_prior(cfg):
    """seed_trees verilince MCTS prior'u alır ve search edebilir."""
    # 3 seed formül üret
    seeds = [cfg.generate(8) for _ in range(3)]
    prior = {str(s): 0.5 for s in seeds}

    mcts = GrammarMCTS(
        cfg, max_K=10, c_puct=1.4, rollouts=4,
        subtree_prior=prior,
    )
    # Prior dict yüklenmiş
    assert len(mcts.subtree_prior) == 3
    # Search bir Node döndürür
    out = mcts.search(iterations=10)
    assert out is not None
    assert hasattr(out, "kind")
