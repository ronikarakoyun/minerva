"""
tests/test_blender.py — Faz 5.2 rejim harmanlama testleri.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from engine.core.alpha_cfg import Node
from engine.execution.blender import (
    BlenderConfig,
    blend_regime_signals,
    load_champions_from_catalog,
)


def _make_synthetic_prob_df(dates: pd.DatetimeIndex, K: int = 2,
                             one_hot_for: int | None = None) -> pd.DataFrame:
    """K rejimli prob_df. one_hot_for verilirse o sütun=1, kalanı 0."""
    if one_hot_for is not None:
        prob = np.zeros((len(dates), K))
        prob[:, one_hot_for] = 1.0
    else:
        prob = np.full((len(dates), K), 1.0 / K)
    return pd.DataFrame(prob, index=dates, columns=[f"regime_{i}" for i in range(K)])


def _make_feature_tree(feature_name: str) -> Node:
    """Tek feature'lı sinyal döndüren AST."""
    return Node("feature", feature_name)


# ──────────────────────────────────────────────────────────────────────
# 1. Uniform prob → blended = ortalama
# ──────────────────────────────────────────────────────────────────────
def test_blend_uniform_prob_equals_average(syn_db, cfg):
    """K rejim için uniform prob → blended = mean(champion_signals)."""
    champions = {
        0: _make_feature_tree("Pclose"),
        1: _make_feature_tree("Phigh"),
    }
    dates = pd.DatetimeIndex(sorted(syn_db["Date"].unique()))
    prob_df = _make_synthetic_prob_df(dates, K=2)

    bcfg = BlenderConfig(use_blending=True, top_k=10, smoothing_alpha=1.0)
    weights = blend_regime_signals(champions, prob_df, syn_db, bcfg, alpha_cfg=cfg)

    # Boş değil, top_k satırı = 10 hisse 0.1 ağırlıkta
    last_row = weights.iloc[-5]  # son birkaç gün arası — warmup hariç
    nonzero = (last_row > 0).sum()
    assert nonzero <= 10
    assert nonzero >= 1
    # Σ ≈ 1.0
    assert abs(last_row.sum() - 1.0) < 0.01 or last_row.sum() == 0


# ──────────────────────────────────────────────────────────────────────
# 2. One-hot prob → blended = o rejim şampiyonu
# ──────────────────────────────────────────────────────────────────────
def test_blend_one_hot_prob_equals_champion(syn_db, cfg):
    """prob = e_k → blend çıktısı sadece champion[k] sinyaline bağlı olmalı."""
    champions = {
        0: _make_feature_tree("Pclose"),
        1: _make_feature_tree("Plow"),
    }
    dates = pd.DatetimeIndex(sorted(syn_db["Date"].unique()))

    # prob = e_0 (her gün %100 rejim 0)
    prob_e0 = _make_synthetic_prob_df(dates, K=2, one_hot_for=0)
    bcfg = BlenderConfig(use_blending=True, top_k=5, smoothing_alpha=1.0)
    w_e0 = blend_regime_signals(champions, prob_e0, syn_db, bcfg, alpha_cfg=cfg)

    # prob = e_1
    prob_e1 = _make_synthetic_prob_df(dates, K=2, one_hot_for=1)
    w_e1 = blend_regime_signals(champions, prob_e1, syn_db, bcfg, alpha_cfg=cfg)

    # İki rejim şampiyonları farklı feature → seçilen tickerlar genelde farklı
    # En azından bir günde top-K seti farklı olmalı
    diff_count = 0
    for d in dates[20:]:
        s0 = set(w_e0.loc[d][w_e0.loc[d] > 0].index)
        s1 = set(w_e1.loc[d][w_e1.loc[d] > 0].index)
        if s0 != s1:
            diff_count += 1

    assert diff_count > 0, "İki farklı rejim şampiyonu aynı portföyü üretti"


# ──────────────────────────────────────────────────────────────────────
# 3. EMA smoothing turnover'ı azaltır
# ──────────────────────────────────────────────────────────────────────
def test_smoothing_reduces_turnover(syn_db, cfg):
    """α=0.3 → ardışık gün ağırlık farkı, α=1.0 versiyondan daha küçük."""
    champions = {
        0: _make_feature_tree("Pclose"),
        1: _make_feature_tree("Vlot"),
    }
    dates = pd.DatetimeIndex(sorted(syn_db["Date"].unique()))

    # Salınan prob — gün gün e_0 / e_1 arası geçiş
    prob_alt = pd.DataFrame(
        {"regime_0": [1.0 if i % 2 == 0 else 0.0 for i in range(len(dates))],
         "regime_1": [0.0 if i % 2 == 0 else 1.0 for i in range(len(dates))]},
        index=dates,
    )

    bcfg_smooth = BlenderConfig(use_blending=True, top_k=5, smoothing_alpha=0.3)
    bcfg_noop   = BlenderConfig(use_blending=True, top_k=5, smoothing_alpha=1.0)

    w_smooth = blend_regime_signals(champions, prob_alt, syn_db, bcfg_smooth, alpha_cfg=cfg)
    w_noop   = blend_regime_signals(champions, prob_alt, syn_db, bcfg_noop,   alpha_cfg=cfg)

    # Toplam ardışık gün ağırlık değişimi (turnover proxy)
    turnover_smooth = w_smooth.diff().abs().sum().sum()
    turnover_noop   = w_noop.diff().abs().sum().sum()

    assert turnover_smooth < turnover_noop, \
        f"smoothing turnover'ı azaltmadı: {turnover_smooth:.2f} >= {turnover_noop:.2f}"


# ──────────────────────────────────────────────────────────────────────
# 4. Catalog round-trip — save_regime_champion → load
# ──────────────────────────────────────────────────────────────────────
def test_load_champions_from_catalog_round_trip(tmp_path, cfg, monkeypatch):
    """save_regime_champion ile yazılmış formüller load_champions ile geri okunabilir."""
    from engine.core import alpha_catalog as ac
    from engine.core.formula_parser import parse_formula

    # Geçici catalog path
    fake_catalog = tmp_path / "alpha_catalog.json"
    monkeypatch.setattr(ac, "CATALOG_PATH", str(fake_catalog))

    # 2 rejim şampiyonu yaz
    f0 = "Pclose"
    f1 = "Phigh"
    t0 = parse_formula(f0, cfg)
    t1 = parse_formula(f1, cfg)
    ac.save_regime_champion(0, f0, t0, ic=0.05, rank_ic=0.05, adj_ic=0.05)
    ac.save_regime_champion(1, f1, t1, ic=0.04, rank_ic=0.04, adj_ic=0.04)

    # Geri yükle
    champions = load_champions_from_catalog(fake_catalog, alpha_cfg=cfg)

    assert set(champions.keys()) == {0, 1}
    assert champions[0].kind == "feature"
    assert champions[0].op == "Pclose"
    assert champions[1].op == "Phigh"
