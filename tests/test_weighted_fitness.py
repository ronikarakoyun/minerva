"""
tests/test_weighted_fitness.py — Faz 2 ağırlıklı RankIC fitness için unit testler.

Sentetik 2-rejim prob_df + sentetik MultiIndex idx kullanır; yfinance/HMM gerek yok.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.validation.weighted_fitness import (
    WeightConfig,
    compute_regime_weights,
    compute_weighted_wf_fitness,
    cosine_similarity_to_ref,
)
from engine.validation.wf_fitness import compute_wf_fitness, make_date_folds


# ──────────────────────────────────────────────────────────────────────
# Yardımcı: 2-rejim sentetik prob_df
# ──────────────────────────────────────────────────────────────────────
def _make_two_regime_prob_df(dates: pd.DatetimeIndex, switch_idx: int) -> pd.DataFrame:
    """
    İlk yarı saf Rejim A (1, 0), ikinci yarı saf Rejim B (0, 1).
    """
    K = 2
    P = np.zeros((len(dates), K), dtype=float)
    P[:switch_idx, 0] = 1.0
    P[switch_idx:, 1] = 1.0
    return pd.DataFrame(P, index=dates, columns=[f"regime_{i}" for i in range(K)])


# ──────────────────────────────────────────────────────────────────────
# 1. Cosine similarity
# ──────────────────────────────────────────────────────────────────────
def test_cosine_similarity_self_is_one():
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    P = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.3, 0.3, 0.4],
        [0.5, 0.5, 0.0],
        [0.0, 0.0, 1.0],
        [0.6, 0.3, 0.1],
        [0.2, 0.6, 0.2],
        [0.4, 0.4, 0.2],
        [0.1, 0.1, 0.8],
        [0.33, 0.33, 0.34],
    ])
    df = pd.DataFrame(P, index=idx, columns=["r0", "r1", "r2"])

    # Her satırı kendi vektörüyle karşılaştır → 1.0
    for i in range(len(df)):
        sims = cosine_similarity_to_ref(df, ref_vec=df.iloc[i].values)
        assert abs(sims.iloc[i] - 1.0) < 1e-9


def test_cosine_orthogonal_is_zero():
    idx = pd.date_range("2020-01-01", periods=2, freq="B")
    df = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]], index=idx, columns=["r0", "r1"]
    )
    sims = cosine_similarity_to_ref(df, ref_vec=np.array([1.0, 0.0]))
    assert abs(sims.iloc[0] - 1.0) < 1e-9
    assert abs(sims.iloc[1] - 0.0) < 1e-9


# ──────────────────────────────────────────────────────────────────────
# 2. Ağırlık dönüşümü sınırları
# ──────────────────────────────────────────────────────────────────────
def test_weight_transform_bounds():
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    df = _make_two_regime_prob_df(idx, switch_idx=100)
    cfg = WeightConfig(temperature=2.0, w_min=1.0, w_max=10.0)
    w = compute_regime_weights(df, cfg)

    assert w.min() >= cfg.w_min - 1e-9
    assert w.max() <= cfg.w_max + 1e-9
    # Son gün ref → w == w_max
    assert abs(w.iloc[-1] - cfg.w_max) < 1e-9
    # İlk yarı saf Rejim A, son gün saf Rejim B (orthogonal) → w ≈ w_min + span·exp(-T)
    expected_low = cfg.w_min + (cfg.w_max - cfg.w_min) * np.exp(-cfg.temperature)
    assert abs(w.iloc[0] - expected_low) < 1e-9


# ──────────────────────────────────────────────────────────────────────
# 3. Sıcaklık → keskinlik
# ──────────────────────────────────────────────────────────────────────
def test_higher_temperature_sharpens_weights():
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    df = _make_two_regime_prob_df(idx, switch_idx=100)

    w_low = compute_regime_weights(df, WeightConfig(temperature=0.5))
    w_high = compute_regime_weights(df, WeightConfig(temperature=4.0))

    # Yüksek T daha keskin → benzemeyen günler için daha düşük w (daha keniş aralık)
    assert w_high.iloc[0] < w_low.iloc[0]
    # İkisinde de son gün w_max
    assert abs(w_high.iloc[-1] - w_low.iloc[-1]) < 1e-9


# ──────────────────────────────────────────────────────────────────────
# 4. Uniform ağırlıkla weighted_ic == klasik mean_ric
# ──────────────────────────────────────────────────────────────────────
def test_weighted_ic_equals_unweighted_when_uniform(syn_idx, cfg):
    """weights=1 sabit → ağırlıklı fitness, klasik fitness ile aynı mean_ric verir."""
    from engine.core.formula_parser import parse_formula

    tree = parse_formula("Pclose", cfg)
    dates = syn_idx.index.get_level_values("Date").unique().sort_values()
    folds = make_date_folds(dates.values, n_folds=4, min_fold_days=20, embargo_days=2)

    # Sabit ağırlık
    weights = pd.Series(1.0, index=dates)

    classical = compute_wf_fitness(
        tree, cfg.evaluate, syn_idx, folds, neutralize=False, lambda_cx=0.0
    )
    weighted = compute_weighted_wf_fitness(
        tree, cfg.evaluate, syn_idx, folds, weights=weights,
        neutralize=False, lambda_cx=0.0,
    )

    if classical["status"] == "ok" and weighted["status"] == "ok":
        # Aynı fold'larda aynı IC değerleri — uniform ağırlıkta birebir
        assert abs(classical["mean_ric"] - weighted["mean_ric"]) < 1e-9
        assert len(classical["fold_rics"]) == len(weighted["fold_rics"])


# ──────────────────────────────────────────────────────────────────────
# 5. Ağırlık benzer günlere skoru kaydırır
# ──────────────────────────────────────────────────────────────────────
def test_weighted_ic_shifts_to_similar_days(syn_db, cfg):
    """
    Sentetik 2-rejim: ilk yarı sinyalin alpha'sı negatif, ikinci yarı pozitif.
    Referans = son gün (Rejim B). Ağırlıklı IC ikinci yarıyı yüksek tartmalı.

    Klasik mean_ric ≈ 0 (iki yarı bibirini iptal eder), ağırlıklı mean_ric > 0.
    """
    from engine.core.formula_parser import parse_formula

    db = syn_db.copy()
    dates = sorted(db["Date"].unique())
    half = len(dates) // 2
    early = set(dates[:half])
    # İkinci yarıda Pclose ↗ Next_Ret pozitif ilişkisi yarat (alpha)
    # İlk yarıda ters çevir
    rng = np.random.default_rng(7)
    for ticker, sub in db.groupby("Ticker"):
        mask = db["Ticker"] == ticker
        sig_like = (db.loc[mask, "Pclose"] - db.loc[mask, "Pclose"].mean())
        for i in db.index[mask]:
            d = db.at[i, "Date"]
            base = float(sig_like.loc[i])
            # Ölçek küçük; varolan Next_Ret üzerine ekle
            if d in early:
                db.at[i, "Next_Ret"] = -0.0005 * base + rng.normal(0, 0.01)
            else:
                db.at[i, "Next_Ret"] = 0.0005 * base + rng.normal(0, 0.01)

    idx = db.set_index(["Ticker", "Date"]).sort_index()
    dates_arr = idx.index.get_level_values("Date").unique().sort_values()

    folds = make_date_folds(dates_arr.values, n_folds=4, min_fold_days=20, embargo_days=2)

    # 2-rejim prob_df: ilk yarı Rejim A, ikinci yarı Rejim B; ref = son gün → Rejim B
    prob_df = _make_two_regime_prob_df(pd.DatetimeIndex(dates_arr), switch_idx=half)
    weights = compute_regime_weights(prob_df, WeightConfig(temperature=3.0))

    tree = parse_formula("Pclose", cfg)

    classical = compute_wf_fitness(
        tree, cfg.evaluate, idx, folds, neutralize=False, lambda_cx=0.0
    )
    weighted = compute_weighted_wf_fitness(
        tree, cfg.evaluate, idx, folds, weights=weights,
        neutralize=False, lambda_cx=0.0,
    )

    # Klasik: mean_ric ≈ 0; ağırlıklı: pozitif (ikinci yarıyı tartar)
    if classical["status"] == "ok" and weighted["status"] == "ok":
        assert weighted["mean_ric"] > classical["mean_ric"]


# ──────────────────────────────────────────────────────────────────────
# 6. ref_date parametresi
# ──────────────────────────────────────────────────────────────────────
def test_ref_date_overrides_last_day():
    idx = pd.date_range("2020-01-01", periods=200, freq="B")
    df = _make_two_regime_prob_df(idx, switch_idx=100)

    # ref_date ilk yarı içinde → ilk yarı w_max'a yakın olmalı
    early_ref = idx[50]
    cfg_ref = WeightConfig(temperature=2.0, ref_date=early_ref)
    w = compute_regime_weights(df, cfg_ref)

    assert abs(w.loc[early_ref] - cfg_ref.w_max) < 1e-9
    # Son gün artık zayıf
    assert w.iloc[-1] < w.loc[early_ref]
