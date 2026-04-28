"""
tests/test_forensics.py — Faz 6.2 adli loglama testleri.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from engine.core.alpha_cfg import Node
from engine.execution.forensics import (
    FORENSIC_COLUMNS,
    ForensicConfig,
    load_decisions,
    log_decision_forensics,
)


def _make_prob_row(K: int = 2, top: int = 0) -> pd.Series:
    """K-rejim olasılık vektörü: top'a 0.8, kalanı eşit dağıt."""
    p = np.full(K, 0.2 / max(K - 1, 1))
    p[top] = 0.8
    return pd.Series(p, index=[f"regime_{i}" for i in range(K)])


def _make_champions(K: int = 2) -> dict:
    return {
        i: Node("feature", "Pclose")
        for i in range(K)
    }


# ──────────────────────────────────────────────────────────────────────
# 1. İki gün log → parquet 2× büyür, kolon şeması doğru
# ──────────────────────────────────────────────────────────────────────
def test_log_decision_forensics_appends_parquet(syn_db, tmp_path):
    out_path = tmp_path / "decisions_log.parquet"
    cfg = ForensicConfig(output_path=out_path, portfolio_capital_TL=1_000_000)

    dates = sorted(syn_db["Date"].unique())
    d1 = dates[100]
    d2 = dates[101]

    weights = pd.Series({"T000": 0.5, "T001": 0.3, "T002": 0.2})
    prob = _make_prob_row(K=2, top=0)
    champions = _make_champions(K=2)

    n1 = log_decision_forensics(
        date=d1, weights=weights, prob_row=prob,
        champions=champions, db=syn_db, formula_id="f1", cfg=cfg,
    )
    n2 = log_decision_forensics(
        date=d2, weights=weights, prob_row=prob,
        champions=champions, db=syn_db, formula_id="f1", cfg=cfg,
    )

    assert n1 == 3
    assert n2 == 3

    df = pd.read_parquet(out_path)
    assert len(df) == 6
    assert list(df.columns) == FORENSIC_COLUMNS
    # execution_id tekil
    assert df["execution_id"].nunique() == 6
    # action ilk gün BUY (prev=0), ikinci gün de yeni weight değeri 0'dan farklı → kayıt
    assert (df["target_weight"] > 0).all()


# ──────────────────────────────────────────────────────────────────────
# 2. load_decisions filtreleme
# ──────────────────────────────────────────────────────────────────────
def test_load_decisions_filters_by_date_and_ticker(syn_db, tmp_path):
    out_path = tmp_path / "decisions_log.parquet"
    cfg = ForensicConfig(output_path=out_path)

    dates = sorted(syn_db["Date"].unique())
    weights = pd.Series({"T000": 0.5, "T001": 0.5})
    prob = _make_prob_row(K=2, top=0)
    champions = _make_champions(K=2)

    for d in dates[100:103]:  # 3 gün
        log_decision_forensics(
            date=d, weights=weights, prob_row=prob,
            champions=champions, db=syn_db, formula_id="f1", cfg=cfg,
        )

    # Tarih filtresi: yalnız ortadaki gün
    mid = dates[101]
    filt = load_decisions(start=mid, end=mid, cfg=cfg)
    assert len(filt) == 2
    assert (filt["date"] == pd.Timestamp(mid)).all()

    # Ticker filtresi
    only_t000 = load_decisions(ticker="T000", cfg=cfg)
    assert len(only_t000) == 3
    assert (only_t000["ticker"] == "T000").all()


# ──────────────────────────────────────────────────────────────────────
# 3. hmm_state JSON round-trip
# ──────────────────────────────────────────────────────────────────────
def test_forensic_hmm_state_json_round_trip(syn_db, tmp_path):
    out_path = tmp_path / "decisions_log.parquet"
    cfg = ForensicConfig(output_path=out_path)

    dates = sorted(syn_db["Date"].unique())
    weights = pd.Series({"T000": 1.0})
    prob = _make_prob_row(K=4, top=2)
    champions = _make_champions(K=4)

    log_decision_forensics(
        date=dates[100], weights=weights, prob_row=prob,
        champions=champions, db=syn_db, formula_id="f1", cfg=cfg,
    )

    df = pd.read_parquet(out_path)
    assert len(df) == 1
    raw = df.iloc[0]["hmm_state"]
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)
    assert len(parsed) == 4
    # Top regime 2'de %80'e yakın
    top_key = max(parsed, key=parsed.get)
    assert "regime_2" in top_key or top_key.endswith("2")
    assert parsed[top_key] > 0.7
    # hmm_top_regime kolonu da 2
    assert int(df.iloc[0]["hmm_top_regime"]) == 2
