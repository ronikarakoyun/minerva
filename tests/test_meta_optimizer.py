"""
tests/test_meta_optimizer.py — Faz 3 Optuna meta-optimizer için unit testler.

Sentetik db + sentetik prob_df ile gerçek Optuna study koşturur
(yfinance/HMM mock'lanır).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import pytest

from engine.strategies.meta_optimizer import (
    MetaOptConfig,
    build_objective,
    run_meta_optimization,
)
from engine.strategies.mining_runner import MiningConfig


# ──────────────────────────────────────────────────────────────────────
# Yardımcı: sentetik prob_df
# ──────────────────────────────────────────────────────────────────────
def _make_synthetic_prob_df(dates: pd.DatetimeIndex, K: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(13)
    raw = rng.dirichlet(np.ones(K), size=len(dates))
    return pd.DataFrame(
        raw, index=dates, columns=[f"regime_{i}" for i in range(K)]
    )


# ──────────────────────────────────────────────────────────────────────
# 1. Objective sonlu skor döndürür
# ──────────────────────────────────────────────────────────────────────
def test_objective_returns_finite(syn_db, cfg):
    """Tek trial koş, NaN/inf olmadığını doğrula."""
    dates = pd.DatetimeIndex(sorted(syn_db["Date"].unique()))
    prob_df = _make_synthetic_prob_df(dates, K=3)
    meta_cfg = MetaOptConfig(
        n_trials=1, trial_num_gen=8, trial_iters_per_root=6,
    )
    objective = build_objective(syn_db, cfg, prob_df, meta_cfg)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    val = study.best_value
    assert np.isfinite(val), f"objective non-finite: {val}"
    # Başarısız trial guard değeri (-1e6) bile sonlu — sadece NaN/inf veto
    assert val > -1e7


# ──────────────────────────────────────────────────────────────────────
# 2. Best-params JSON persist
# ──────────────────────────────────────────────────────────────────────
def test_best_params_persisted(syn_db, cfg, tmp_path: Path, monkeypatch):
    """Tam pipeline: meta-opt çalıştır → JSON oluşur → anahtarlar doğru."""
    dates = pd.DatetimeIndex(sorted(syn_db["Date"].unique()))
    prob_df = _make_synthetic_prob_df(dates, K=3)

    # _load_db ve _load_prob_df'i mock et
    from engine.strategies import meta_optimizer as mo
    monkeypatch.setattr(mo, "_load_db", lambda *a, **k: syn_db)
    monkeypatch.setattr(mo, "_load_prob_df", lambda *a, **k: prob_df)
    # AlphaCFG sentetik veri için cfg fixture'ı kullan
    monkeypatch.setattr(mo, "AlphaCFG", lambda: cfg)

    out = tmp_path / "best.json"
    meta_cfg = MetaOptConfig(
        n_trials=2,
        trial_num_gen=8,
        trial_iters_per_root=6,
        output_path=out,
        db_path=tmp_path / "fake.parquet",
        prob_df_pkl=tmp_path / "fake.pkl",
    )
    result = run_meta_optimization(meta_cfg)

    assert out.exists()
    payload = json.loads(out.read_text())
    required = {"trained_at", "study_name", "n_trials", "best_value",
                "best_params", "train_window_days"}
    assert required.issubset(payload.keys())
    assert payload["n_trials"] == 2

    # 6 kritik parametrenin tümü best_params'ta
    expected_keys = {"max_K", "c_puct", "mcts_rollouts",
                     "lambda_std", "lambda_cx", "temperature"}
    assert expected_keys.issubset(payload["best_params"].keys())


# ──────────────────────────────────────────────────────────────────────
# 3. from_best_params round-trip
# ──────────────────────────────────────────────────────────────────────
def test_from_best_params_round_trip(tmp_path: Path):
    """JSON → MiningConfig — değerler doğru yüklensin."""
    payload = {
        "trained_at": "2026-04-27T10:00:00+00:00",
        "study_name": "test",
        "n_trials": 5,
        "best_value": 0.123,
        "best_params": {
            "max_K": 12,
            "c_puct": 2.1,
            "mcts_rollouts": 20,
            "lambda_std": 1.7,
            "lambda_cx": 0.0042,
            "temperature": 3.3,
        },
    }
    json_path = tmp_path / "best.json"
    json_path.write_text(json.dumps(payload))

    mcfg = MiningConfig.from_best_params(json_path, num_gen=80)
    assert mcfg.search_mode == "mcts"
    assert mcfg.max_K == 12
    assert abs(mcfg.c_puct - 2.1) < 1e-9
    assert mcfg.mcts_rollouts == 20
    assert abs(mcfg.lambda_std - 1.7) < 1e-9
    assert abs(mcfg.lambda_cx - 0.0042) < 1e-9
    assert abs(mcfg.weight_cfg.temperature - 3.3) < 1e-9
    assert mcfg.use_regime_weighting is True
    # override geçti
    assert mcfg.num_gen == 80


# ──────────────────────────────────────────────────────────────────────
# 4. Optuna SQLite storage resume
# ──────────────────────────────────────────────────────────────────────
def test_optuna_storage_resume(syn_db, cfg, tmp_path: Path, monkeypatch):
    """SQLite storage → 2. run'da study yüklenir, trial sayısı birikir."""
    dates = pd.DatetimeIndex(sorted(syn_db["Date"].unique()))
    prob_df = _make_synthetic_prob_df(dates, K=3)

    from engine.strategies import meta_optimizer as mo
    monkeypatch.setattr(mo, "_load_db", lambda *a, **k: syn_db)
    monkeypatch.setattr(mo, "_load_prob_df", lambda *a, **k: prob_df)
    monkeypatch.setattr(mo, "AlphaCFG", lambda: cfg)

    storage = f"sqlite:///{tmp_path}/optuna.db"
    out = tmp_path / "best.json"
    common = dict(
        trial_num_gen=8,
        trial_iters_per_root=6,
        output_path=out,
        db_path=tmp_path / "fake.parquet",
        prob_df_pkl=tmp_path / "fake.pkl",
        optuna_storage=storage,
        study_name="resume_test",
    )

    # 1. run: 2 trial
    run_meta_optimization(MetaOptConfig(n_trials=2, **common))
    payload1 = json.loads(out.read_text())
    assert payload1["n_trials"] == 2

    # 2. run: 2 trial daha → toplam 4 birikir
    run_meta_optimization(MetaOptConfig(n_trials=2, **common))
    payload2 = json.loads(out.read_text())
    assert payload2["n_trials"] == 4
