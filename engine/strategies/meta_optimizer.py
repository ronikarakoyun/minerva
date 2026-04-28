"""
engine/strategies/meta_optimizer.py — Faz 3 dış döngü: Optuna ile haftalık
hyperparameter meta-tuning.

MCTS, formül üretmek için "ne kadar derin arasın", "ne kadar keşif yapsın",
"rejim ağırlığı ne kadar keskin olsun" gibi soruları cevaplayamaz. Bu modül,
Bayesian optimization ile bu hyperparametreleri haftalık olarak BIST'in güncel
karakterine göre tunelarak `data/best_params.json` üretir.

Kullanım (CLI):
    python -m engine.strategies.meta_optimizer
    python -m engine.strategies.meta_optimizer --n-trials 30
    python -m engine.strategies.meta_optimizer --storage sqlite:///data/optuna.db

Kullanım (programmatic):
    from engine.strategies.meta_optimizer import MetaOptConfig, run_meta_optimization
    best = run_meta_optimization(MetaOptConfig(n_trials=20))

Akış:
    1. data/market_db.parquet → train pencere (son N gün)
    2. data/regime_hmm.pkl → prob_df (Faz 1 çıktısı)
    3. Optuna 6 parametre üzerinde n_trials denemeye → her trial = kısa MCTS mining
    4. Objective: top-10 formülün medyan mean_ric'i (yüksek = iyi)
    5. Çıktı: data/best_params.json (timestamp + best_value + best_params)

Faz 3 kuralı: meta-tuning sadece kapalı geçmiş train window'da yapılır;
production'da bir sonraki haftaya uygulanır (look-ahead yok).
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import optuna
import pandas as pd

from ..core.alpha_cfg import AlphaCFG
from ..validation.weighted_fitness import WeightConfig
from .mining_runner import MiningConfig, run_mining_window

logger = logging.getLogger(__name__)


@dataclass
class MetaOptConfig:
    """Optuna study parametreleri."""
    n_trials: int = 50
    train_window_days: int = 252         # ~1 iş yılı
    db_path: Path = field(default_factory=lambda: Path("data/market_db.parquet"))
    prob_df_pkl: Path = field(default_factory=lambda: Path("data/regime_hmm.pkl"))
    output_path: Path = field(default_factory=lambda: Path("data/best_params.json"))
    optuna_storage: Optional[str] = None  # SQLite: "sqlite:///data/optuna.db"
    study_name: str = "minerva_v3_mcts"
    trial_num_gen: int = 50               # her trial için kısa mining bütçesi
    trial_iters_per_root: int = 30
    seed: int = 42


def _load_prob_df(pkl_path: Path) -> pd.DataFrame:
    """Faz 1 çıktısını üret: HMM modelini yükle, prob_df'i hesapla."""
    import joblib

    from ..data.regime_detector import (
        RegimeConfig,
        compute_features,
        compute_probability_vector,
        fetch_bist_data,
    )
    cfg = RegimeConfig(model_path=pkl_path)
    df = fetch_bist_data(cfg)
    features = compute_features(df, cfg)
    model = joblib.load(pkl_path)
    return compute_probability_vector(model, features)


def _load_db(db_path: Path, train_window_days: int) -> pd.DataFrame:
    """market_db.parquet'i yükle ve son train_window_days iş gününü kes."""
    db = pd.read_parquet(db_path)
    db["Date"] = pd.to_datetime(db["Date"])
    unique_dates = sorted(db["Date"].unique())
    if len(unique_dates) > train_window_days:
        cutoff = unique_dates[-train_window_days]
        db = db[db["Date"] >= cutoff].copy()
    return db


def build_objective(
    db: pd.DataFrame,
    alpha_cfg: AlphaCFG,
    prob_df: pd.DataFrame,
    meta_cfg: MetaOptConfig,
):
    """Optuna objective factory: closure ile veriyi yakalar."""

    def objective(trial: optuna.Trial) -> float:
        mcfg = MiningConfig(
            search_mode="mcts",
            num_gen=meta_cfg.trial_num_gen,
            max_K=trial.suggest_int("max_K", 8, 20),
            c_puct=trial.suggest_float("c_puct", 0.5, 3.0),
            mcts_rollouts=trial.suggest_int("mcts_rollouts", 8, 32),
            mcts_iterations_per_root=meta_cfg.trial_iters_per_root,
            lambda_std=trial.suggest_float("lambda_std", 0.5, 4.0),
            lambda_cx=trial.suggest_float("lambda_cx", 0.001, 0.01, log=True),
            use_regime_weighting=True,
            weight_cfg=WeightConfig(
                temperature=trial.suggest_float("temperature", 0.5, 5.0),
            ),
            prob_df=prob_df,
            seed=meta_cfg.seed,
        )

        try:
            results = run_mining_window(db, alpha_cfg, mcfg)
        except Exception as e:
            logger.warning("trial %d failed: %s", trial.number, e)
            return -1e6

        if len(results) < 5:
            return -1e6

        top = sorted([r.mean_ric for r in results], reverse=True)[:10]
        score = float(np.median(top))
        if not np.isfinite(score):
            return -1e6
        return score

    return objective


def run_meta_optimization(meta_cfg: Optional[MetaOptConfig] = None) -> dict:
    """Optuna study çalıştır, en iyi parametreleri JSON'a yaz."""
    meta_cfg = meta_cfg or MetaOptConfig()
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )

    logger.info("=== Faz 3: Meta-Optimization (Optuna) ===")
    logger.info("DB: %s | window: %d gün | n_trials: %d",
                meta_cfg.db_path, meta_cfg.train_window_days, meta_cfg.n_trials)

    db = _load_db(meta_cfg.db_path, meta_cfg.train_window_days)
    prob_df = _load_prob_df(meta_cfg.prob_df_pkl)
    alpha_cfg = AlphaCFG()

    sampler = optuna.samplers.TPESampler(seed=meta_cfg.seed)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        storage=meta_cfg.optuna_storage,
        load_if_exists=True,
        study_name=meta_cfg.study_name,
    )

    objective = build_objective(db, alpha_cfg, prob_df, meta_cfg)
    study.optimize(objective, n_trials=meta_cfg.n_trials, show_progress_bar=False)

    best = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "study_name": meta_cfg.study_name,
        "n_trials": len(study.trials),
        "best_value": float(study.best_value),
        "best_params": dict(study.best_params),
        "train_window_days": meta_cfg.train_window_days,
    }
    meta_cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(meta_cfg.output_path) + ".lock"
    import fcntl
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            meta_cfg.output_path.write_text(json.dumps(best, indent=2, ensure_ascii=False))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)
    logger.info("=== DONE — best_value=%.4f ===", best["best_value"])
    logger.info("best_params=%s", best["best_params"])
    logger.info("Saved: %s", meta_cfg.output_path)
    return best


def _parse_args() -> MetaOptConfig:
    p = argparse.ArgumentParser(description="Minerva v3 Faz 3 — Optuna meta-tuning")
    p.add_argument("--n-trials", type=int, default=50)
    p.add_argument("--window", type=int, default=252, help="train window (iş günü)")
    p.add_argument("--storage", type=str, default=None,
                   help="Optuna storage (örn. sqlite:///data/optuna.db)")
    p.add_argument("--db", type=Path, default=Path("data/market_db.parquet"))
    p.add_argument("--output", type=Path, default=Path("data/best_params.json"))
    args = p.parse_args()
    return MetaOptConfig(
        n_trials=args.n_trials,
        train_window_days=args.window,
        optuna_storage=args.storage,
        db_path=args.db,
        output_path=args.output,
    )


if __name__ == "__main__":
    cfg = _parse_args()
    run_meta_optimization(cfg)
