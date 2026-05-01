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
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from filelock import FileLock

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
    output_dir = meta_cfg.output_path.parent

    with FileLock(lock_path, timeout=15):
        fd, tmp_path = tempfile.mkstemp(dir=output_dir, suffix=".tmp")
        try:
            with open(fd, "w", encoding="utf-8") as f:
                json.dump(best, f, indent=2, ensure_ascii=False)
            Path(tmp_path).replace(meta_cfg.output_path)
        except Exception:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except OSError:
                pass
            raise
    logger.info("=== DONE — best_value=%.4f ===", best["best_value"])
    logger.info("best_params=%s", best["best_params"])
    logger.info("Saved: %s", meta_cfg.output_path)
    return best


@dataclass
class WindowTuneResult:
    """Tek bir eğitim penceresi için ayar sonucu."""
    window_index: int
    start_date: Any          # pd.Timestamp
    end_date: Any            # pd.Timestamp
    best_value: float
    best_params: dict


@dataclass
class RollingTuneResult:
    """rolling_tune() çıktısı: pencere bazlı sonuçlar + mutabakat parametreler."""
    window_results: list[WindowTuneResult]
    consensus_params: dict   # Her parametre üzerinden medyan
    best_window_index: int   # En yüksek best_value'ye sahip pencere
    n_windows: int


def rolling_tune(
    db: pd.DataFrame,
    cfg: Optional[MetaOptConfig] = None,
    n_windows: int = 3,
    window_days: int = 252,
) -> RollingTuneResult:
    """N14 — Çakışan birden fazla eğitim penceresinde tune() çalıştır.

    Her pencere için ayrı bir Optuna study çalıştırılır; ardından her
    hiperparametrenin medyanı "mutabakat" (consensus) parametre seti olarak
    döndürülür. Bu sayede tek bir döneme aşırı uyum (overfitting) azaltılır.

    Parametreler
    ----------
    db          : market_db.parquet içeriği (Date sütunu datetime).
    cfg         : MetaOptConfig; None ise varsayılan kullanılır.
    n_windows   : Oluşturulacak pencere sayısı (varsayılan 3).
    window_days : Her pencerenin iş günü cinsinden uzunluğu (varsayılan 252).

    Döndürür
    -------
    RollingTuneResult
    """
    cfg = cfg or MetaOptConfig()
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )

    db = db.copy()
    db["Date"] = pd.to_datetime(db["Date"])
    unique_dates = sorted(db["Date"].unique())
    total_days = len(unique_dates)

    if total_days < window_days:
        raise ValueError(
            f"DB'de {total_days} gün var, ancak window_days={window_days} gerekli."
        )

    # Pencereleri oluştur: son pencere en sondaki window_days günü kapsar;
    # önceki pencereler geriye doğru kaydırılır.
    step = max(1, (total_days - window_days) // max(1, n_windows - 1))
    window_results: list[WindowTuneResult] = []
    prob_df = _load_prob_df(cfg.prob_df_pkl)
    alpha_cfg_obj = AlphaCFG()

    for i in range(n_windows):
        end_idx = total_days - i * step
        start_idx = max(0, end_idx - window_days)
        window_dates = unique_dates[start_idx:end_idx]
        if len(window_dates) < 2:
            logger.warning("Pencere %d çok kısa, atlanıyor.", i)
            continue

        start_date, end_date = window_dates[0], window_dates[-1]
        window_db = db[db["Date"].isin(window_dates)].copy()
        logger.info(
            "Rolling tune pencere %d/%d: %s → %s (%d gün)",
            i + 1, n_windows, start_date.date(), end_date.date(), len(window_dates),
        )

        sampler = optuna.samplers.TPESampler(seed=cfg.seed + i)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        objective = build_objective(window_db, alpha_cfg_obj, prob_df, cfg)
        study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=False)

        window_results.append(WindowTuneResult(
            window_index=i,
            start_date=start_date,
            end_date=end_date,
            best_value=float(study.best_value),
            best_params=dict(study.best_params),
        ))
        logger.info("Pencere %d tamamlandı — best_value=%.4f", i + 1, study.best_value)

    if not window_results:
        raise RuntimeError("Hiçbir pencere başarıyla tamamlanamadı.")

    # Mutabakat: sayısal parametreler için medyan al
    all_params_keys = window_results[0].best_params.keys()
    consensus: dict = {}
    for key in all_params_keys:
        values = [wr.best_params[key] for wr in window_results]
        if all(isinstance(v, int) for v in values):
            consensus[key] = int(np.median(values))
        else:
            consensus[key] = float(np.median([float(v) for v in values]))

    best_window_index = int(
        np.argmax([wr.best_value for wr in window_results])
    )
    logger.info(
        "Rolling tune tamamlandı — consensus=%s, best_window=%d",
        consensus, best_window_index,
    )

    return RollingTuneResult(
        window_results=window_results,
        consensus_params=consensus,
        best_window_index=best_window_index,
        n_windows=len(window_results),
    )


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
