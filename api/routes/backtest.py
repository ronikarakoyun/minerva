"""POST /api/backtest/* — Faz 2 doğrulama modülleri + Workbench job-based backtest."""
from __future__ import annotations

import asyncio
import itertools
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter

from api.deps import get_benchmark, get_cfg, get_market_db, get_split_date
from api.jobs import registry
from api.schemas import EvaluateRequest, EvaluateResponse
from engine.api_helpers import (
    evaluate_ic,
    parse_or_raise,
    prepare_eval_idx,
    run_full_evaluate,
    slice_db_by_window,
)
from pydantic import BaseModel

router = APIRouter()


# ─── Multi-parse (LLM Trainer) ──────────────────────────────────────────────

class MultiParseRequest(BaseModel):
    formulas: list[str]
    window: str = "test"
    wf_fitness: bool = False


class ParseResult(BaseModel):
    formula: str
    ok: bool
    ic: Optional[float] = None
    rank_ic: Optional[float] = None
    error: Optional[str] = None


@router.post("/parse-multi", response_model=list[ParseResult])
def parse_multi(req: MultiParseRequest) -> list[ParseResult]:
    """Birden fazla formülü parse et + IC hesapla (LLM Trainer pasta alanı)."""
    cfg = get_cfg()
    db = get_market_db()
    split_ts = get_split_date()

    df_window = slice_db_by_window(db, req.window, split_ts)
    idx = prepare_eval_idx(df_window)

    from api.deps import get_brain
    brain = get_brain()
    buf = brain["buffer"]

    results: list[ParseResult] = []
    for formula in req.formulas:
        formula = formula.strip()
        if not formula:
            continue
        try:
            tree = parse_or_raise(formula, cfg)
            ic_stats = evaluate_ic(tree, idx, cfg)
            rank_ic = ic_stats.get("rank_ic") or 0.0
            results.append(ParseResult(
                formula=formula,
                ok=True,
                ic=ic_stats.get("ic"),
                rank_ic=rank_ic,
            ))
            if abs(rank_ic) > 0.002:
                buf.add(tree, float(rank_ic))
        except Exception as e:
            results.append(ParseResult(formula=formula, ok=False, error=str(e)))

    return results


# ─── DSR ────────────────────────────────────────────────────────────────────

class DSRRequest(BaseModel):
    formula: str
    window: str = "test"
    n_trials: int = 0       # 0 = katalog büyüklüğü


class DSRResponse(BaseModel):
    sharpe_obs: Optional[float] = None
    dsr: Optional[float] = None
    p_value: Optional[float] = None
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    error: Optional[str] = None


@router.post("/dsr", response_model=DSRResponse)
def dsr(req: DSRRequest) -> DSRResponse:
    """Deflated Sharpe Ratio — Bailey & López de Prado (2014)."""
    cfg = get_cfg()
    db = get_market_db()
    split_ts = get_split_date()
    benchmark = get_benchmark()

    result = run_full_evaluate(req.formula, db, split_ts, cfg, req.window, benchmark)
    if "error" in result and result.get("sharpe") is None:
        return DSRResponse(error=result["error"])

    sharpe_obs = result.get("sharpe") or 0.0
    n_trials = req.n_trials or 362  # katalog büyüklüğü fallback

    # Bailey LdP (2014) Eq. 8 — simplified DSR
    # DSR = SR * (1 - skew/6*SR + (kurt-3)/24*SR^2)^(-1)
    # için basit versiyon: havuz büyüklüğü düzeltmesi
    gamma = float(np.euler_gamma)
    e_max = (
        (1 - gamma) * float(np.sqrt(2 * np.log(n_trials)))
        + gamma / float(np.sqrt(2 * np.log(n_trials)))
    ) if n_trials > 1 else 0.0

    # Equity curve'den daily returns hesapla
    equity_list = result.get("equity_curve", [])
    if len(equity_list) >= 4:
        equities = np.array([p["equity"] for p in equity_list])
        rets = np.diff(np.log(equities + 1e-10))
        skewness = float(pd.Series(rets).skew())
        kurt = float(pd.Series(rets).kurt())  # excess kurtosis
    else:
        skewness = 0.0
        kurt = 0.0

    T = max(len(equity_list), 252)
    # DSR = (SR_obs - e_max) * sqrt(T) / sigma
    # Approximation using t-stat approach
    sigma_sr = float(np.sqrt((1 + 0.5 * sharpe_obs**2 - skewness * sharpe_obs + (kurt / 4) * sharpe_obs**2) / T))
    dsr_val = (sharpe_obs - e_max) / (sigma_sr + 1e-10) if sigma_sr > 0 else 0.0
    # Convert z-score to p-value (one-sided)
    from scipy.special import ndtr
    p_value = float(ndtr(dsr_val))

    return DSRResponse(
        sharpe_obs=sharpe_obs,
        dsr=float(dsr_val),
        p_value=p_value,
        skewness=skewness,
        kurtosis=kurt,
    )


# ─── PBO ────────────────────────────────────────────────────────────────────

class PBORequest(BaseModel):
    formula: str
    window: str = "test"
    n_splits: int = 8
    max_combinations: int = 500


class PBOResponse(BaseModel):
    pbo: Optional[float] = None
    n_combinations: Optional[int] = None
    error: Optional[str] = None


@router.post("/pbo", response_model=PBOResponse)
def pbo(req: PBORequest) -> PBOResponse:
    """Probability of Backtest Overfitting (CSCV) — Bailey, LdP & Zhu (2014)."""
    cfg = get_cfg()
    db = get_market_db()
    split_ts = get_split_date()

    try:
        df_window = slice_db_by_window(db, req.window, split_ts)
        tree = parse_or_raise(req.formula, cfg)
        idx = prepare_eval_idx(df_window)

        # Veriyi n_splits zaman dilimine böl
        dates = sorted(df_window["Date"].unique())
        n = len(dates)
        if n < req.n_splits * 2:
            return PBOResponse(error=f"Yeterli gün yok: {n} < {req.n_splits * 2}")

        chunks = np.array_split(dates, req.n_splits)

        # IC'yi her chunk'ta hesapla
        chunk_ics: list[float] = []
        for chunk in chunks:
            chunk_set = set(chunk)
            chunk_idx = idx[idx.index.get_level_values("Date").isin(chunk_set)]
            if len(chunk_idx) == 0:
                chunk_ics.append(0.0)
                continue
            try:
                ics = evaluate_ic(tree, chunk_idx, cfg)
                chunk_ics.append(ics.get("rank_ic") or 0.0)
            except Exception:
                chunk_ics.append(0.0)

        # CSCV: C(M, M/2) kombinasyonları
        M = req.n_splits
        half = M // 2
        all_combos = list(itertools.combinations(range(M), half))
        if len(all_combos) > req.max_combinations:
            rng = np.random.default_rng(42)
            idx_sel = rng.choice(len(all_combos), req.max_combinations, replace=False)
            combos = [all_combos[i] for i in idx_sel]
        else:
            combos = all_combos

        # IS'te en iyi seçimi OOS'ta rank'le (tek formül → degenerate case)
        # Tek formül PBO: IS IC > 0 ise OOS IC > 0 olup olmadığına bak
        overfit_count = 0
        for combo in combos:
            is_idx = list(combo)
            oos_idx = [i for i in range(M) if i not in is_idx]
            is_ic = np.mean([chunk_ics[i] for i in is_idx])
            oos_ic = np.mean([chunk_ics[i] for i in oos_idx])
            # IS'te pozitif seçim, OOS'ta negatif → overfit
            if is_ic > 0 and oos_ic <= 0:
                overfit_count += 1

        pbo_val = overfit_count / len(combos) if combos else 0.0
        return PBOResponse(pbo=float(pbo_val), n_combinations=len(combos))

    except Exception as e:
        return PBOResponse(error=str(e))


# ─── Rolling Walk-Forward ────────────────────────────────────────────────────

class RollingWFRequest(BaseModel):
    formula: str
    test_window_months: int = 6
    min_train_months: int = 18
    window: str = "all"
    mode: int = 1  # 1=Anchored, 2=Rolling Re-fit


class RollingWFResult(BaseModel):
    period: str
    ic: Optional[float] = None
    rank_ic: Optional[float] = None


class RollingWFResponse(BaseModel):
    periods: list[RollingWFResult] = []
    avg_ic: Optional[float] = None
    avg_rank_ic: Optional[float] = None
    n_positive: Optional[int] = None
    error: Optional[str] = None


@router.post("/rolling-wf", response_model=RollingWFResponse)
def rolling_wf(req: RollingWFRequest) -> RollingWFResponse:
    """Rolling Walk-Forward backtest."""
    cfg = get_cfg()
    db = get_market_db()
    split_ts = get_split_date()

    try:
        df_window = slice_db_by_window(db, req.window, split_ts)
        tree = parse_or_raise(req.formula, cfg)

        dates = pd.DatetimeIndex(sorted(df_window["Date"].unique()))
        if len(dates) == 0:
            return RollingWFResponse(error="Veri yok")

        start = dates[0]
        end = dates[-1]

        results: list[RollingWFResult] = []
        test_delta = pd.DateOffset(months=req.test_window_months)
        train_delta = pd.DateOffset(months=req.min_train_months)

        # Rolling window
        test_start = start + train_delta
        while test_start < end:
            test_end = min(test_start + test_delta, end)
            if req.mode == 1:
                train_mask = (df_window["Date"] >= start) & (df_window["Date"] < test_start)
            else:
                train_start = test_start - train_delta
                train_mask = (df_window["Date"] >= train_start) & (df_window["Date"] < test_start)

            test_mask = (df_window["Date"] >= test_start) & (df_window["Date"] < test_end)
            df_test = df_window[test_mask]

            if len(df_test) == 0:
                test_start = test_end
                continue

            try:
                idx_test = prepare_eval_idx(df_test)
                ics = evaluate_ic(tree, idx_test, cfg)
                results.append(RollingWFResult(
                    period=f"{test_start.strftime('%Y-%m')} → {test_end.strftime('%Y-%m')}",
                    ic=ics.get("ic"),
                    rank_ic=ics.get("rank_ic"),
                ))
            except Exception:
                results.append(RollingWFResult(
                    period=f"{test_start.strftime('%Y-%m')} → {test_end.strftime('%Y-%m')}",
                ))
            test_start = test_end

        if not results:
            return RollingWFResponse(error="Yeterli pencere oluşturulamadı")

        ics = [r.rank_ic for r in results if r.rank_ic is not None]
        avg_ic = float(np.mean([r.ic for r in results if r.ic is not None])) if results else None
        avg_ric = float(np.mean(ics)) if ics else None
        n_pos = sum(1 for v in ics if v > 0)

        return RollingWFResponse(
            periods=results,
            avg_ic=avg_ic,
            avg_rank_ic=avg_ric,
            n_positive=n_pos,
        )
    except Exception as e:
        return RollingWFResponse(error=str(e))


# ─── Ensemble ───────────────────────────────────────────────────────────────

class EnsembleRequest(BaseModel):
    formulas: list[str]
    window: str = "test"
    weighting: str = "equal"
    max_corr: float = 0.70


class EnsembleResponse(BaseModel):
    ic: Optional[float] = None
    rank_ic: Optional[float] = None
    n_formulas_used: Optional[int] = None
    dropped_correlated: Optional[int] = None
    error: Optional[str] = None


@router.post("/ensemble", response_model=EnsembleResponse)
def ensemble(req: EnsembleRequest) -> EnsembleResponse:
    """Ensemble backtest — CSRank normalize + ağırlıklı ortalama sinyal."""
    cfg = get_cfg()
    db = get_market_db()
    split_ts = get_split_date()

    try:
        df_window = slice_db_by_window(db, req.window, split_ts)
        idx = prepare_eval_idx(df_window)

        signals: list[pd.Series] = []
        valid_formulas: list[str] = []

        for formula in req.formulas:
            formula = formula.strip()
            if not formula:
                continue
            try:
                tree = parse_or_raise(formula, cfg)
                sig = cfg.evaluate(tree, idx)
                # CSRank normalize (cross-section)
                sig_norm = sig.groupby(level="Date").rank(pct=True)
                signals.append(sig_norm)
                valid_formulas.append(formula)
            except Exception:
                continue

        if not signals:
            return EnsembleResponse(error="Geçerli formül yok")

        # Korelasyon filtresi
        dropped = 0
        if len(signals) > 1 and req.max_corr < 1.0:
            kept_sigs = [signals[0]]
            for sig in signals[1:]:
                # existing sigs ile ortalama korelasyon
                corrs = [kept_sigs[i].corr(sig) for i in range(len(kept_sigs))]
                if all(abs(c) < req.max_corr for c in corrs if not np.isnan(c)):
                    kept_sigs.append(sig)
                else:
                    dropped += 1
            signals = kept_sigs

        # Ensemble sinyal
        if req.weighting == "equal":
            ensemble_sig = pd.concat(signals, axis=1).mean(axis=1)
        else:
            ensemble_sig = pd.concat(signals, axis=1).mean(axis=1)

        # IC hesapla
        tmp = pd.DataFrame({
            "Date": idx.index.get_level_values("Date"),
            "Signal": ensemble_sig.values,
            "Next_Ret": idx["Next_Ret"].values,
        }).dropna()

        if len(tmp) == 0:
            return EnsembleResponse(error="Sinyal boş")

        ic = float(tmp.groupby("Date").apply(lambda g: g["Signal"].corr(g["Next_Ret"])).mean())
        ric = float(tmp.groupby("Date").apply(
            lambda g: g["Signal"].corr(g["Next_Ret"], method="spearman")
        ).mean())

        return EnsembleResponse(
            ic=None if np.isnan(ic) else ic,
            rank_ic=None if np.isnan(ric) else ric,
            n_formulas_used=len(signals),
            dropped_correlated=dropped,
        )
    except Exception as e:
        return EnsembleResponse(error=str(e))


# ─── Overfit (time-based) ───────────────────────────────────────────────────

class OverfitRequest(BaseModel):
    n_top: int = 10
    n_folds: int = 5
    mode: str = "walk_forward"  # "single_split" | "walk_forward"


class OverfitResult(BaseModel):
    formula: str
    train_ic: Optional[float] = None
    test_ic: Optional[float] = None
    degradation: Optional[float] = None
    pass_: Optional[bool] = None


class OverfitResponse(BaseModel):
    results: list[OverfitResult] = []
    avg_train_ic: Optional[float] = None
    avg_test_ic: Optional[float] = None
    avg_degradation: Optional[float] = None
    passing: Optional[int] = None
    error: Optional[str] = None


@router.post("/overfit", response_model=OverfitResponse)
def overfit(req: OverfitRequest) -> OverfitResponse:
    """Zaman-bazlı overfit validasyonu — train vs test IC degradation."""
    from engine.alpha_catalog import load_catalog

    cfg = get_cfg()
    db = get_market_db()
    split_ts = get_split_date()

    try:
        records = load_catalog()
        top_records = sorted(records, key=lambda r: r.get("rank_ic", 0) or 0, reverse=True)[:req.n_top]

        df_train = slice_db_by_window(db, "train", split_ts)
        df_test = slice_db_by_window(db, "test", split_ts)
        idx_train = prepare_eval_idx(df_train)
        idx_test = prepare_eval_idx(df_test)

        results: list[OverfitResult] = []
        for rec in top_records:
            formula = rec.get("formula", "")
            try:
                tree = parse_or_raise(formula, cfg)
                train_ics = evaluate_ic(tree, idx_train, cfg)
                test_ics = evaluate_ic(tree, idx_test, cfg)
                train_ic = train_ics.get("rank_ic") or 0.0
                test_ic = test_ics.get("rank_ic") or 0.0
                deg = (test_ic - train_ic) / abs(train_ic) if train_ic != 0 else None
                results.append(OverfitResult(
                    formula=formula,
                    train_ic=train_ic,
                    test_ic=test_ic,
                    degradation=deg,
                    pass_=deg is not None and deg > -0.5,
                ))
            except Exception as e:
                results.append(OverfitResult(formula=formula))

        passing = sum(1 for r in results if r.pass_)
        train_ics = [r.train_ic for r in results if r.train_ic is not None]
        test_ics = [r.test_ic for r in results if r.test_ic is not None]
        degs = [r.degradation for r in results if r.degradation is not None]

        return OverfitResponse(
            results=results,
            avg_train_ic=float(np.mean(train_ics)) if train_ics else None,
            avg_test_ic=float(np.mean(test_ics)) if test_ics else None,
            avg_degradation=float(np.mean(degs)) if degs else None,
            passing=passing,
        )
    except Exception as e:
        return OverfitResponse(error=str(e))


# ─── Job-based Backtest (Workbench "▸ Run Backtest") ───────────────────────

class BacktestRunRequest(BaseModel):
    formula: str
    window: str = "test"
    validations: dict = {}


class BacktestRunResponse(BaseModel):
    job_id: str


@router.post("/run", response_model=BacktestRunResponse)
async def run_backtest_job(req: BacktestRunRequest) -> BacktestRunResponse:
    """Workbench'in "▸ Run Backtest" butonu — async job döner, WS üzerinden takip edilir."""
    job = registry.create()
    job.status = "running"

    async def _run():
        try:
            await job.emit_log(f"Başlatılıyor: {req.formula[:60]}…")
            await job.emit_progress(0.05)

            db = get_market_db()
            split_ts = get_split_date()
            cfg = get_cfg()
            bm = get_benchmark()

            result = await asyncio.to_thread(
                run_full_evaluate,
                req.formula,
                db,
                split_ts,
                cfg,
                req.window,
                bm,
            )

            await job.emit_progress(0.95)
            await job.emit_log("Backtest tamamlandı")
            await job.finish(result)

        except Exception as exc:
            await job.emit_log(f"HATA: {exc}")
            await job.fail(str(exc))

    asyncio.create_task(_run())
    return BacktestRunResponse(job_id=job.id)
