"""
engine/api_helpers.py — FastAPI ve Streamlit'in ortak kullandığı helper'lar.

Mevcut Streamlit `app.py`'deki LLM evaluate akışı (411-559 satırları)
buradan import edilerek hem eski UI hem de yeni SPA'da aynı sonucu üretir.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from .alpha_cfg import AlphaCFG, Node
from .backtest_engine import run_pro_backtest
from .formula_parser import ParseError, parse_formula


def prepare_eval_idx(db: pd.DataFrame) -> pd.DataFrame:
    """
    Flat market_db DataFrame → (Ticker, Date) MultiIndex DataFrame.
    Next_Ret = Pclose_{t+2}/Pclose_{t+1}-1 hedef sütunu eklenir.

    Streamlit `app.py:417-423` ve `app.py:583-589` ile aynı mantık.
    """
    df = db.sort_values(["Ticker", "Date"]).copy()
    df["Pclose_t1"] = df.groupby("Ticker")["Pclose"].shift(-1)
    df["Pclose_t2"] = df.groupby("Ticker")["Pclose"].shift(-2)
    df["Next_Ret"] = df["Pclose_t2"] / df["Pclose_t1"] - 1

    cols = ["Date", "Ticker", "Popen", "Phigh", "Plow",
            "Pclose", "Vlot", "Pvwap", "Next_Ret"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].set_index(["Ticker", "Date"]).sort_index()


def evaluate_ic(
    tree: Node,
    idx: pd.DataFrame,
    cfg: AlphaCFG,
) -> dict[str, Optional[float]]:
    """
    Bir AST için IC + RankIC hesapla. NaN → None.

    Streamlit `app.py:436-451` ile aynı formül.
    """
    sig = cfg.evaluate(tree, idx)
    tmp = pd.DataFrame({
        "Date":     idx.index.get_level_values("Date"),
        "Signal":   sig.values,
        "Next_Ret": idx["Next_Ret"].values,
    }).dropna()

    if len(tmp) == 0:
        return {"ic": None, "rank_ic": None, "n": 0}

    def _corr(g: pd.DataFrame, method: str) -> float:
        if g["Signal"].std() == 0:
            return 0.0
        return float(g["Signal"].corr(g["Next_Ret"], method=method))

    ic      = float(tmp.groupby("Date").apply(lambda g: _corr(g, "pearson")).mean())
    rank_ic = float(tmp.groupby("Date").apply(lambda g: _corr(g, "spearman")).mean())

    return {
        "ic":       None if np.isnan(ic) else ic,
        "rank_ic":  None if np.isnan(rank_ic) else rank_ic,
        "n":        int(len(tmp)),
    }


def slice_db_by_window(
    db: pd.DataFrame,
    window: str,
    split_ts: pd.Timestamp,
) -> pd.DataFrame:
    """window ∈ {test, train, all} → filtrelenmiş DataFrame."""
    if window == "train":
        return db[db["Date"] < split_ts].copy()
    if window == "test":
        return db[db["Date"] >= split_ts].copy()
    return db.copy()


def parse_or_raise(formula: str, cfg: AlphaCFG) -> Node:
    """Formül parse et; hata fırlatırsa human-readable mesaj."""
    try:
        return parse_formula(formula, cfg)
    except ParseError as e:
        raise ValueError(f"Parse hatası: {e}") from e


def run_full_evaluate(
    formula: str,
    db: pd.DataFrame,
    split_ts: pd.Timestamp,
    cfg: AlphaCFG,
    window: str = "test",
    benchmark: Optional[pd.Series] = None,
    top_k: int = 50,
    n_drop: int = 5,
    buy_fee: float = 0.0005,
    sell_fee: float = 0.0015,
) -> dict[str, Any]:
    """
    Tek formül için tam pipeline: parse → evaluate → IC → backtest.

    Returns: { ic, rank_ic, sharpe (=IR), annual, mdd, net_return,
               equity_curve: [{date, equity, benchmark?}], error?, ... }
    """
    try:
        tree = parse_or_raise(formula, cfg)
    except ValueError as e:
        return {"error": str(e)}

    df_window = slice_db_by_window(db, window, split_ts)
    if len(df_window) == 0:
        return {"error": f"window={window} için veri yok"}

    idx = prepare_eval_idx(df_window)

    try:
        ic_stats = evaluate_ic(tree, idx, cfg)
    except Exception as e:
        return {"error": f"IC hesaplanamadı: {e}"}

    # Backtest için sinyal serisini df_window sırasıyla hizala
    try:
        sig = cfg.evaluate(tree, idx)
        # idx (Ticker, Date) → df_window (Date, Ticker) sıralı
        df_bt = df_window.sort_values(["Date", "Ticker"]).reset_index(drop=True)
        sig_aligned = (
            sig.reset_index()
               .rename(columns={0: "Signal"})
               .set_index(["Ticker", "Date"])
               .reindex(pd.MultiIndex.from_arrays(
                   [df_bt["Ticker"], df_bt["Date"]], names=["Ticker", "Date"]
               ))
        )
        sig_arr = sig_aligned.iloc[:, 0].values
    except Exception as e:
        return {
            "error": f"Sinyal hizalanamadı: {e}",
            "ic": ic_stats["ic"],
            "rank_ic": ic_stats["rank_ic"],
        }

    try:
        curve, metrics = run_pro_backtest(
            df_bt, sig_arr,
            top_k=top_k, n_drop=n_drop,
            buy_fee=buy_fee, sell_fee=sell_fee,
            benchmark=benchmark,
        )
    except Exception as e:
        return {
            "error": f"Backtest hatası: {e}",
            "ic": ic_stats["ic"],
            "rank_ic": ic_stats["rank_ic"],
        }

    # Equity curve serisi → liste of dicts (frontend-friendly)
    equity_curve = []
    for _, row in curve.iterrows():
        point = {
            "date":   pd.Timestamp(row["Date"]).strftime("%Y-%m-%d"),
            "equity": float(row["Equity"]),
        }
        if "BenchmarkEquity" in curve.columns:
            bm_v = row.get("BenchmarkEquity")
            if pd.notna(bm_v):
                point["benchmark"] = float(bm_v)
        equity_curve.append(point)

    return {
        "ic":         ic_stats["ic"],
        "rank_ic":    ic_stats["rank_ic"],
        "sharpe":     float(metrics.get("IR", 0.0)),
        "annual":     float(metrics.get("Yıllık", 0.0)),
        "mdd":        float(metrics.get("MDD", 0.0)),
        "net_return": float(metrics.get("Net Getiri (%)", 0.0)),
        "alpha_ir":   metrics.get("Alfa IR"),
        "beta":       metrics.get("Beta"),
        "equity_curve": equity_curve,
        "n_observations": ic_stats["n"],
    }
