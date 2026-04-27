"""Singleton servisler — market_db, benchmark, brain.

`@functools.lru_cache(None)` ile her singleton bir kez yüklenir.
Streamlit `app.py:51-114`'teki cache yapısının FastAPI eşdeğeri.
"""
from __future__ import annotations

import functools
import os
from typing import Optional

import pandas as pd


@functools.lru_cache(maxsize=1)
def get_market_db() -> pd.DataFrame:
    """data/market_db.parquet → DataFrame, duplicate (Ticker,Date) temizlendi."""
    from config import load_paths

    paths = load_paths()
    df = pd.read_parquet(paths["market_db"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop_duplicates(subset=["Ticker", "Date"], keep="first")
    if "Pvwap" not in df.columns:
        df["Pvwap"] = (df["Phigh"] + df["Plow"] + df["Pclose"]) / 3
    return df


@functools.lru_cache(maxsize=1)
def get_benchmark() -> Optional[pd.Series]:
    """data/bist100.{parquet,csv} → Date-indexed close series. Yoksa None."""
    from config import load_paths

    paths = load_paths()
    for path in [paths.get("benchmark"), paths.get("bm_csv")]:
        if not path or not os.path.exists(path):
            continue
        try:
            if path.endswith(".parquet"):
                bm = pd.read_parquet(path)
            else:
                bm = pd.read_csv(path)
            bm["Date"] = pd.to_datetime(bm["Date"])
            close_col = next(
                (c for c in ["Close", "Pclose", "close", "CLOSE"] if c in bm.columns),
                None,
            )
            if close_col is None:
                continue
            return bm.set_index("Date")[close_col].sort_index()
        except Exception:
            continue
    return None


@functools.lru_cache(maxsize=1)
def get_cfg():
    """AlphaCFG singleton — formül evaluate'i için."""
    from engine.core.alpha_cfg import AlphaCFG

    return AlphaCFG()


@functools.lru_cache(maxsize=1)
def get_brain():
    """Tree-LSTM + trainer + buffer (lazy — sadece gerektiğinde yüklenir)."""
    from engine.core.alpha_cfg import AlphaCFG
    from engine.ml.replay_buffer import ReplayBuffer
    from engine.ml.trainer import TreeLSTMTrainer
    from engine.ml.tree_lstm import (
        PolicyValueNet,
        build_action_vocab,
        build_token_vocab,
    )

    cfg = AlphaCFG()
    token_vocab = build_token_vocab(cfg)
    action_vocab = build_action_vocab(cfg)
    net = PolicyValueNet(
        token_vocab_size=len(token_vocab),
        action_size=len(action_vocab),
    )
    trainer = TreeLSTMTrainer(net, token_vocab, device="cpu").load()
    buffer = ReplayBuffer().load()
    return {
        "cfg": cfg,
        "vocab": token_vocab,
        "actions": action_vocab,
        "net": net,
        "trainer": trainer,
        "buffer": buffer,
    }


def get_split_date() -> pd.Timestamp:
    """Default split date: train veri aralığının %70'i (Streamlit default'uyla aynı)."""
    db = get_market_db()
    date_min = pd.to_datetime(db["Date"].min())
    date_max = pd.to_datetime(db["Date"].max())
    return date_min + (date_max - date_min) * 0.7
