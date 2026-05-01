"""Singleton servisler — market_db, benchmark, brain, auth."""
from __future__ import annotations

import functools
import os
import sys
from typing import Optional

import pandas as pd
from fastapi import Header, HTTPException, status

# .env dosyası varsa yükle
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 1. Ana dizini tespit et
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 2. Config.py'yi TAMAMEN İPTAL EDİYORUZ. Yolları doğrudan data/ klasörüne bağlıyoruz.
DATA_DIR = os.path.join(ROOT_DIR, "data")
MARKET_DB_PATH = os.path.join(DATA_DIR, "market_db.parquet")
BM_PARQUET_PATH = os.path.join(DATA_DIR, "bist100.parquet")
BM_CSV_PATH = os.path.join(DATA_DIR, "bist100.csv")

# N28: lru_cache(maxsize=1) multi-worker'da her process N×bellek kullanır.
# Workaround: mtime kontrollü module-level cache — aynı process içinde tek kopya.
# Gerçek multi-worker çözümü: Redis veya shared-memory (LT-14).
_market_db_cache: Optional[pd.DataFrame] = None
_market_db_mtime: float = 0.0


def get_market_db() -> pd.DataFrame:
    """
    market_db.parquet'i yükle. Dosya değişmediyse bellekteki kopyayı döndür.
    N28: lru_cache kaldırıldı — mtime tabanlı lazy reload.
    """
    global _market_db_cache, _market_db_mtime
    if not os.path.exists(MARKET_DB_PATH):
        raise FileNotFoundError(
            f"VERİ BULUNAMADI: {MARKET_DB_PATH}\n"
            f"Lütfen verilerin Minerva_v3_Studio/data/ klasörü içinde olduğundan emin ol!"
        )
    mtime = os.path.getmtime(MARKET_DB_PATH)
    if _market_db_cache is None or mtime != _market_db_mtime:
        df = pd.read_parquet(MARKET_DB_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.drop_duplicates(subset=["Ticker", "Date"], keep="first")
        if "Pvwap" not in df.columns:
            df["Pvwap"] = (df["Phigh"] + df["Plow"] + df["Pclose"]) / 3
        _market_db_cache = df
        _market_db_mtime = mtime
    return _market_db_cache


@functools.lru_cache(maxsize=1)
def get_benchmark() -> Optional[pd.Series]:
    for path in [BM_PARQUET_PATH, BM_CSV_PATH]:
        if not os.path.exists(path):
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
    try:
        from engine.alpha_cfg import AlphaCFG
    except ModuleNotFoundError:
        from engine.core.alpha_cfg import AlphaCFG
    return AlphaCFG()


@functools.lru_cache(maxsize=1)
def get_brain():
    try:
        from engine.alpha_cfg import AlphaCFG
        from engine.replay_buffer import ReplayBuffer
        from engine.trainer import TreeLSTMTrainer
        from engine.tree_lstm import PolicyValueNet, build_action_vocab, build_token_vocab
    except ModuleNotFoundError:
        from engine.core.alpha_cfg import AlphaCFG
        from engine.ml.replay_buffer import ReplayBuffer
        from engine.ml.trainer import TreeLSTMTrainer
        from engine.ml.tree_lstm import PolicyValueNet, build_action_vocab, build_token_vocab

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


def verify_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """
    Yazma/işlem endpoint'leri için X-API-Key header doğrulaması.

    - API_KEY ortam değişkeni tanımlıysa: header zorunlu ve eşleşmeli.
    - Tanımlı değilse: geliştirme modunda çalışır (kontrol atlanır).
    """
    expected = os.getenv("API_KEY")
    if not expected:
        # Dev modu — anahtar yoksa auth devre dışı
        return
    if x_api_key != expected:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Geçersiz veya eksik API anahtarı (X-API-Key header).",
        )


def get_split_date() -> pd.Timestamp:
    db = get_market_db()
    date_min = pd.to_datetime(db["Date"].min())
    date_max = pd.to_datetime(db["Date"].max())
    return date_min + (date_max - date_min) * 0.7