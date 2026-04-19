"""
tests/conftest.py — Paylaşılan test fikstürleri.

Tüm test modülleri tarafından pytest fixture mekanizmasıyla kullanılır.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from engine.alpha_cfg import AlphaCFG


# ─── Sentetik veri üretici ────────────────────────────────────────────────────

def make_synthetic_db(
    n_tickers: int = 30,
    n_days: int = 400,
    seed: int = 42,
    price_range: tuple = (5.0, 300.0),   # cross-sectional size çeşitliliği için
) -> pd.DataFrame:
    """
    Test için deterministik sentetik piyasa verisi üretir.

    Özellikler:
    - Her ticker farklı başlangıç fiyatı → güçlü cross-sectional size varyasyonu
    - Geometric Brownian Motion fiyat süreci
    - Pclose, Phigh, Plow, Popen, Vlot, Pvwap, Next_Ret sütunları
    - Flat DataFrame (Date, Ticker, ...) formatı
    """
    rng = np.random.default_rng(seed)
    dates   = pd.date_range("2020-01-01", periods=n_days, freq="B")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    # Her ticker için lineer aralıkta farklı başlangıç fiyatı
    base_prices = np.linspace(price_range[0], price_range[1], n_tickers)

    rows = []
    for i, ticker in enumerate(tickers):
        # Geometric Brownian Motion
        drift = rng.uniform(-0.0001, 0.0003)
        vol   = rng.uniform(0.01, 0.03)
        log_returns = rng.normal(drift, vol, n_days)
        prices = base_prices[i] * np.exp(np.cumsum(log_returns))

        for j, date in enumerate(dates):
            p = float(prices[j])
            h = p * (1.0 + abs(rng.normal(0.0, 0.007)))
            l = p * (1.0 - abs(rng.normal(0.0, 0.007)))
            o = p * (1.0 + rng.normal(0.0, 0.004))
            v = float(abs(rng.normal(1e6, 2e5)))
            rows.append({
                "Ticker": ticker,
                "Date":   date,
                "Pclose": p,
                "Phigh":  h,
                "Plow":   l,
                "Popen":  o,
                "Vlot":   v,
                "Pvwap":  (h + l + p) / 3.0,
            })

    df = pd.DataFrame(rows).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Next_Ret: t+2 kapanış / t+1 kapanış - 1  (alış ertesi gün açılış ≈ Popen)
    df["Pclose_t1"] = df.groupby("Ticker")["Pclose"].shift(-1)
    df["Pclose_t2"] = df.groupby("Ticker")["Pclose"].shift(-2)
    df["Next_Ret"]  = df["Pclose_t2"] / df["Pclose_t1"] - 1
    df = df.drop(columns=["Pclose_t1", "Pclose_t2"])
    return df


def make_synthetic_idx(
    n_tickers: int = 30,
    n_days: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    """Flat df'yi MultiIndex (Ticker, Date) idx'e çevirir."""
    df = make_synthetic_db(n_tickers=n_tickers, n_days=n_days, seed=seed)
    return df.set_index(["Ticker", "Date"]).sort_index()


# ─── Pytest CLI Options ───────────────────────────────────────────────────────

def pytest_addoption(parser):
    """Regresyon altın dosyası için --update-golden seçeneği ekle."""
    try:
        parser.addoption(
            "--update-golden",
            action="store_true",
            default=False,
            help="Regresyon altın dosyasını (tests/data/golden.json) yeniden oluştur.",
        )
    except ValueError:
        pass  # Zaten eklenmiş (multi-collection senaryosu)


# ─── Pytest Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cfg():
    """Tek AlphaCFG örneği — tüm testler için paylaşılır."""
    return AlphaCFG()


@pytest.fixture(scope="session")
def syn_db():
    """Sentetik flat DataFrame (30 ticker × 400 gün)."""
    return make_synthetic_db(n_tickers=30, n_days=400, seed=42)


@pytest.fixture(scope="session")
def syn_idx(syn_db):
    """Sentetik MultiIndex idx (30 ticker × 400 gün)."""
    return syn_db.set_index(["Ticker", "Date"]).sort_index()
