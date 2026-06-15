"""Veri Terminali (Arkhimedes) kaynaklı external veri yükleyicileri.

`scripts/import_from_veri_terminali.py` ile data/external/ altına kopyalanmış
parquet/csv/xlsx dosyalarını lazy-load eder ve schema normalize uygular.

PIT (point-in-time) discipline:
- Fundamental: announce_date = period_end + 60 gün (bilanço açıklama gecikmesi)
- Macro:       as_of_date    = period_end + 21 gün (TÜİK/TCMB yayın gecikmesi)
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
EXTERNAL_DIR = ROOT / "data" / "external"


@lru_cache(maxsize=1)
def load_features_db() -> pd.DataFrame:
    """67 hazır teknik feature (mom, vol, xu100_*, usd_*, sector_*, cust_*).

    Veri Terminali'nde 2017-05-10 → 2026-06-09, ~977K satır, 581+ ticker.
    """
    path = EXTERNAL_DIR / "features_db.parquet"
    df = pd.read_parquet(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@lru_cache(maxsize=1)
def load_fundamentals(pit_delay_days: int = 60) -> pd.DataFrame:
    """42 kolon fundamental (F_K, PD_DD, FAVOK_TTM, ROE_pct, …).

    `Tarih` kolonu period_end olarak yorumlanır; announce_date = Tarih + 60g.

    Kolon ismi normalize:
      Symbol → Ticker
      ROE_%  → ROE_pct (% karakteri sorunlu)
      Brut_Kar_Marji_%  → Brut_Kar_Marji_pct
    """
    df = pd.read_parquet(EXTERNAL_DIR / "BIST_Tarihsel_Temel_Analiz.parquet")
    df = df.rename(columns={
        "Symbol": "Ticker",
        "ROE_%":  "ROE_pct",
        "ROA_%":  "ROA_pct",
        "Brut_Kar_Marji_%":  "Brut_Kar_Marji_pct",
        "Net_Kar_Marji_%":   "Net_Kar_Marji_pct",
    })
    df["Tarih"] = pd.to_datetime(df["Tarih"], errors="coerce")
    df["announce_date"] = df["Tarih"] + pd.Timedelta(days=pit_delay_days)
    return df


@lru_cache(maxsize=1)
def load_macro(pit_delay_days: int = 21) -> pd.DataFrame:
    """EVDS TÜFE/ÜFE/Politika Faizi, aylık. PIT gecikme +21 gün."""
    df = pd.read_excel(EXTERNAL_DIR / "EVDS_Verileri_2016_2026.xlsx")
    df["Tarih"] = pd.to_datetime(df["Tarih"], errors="coerce")
    df["as_of_date"] = df["Tarih"] + pd.Timedelta(days=pit_delay_days)
    return df


@lru_cache(maxsize=1)
def load_custody() -> pd.DataFrame:
    """15 custody feature (yatırımcı sayısı momentum, kurumsal % trendi…).

    Veri Terminali'nde 2022-2026 (pre-2022 dönemde NaN olacak).
    """
    df = pd.read_parquet(EXTERNAL_DIR / "custody_features_db.parquet")
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@lru_cache(maxsize=1)
def load_sector_map() -> pd.DataFrame:
    """611 hisse → sektör / üst-sektör tablosu."""
    return pd.read_csv(EXTERNAL_DIR / "sector_map.csv")


@lru_cache(maxsize=1)
def load_delisted_tickers() -> frozenset[str]:
    """159 delisted ticker (.IS suffix kaldırılmış)."""
    txt = (EXTERNAL_DIR / "delisted_tickers.txt").read_text()
    return frozenset(
        line.strip().replace(".IS", "")
        for line in txt.splitlines()
        if line.strip()
    )


def available() -> dict[str, bool]:
    """Hangi external veriler mevcut?"""
    return {
        "features_db":   (EXTERNAL_DIR / "features_db.parquet").exists(),
        "fundamentals":  (EXTERNAL_DIR / "BIST_Tarihsel_Temel_Analiz.parquet").exists(),
        "macro":         (EXTERNAL_DIR / "EVDS_Verileri_2016_2026.xlsx").exists(),
        "custody":       (EXTERNAL_DIR / "custody_features_db.parquet").exists(),
        "sector_map":    (EXTERNAL_DIR / "sector_map.csv").exists(),
        "delisted":      (EXTERNAL_DIR / "delisted_tickers.txt").exists(),
    }
