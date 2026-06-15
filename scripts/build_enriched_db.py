"""market_db.parquet + external sources → enriched_market_db.parquet

Tek seferlik derived artifact üretimi. Mining pipeline bu birleştirilmiş
dosyayı kullanır (`run_historical_paper_trade.py` enriched varsa onu yükler).

Çıktı:
    data/enriched_market_db.parquet
        (Ticker, Date) + ~60 feature
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.data import external_data  # noqa: E402

OUT_PATH = ROOT / "data" / "enriched_market_db.parquet"


def build() -> pd.DataFrame:
    print("=== Enriched Market DB Builder ===\n")
    avail = external_data.available()
    print("External veri:")
    for k, v in avail.items():
        print(f"  {k:14s} {'✓' if v else '✗'}")
    print()

    # 1. Base: market_db
    base_path = ROOT / "data" / "market_db.parquet"
    base = pd.read_parquet(base_path)
    base["Date"] = pd.to_datetime(base["Date"])
    # Ptyp = (Phigh + Plow + Pclose) / 3 — backward compat (v11)
    if "Ptyp" not in base.columns:
        base["Ptyp"] = (base["Phigh"] + base["Plow"] + base["Pclose"]) / 3.0
    print(f"[1/6] market_db: {base.shape} ({base['Date'].min().date()} → {base['Date'].max().date()})")

    # 2. features_db (67 hazır feature)
    if avail["features_db"]:
        feats = external_data.load_features_db()
        # Dup kolon yönetimi: market_db'deki master; features_db'den drop et
        dup_cols = [c for c in feats.columns
                    if c in base.columns and c not in ["Ticker", "Date"]]
        feats_use = feats.drop(columns=dup_cols)
        base = base.merge(feats_use, on=["Ticker", "Date"], how="left")
        print(f"[2/6] features_db merge: +{feats_use.shape[1] - 2} kolon → {base.shape[1]} total "
              f"(duplicates dropped: {len(dup_cols)})")
    else:
        print("[2/6] features_db SKIPPED (mevcut değil)")

    # 3. fundamentals (PIT-aware as-of merge)
    if avail["fundamentals"]:
        fund = external_data.load_fundamentals(pit_delay_days=60)
        # Yararlı kolonları seç
        fund_cols = [
            "F_K", "PD_DD", "FD_FAVOK", "ROE_pct", "ROA_pct",
            "Brut_Kar_Marji_pct", "Net_Kar_Marji_pct", "Cari_Oran",
            "Halka_Aciklik_Orani", "Net_Borc",
        ]
        fund_cols = [c for c in fund_cols if c in fund.columns]
        fund_subset = fund[["Ticker", "announce_date"] + fund_cols].dropna(subset=["Ticker"]).copy()
        # Date dtype normalize ([ms] match)
        fund_subset["Date"] = pd.to_datetime(fund_subset["announce_date"]).astype("datetime64[ms]")
        fund_subset = fund_subset.drop(columns=["announce_date"])
        base["Date"] = base["Date"].astype("datetime64[ms]")
        # asof merge per ticker
        base = pd.merge_asof(
            base.sort_values("Date"),
            fund_subset.sort_values("Date"),
            on="Date",
            by="Ticker",
            direction="backward",
            allow_exact_matches=True,
        )
        print(f"[3/6] fundamentals merge: +{len(fund_cols)} kolon (PIT +60g) → {base.shape[1]} total")
    else:
        print("[3/6] fundamentals SKIPPED")

    # 4. custody (zaten features_db içinde 15 cust_* var — fakat custody_features_db tekil
    #    olabilir; eğer features_db'de yoksa burada ekleriz)
    if avail["custody"]:
        custody = external_data.load_custody()
        custody_new = [c for c in custody.columns
                       if c.startswith("cust_") and c not in base.columns]
        if custody_new:
            base = base.merge(
                custody[["Ticker", "Date"] + custody_new],
                on=["Ticker", "Date"], how="left",
            )
            print(f"[4/6] custody ek kolon: +{len(custody_new)} → {base.shape[1]} total")
        else:
            print("[4/6] custody: tüm kolonlar zaten features_db'den geldi")
    else:
        print("[4/6] custody SKIPPED")

    # 5. sector_map (eğer features_db'de Sector kolonu yoksa)
    if avail["sector_map"] and "Sector" not in base.columns:
        smap = external_data.load_sector_map()
        base = base.merge(smap, on="Ticker", how="left")
        print(f"[5/6] sector_map merge: +{smap.shape[1] - 1} kolon → {base.shape[1]} total")
    else:
        print("[5/6] sector_map zaten mevcut veya yok")

    # 6. delisted flag
    if avail["delisted"]:
        delisted = external_data.load_delisted_tickers()
        base["is_delisted"] = base["Ticker"].isin(delisted)
        n_del = base["is_delisted"].sum()
        print(f"[6/6] is_delisted flag: {n_del} satır işaretlendi")
    else:
        print("[6/6] delisted SKIPPED")

    # Float optimization — küçük dosya
    for c in base.select_dtypes(include="float64").columns:
        base[c] = base[c].astype("float32")

    print(f"\nFinal: {base.shape} | {base['Date'].min().date()} → {base['Date'].max().date()}")
    print(f"Ticker count: {base['Ticker'].nunique()}")

    # NaN raporu
    nan_pct = (base.isna().sum() / len(base) * 100).round(1)
    high_nan = nan_pct[nan_pct > 30].sort_values(ascending=False)
    if len(high_nan):
        print(f"\nYüksek NaN (%30+) kolonlar:")
        for col, pct in high_nan.head(15).items():
            print(f"  {col:35s} {pct:5.1f}% NaN")

    return base


def main() -> int:
    df = build()
    df.to_parquet(OUT_PATH, index=False)
    sz_mb = OUT_PATH.stat().st_size / 1024 / 1024
    print(f"\nKaydedildi: {OUT_PATH} ({sz_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
