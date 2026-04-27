"""
Minerva v3 — Master Data Pipeline
===================================
Kaynaklar:
  1. yfinance       — OHLCV (günlük)
  2. isyatirimhisse — Bilanço: ROE, F/K, PD/DD (çeyreklik → ffill)
  3. TCMB EVDS      — USD/TRY kuru (günlük, yıllık chunk)
  4. TEFAS          — Top-5 hisse fonu getirisi (günlük, 3 aylık chunk)

Çıktı: data/market_db_master.parquet
"""

import os
import time
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import evds
from tefas import Crawler
from isyatirimhisse import fetch_financials

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# YARDIMCI: tarih aralığını N-günlük dilimlere böl
# ─────────────────────────────────────────────
def _date_chunks(start: str, end: str, days: int = 365):
    """'YYYY-MM-DD' aralığını `days` günlük dilimler halinde döndürür."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    while s <= e:
        chunk_end = min(s + pd.Timedelta(days=days - 1), e)
        yield s.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")
        s = chunk_end + pd.Timedelta(days=1)


# ─────────────────────────────────────────────
# 1. OHLCV — yfinance
# ─────────────────────────────────────────────
def fetch_ohlcv(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    print("\n[OHLCV] yfinance'ten fiyat verisi çekiliyor...")
    frames = []
    for ticker in tickers:
        try:
            df = yf.download(
                ticker + ".IS",
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                print(f"  ✗ {ticker}: boş")
                continue

            # MultiIndex düzelt
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            df["Ticker"] = ticker
            frames.append(df[["Date", "Close", "High", "Low", "Open", "Volume", "Ticker"]])
            print(f"  ✓ {ticker}: {len(df)} satır")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")

    if not frames:
        raise RuntimeError("Hiçbir hisseden OHLCV verisi gelmedi.")
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────
# 2. BİLANÇO — isyatirimhisse (look-ahead: +45 gün)
# ─────────────────────────────────────────────
BANKALAR = {
    "AKBNK", "GARAN", "YKBNK", "ISCTR", "VAKBN", "HALKB",
    "QNBFB", "TSKB", "ALBRK", "SKBNK",
}

def fetch_fundamentals(ticker: str, start_year: int, end_year: int) -> pd.DataFrame:
    fin_group = "2" if ticker in BANKALAR else "1"
    try:
        veri = fetch_financials(
            symbols=ticker,
            start_year=str(start_year),
            end_year=str(end_year),
            exchange="TRY",
            financial_group=fin_group,
        )
        df = veri[ticker] if isinstance(veri, dict) else veri
        if df is None or df.empty:
            return pd.DataFrame()

        # Sabit sütun: FINANCIAL_ITEM_NAME_TR
        label_col = "FINANCIAL_ITEM_NAME_TR"
        if label_col not in df.columns:
            label_col = next(
                (c for c in df.columns if df[c].dtype == object), df.columns[0]
            )

        if fin_group == "2":
            # Bankalar — BDDK/IFRS format (XI_59)
            ozk_lbl = "XVI. ÖZKAYNAKLAR"
            kar_lbl = "XXIII. NET DÖNEM KARI/ZARARI"
            ser_lbl = "16.1 Ödenmiş Sermaye"
        else:
            # Sanayi/Hizmet — XI_29 format
            ozk_lbl = "Ana Ortaklığa Ait Özkaynaklar"
            kar_lbl = "DÖNEM KARI (ZARARI)"
            ser_lbl = "Ödenmiş Sermaye"

        def _row(lbl):
            r = df[df[label_col].str.contains(lbl, case=False, na=False, regex=False)]
            return r if not r.empty else None

        ozk_row = _row(ozk_lbl)
        kar_row = _row(kar_lbl)
        ser_row = _row(ser_lbl)
        if any(r is None for r in [ozk_row, kar_row, ser_row]):
            return pd.DataFrame()

        # Dönem sütunları: '2023/3', '2023/6', ...
        date_cols = [c for c in df.columns if "/" in str(c)]

        def _vals(row):
            return pd.to_numeric(row[date_cols].iloc[0], errors="coerce").values

        # isyatirimhisse en yeniden en eskiye sıralar → ters çevir
        ozkaynak = _vals(ozk_row)[::-1]
        net_kar  = _vals(kar_row)[::-1]
        sermaye  = _vals(ser_row)[::-1]
        d_cols   = date_cols[::-1]

        dates = []
        for d in d_cols:
            yr, mo = d.split("/")
            dates.append(
                pd.to_datetime(f"{yr}-{int(mo):02d}-01") + pd.offsets.MonthEnd(0)
            )

        fund = pd.DataFrame(
            {"OZKAYNAK": ozkaynak, "NET_KAR": net_kar, "SERMAYE": sermaye},
            index=dates,
        )
        fund["ROE"] = np.where(
            fund["OZKAYNAK"] > 0, fund["NET_KAR"] / fund["OZKAYNAK"] * 100, np.nan
        )
        fund.index.name = "Date"
        # 👑 Look-ahead bias koruması: KAP bildirimi ~45 gün gecikir
        fund.index = fund.index + pd.Timedelta(days=45)
        return fund

    except Exception as e:
        print(f"  ✗ [{ticker}] bilanço hatası: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# 3. MAKRO — TCMB EVDS (yıllık chunk, 1000 satır limitini aşar)
# ─────────────────────────────────────────────
def fetch_macro(evds_api_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Çekilen seriler:
      Günlük  : USD/TRY, EUR/TRY  (yıllık chunk — 1000 satır limitini aşar)
      Aylık   : TÜFE yıllık%, TÜFE aylık%, ÜFE  (ffill ile günlüğe dönüşür)
    """
    print("\n[EVDS] Makro veriler çekiliyor...")
    api = evds.evdsAPI(evds_api_key)

    # ── Günlük seriler (yıllık chunk) ──────────────────────────────────────
    GUNLUK_SERILER = {
        "TP.DK.USD.A": "USD_TRY",
        "TP.DK.EUR.A": "EUR_TRY",
        "TP.APIFON4":  "AOF",       # TCMB Ağırlıklı Ortalama Fonlama Maliyeti
    }
    daily_frames = {}

    for seri, kolon in GUNLUK_SERILER.items():
        chunks = []
        for chunk_s, chunk_e in _date_chunks(start_date, end_date, days=365):
            cs_dmy = datetime.strptime(chunk_s, "%Y-%m-%d").strftime("%d-%m-%Y")
            ce_dmy = datetime.strptime(chunk_e, "%Y-%m-%d").strftime("%d-%m-%Y")
            try:
                df = api.get_data([seri], startdate=cs_dmy, enddate=ce_dmy)
                if df is not None and not df.empty:
                    chunks.append(df)
            except Exception:
                pass
            time.sleep(0.2)

        if chunks:
            df_all = pd.concat(chunks, ignore_index=True).drop_duplicates("Tarih")
            df_all["Tarih"] = pd.to_datetime(df_all["Tarih"], format="%d-%m-%Y")
            seri_col = seri.replace(".", "_")
            df_all = df_all.rename(columns={"Tarih": "Date", seri_col: kolon, seri: kolon})
            df_all = df_all.set_index("Date").sort_index()
            if kolon in df_all.columns:
                daily_frames[kolon] = df_all[[kolon]]
                print(f"  ✓ {kolon:<10}: {len(df_all)} satır (günlük)")

    # ── Aylık seriler (tek chunk yeterli — aylık veri azdır) ───────────────
    AYLIK_SERILER = {
        "TP.FG.J0":      "TUFE_YOY",   # TÜFE Yıllık %
        "TP.FE.OKTG01":  "TUFE_MOM",   # TÜFE Aylık %
        "TP.FE.OKTG02":  "UFE_MOM",    # ÜFE Aylık %
    }
    monthly_frames = {}

    cs_dmy = datetime.strptime(start_date, "%Y-%m-%d").strftime("%d-%m-%Y")
    ce_dmy = datetime.strptime(end_date,   "%Y-%m-%d").strftime("%d-%m-%Y")

    for seri, kolon in AYLIK_SERILER.items():
        try:
            df = api.get_data([seri], startdate=cs_dmy, enddate=ce_dmy)
            if df is not None and not df.empty:
                seri_col = seri.replace(".", "_")
                df = df.rename(columns={"Tarih": "Date", seri_col: kolon, seri: kolon})
                # Aylık veri "YYYY-M" formatında gelir (örn. "2021-1") → ayın 1'i yap
                df["Date"] = pd.to_datetime(
                    df["Date"].astype(str).str.strip() + "-01",
                    format="%Y-%m-%d",
                    errors="coerce",
                )
                df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
                if kolon in df.columns:
                    df[kolon] = pd.to_numeric(df[kolon], errors="coerce")
                    monthly_frames[kolon] = df[[kolon]]
                    print(f"  ✓ {kolon:<10}: {len(df)} satır (aylık → ffill)")
        except Exception as e:
            print(f"  ✗ {kolon:<10}: {e}")
        time.sleep(0.2)

    # ── Birleştir ──────────────────────────────────────────────────────────
    all_frames = list(daily_frames.values()) + list(monthly_frames.values())
    if not all_frames:
        print("  [EVDS] Hiçbir seri gelmedi.")
        return pd.DataFrame()

    result = all_frames[0]
    for f in all_frames[1:]:
        result = result.join(f, how="outer")

    result = result.sort_index()
    print(f"  [EVDS] Toplam: {len(result)} satır × {result.shape[1]} sütun")
    return result


# ─────────────────────────────────────────────
# 4. TEFAS — Top-5 hisse fonu (3 aylık chunk, ~90 gün limit)
# ─────────────────────────────────────────────
def _discover_equity_funds(crawler: Crawler, top_n: int = 5) -> list:
    """Bugünden 5 iş günü önceki referans günde en büyük hisse fonlarını keşfeder."""
    ref = pd.bdate_range(end=datetime.today(), periods=5)[0].strftime("%Y-%m-%d")
    print(f"  [TEFAS] Keşif tarihi: {ref}")
    try:
        df = crawler.fetch(start=ref, end=ref)
        if df is None or df.empty:
            print("  [TEFAS] Keşif boş döndü.")
            return []

        df["stock"]      = pd.to_numeric(df.get("stock",      0), errors="coerce").fillna(0)
        df["market_cap"] = pd.to_numeric(df.get("market_cap", 0), errors="coerce").fillna(0)

        print(f"  [TEFAS] stock≥70: {(df['stock']>=70).sum()} fon  "
              f"| max={df['stock'].max():.0f}")

        for thr in [70, 50, 30, 0]:
            sub = df[df["stock"] >= thr].sort_values("market_cap", ascending=False).head(top_n)
            if not sub.empty:
                codes = sub["code"].tolist()
                print(f"  [TEFAS] Eşik≥{thr}% → seçilen: {codes}")
                return codes
    except Exception as e:
        print(f"  [TEFAS] Keşif hatası: {e}")
    return []


def fetch_tefas(start_date: str, end_date: str, top_n: int = 5) -> pd.DataFrame:
    print("\n[TEFAS] Hisse fonu verileri çekiliyor (3 aylık chunk)...")
    crawler = Crawler()

    funds = _discover_equity_funds(crawler, top_n=top_n)
    if not funds:
        print("  [TEFAS] Fon bulunamadı — atlanıyor.")
        return pd.DataFrame()

    # TEFAS ~90 gün veriyor; 3 aylık (91 günlük) dilimlerle tüm aralığı tara
    all_frames = []
    for fund in funds:
        fund_chunks = []
        for chunk_s, chunk_e in _date_chunks(start_date, end_date, days=91):
            try:
                df = crawler.fetch(
                    start=chunk_s,
                    end=chunk_e,
                    name=fund,
                    columns=["date", "code", "price"],
                )
                if df is not None and not df.empty:
                    fund_chunks.append(df[["date", "code", "price"]])
            except Exception:
                pass
            time.sleep(0.2)

        if fund_chunks:
            combined = pd.concat(fund_chunks, ignore_index=True).drop_duplicates("date")
            print(f"  ✓ {fund}: {len(combined)} satır")
            all_frames.append(combined)
        else:
            print(f"  ✗ {fund}: veri gelmedi")

    if not all_frames:
        print("  [TEFAS] Hiçbir fondan veri gelmedi.")
        return pd.DataFrame()

    df_tefas = pd.concat(all_frames, ignore_index=True)
    df_tefas["date"] = pd.to_datetime(df_tefas["date"])

    pivot = df_tefas.pivot_table(index="date", columns="code", values="price", aggfunc="last")
    pct   = pivot.pct_change()
    pct.columns = [f"{c}_RET" for c in pct.columns]
    pct.index.name = "Date"
    print(f"  [TEFAS] Toplam: {len(pct)} satır × {pct.shape[1]} fon sütunu")
    return pct


# ─────────────────────────────────────────────
# 5. MASTER MERGE
# ─────────────────────────────────────────────
def build_master_database(
    tickers:      list,
    evds_api_key: str,
    start_date:   str = "2021-01-01",
    end_date:     str = None,
    fund_start_year: int = 2020,
):
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")

    print(f"\n{'='*55}")
    print(f"  Minerva Master Pipeline  {start_date} → {end_date}")
    print(f"  Hisseler : {tickers}")
    print(f"{'='*55}")

    end_year = int(end_date[:4])

    # ── Global veriler (bir kez çek) ──────────────────────────
    df_macro  = fetch_macro(evds_api_key, start_date, end_date)
    df_tefas  = fetch_tefas(start_date, end_date)

    # ── Hisse bazlı işlem ─────────────────────────────────────
    print("\n[BİLANÇO + MERGE] Hisseler işleniyor...")
    master_rows = []

    for ticker in tickers:
        print(f"\n  → {ticker}")

        # OHLCV tek hisse
        df_price = yf.download(
            ticker + ".IS", start=start_date, end=end_date,
            progress=False, auto_adjust=True,
        )
        if df_price.empty:
            print(f"    ✗ Fiyat verisi yok, atlanıyor.")
            continue
        if isinstance(df_price.columns, pd.MultiIndex):
            df_price.columns = df_price.columns.get_level_values(0)
        df_price = df_price.reset_index()
        df_price["Date"]   = pd.to_datetime(df_price["Date"]).dt.tz_localize(None)
        df_price["Ticker"] = ticker
        df_price = df_price[["Date", "Close", "High", "Low", "Open", "Volume", "Ticker"]]

        # Bilanço (start_year bir yıl erken → F/K geçmişi dolsun)
        print(f"    isyatirimhisse bilanço çekiliyor ({fund_start_year}-{end_year})...")
        df_fund = fetch_fundamentals(ticker, fund_start_year, end_year)
        if not df_fund.empty:
            print(f"    ✓ Bilanço: {len(df_fund)} dönem")

        # Merge: Date indexi üzerinden left-join
        df_merged = df_price.set_index("Date")

        if not df_macro.empty:
            df_merged = df_merged.join(df_macro, how="left")
        if not df_tefas.empty:
            df_merged = df_merged.join(df_tefas, how="left")
        if not df_fund.empty:
            df_merged = df_merged.join(df_fund, how="left")

        # Sadece ileriye doldur (look-ahead koruması — asla bfill kullanma)
        df_merged.ffill(inplace=True)

        # Dinamik çarpanlar
        if {"SERMAYE", "OZKAYNAK", "NET_KAR"}.issubset(df_merged.columns):
            piyasa = df_merged["Close"] * df_merged["SERMAYE"]
            df_merged["PD_DD"] = piyasa / df_merged["OZKAYNAK"]
            df_merged["FK"]    = np.where(
                df_merged["NET_KAR"] > 0, piyasa / df_merged["NET_KAR"], np.nan
            )
            df_merged.drop(
                columns=["SERMAYE", "OZKAYNAK", "NET_KAR"], errors="ignore", inplace=True
            )

        master_rows.append(df_merged.reset_index())
        time.sleep(0.3)

    # ── Kaydet ───────────────────────────────────────────────
    if not master_rows:
        print("\n❌ Hiçbir hisse işlenemedi.")
        return

    final = pd.concat(master_rows, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    final.to_parquet("data/market_db_master.parquet", engine="pyarrow")

    print(f"\n{'='*55}")
    print(f"  ✅ Veri tabanı kaydedildi → data/market_db_master.parquet")
    print(f"  Toplam satır : {len(final):,}")
    print(f"  Hisse sayısı : {final['Ticker'].nunique()}")
    print(f"  Sütunlar     : {list(final.columns)}")
    print(f"  Tarih aralığı: {final['Date'].min().date()} → {final['Date'].max().date()}")

    # Doluluk raporu
    print(f"\n  {'Sütun':<18} {'Dolu':>8} {'Oran':>7}")
    print(f"  {'-'*35}")
    for col in final.columns:
        n = final[col].notna().sum()
        print(f"  {col:<18} {n:>8,} {n/len(final)*100:>6.1f}%")
    print(f"{'='*55}\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    TICKERS  = ["THYAO", "TUPRS", "FROTO", "AKBNK", "TSKB"]
    EVDS_KEY = "TtvxPNsuO2"

    build_master_database(
        tickers      = TICKERS,
        evds_api_key = EVDS_KEY,
        start_date   = "2021-01-01",
        end_date     = datetime.today().strftime("%Y-%m-%d"),
        fund_start_year = 2020,
    )
