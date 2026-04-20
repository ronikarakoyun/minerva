import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import evds
from tefas import Crawler
import time
from datetime import datetime

# ==========================================
# 1. TEMEL ANALİZ (İŞ YATIRIM) KATMANI
# ==========================================
def fetch_fundamentals(ticker, start_year, end_year):
    """İş Yatırım'dan çeyreklik bilanço kalemlerini (YIL BAZINDA DÖNGÜ İLE) çeker."""
    print(f"[{ticker}] İş Yatırım Temel Verileri Çekiliyor...")
    
    BANKALAR = ["AKBNK", "GARAN", "YKBNK", "ISCTR", "VAKBN", "HALKB", "QNBFB", "TSKB", "ALBRK", "SKBNK"]
    fin_group = "XI_59" if ticker in BANKALAR else "XI_29"
    headers = {"User-Agent": "Mozilla/5.0"}
    all_years_df = []
    
    for year in range(start_year, end_year + 1):
        url = f"https://www.isyatirim.com.tr/_layouts/15/IsYatirim.Website/Common/Data.aspx/MaliTablo?companyCode={ticker}&exchange=TRY&financialGroup={fin_group}&year1={year}&period1=3&year2={year}&period2=12"
        
        try:
            r = requests.get(url, headers=headers, timeout=10)
            data = r.json()
            if 'value' not in data or not data['value']:
                continue
                
            df_json = pd.DataFrame(data['value'])
            
            # Kırılgan positional index yerine güvenli sütun seçimi (Bug 2 Fix)
            value_cols = [c for c in df_json.columns if c.startswith('value')]
            
            ozk_isim = "Özkaynaklar" if fin_group == "XI_59" else "Ana Ortaklığa Ait Özkaynaklar"
            kar_isim = "Dönem Net Kar veya Zararı" if fin_group == "XI_59" else "DÖNEM KARI (ZARARI)"
            sermaye_isim = "Ödenmiş Sermaye"
            
            ozk_row = df_json[df_json['itemDescTr'] == ozk_isim]
            kar_row = df_json[df_json['itemDescTr'] == kar_isim]
            ser_row = df_json[df_json['itemDescTr'] == sermaye_isim]
            
            if ozk_row.empty or kar_row.empty or ser_row.empty:
                continue
                
            ozkaynak = pd.to_numeric(ozk_row[value_cols].iloc[0], errors='coerce').values
            net_kar = pd.to_numeric(kar_row[value_cols].iloc[0], errors='coerce').values
            sermaye = pd.to_numeric(ser_row[value_cols].iloc[0], errors='coerce').values
            
            n_periods = len(ozkaynak)
            
            # İş Yatırım veriyi yeniden eskiye verir, zaman akışı için ters çeviriyoruz
            ozkaynak = ozkaynak[::-1]
            net_kar = net_kar[::-1]
            sermaye = sermaye[::-1]
            
            months = [3, 6, 9, 12][:n_periods]
            dates = [pd.to_datetime(f"{year}-{m:02d}-01") + pd.offsets.MonthEnd(0) for m in months]
            
            df_year = pd.DataFrame({
                "OZKAYNAK": ozkaynak,
                "NET_KAR": net_kar,
                "SERMAYE": sermaye
            }, index=dates)
            
            all_years_df.append(df_year)
            
        except Exception as e:
            pass 
            
    if not all_years_df:
        print(f"[{ticker}] Tüm yıllar boş döndü. (Sistemde veri yok)")
        return pd.DataFrame()
        
    df_fund = pd.concat(all_years_df)
    df_fund.sort_index(inplace=True)
    
    df_fund['ROE'] = np.where(df_fund['OZKAYNAK'] > 0, (df_fund['NET_KAR'] / df_fund['OZKAYNAK']) * 100, np.nan)
    df_fund.index.name = "Date"
    
    # 👑 LOOK-AHEAD BİAS KORUMASI (45 Gün Gecikme)
    df_fund.index = df_fund.index + pd.Timedelta(days=45)
    
    return df_fund

# ==========================================
# 2. MAKRO VERİ (TCMB EVDS) KATMANI
# ==========================================
def fetch_macro(evds_api_key, start_date, end_date):
    """TCMB'den Dolar ve Faiz verisini çeker"""
    print(f"[MAKRO] TCMB EVDS Verileri Çekiliyor ({start_date} - {end_date})...")
    try:
        evds_api = evds.evdsAPI(evds_api_key)
        df_macro = evds_api.get_data(['TP.DK.USD.A', 'TP.POLITEZ.FAIZ'], startdate=start_date, enddate=end_date)
        
        df_macro['Tarih'] = pd.to_datetime(df_macro['Tarih'], format="%d-%m-%Y")
        
        # Orijinal noktalı isimleri eşleştirme (Bug 1 Fix)
        df_macro = df_macro.rename(columns={
            'Tarih': 'Date', 
            'TP.DK.USD.A': 'USD_TRY', 
            'TP.POLITEZ.FAIZ': 'TCMB_FAIZ'
        })
        df_macro.set_index('Date', inplace=True)
        
        available_cols = [col for col in ['USD_TRY', 'TCMB_FAIZ'] if col in df_macro.columns]
        if 'TCMB_FAIZ' not in available_cols:
            print("[MAKRO Uyarı] TCMB_FAIZ verisi API'den gelmedi, sadece olan veriler eklenecek.")
            
        return df_macro[available_cols]
    except Exception as e:
        print(f"EVDS Kritik Hata: {e}")
        return pd.DataFrame()

# ==========================================
# 3. KURUMSAL PARA AKIŞI (TEFAS) KATMANI
# ==========================================
def fetch_tefas_funds(start_date, end_date):
    """TEFAS'tan Hisse Senedi Yoğun Fonların fiyatlarını çeker"""
    print("[TEFAS] Fon Verileri Çekiliyor...")
    try:
        crawler = Crawler()
        funds = ['MAC', 'NNF', 'TKF'] 
        
        # Yeni TEFAS API formatı (Bug 4 Fix)
        df_tefas = crawler.fetch(start=start_date, end=end_date)
        df_tefas = df_tefas[['date', 'code', 'price']]
        
        df_tefas = df_tefas[df_tefas['code'].isin(funds)]
        df_tefas['date'] = pd.to_datetime(df_tefas['date'])
        
        df_tefas_pivot = df_tefas.pivot(index='date', columns='code', values='price')
        df_tefas_pct = df_tefas_pivot.pct_change()
        df_tefas_pct.columns = [f"{col}_RET" for col in df_tefas_pct.columns]
        df_tefas_pct.index.name = "Date"
        return df_tefas_pct
    except Exception as e:
        print(f"TEFAS Hatası: {e}")
        return pd.DataFrame()

# ==========================================
# 4. MASTER MERGE (BÜYÜK BİRLEŞTİRME)
# ==========================================
def build_master_database(tickers, evds_api_key, start_date="2021-01-01", end_date="2024-01-01"):
    print(f"🚀 Master Pipeline Başlıyor ({start_date} -> {end_date})...")
    
    # Dinamik Tarih Formatları (Bug 3 Fix)
    _evds_start = datetime.strptime(start_date, "%Y-%m-%d").strftime("%d-%m-%Y")
    _evds_end   = datetime.strptime(end_date, "%Y-%m-%d").strftime("%d-%m-%Y")
    
    start_year = int(start_date[:4])
    end_year   = int(end_date[:4])
    
    df_macro = fetch_macro(evds_api_key, _evds_start, _evds_end) 
    df_tefas = fetch_tefas_funds(start_date, end_date)
    
    master_rows = []
    
    for ticker in tickers:
        print(f"\n---> {ticker} İşleniyor...")
        
        df_price = yf.download(ticker + ".IS", start=start_date, end=end_date, progress=False)
        if df_price.empty:
            continue
            
        if isinstance(df_price.columns, pd.MultiIndex):
            df_price.columns = df_price.columns.get_level_values(0)
            
        df_price = df_price.reset_index()
        df_price['Ticker'] = ticker
        
        df_fund = fetch_fundamentals(ticker, start_year, end_year)
        
        df_merged = df_price.copy()
        df_merged.set_index('Date', inplace=True)
        
        if not df_macro.empty:
            df_merged = df_merged.join(df_macro, how='left')
        if not df_tefas.empty:
            df_merged = df_merged.join(df_tefas, how='left')
        if not df_fund.empty:
            df_merged = df_merged.join(df_fund, how='left')
        
        # İleriye Sarma (Look-Ahead engeli)
        df_merged.ffill(inplace=True)
        
        # 👑 DİNAMİK ÇARPAN (F/K & PD/DD) HESAPLAYICI
        if 'SERMAYE' in df_merged.columns and 'OZKAYNAK' in df_merged.columns:
            df_merged['Piyasa_Degeri'] = df_merged['Close'] * df_merged['SERMAYE']
            df_merged['PD_DD'] = df_merged['Piyasa_Degeri'] / df_merged['OZKAYNAK']
            
            if 'NET_KAR' in df_merged.columns:
                df_merged['FK'] = np.where(df_merged['NET_KAR'] > 0, 
                                           df_merged['Piyasa_Degeri'] / df_merged['NET_KAR'], 
                                           np.nan)
            
            df_merged.drop(columns=['Piyasa_Degeri', 'NET_KAR', 'OZKAYNAK', 'SERMAYE'], inplace=True, errors='ignore')

        df_merged = df_merged.reset_index()
        master_rows.append(df_merged)
        time.sleep(0.5)

    final_db = pd.concat(master_rows, ignore_index=True)
    os.makedirs("data", exist_ok=True)
    final_db.to_parquet("data/market_db_master.parquet", engine="pyarrow")
    print("\n✅ Veri Tabanı Başarıyla Oluşturuldu! Kayıt: data/market_db_master.parquet")

if __name__ == "__main__":
    # Test Listesine TSKB de eklendi
    bist_tickers = ["THYAO", "TUPRS", "FROTO", "AKBNK", "TSKB"] 
    EVDS_KEY = "TtvxPNsuO2"
    # Tüm datayı dinamik olarak 2021'den bugüne çekiyoruz
    today_str = datetime.today().strftime("%Y-%m-%d")
    build_master_database(bist_tickers, EVDS_KEY, start_date="2021-01-01", end_date=today_str)