import pandas as pd
import numpy as np

db = pd.read_parquet("data/market_db.parquet")
db["Date"] = pd.to_datetime(db["Date"])
if "Pvwap" not in db.columns:
    db["Pvwap"] = (db["Phigh"] + db["Plow"] + db["Pclose"]) / 3

# SADECE TRAIN — sidebar'daki split noktasıyla aynı
split = db["Date"].min() + (db["Date"].max() - db["Date"].min()) * 0.7
train = db[db["Date"] < split].copy()
print(f"Train dönemi: {train['Date'].min().date()} → {train['Date'].max().date()}")
print(f"Train satır: {len(train):,}")

# Aynı istatistikleri yalnız train'de
train_s = train.sort_values(["Ticker","Date"])
train_s["Ret"] = train_s.groupby("Ticker")["Pclose"].pct_change()

print("\n=== TRAIN GÜNLÜK GETİRİ ===")
print(train_s["Ret"].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).round(4))
print(f"Skew: {train_s['Ret'].skew():.2f}, Kurt: {train_s['Ret'].kurt():.2f}")

print("\n=== TRAIN HACİM ===")
print(train["Vlot"].describe(percentiles=[.1,.5,.9]).round(0))

print("\n=== TRAIN AYLIK VOL ===")
train_s["YM"] = train_s["Date"].dt.to_period("M")
print(train_s.groupby("YM")["Ret"].std().describe().round(4))

print("\n=== TRAIN FİYAT ===")
print(train["Pclose"].describe(percentiles=[.1,.5,.9]).round(2))