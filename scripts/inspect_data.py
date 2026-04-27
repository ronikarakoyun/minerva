import pandas as pd

# Parquet dosyasını oku
print("Veri yükleniyor...")
df = pd.read_parquet("data/market_db_master.parquet")

print("\n📊 --- VERİ TABANI ÖZETİ (INFO) ---")
df.info()

print("\n📈 --- İLK 10 SATIR (THYAO'nun başlangıcı) ---")
# Sütunların hepsi ekrana sığsın diye pandas ayarlarını açıyoruz
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.head(10))

print("\n📉 --- SON 5 SATIR (En güncel veriler) ---")
print(df.tail(5))