# Minerva v12 — Veri Zenginleştirme Planı

## Bağlam

v11 maratonu tamamlandığında elde edeceğimiz beklenen sonuç: **CAGR ~%5-10, Sharpe 0.5-0.8**. Bu değerler mevcut feature space'in (5 değişken: `Pclose, Phigh, Plow, Popen, Vlot`) **fiziksel sınırıdır**. Filter sıkılaştırarak veya threshold ayarlayarak çıkarılacak ek alfa marjinaldir.

**Çözüm yönü:** Feature space'i sektörel/makro/microstructure verilerle genişletmek. Bu doküman v12 maratonu için veri kaynakları, pipeline mimarisi ve entegrasyon planını tanımlar.

---

## 1. Mevcut Durum

### 1.1 Şu anki feature'lar

`engine/core/alpha_cfg.py:67`:
```python
FEATURES = ["Popen", "Phigh", "Plow", "Pclose", "Vlot", "Ptyp"]
```

Mining MCTS bu 6 feature'dan kombinasyon arar. v11 sonucunda en iyi formüller:
- `Delta(Phigh, 30)` (RIC=0.0118) — 30 günlük zirve değişimi
- `Cov(Sub(0.01, Plow), Vlot, 30)` (RIC=0.0153) — fiyat-volatilite kovaryansı

RIC ~0.012-0.015 → marjinal alfa. Bu, OHLCV'den çıkarılabilecek üst sınıra yakın.

### 1.2 Mevcut veri şeması

`data/market_db.parquet`:
```
columns: [Date, Ticker, Popen, Phigh, Plow, Pclose, Vlot, ...]
shape:   ~13 yıl × ~500 hisse ≈ 1.5 GB
index:   (Ticker, Date) — cross-sectional ranking için ideal
```

---

## 2. Yeni Veri Kaynakları — Tier'lara Göre

Her tier için: **Ne**, **Nereden**, **Frekans**, **Schema entegrasyonu**, **Mining'de açtığı formül örnekleri**.

### 🥇 Tier 1 — Yüksek ROI, Kolay Erişim

#### 1.1 USD/TRY ve EUR/TRY

| Alan | Değer |
|---|---|
| **Ne** | Günlük FX kapanış kuru |
| **Nereden** | TCMB EVDS API (TP.DK.USD.A.YTL serisi), yfinance `USDTRY=X` |
| **Frekans** | Günlük (10:30 ve 15:30 TCMB resmi noktalar) |
| **Look-ahead riski** | Yok (T günü kapanışı = T+1 sabah kullanılabilir) |

**Schema:**
- Yeni dosya: `data/macro_db.parquet` (Date, indicator, value) — market-wide tablolar için ayrı
- VEYA `market_db.parquet`'e `USDTRY` kolonu olarak broadcast (her ticker için aynı değer)

**Mining'de açacağı formüller:**
- `Pclose / USDTRY` → TL bazlı hisseyi dolar'a çevir, TL düştüğünde yükselen hisseleri ayırt et
- `Beta(Pclose, USDTRY, 60)` → 60 gün kur duyarlılığı (ihracatçı vs ithalatçı ayrımı)
- `Corr(Vlot, Delta(USDTRY, 1), 20)` → kur şokunda hacim değişimi

**Faydalı kavram:** "TL beta" — bazı hisseler USD/TRY yükselince yükselir (EREGL, KCHOL gibi ihracatçılar), bazıları düşer (THYAO yakıt maliyeti). Bu ayrım v11'de YOK.

#### 1.2 BIST Endeks ve Sektör Endeksleri

| Alan | Değer |
|---|---|
| **Ne** | XU100, XU030, XBANK, XGIDA, XELKT, XKMYA, XHOLD vb. (toplam ~20 endeks) |
| **Nereden** | yfinance (`XU100.IS`), Borsa İstanbul resmi günlük bülten |
| **Frekans** | Günlük OHLC |
| **Schema** | `data/macro_db.parquet` (indicator: XU100, value: kapanış) |

**Mining'de açacağı formüller:**
- `Pclose / XU100` → market-relative güç (göreceli momentum)
- `Beta(Pclose, XU100, 120)` → 120-gün market betası
- `Pclose / XBANK` (banka için) → sektör-içi göreceli pozisyon
- `Rank(Pclose) - Rank(XU100)` → cross-sectional vs index relative

**Kritik:** Sektör endeksi seçimi her ticker için farklı olmalı. AKBNK için XBANK, EREGL için XKMYA. Bu eşleme tablosu gerekli (Bölüm 5).

#### 1.3 TR Makro Göstergeler

| Alan | Değer |
|---|---|
| **Ne** | TCMB politika faizi, 10Y tahvil getirisi, manşet enflasyon, ÜFE |
| **Nereden** | TCMB EVDS REST API (`https://evds2.tcmb.gov.tr/service/evds`) |
| **Frekans** | Faiz: günlük, Enflasyon: aylık (3'ünde yayınlanır) |
| **Look-ahead riski** | **VAR** — aylık veri ay sonunda kapatılır ama 3 hafta sonra açıklanır. PIT discipline şart. |

**Schema:**
- `data/macro_db.parquet` (Date, indicator, value, **as_of_date**)
- `as_of_date` = veri *kullanılabilir* olduğu tarih (açıklama tarihi). Look-ahead'i bu kolon önler.
- Mining'de kullanım: `db[db.Date >= macro.as_of_date]` join

**Mining'de açacağı formüller:**
- `Delta(InterestRate, 5)` → 5 günlük faiz değişimi (faiz şoku)
- `Inflation_yoy / Pclose_yoy` → reel hisse getirisi
- `Sub(InterestRate, CPI_yoy)` → reel faiz (bankalara duyarlı)

#### 1.4 Global Risk-On/Off

| Alan | Değer |
|---|---|
| **Ne** | S&P 500 (SPX), MSCI EM (EEM), VIX, DXY |
| **Nereden** | yfinance |
| **Frekans** | Günlük (TR ile takvim farkı dikkat — ABD borsası kapalıyken TR açık olabilir) |

**Saat dilimi tuzağı:** Bugün TR borsası açılırken SPX dün kapandı. Bu yüzden `SPX_t-1` kullanılır, `SPX_t` değil. EVDS API'da bu otomatik değil, manuel offset gerekli.

**Mining formülleri:**
- `Delta(SPX, 1)` → dün gece ABD'den gelen yön
- `Corr(Pclose, MSCI_EM, 60)` → 60-gün EM korelasyonu

---

### 🥈 Tier 2 — Yüksek Etki, Orta İş

#### 2.1 Fundamental Veri (Bilanço)

| Alan | Değer |
|---|---|
| **Ne** | P/E, P/B, ROE, Net Borç/EBITDA, Net Marj |
| **Nereden** | KAP (kap.org.tr) bilanço PDF'leri → Foreks/Mynet aggregator API, ücretliyse fintables.com |
| **Frekans** | Çeyreklik (3,6,9,12. aylar) |
| **Açıklama gecikmesi** | **KRİTİK: ~45-60 gün.** Q1 (Mart sonu) bilançosu Mayıs ortası açıklanır. PIT discipline ZORUNLU. |

**Schema:**
```
data/fundamentals.parquet
columns: [Ticker, period_end, announce_date, PE, PB, ROE, NetDebt_EBITDA, NetMargin]
```

Marathon'da kullanım:
```python
# T günü için T'den önce ANNOUNCE edilmiş son bilanço
fund_pit = fundamentals[fundamentals.announce_date <= T] \
    .groupby('Ticker').tail(1)
```

**Mining formülleri:**
- `1 / PE` → earnings yield (Fama-French value faktörü)
- `Delta(ROE, 4q)` → ROE momentum (Asness Quality faktörü)
- `Rank(Inv(PE)) - Rank(Inv(XU100_PE))` → relative valuation

**Faktör nötralizasyon:** v11'deki `factor_neutralize` modülü zaten size factor kullanıyor. Buna PE, PB factor'leri eklenir — alfa sinyali bilinen value/size faktörlerinden ARTAKALAN'ı bulur, başka yerde halihazırda fiyatlanmış faktörü tekrar bulmaz.

#### 2.2 Hacim Mikroyapısı

| Alan | Değer |
|---|---|
| **Ne** | Günlük lotlu işlem hacmi (TL), işlem sayısı, ortalama işlem büyüklüğü, VWAP |
| **Nereden** | Borsa İstanbul günlük bülten (https://www.borsaistanbul.com/data/bulten/) |
| **Frekans** | Günlük |

**Mining formülleri:**
- `Vlot_TL / ADV_20` → ortalama göreceli hacim (likidite artışı sinyali)
- `Amihud = |Pclose_ret| / Vlot_TL` (Amihud 2002 illiquidity)
- `Rank(TradeSize) × Rank(Pclose_ret)` → kurumsal alıcı işareti (büyük lot + fiyat artışı)

**Faydalı:** Mevcut slipaj modeli zaten ADV kullanıyor (`engine/execution/cont_stoikov_slippage.py`). Bu data zaten kısmen var, sadece mining feature olarak ekspoze edilmemiş.

---

### 🥉 Tier 3 — Güçlü Sinyal, Yüksek İş

#### 3.1 Yabancı Akışı (Foreign Flow)

| Alan | Değer |
|---|---|
| **Ne** | Hisse bazında net yabancı işlem (TL) |
| **Nereden** | MKK günlük raporları, Borsa İstanbul Veri Mağazası (ücretli) |
| **Frekans** | Günlük (T+1 açıklanır → 1 gün gecikme) |

**Schema:**
```
data/foreign_flow.parquet
columns: [Date, Ticker, foreign_net_TL, foreign_share_pct]
```

**Mining formülleri:**
- `Cumsum(foreign_net_TL, 5)` → 5 günlük birikimli yabancı alımı
- `foreign_share_pct - SMA(foreign_share_pct, 60)` → yabancı oranı değişimi
- `Sign(foreign_net_TL) × Vlot` → yabancı yönlü hacim

**Akademik kanıt:** Froot, O'Connell, Seasholes (2001) "The Portfolio Flows of International Investors" — yabancı akışı emerging markets'ta short-term price impact yaratır, alpha kaynağı.

#### 3.2 Sentiment ve Haber Verisi

| Alan | Değer |
|---|---|
| **Ne** | KAP özel durum açıklamaları, finansal medya haberleri |
| **Nereden** | KAP API (https://www.kap.org.tr) — bedava, Anadolu Ajansı haber feed (ücretli) |
| **Frekans** | Event-driven (her açıklama anında) |

**Çıkarım:**
- KAP açıklamasından materiality skoru (insider trading, temettü, yönetim değişikliği, vb.)
- Türkçe FinBERT (varsa) ile haber sentiment skoru — eğer Turkish model yoksa, çeviri pipeline gerekli

**Schema:**
```
data/events.parquet
columns: [Ticker, datetime, event_type, sentiment_score, materiality]
```

**Mining formülleri:**
- `Decay(materiality, halflife=5)` → 5 gün yarılanma ömürlü event score
- `Sum(sentiment_score, 7)` → haftalık sentiment akümülasyonu

**Bu Tier 3 çünkü:** NLP pipeline kurulması zaman alır, Türkçe model sınırlı, sentiment ground truth doğrulaması zor.

---

## 3. Pipeline Mimarisi

### 3.1 Veri Akışı

```
[Kaynaklar]                  [Fetch]                [Storage]                  [Birleştirme]            [Mining]
EVDS API     ───┐
yfinance     ───┤      scripts/fetch_macro.py  →  data/macro_db.parquet  ───┐
KAP          ───┤      scripts/fetch_fund.py   →  data/fundamentals.pq   ───┤
MKK          ───┘      scripts/fetch_flow.py   →  data/foreign_flow.pq   ───┤
                                                                            ↓
                                                              engine/data/data_merger.py
                                                                            ↓
                                                              data/enriched_market_db.parquet
                                                              ((Ticker, Date) × N features)
                                                                            ↓
                                                              MCTS mining (FEATURES listesi genişler)
```

### 3.2 Yeni dosyalar

| Yol | Sorumluluk |
|---|---|
| `scripts/fetch_macro.py` | EVDS + yfinance ile makro/endeks veri çek, `data/macro_db.parquet` |
| `scripts/fetch_fundamentals.py` | KAP scraper veya Foreks API, `data/fundamentals.parquet` |
| `scripts/fetch_foreign_flow.py` | MKK günlük raporları, `data/foreign_flow.parquet` |
| `engine/data/data_merger.py` | (Ticker, Date) join + PIT discipline + sector mapping |
| `data/sector_mapping.csv` | Ticker → sektör endeksi (manuel tutulur, çeyreklik update) |
| `data/macro_db.parquet` | Market-wide göstergeler (FX, faiz, endeks) |
| `data/fundamentals.parquet` | Çeyreklik bilanço, announce_date ile |
| `data/foreign_flow.parquet` | Günlük yabancı akış |

### 3.3 Schema sözleşmesi

**Market-wide data (`macro_db.parquet`):**
```
Date         | indicator | value    | as_of_date
2026-05-14   | USDTRY    | 34.21    | 2026-05-14  (eş-zamanlı)
2026-04-30   | CPI_yoy   | 0.681    | 2026-05-03  (gecikmeli)
```

**Ticker-specific time series (`fundamentals.parquet`):**
```
Ticker | period_end | announce_date | PE   | PB  | ...
AKBNK  | 2026-03-31 | 2026-05-12    | 5.2  | 0.8 | ...
```

**Birleştirilmiş feature db (`enriched_market_db.parquet`):**
```
Date | Ticker | Popen | Phigh | ... | USDTRY | XU100 | XBANK | PE | foreign_net_TL
```

Burada her gün, her ticker için tüm feature'lar **PIT geçerli** olmalı.

### 3.4 PIT Discipline (Look-ahead Önleme)

**Kuralı:** T günü mining'inde **announce_date ≤ T** olan veriler kullanılır.

```python
# data_merger.py - kavramsal kod
def build_enriched_db(date_t, market_db, fundamentals, macro_db):
    base = market_db[market_db.Date <= date_t].copy()

    # Fundamental: T'den önce announce edilmiş son bilanço
    fund_pit = fundamentals[fundamentals.announce_date <= date_t] \
        .sort_values(['Ticker', 'announce_date']) \
        .groupby('Ticker').tail(1)

    base = base.merge(fund_pit[['Ticker', 'PE', 'PB', 'ROE']], on='Ticker', how='left')

    # Macro: as_of_date ≤ T olan en güncel değer
    for ind in ['USDTRY', 'XU100', 'CPI_yoy']:
        macro_ind = macro_db[(macro_db.indicator == ind) &
                             (macro_db.as_of_date <= date_t)]
        latest = macro_ind.sort_values('as_of_date').groupby('Date').tail(1)
        base = base.merge(latest[['Date', 'value']].rename(columns={'value': ind}),
                          on='Date', how='left')

    return base
```

Bu fonksiyon `run_historical_paper_trade.py`'de **her gün için ayrı çağrılır** (snapshot). Hiçbir zaman geleceğin verisi kullanılmaz.

---

## 4. Kod Entegrasyonu

### 4.1 `alpha_cfg.py` Genişletme

```python
FEATURES_OHLCV = ["Popen", "Phigh", "Plow", "Pclose", "Vlot", "Ptyp"]
FEATURES_MACRO = ["USDTRY", "XU100", "XBANK_REL", "InterestRate", "CPI_yoy"]
FEATURES_FUND  = ["PE", "PB", "ROE", "NetDebt_EBITDA"]
FEATURES_FLOW  = ["foreign_net_TL_5d", "foreign_share_pct"]
FEATURES = FEATURES_OHLCV + FEATURES_MACRO + FEATURES_FUND + FEATURES_FLOW
```

**Dikkat:** MCTS arama uzayı feature sayısı ile **kuadratik** büyür. 6 → 18 feature: kombinasyon sayısı ~9× artar. `n_trials` ve `num_gen` ayarları gözden geçirilmeli.

### 4.2 Sektör Bazlı Beta

Sektör endeksi her ticker için farklı:
```python
# data/sector_mapping.csv kullanımı
def add_sector_features(base, sector_map, macro_db):
    base = base.merge(sector_map[['Ticker', 'sector_index']], on='Ticker')
    for sec_idx in base['sector_index'].unique():
        sec_data = macro_db[macro_db.indicator == sec_idx]
        # ... her ticker'ı kendi sektör endeksiyle eşle
    return base
```

`engine/core/alpha_cfg.py`'da `SectorRel(Pclose)` gibi sektör-rölatif operatör eklenebilir.

### 4.3 Factor Neutralization Genişletme

Mevcut `engine/data/factor_neutralize.py` size factor kullanıyor. Genişletilebilir:
```python
KNOWN_FACTORS = ["log_mktcap", "1_over_PE", "PB", "beta_USDTRY", "sector_dummy"]
```

DML neutralization bu faktörleri orthogonalize eder → mining ARTAKALAN alfa'yı bulur (already-priced faktörleri tekrar bulmaz).

---

## 5. Sektör Eşleme Tablosu

`data/sector_mapping.csv` örneği:
```csv
Ticker,sector_name,sector_index,industry
AKBNK,Bankacılık,XBANK,Mevduat Bankası
GARAN,Bankacılık,XBANK,Mevduat Bankası
EREGL,Demir-Çelik,XMETL,Yassı Çelik
KCHOL,Holding,XHOLD,Sanayi Holding
THYAO,Ulaştırma,XULAS,Havayolu
...
```

**Bakım:**
- Manuel tutuluyor (~500 satır)
- Çeyreklik kontrol (sektör değişiklikleri, yeni listelemeler, delisting)
- Versiyon kontrolü altında (git'te)

---

## 6. Aşamalı Geçiş Planı

### Faz A — Veri Çekme (1-2 gün)
- [ ] `scripts/fetch_macro.py` (EVDS + yfinance) — Tier 1.1, 1.2, 1.3, 1.4
- [ ] Geçmiş 13 yıl tarihsel doldurma
- [ ] `data/macro_db.parquet` üretimi
- [ ] Smoke test: `Pclose / USDTRY` kolonu doğru hesaplanıyor mu?

### Faz B — Birleştirme Pipeline (1 gün)
- [ ] `engine/data/data_merger.py`
- [ ] PIT discipline testleri
- [ ] `enriched_market_db.parquet` üretimi

### Faz C — Sektör Eşleme (0.5 gün)
- [ ] `data/sector_mapping.csv` (manuel + KAP'tan crosscheck)
- [ ] Sektör endeksi join testi

### Faz D — `alpha_cfg.py` Güncelleme (0.5 gün)
- [ ] FEATURES listesini genişlet
- [ ] Sektör-rölatif operatör (`SectorRel`)
- [ ] Beta operatörü (`Beta(x, y, window)`)
- [ ] Birim testler

### Faz E — Fundamental Veri (2-3 gün, Tier 2)
- [ ] `scripts/fetch_fundamentals.py` (KAP veya Foreks)
- [ ] PIT discipline (announce_date)
- [ ] Faktör nötralizasyona PE, PB ekle

### Faz F — Yabancı Akış (2 gün, Tier 3)
- [ ] `scripts/fetch_foreign_flow.py`
- [ ] Cache stratejisi (MKK günlük rapor parser)

### Faz G — v12 Marathon (gece çalışır)
- [ ] Yeni FEATURES ile mining çalıştır
- [ ] v11 vs v12 karşılaştırması (aynı dönem, farklı feature set)
- [ ] Per-feature importance analizi (hangi yeni feature gerçekten katkı sağladı?)

---

## 7. Risk ve Tuzaklar

### 7.1 Look-Ahead Bias
**En büyük risk.** Özellikle:
- Fundamental veri: bilanço açıklama tarihi vs period_end (~45 gün gecikme)
- Makro veri: TÜFE Aralık verisi 3 Ocak'ta açıklanır → 31 Aralık mining'de KULLANILAMAZ
- Sektör değişiklikleri: AKBNK her zaman XBANK içinde mi? Tarihsel olarak değişmiş olabilir.

**Önlem:** `as_of_date` ve `announce_date` her tabloda **zorunlu**.

### 7.2 Veri Kalitesi
- **Eksik veri:** Bazı küçük hisseler için yabancı akış olmayabilir
- **Restated bilanço:** Şirket Q2'de Q1'i revize edebilir → "as-reported" tutulur, never overwrite
- **Sektör değişikliği:** Holding'in sektörü zaman içinde değişebilir

**Önlem:** Eksik veri için NaN, mining'de NaN-aware operatörler.

### 7.3 Stationarity
Yeni feature'lar non-stationary (PE çarpanı zaman içinde değişir, USDTRY trend etmek). Mevcut `fracdiff` modülü zaten var ama yeni feature'lar için elle d-parametresi seçilmeli.

### 7.4 Hesaplama Maliyeti
- 6 feature → 18 feature: arrow DB boyutu **3× artar** (~75MB → 225MB)
- MCTS arama uzayı **9× artar**
- Marathon süresi tahmini: ~12-15 saat → 20-25 saat (single instance, 8 worker)
- **Önlem:** Sektör/macro feature'ları cache'lemek (her ticker için tekrar tekrar hesaplamamak)

### 7.5 Overfitting Riski
Daha fazla feature = daha fazla overfitting şansı. PBO (Bailey 2014) ve OOS holdout filtreleri zaten var (S1, S2 fixes), bunlar daha agresif çalışmalı:
- `min_mean_ric=0.008` korunabilir
- PBO threshold 0.5 → 0.4 (daha sıkı)
- Holdout filtresi mandatory (atlanmaz)

---

## 8. Beklenen Etki

v11 vs v12 karşılaştırması (hipotez):

| Metrik | v11 (OHLCV only) | v12 (zenginleştirilmiş) | Kaynak |
|---|---|---|---|
| Unique formül kabul | 26 | 100-300 | Daha geniş feature space |
| Mean RIC (top-1) | 0.012 | 0.025-0.040 | Sektör-rölatif + makro |
| CAGR | %5-10 | %15-25 | Yeni alpha kaynakları |
| Sharpe | 0.60 | 1.2-1.8 | Daha iyi risk-ayarlı getiri |
| Max DD | %10-15 | %8-12 | Sektör/makro hedge etkisi |

Bu hedefler agresif; iyimser senaryoyu gösterir. Gerçekçi: **Sharpe 1.0+, CAGR %12-15** yeterli kabul edilebilir.

---

## 9. Kaynak/Referans İhtiyaçları

Aşağıdaki konularda NotebookLM kaynakları doğrulama için faydalı olur:

| Konu | Önerilen Kaynak |
|---|---|
| PIT discipline, look-ahead bias | López de Prado "Advances in Financial ML" Böl. 4 |
| Fama-French faktör modelleri | Fama-French 1993, Asness "Quality Minus Junk" 2018 |
| Foreign flow alpha | Froot, O'Connell, Seasholes 2001 |
| Sentiment finansta | Tetlock 2007 "Giving Content to Investor Sentiment" |
| Amihud illiquidity | Amihud 2002 "Illiquidity and Stock Returns" |
| BIST mikroyapı | Borsa İstanbul İşlem Kuralları (resmi PDF) |
| TR FX-equity beta | Dornbusch-Fischer "Currency Crisis" modeli |
| Cross-sectional asset pricing | Bali-Engle-Murray "Empirical Asset Pricing" |

NotebookLM'de bunlardan eksik olanlar varsa eklenmesi v12 dokümanasyonunu güçlendirir.

---

## 10. Veri Dışı Getiri Artırıcılar

Veri zenginleştirme tek lever değil. Mevcut mimarinin **strategy/execution/sizing** katmanlarında da 8-10 ek alfa kaynağı var. Bu bölüm onları sıralar; v13+ marathon'larda devreye alınacak.

### 10.1 Tier 1 — Yüksek Etki (+5-15% CAGR)

#### 10.1.1 Long/Short (Equity Market Neutral)

**Sorun:** Mevcut sistem **long-only**. Alfa formülü hem yükseliş hem düşüş tahmin eder ama yarısını kullanıyoruz.

**Çözüm:** Cross-sectional ranking'te:
- **Long:** signal rank top %20
- **Short:** signal rank bottom %20 (VIOP individual stock futures veya XU100 endeks)
- **Neutral:** ortadaki %60 pas

**BIST sınırlamaları:**
- VIOP'ta sadece **liquid 100 hisse** futures var
- Ortalama BIST hissesini tek tek shortlamak imkansız → **endeks short** (XU100 future) ile market-neutral yapılabilir

**Beklenen etki:** Alfa **2×**, Sharpe 0.6 → 1.2-1.5.

#### 10.1.2 VIOP Beta Hedging — Saf Alfa

**Mevcut:** Portföy beta'sı ~1.0 (XU100 ile aynı yönde hareket).

**Önerilen:** XU100 vadeli (VIOP F_XU100) short pozisyon ile beta'yı 0'a indir:
```
hedge_size_TL = portfolio_value × portfolio_beta × XU100_future_multiplier
```

**Etki:**
- CAGR aynı kalabilir, **volatilite yarıya iner**
- Sharpe 0.6 → 1.5-2.0
- 2018 kur krizi (-%20) gibi market şokları nötralize edilir

**Uygulama maliyeti düşük:**
- Sadece XU100 futures fiyat verisi gerekli (yfinance: `F_XU100.IS` veya finnet API)
- Mevcut paper_trader'a ek bir "hedge_position" kolonu yeter

#### 10.1.3 Multi-Frequency Mining — Birden Fazla Zaman Ölçeği

**Mevcut:** Aylık rebalance (hold_days=21). Tek frekans, sinyal kalitesi bu zaman ölçeğine bağımlı.

**Önerilen 3 katmanlı ensemble:**
- **Aylık layer** (mevcut): Value/fundamental sinyaller — yavaş döner
- **Haftalık layer** (yeni): Momentum/teknik — orta hız
- **Günlük layer** (yeni): Mean-reversion, event-driven — hızlı

Üç katmanın sinyalleri ağırlıklı birleşir. **Akademik kanıt:** Moskowitz-Ooi-Pedersen 2012 "Time Series Momentum" — farklı frekansların alfaları zayıf-korelasyon, ensemble eder.

**Etki:** Sharpe **1.5-2× artar** (uncorrelated alpha sources additive).

**Uygulama maliyeti yüksek:** 3× mining süresi, kompleks blender mimarisi.

---

### 10.2 Tier 2 — Orta Etki (+2-5% CAGR)

#### 10.2.1 Fractional Kelly Position Sizing

**Mevcut:** Equal Risk Contribution (ERC) — her pozisyon aynı risk katkısı.

**Sorun:** Top-conviction sinyal (RIC=0.05) ile zayıf sinyal (RIC=0.01) aynı ağırlığı alıyor.

**Kelly formülü:**
```
optimal_weight = (expected_return / variance) × (1 / max_drawdown_tolerance)
```

**Quarter Kelly** (Kelly/4) tipik kullanım — full Kelly çok agresif. Conviction-weighted sizing.

**Etki:** CAGR +3-5%, MDD biraz artar.

**Geliştirme süresi:** 1-2 gün, mevcut ERC ile yan yana koş, A/B karşılaştır.

#### 10.2.2 ATR-Based Stop-Loss / Take-Profit

**Mevcut:** Sabit hold_days=21 — pozisyon iyi/kötü olsun 21 gün tutulur.

**Önerilen — Dinamik trailing stop:**
```python
trailing_stop = max(trailing_stop, current_px - 2 × ATR(20))
take_profit  = entry_px + 4 × ATR(20)   # 2:1 R:R
```

**Etki:**
- Kötü trade erken kes → MDD %10-15 azalır
- İyi trade trail et → kazanan yüzdesi artar
- BIST 2018 kur krizinde -%20 yerine -%8'de çıkış mümkün

**Geliştirme süresi:** 1-2 gün, paper_trader içinde exit logic.

#### 10.2.3 Sector Rotation Overlay

**Mevcut:** Sektör bilgisi mining'de yok (v12 data planı bunu çözer).

**Eklenirse iki katmanlı strateji:**
- **Macro layer:** Hangi sektör outperform edecek? (sektör momentum, makro overlay)
- **Micro layer:** O sektör içinde hangi hisse? (mevcut MCTS)

**Akademik kanıt:** Cohen & Polk 2009 — sektör momentum + within-sector pick güçlü kombinasyon (Sharpe ~+0.3).

**Geliştirme süresi:** 3-5 gün (v12 sonrası, sektör data hazır olunca).

---

### 10.3 Tier 3 — Mimari İyileştirmeler

#### 10.3.1 Hierarchical Risk Parity (HRP)

**Mevcut:** ERC korelasyon-aware değil — her pozisyona eşit risk dağıtır, korelasyonlu pozisyonlar overweight olabilir.

**HRP (López de Prado 2016):** Korelasyon kümelerini hiyerarşik kümeleyerek dağıtım yapar.

**Etki:** Drawdown %10-20 azalır, Sharpe biraz artar.

**Geliştirme süresi:** 1-2 gün (mevcut `engine/risk/portfolio_allocator.py` genişletilir).

#### 10.3.2 Walk-Forward Hyperparameter Tuning

**Mevcut:** `min_mean_ric=0.008`, `λ_std=0.74`, `hold_days=21` — hardcoded sabitler.

**Önerilen:** Her quarter'da bu parametreler bir önceki **2 yıl OOS** üzerinde Optuna ile optimize edilir.

**Etki:** Adaptive sistem — 2018 kur krizi gibi rejim değişimlerinde parametreler otomatik adapt eder.

**Geliştirme süresi:** 2-3 gün (Optuna pipeline + walk-forward CV).

#### 10.3.3 Meta-Learning — Formula × Regime Performance

**Mevcut:** HMM rejimi belirler, mining her rejime top-K şampiyon atar. Atama sezgisel.

**Önerilen:** "Hangi rejimde hangi formül tarihsel olarak iyi çalıştı?" matrisi öğrenilir. Meta-model (LogisticRegression veya GBM) rejim → formül eşlemesi yapar.

**Etki:** +%2-4 CAGR.

**Geliştirme süresi:** 5-7 gün.

#### 10.3.4 RL'i Sisteme Yay (Online Learning)

**Mevcut:** RL sadece kaldıraç (leverage 0.5×-1.0×) için. Formül seçimi statik.

**Önerilen:** RL agent şunları da öğrensin:
- Formül seçimi ağırlığı (top-K içinde dinamik dağılım)
- Stop-loss eşiği (volatilite rejimine göre)
- Sektör tilt (overweight/underweight)

**Etki:** Marjinal ama uzun vadede otomatik adaptasyon.

---

### 10.4 Etki-Maliyet Matrisi

| # | Öneri | CAGR Artışı | Geliştirme | Risk | Akademik kanıt |
|---|---|---|---|---|---|
| 10.1.1 | Long/Short | +5-10% | 3-5 gün | Orta | Fama-French long-short faktörler |
| 10.1.2 | **VIOP Beta Hedge** | Sharpe 2× | **2-3 gün** | **Düşük** | Treynor-Black 1973 |
| 10.1.3 | Multi-frequency | +3-7% | 4-7 gün | Düşük | Moskowitz et al. 2012 |
| 10.2.1 | Kelly Sizing | +3-5% | 1-2 gün | Düşük | Kelly 1956, Thorp 1969 |
| 10.2.2 | ATR Stop-Loss | +1-3% (MDD↓) | 1-2 gün | Düşük | Kaufman "Smarter Trading" |
| 10.2.3 | Sector Rotation | +2-4% | 3-5 gün | Orta | Cohen & Polk 2009 |
| 10.3.1 | HRP | MDD↓%10-20 | 1-2 gün | Düşük | López de Prado 2016 |
| 10.3.2 | Hyperparameter | +1-3% | 2-3 gün | Düşük | Bergstra & Bengio 2012 |
| 10.3.3 | Meta-Learning | +2-4% | 5-7 gün | Orta | Joulin-Lefèvre 2008 |
| 10.3.4 | RL Genişletme | Marjinal | 5-7 gün | Yüksek | Pippas 2025 (RL trading) |

---

### 10.5 Önerilen Marathon Yol Haritası

```
v11 (mevcut)     OHLCV-only baseline                          → CAGR 5-10%, Sharpe 0.6
   ↓
v12  (Bölüm 1-9) Tier 1 veri zenginleştirme (FX, endeks, makro) → CAGR 15-20%, Sharpe 0.9-1.2
   ↓
v13              VIOP Beta Hedge + Kelly Sizing                → Sharpe 2× (CAGR aynı, vol↓)
   ↓
v14              ATR Stop-Loss + HRP + Hyperparameter Tuning   → MDD↓, Sharpe +%20
   ↓
v15              Long/Short (VIOP single-stock futures)        → CAGR 2×
   ↓
v16              Multi-frequency ensemble (haftalık + günlük)  → Sharpe +%30-50
```

**Kritik:** Her adım bir öncekinin ÜZERİNE eklenir. Compound etki bekleniyor.

**Toplam hedef (v16'da):** Net CAGR %30-40, Sharpe 2.5-3.0, MDD <%10.

Bu agresif hedeftir; gerçekçi konservatif tahmin **CAGR %20-25, Sharpe 1.8-2.2** civarında.

---

## 11. Sonuç ve Sonraki Adımlar

**Önerilen sıra:**
1. v11 marathon tamamlanmasını bekle (yarın sabaha kadar)
2. v11 tam P&L analizi (10 yıl)
3. Bu dokümanın Faz A-D'sini uygula (Tier 1 veri ekle)
4. v12 marathon Tier 1 ile çalıştır (~3-4 gün dev + 1 gece marathon)
5. v12 sonucuna göre Bölüm 10'daki strateji iyileştirmelerini sıraya koy

**Yakın vadeli prioritized backlog:**
- v12: Tier 1 data (FX, endeks, makro) — Bölüm 1-9 (5-7 gün dev)
- v13: VIOP Beta Hedge + Kelly Sizing — Bölüm 10.1.2 + 10.2.1 (3-4 gün dev)
- v14: ATR Stop + HRP + Hyperparameter — Bölüm 10.2.2 + 10.3.1 + 10.3.2 (4-5 gün dev)

**Toplam zaman tahmini (v12'ye kadar):** **5-7 gün** geliştirme + 1 gece marathon.
**Toplam zaman tahmini (v14'e kadar):** ~3-4 hafta geliştirme + 3 gece marathon.

**Final hedef:** v11'in net %5-10 CAGR'sini, kademeli olarak **v14'te net %20-25 CAGR, Sharpe 1.8-2.2**'ye çekmek.

---

## 12. Veri Terminali Keşfi — Plan Revizyonu

**KRİTİK GÜNCELLEMEsi:** `/Users/unalronikarakoyun/Desktop/Veri` altında çalışan bir veri terminali (Arkhimedes) sistemi keşfedildi. Bu sistem **Bölüm 1-9'da fetch edilmesi planlanan verilerin neredeyse tamamına** ve **fazlasına** sahip. Plan revize ediliyor: **fetch fazı (A) tamamen iptal**, sadece entegrasyon kalıyor.

### 12.1 Mevcut Varlık Envanteri

| Veri | Dosya (Veri Terminali) | Detay | v12 Plan'da Karşılığı |
|---|---|---|---|
| OHLCV + VWAP | `market_db.parquet` | 2016-2026, 1.13M satır, 8 kolon | Tier 0 (mevcut) |
| **42-kolon fundamental** | `BIST_Tarihsel_Temel_Analiz.parquet` | F/K, PD/DD, FD/FAVÖK, ROE, ROA, Brüt Marj, Net Marj, Cari Oran, Net Borç, FAVÖK, Net Kar TTM, Satış TTM, Özkaynaklar, Toplam Varlıklar, Donen Varlıklar, Kısa V. Yükümlülükler vb. (56k satır) | **Tier 2.1 ✅** |
| **TR Makro** | `EVDS_Verileri_2016_2026.xlsx` | TÜFE Genel, Yİ-ÜFE, Politika Faizi AÖFM (aylık, 126 satır) | **Tier 1.3 ✅** |
| **15 hazır Custody feature** | `custody_features_db.parquet` | Yatırımcı sayısı momentum, kurumsal % trendi, retail capitulation, HHI değişim, breadth değişim, accumulator cost gap, AKD dominance, gross buy (581 hisse, 2022-2026) | **Tier 3 SUPER ✅** |
| **Sector Map** | `sector_map.csv` | 611 hisse, 6 üst-sektör (SINAİ/MALİ/HİZMETLER/TEKNOLOJİ/DİĞER/MENKUL KIYM YO) | **Bölüm 5 ✅** |
| **Delisted Tickers** | `delisted_tickers.txt` | 159 ticker | **Survivorship Bias FİX ✅** |
| **Fiili Dolaşım** | `bist_fiili_dolasim_gunluk_2014_2026.csv` | Günlük 2014-2026, lot bazlı | **Likidite/Float feature ✅** |
| **MKK Demografi** | `mkk_hisse_demografi_2014_2025.csv` | Yıllık yerli/yabancı × fon/tüzel/gerçek × değer_TL/yatırımcı_sayısı | **Tier 3 yabancı flow ✅** |
| **67 hazır teknik feature** | `features_db.parquet` | (Aşağıda detay) | **Tier 1.1, 1.2, 1.4, 2.2 ✅** |
| KAP Açıklamaları | `kap_disclosures.parquet` | 80k disclosure (sadece 2023!) | Tier 3 (kısıtlı) |
| TEFAS Fon Akışı | `arkhimedes.duckdb` içinde | (incelenecek) | Tier 2.2 |
| IPO Veritabanı | `ipo_database.json` | IPO event verisi | Bonus |
| Knowledge Pools | `knowledge_success_pool.parquet` + `_failure_pool.parquet` + `_transitions.parquet` | Önceden bulunmuş başarılı/başarısız setup'lar | **Meta-learning için altın ✅** |

### 12.2 features_db.parquet — 67 Hazır Feature Detayı

Bu tek dosya v12 planımızdaki Tier 1.1-1.4 ve 2.2'nin **TAMAMINI** içeriyor:

**Mevcut OHLCV (7):** `Pclose, Pvwap, Vlot, Phigh, Plow` + Ticker, Date

**Momentum/Volatilite (12):** `mom_30, mom_60, mom_120, mom_252, vol_30, vol_60, vol_120, vol_252, cv_60, cv_120, cv_252, v_roc`

**Teknik pozisyon (7):** `vwap_dist_avg, dist_52w_high, dist_52w_low, price_pos_20d, vwap_band_pos, pivot_dist, vol_mean_revert`

**BIST 100 & Endeks (5):** ← Tier 1.2 ✅
- `xu100_mom_60`, `xu100_mom_120` — endeks momentum
- `xu100_above_ma200` — uzun-vade trend
- `xu100_drawdown` — endeks risk durumu
- `bm_vol_120` — benchmark volatilite

**USD/TRY (3):** ← Tier 1.1 ✅
- `usd_mom_30`, `usd_mom_60` — kur momentum
- `usd_vol_30` — kur volatilitesi

**Sektör Endeksleri (2):** ← Tier 1.2 ✅
- `xbank_mom_60` — bankacılık momentum
- `xusin_mom_60` — sanayi momentum

**Göreceli Güç & Sektör (10):** ← Tier 2.3 (Sector Rotation) ✅
- `rel_vol_120, cv_compression, rel_strength_60, rel_strength_120, mom_divergence`
- `Sector, SuperSector` (kategorik)
- `sector_mom_60, sector_mom_120, sector_vol_120`
- `sector_rel_60, sector_rel_120` — sektör-rölatif performans

**Fiili Dolaşım / Float (4):**
- `float_lot_z_180d, float_lot_yoy_change, float_lot_mom_60, float_change_streak`

**Custody/Yatırımcı (15):** ← Tier 3.1 yabancı/kurumsal akış ✅
- `cust_invcount_mom_60, cust_invcount_accel` — yatırımcı sayısı dinamiği
- `cust_inst_pct_trend_120, cust_inst_pct_accel` — kurumsal yüzde trendi
- `cust_retail_capitulation` — perakende panik metriği
- `cust_concentration, cust_entropy_now, cust_entropy_chg_60` — yoğunlaşma
- `cust_hhi_chg_60, cust_top5_chg_60` — Herfindahl + top-5 değişim
- `cust_breadth_chg_60` — yatırımcı genişliği
- `cust_netcost_trend_60, cust_accumulator_costgap` — akümülatör analizi
- `cust_akd_dominance_20` — AKD dominans
- `cust_gross_buy_20` — gross buy

**Mikroyapı (1):** `ats_z` — average trade size z-score ← Tier 2.2 ✅

### 12.3 Revize Plan — Faz Listesi Güncel

| Faz | Eski Durum | Yeni Durum | Açıklama |
|---|---|---|---|
| A. Veri Çekme | 1-2 gün | **İPTAL** | Veri zaten hazır |
| B. Birleştirme | 1 gün | **0.5 gün** | Sadece path-bağlama + parquet merge |
| C. Sektör Eşleme | 0.5 gün | **İPTAL** | `sector_map.csv` hazır (611 hisse) |
| D. `alpha_cfg.py` | 0.5 gün | **1 gün** | 6 → ~50 feature (MCTS search space genişler) |
| E. Fundamental | 2-3 gün | **0.5 gün** | 42 kolon hazır, sadece PIT discipline + merge |
| F. Yabancı Akış | 2 gün | **0 gün** | Custody features hazır + MKK demografi |
| G. v12 Marathon | 1 gece | 1 gece | Aynı |

**Toplam revize tahmin:** 5-7 gün → **2-3 gün**

### 12.4 Acil Yapılacaklar — 3 Gün İçin Plan

#### Gün 1 — Veri Entegrasyonu
1. `engine/data/external_data.py` yeni modül:
   ```python
   VERI_TERMINALI_PATH = "/Users/unalronikarakoyun/Desktop/Veri/data"

   def load_features_db() -> pd.DataFrame:
       """67 hazır feature'ı yükle."""
       return pd.read_parquet(f"{VERI_TERMINALI_PATH}/features_db.parquet")

   def load_fundamentals() -> pd.DataFrame:
       """42 kolon fundamental, PIT-aware merge."""
       df = pd.read_parquet(f"{VERI_TERMINALI_PATH}/BIST_Tarihsel_Temel_Analiz.parquet")
       # Tarih kolonu HGDG_TARIH veya Tarih; period_end mantığı
       return df

   def load_macro() -> pd.DataFrame:
       """TÜFE, ÜFE, Faiz — aylık, 1 ay PIT gecikme."""
       df = pd.read_excel(f"{VERI_TERMINALI_PATH}/EVDS_Verileri_2016_2026.xlsx")
       # as_of_date = açıklama tarihi (1 ay gecikme uygulanır)
       return df

   def load_sector_map() -> pd.DataFrame:
       return pd.read_csv(f"{VERI_TERMINALI_PATH}/sector_map.csv")

   def load_delisted() -> set:
       """Survivorship bias için delisted ticker'lar."""
       with open(f"{VERI_TERMINALI_PATH}/delisted_tickers.txt") as f:
           return {line.strip().replace(".IS", "") for line in f}
   ```

2. PIT discipline kontrol — `BIST_Tarihsel_Temel_Analiz` içinde `period_end` ile `announce_date` ayrımı var mı? Yoksa konservatif 60 gün ekle.

3. Survivorship bias düzeltmesi — `delisted_tickers` listesini market_db'de tutarak mining'e dahil et.

#### Gün 2 — `alpha_cfg.py` Genişletme

```python
# OHLCV core (mevcut)
FEATURES_OHLCV = ["Popen", "Phigh", "Plow", "Pclose", "Vlot", "Ptyp", "Pvwap"]

# Momentum/Volatilite (features_db'den)
FEATURES_MOM = ["mom_30", "mom_60", "mom_120", "mom_252",
                "vol_30", "vol_60", "vol_120",
                "cv_60", "cv_120", "v_roc"]

# Teknik pozisyon
FEATURES_TECH = ["vwap_dist_avg", "dist_52w_high", "dist_52w_low",
                  "price_pos_20d", "vwap_band_pos", "vol_mean_revert"]

# Endeks & FX (Tier 1)
FEATURES_MARKET = ["xu100_mom_60", "xu100_mom_120", "xu100_drawdown",
                    "usd_mom_30", "usd_mom_60", "usd_vol_30",
                    "xbank_mom_60", "xusin_mom_60"]

# Sektör (Tier 2.3)
FEATURES_SECTOR = ["sector_mom_60", "sector_mom_120", "sector_vol_120",
                    "sector_rel_60", "sector_rel_120", "rel_strength_60"]

# Custody (Tier 3.1 — ÖZEL DEĞER)
FEATURES_CUSTODY = ["cust_invcount_mom_60", "cust_inst_pct_trend_120",
                     "cust_retail_capitulation", "cust_concentration",
                     "cust_entropy_chg_60", "cust_breadth_chg_60",
                     "cust_accumulator_costgap", "cust_akd_dominance_20"]

# Fundamental (Tier 2.1 — PIT-aware!)
FEATURES_FUND = ["F_K", "PD_DD", "FD_FAVOK", "ROE_pct", "ROA_pct",
                  "Brut_Kar_Marji_pct", "Net_Kar_Marji_pct", "Cari_Oran",
                  "Halka_Aciklik_Orani"]

# Float
FEATURES_FLOAT = ["float_lot_z_180d", "float_lot_yoy_change", "float_lot_mom_60"]

FEATURES = (FEATURES_OHLCV + FEATURES_MOM + FEATURES_TECH +
            FEATURES_MARKET + FEATURES_SECTOR +
            FEATURES_CUSTODY + FEATURES_FUND + FEATURES_FLOAT)
# Toplam: 6 → ~55 feature
```

**Önemli:** MCTS search space 55 feature ile **kuadratik patlar**. `n_trials` veya `num_gen`'i azaltmak gerekebilir. Optimal: `n_trials=30`, `num_gen=20` (eski 50, 30'dan düşür).

#### Gün 3 — v12 Marathon Hazırlık + Smoke Test

1. `scripts/run_historical_paper_trade.py` küçük güncelleme:
   - `external_data.load_features_db()` çağrısı (mining öncesi)
   - `enriched_market_db` üretimi → `data/enriched_market_db.parquet` cache

2. Smoke test (3 ay paper trade, küçük örnek):
   ```bash
   venv/bin/python scripts/run_historical_paper_trade.py \
       --trading-start 2018-01-01 --trading-end 2018-03-31 \
       --workers 4 --n-trials 20 --use-enriched-features
   ```

3. v11 vs smoke test karşılaştırması — yeni feature'lar gerçekten katkı sağlıyor mu?

### 12.5 Beklenen Etki Güncellemesi

| Metrik | v11 (6 feature) | v12 (55 feature — VT entegrasyonu) | Açıklama |
|---|---|---|---|
| Unique formül kabul | 26 | 200-500 | Daha geniş feature space |
| Mean RIC (top-1) | 0.012 | **0.025-0.045** | Custody + sektör-rölatif güçlü |
| CAGR | %5-10 | **%20-30** | Çoklu alpha kaynakları compound |
| Sharpe | 0.60 | **1.2-2.0** | Daha iyi risk-ayarlı getiri |
| Max DD | %10 | **%6-9** | Sektör + custody hedge etkisi |

Bu tahminler **eskisinden daha yukarı** çünkü:
- 67 hazır feature, 6 orijinal feature'ın 10×'u
- Custody features kompleks ve alfa açısından zengin (akümülatör cost gap, retail capitulation gibi metrikler akademide kanıtlanmış)
- Fundamental 42 kolon (sadece P/E değil, FD/FAVÖK, ROE momentum, Net Marj trend dahil)

### 12.6 Henüz Kullanılmayacaklar — v13 ve Sonrası İçin

- `kap_disclosures.parquet` — sadece 2023 var, az veri. Önce KAP geçmiş 10 yılı doldurulmalı, sonra v13'te event-driven layer.
- `knowledge_*_pool.parquet` — Arkhimedes'in başarılı setup havuzları. Bunlar meta-learning için altın (Bölüm 10.3.3) — v14'te değerlendirilecek.
- `arkhimedes.duckdb` — TEFAS fon akışları içeriyor, henüz tam araştırılmadı. v13'te dahil edilebilir.
- `mkk_hisse_demografi` — Yıllık frekans, marathon'un günlük cadence'ı için yeterli granular değil. Forward-fill ile feature olabilir ama düşük öncelik.

### 12.7 Risk ve Dikkat Edilecekler

1. **PIT discipline kritik:** `BIST_Tarihsel_Temel_Analiz` "as-reported" mu yoksa "restated" mı? Belirsiz. Konservatif: tarih + 60 gün gecikme uygula.

2. **features_db.parquet 2017-05'te başlıyor:** v12 marathon başlangıcı bu tarihten önce olamaz. Aralık 2016 → Mayıs 2017 dönemi sadece market_db kullanır.

3. **Custody features 2022-2026:** 2016-2021 dönemi için NaN. Mining NaN-aware olmalı, bu featuresleri eski dönemlerde devre dışı bırakmalı.

4. **Schema farkı:** `market_db.parquet` (Veri Terminali) bizim olan ile aynı ama Pvwap kolonu fazladan var. Birleştirme kolay.

5. **Veri yolu hardcoded — taşınabilirlik:** `VERI_TERMINALI_PATH` env var ile yapılmalı. Production deployment'a dikkat.

---
