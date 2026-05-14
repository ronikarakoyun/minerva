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

## 10. Sonuç ve Sonraki Adımlar

**Önerilen sıra:**
1. v11 marathon tamamlanmasını bekle (yarın sabaha kadar)
2. v11 tam P&L analizi (10 yıl)
3. Bu dokümanın Faz A-D'sini uygula (Tier 1 ekle)
4. v12 marathon Tier 1 ile çalıştır (~3-4 gün dev + 1 gece marathon)
5. v12 sonucuna göre Tier 2 (fundamental) veya Tier 3 (sentiment/flow) eklenir

**Toplam zaman tahmini:** Tier 1 entegrasyonu **5-7 gün** geliştirme + 1 gece marathon. Hedef: v11'in net %5-10 CAGR'sini v12'de net %15-20'ye çekmek.
