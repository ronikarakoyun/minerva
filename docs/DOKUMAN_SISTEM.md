# Minerva v3 Studio — Sistem Dokümanı

> BIST için evrimsel alpha keşif sistemi. AlphaCFG (formül grameri) +
> Walk-Forward validasyon + Factor Neutralization + Triple-Barrier labeling +
> Tree-LSTM (öğrenen değerlendirici) + MCTS (rehberli arama).
>
> **Mevcut sürüm:** v3.8 — paralel Faz 3, çeşitlilik filtresi, LLM WF-Fitness,
> merkezi komisyon, `config.yaml`, sidebar DB yenileme, 89 test + gerçek BIST altın dosyası.
>
> Bu doküman üç bölümden oluşur:
>
> 1. **Kod Akışı** — veriyi parke dosyasından backtest eşitlik eğrisine kadar takip eden mimari
> 2. **Kullanım Kılavuzu** — operatör olarak sistemi nasıl sürersin
> 3. **Referans** — tüm sidebar parametreleri, katalog JSON şeması, dosya yapısı, test süiti

---

## BÖLÜM 1 — KOD AKIŞI

### 1.0 Yüksek seviye görünüm

```
┌──────────────────────────────────────────────────────────────────────┐
│  fetch_bist_data.py  →  data/market_db.parquet  (günlük OHLCV + Vlot) │
└──────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│  app.py yüklenir                                                      │
│  ├─ load_data()           → db                                        │
│  ├─ split_date uygulanır  → db_train (mining için), db_test (kilitli) │
│  └─ get_brain()           → cfg, tree_lstm, trainer, replay_buffer    │
└──────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼──────────────────────┐
        ▼                       ▼                      ▼
   [LLM Paneli]          [Evrimsel Döngü]        [Katalog + Backtest]
                                │
                        ┌───────┴────────┐
                        ▼                ▼
                   Faz 0 (hazırlık)  Faz 1/2/3
```

### 1.1 Veri boru hattı (`fetch_bist_data.py` → `data/market_db.parquet`)

- Kaynak: **yfinance** (Yahoo Finance)
- Evren: `db_builder.py` içinde hardkodlu ~700 BIST tickerı (`.IS` uzantısı)
- Artımsal mod: `load_existing()` parke dosyasındaki son tarihi bulur, yalnızca o tarihten bugüne kadar olan veriyi indirir → eski veri korunur
- Pencere: şu anda son 10 yıl (2016-04 → 2026-04)
- Çıktı şeması (long format):

| Date | Ticker | Popen | Phigh | Plow | Pclose | Vlot |
|---|---|---|---|---|---|---|

- **Eksik olan:** `Pvwap` sütunu. `app.py/load_data()` içinde `(Phigh + Plow + Pclose) / 3` yaklaşımıyla türetiliyor (proxy VWAP, gerçek intraday VWAP değil).
- **Veri yolu yapılandırması:** `load_data()` içinden `config.load_paths()` çağrılır. Eğer kök dizinde `config.yaml` varsa `paths:` bölümü okunur, yoksa varsayılan `data/market_db.parquet` kullanılır. Böylece veri konumu koddan değil konfigürasyondan gelir.
- **Sidebar DB yenileme:** `🗄️ Veri Tabanı → 🔄 Veritabanını Yenile (yfinance)` butonu `engine/db_builder.build_database()`'i tetikler, parke dosyasını günceller ve Streamlit cache'ini temizler.

### 1.2 Uygulama başlatma (`app.py` üst kısım)

```python
db = load_data()                 # pd.read_parquet + Date cast + Pvwap proxy
split_ts = pd.to_datetime(...)   # kullanıcı tarih seçer (default %70)
db_train = db[db["Date"]  < split_ts]
db_test  = db[db["Date"] >= split_ts]
```

**Kural:** `db_test`'e sidebar harici hiçbir mining aracı dokunmaz. Sadece iki yerde okunur:
- Backtest kutusunda kullanıcı "TEST" seçtiğinde
- Overfit Testi expander'ında

Bu kural doğrudan kodun yapısında korunuyor — Faz 0 `db_train` üzerinden `idx` üretir.

`get_brain()` Streamlit cache içindedir (`@st.cache_resource`), hot-reload'larda Tree-LSTM ağırlıkları ve replay buffer korunur.

### 1.3 Evrimsel Döngü — 4 Faz

Kullanıcı **⛏️ Evrimsel Döngüyü Başlat** butonuna bastığında aşağıdaki akış çalışır.

#### Faz 0 — Veri hazırlığı (train-only)

```python
db_sorted = db_train.sort_values(["Ticker","Date"])
db_sorted["Pclose_t1"] = groupby("Ticker")["Pclose"].shift(-1)
db_sorted["Pclose_t2"] = groupby("Ticker")["Pclose"].shift(-2)
db_sorted["Next_Ret"]  = Pclose_t2 / Pclose_t1 - 1       # t+1 alım, t+2 satış getirisi
idx = set_index(["Ticker","Date"]).sort_index()           # MultiIndex
```

**Hedef seçimi:**
- `target_mode = "📈 Next_Ret"` → `_target_col = "Next_Ret"`
- `target_mode = "🎯 Triple-Barrier"` → `add_triple_barrier_to_idx()` çağrılır
  - Her hisse için günlük vol hesaplanır, ±k·σ bariyer konur
  - Ufuk (horizon) içinde üst bariyere çarparsa `+1`, alt bariyere çarparsa `-1`, timeout olursa `0`
  - `long_only=True` ise `-1` etiketleri `0` olur → IC binary olur (1=güçlü yükseliş, 0=kaçın/bekle)
  - `_target_col = "TB_Label"`

**Faktör cache (yalnızca WF + nötralize/size_penalty aktifse):**

```python
_factor_cache = build_factors_cache(idx)   # size, vol, mom matrisi
```

Tek sefer hesaplanır, pool'un tamamı boyunca yeniden kullanılır.

#### Faz 1 — Planlama / Başlangıç havuzu (num_gen/2 formül)

İki yolu var:

**(a) Grammar rastgele üretim (`use_mcts=False`, hızlı):**
```python
for _ in range(num_gen // 2):
    pool.append(cfg.generate(max_K))
```

`AlphaCFG.generate()` α-Sem-k uyarlamalı olasılıklı üretici — her production rule'da Δk ceza maliyeti var, uzun formül üretme olasılığı kısa olanlardan düşük.

**(b) MCTS yönlendirmeli (`use_mcts=True`, yavaş ama kaliteli):**
```python
mcts = GrammarMCTS(cfg, max_K=max_K, value_fn=value_fn)
for _ in range(num_gen // 2):
    pool.append(mcts.search(iterations=mcts_iters))
```

`value_fn` Tree-LSTM tahminidir (eğer `use_value_fn=True`). PUCT (policy UCB tree) seçim kuralıyla "AST yapısında gelecek vaadi olan dallar"ı derinleştirir.

#### Faz 2 — Mutasyon & Crossover (ikinci num_gen/2)

```python
for _ in range(num_gen // 2):
    if random() < 0.7:
        p1, p2 = sample(pool[:20], 2)
        pool.append(cfg.crossover(p1, p2))     # iki ağacın altağaçlarını takas
    else:
        pool.append(cfg.mutate(random.choice(pool[:20])))
                                              # operatör/sabit/pencere mutasyonu
```

Top 20'den seçim yapılıyor ama Faz 1 sonunda pool sıralı değil — yani aslında ilk 20 rastgele bir alt küme. Bu teknik bir zayıflıktır (aşağıda Öneriler dokümanında).

#### Faz 3 — Değerlendirme + Seçim + Katalog

Faz 3 iki **alt-fazdan** oluşur (≥ v3.8):

- **Faz A (Hesaplama):** `compute_wf_fitness` tüm pool için çağrılır. Sidebar'daki `⚡ Paralel Faz 3 (joblib)` checkbox'ı açıksa `joblib.Parallel(n_jobs=N, prefer="threads")` ile paralel koşturulur (numpy GIL bıraktığı için thread-tabanlı gerçek hızlanma verir). Joblib yoksa veya hata alırsa otomatik olarak seri moda düşer.
- **Faz B (UI + Katalog):** Sonuçlar `zip(pool, stats)` ile tek tek dolaşılarak seçim filtresi, benzerlik cezası, `save_alpha()` ve Tree-LSTM buffer güncellemesi yapılır. Bu alt-faz sıralı — çünkü `validated` listesi artımsal büyür.

Her formül için iki yol:

**(a) WF-Fitness modu (`use_wf_fitness=True`, önerilen):**

```python
stats = compute_wf_fitness(
    tree, cfg.evaluate, idx, mining_folds,
    lambda_std=0.5, lambda_cx=0.001,
    target_col=_target_col,
    neutralize=use_neutralize,
    factor_cache=_factor_cache,
    size_corr_hard_limit=0.7,
)
```

İç akış (`engine/wf_fitness.py`):
1. `sig = cfg.evaluate(tree, idx)` — formülü tüm (Ticker, Date) çiftlerinde değerlendir, Series döndür
2. `compute_size_corr(sig, ...)` — sinyalin cross-sectional Spearman korelasyonunu `log(Pclose)` size faktörüyle ölç
3. **Hard filter:** `|size_corr| > 0.7` ise status = `"size_factor"`, fitness = -∞ → erken çıkış
4. Nötralize açıksa: `sig = neutralize_signal(sig, idx, factors=factor_cache)` — her tarih için rank-space OLS, artık (residual) sinyal
5. `tmp = DataFrame(Date, Signal, Target).dropna()` — IC tablosunu oluştur
6. **Tüm train IC:** `tmp.groupby("Date").apply(spearman)` → `rank_ic`
7. **Per-fold IC:** K fold için aynı işlem → `fold_rics[]`
8. Fitness formülü:
   ```
   fitness = mean_ric
           - λ_std · std(fold_rics)
           - λ_cx  · |AST|
           - size_penalty     # yalnızca nötralize KAPALIYKEN
   ```

**Seçim filtresi** (app.py Faz 3):
```python
passes = (status == "ok"
          AND mean_ric > 0.003
          AND pos_ratio >= 0.4)
```

Ardından **benzerlik cezası:**
```python
max_sim = max(cfg.similarity(tree, f) for f in validated)
adjusted = fit * (1 - max_sim)
```
`cfg.similarity` ortak altağaç izomorfizmi üzerinden hesaplanıyor (0 = tamamen farklı, 1 = aynı).

`adjusted > 0` olan formül:
- `results` listesine eklenir (UI tablosuna gidecek)
- `validated` listesine eklenir (sonraki formüller bununla benzerliğe bakar)
- `save_alpha()` ile `data/alpha_catalog.json`'a kalıcı kaydedilir
- `buffer.add(tree, rank_ic)` ile Tree-LSTM replay buffer'a eklenir

**(b) Klasik mod (`use_wf_fitness=False`, geriye uyumluluk):**
Tek train üzerinden IC, RankIC hesaplanır. Fold yok, stabilite testi yok. Sadece eski akışla uyum için duruyor.

#### Faz 3 sonu

```python
buffer.save()                                      # data/replay_buffer.pkl
df_res = DataFrame(results).sort_values("Fitness") # UI için
st.session_state.alphas = df_res
```

### 1.4 LLM Paneli (alternatif formül girişi)

Kullanıcı ChatGPT/Claude'dan formül yapıştırır. Akış:

1. `parse_many(text, cfg)` → `[(raw_str, Node|None, err|None), ...]`
2. Her başarılı parse için `cfg.evaluate` + klasik IC/RankIC hesaplanır
3. **Opsiyonel WF-Fitness** (≥ v3.8): Panelde `LLM formüller için WF-Fitness hesapla` checkbox'ı açıksa, her formül için ana mining ayarlarıyla (fold, λ_std, λ_cx, nötralize) walk-forward fitness da hesaplanır; `wf_mean_ric`, `wf_fitness`, `wf_pos_folds`, `wf_fold_rics` kataloğa yazılır. `target_col` her zaman `Next_Ret` (LLM panelinde TB_Label üretilmez).
4. `buffer.add(tree, rank_ic)` + `save_alpha(..., source="llm", **wf_kwargs)`

### 1.5 Tree-LSTM eğitimi (manuel)

Kullanıcı `🏋️ Tree-LSTM eğit` butonuna bastığında:

```python
hist = trainer.train_epochs(buffer, epochs=train_epochs, batch_size=batch_size)
trainer.save()   # data/tree_lstm.pt
```

`TreeLSTMTrainer`:
- `train_step()` — buffer'dan örnek sampler
- Loss = MSE(value_pred, tanh(ic × ic_scale)) + opt. policy CE
- Adam optimizer

Eğitilmiş ağ bir sonraki MCTS aramasında `value_fn` olarak kullanılır — "bu AST'nin tamamlanınca iyi IC vermesi olası mı?" sorusunu cevaplar.

### 1.6 Alpha Kataloğu (`data/alpha_catalog.json`)

Her kayıt:

```json
{
  "formula":       "Mul(-1, CSRank(Div(Delta(Pclose, 5), Delay(Pclose, 5))))",
  "ast":           {...},          // _tree_to_dict(tree) — yeniden yüklenebilir
  "ic":            0.018,
  "rank_ic":       0.024,
  "adj_ic":        0.020,
  "source":        "evolution",    // "llm" | "evolution"
  "discovered_at": "2026-04-18T09:55:00",
  "updated_at":    "2026-04-18T14:57:00",

  "split_date":    "2023-07-01",
  "max_k":         15,
  "population":    400,
  "mcts_iters":    null,
  "complexity":    7,

  "wf": {
    "mean_ric":  0.024,
    "std_ric":   0.008,
    "pos_folds": 5,
    "n_folds":   5,
    "fitness":   0.020,
    "fold_rics": [0.021, 0.018, 0.026, 0.029, 0.027]
  },

  "overfit": {                     // Overfit Testi çalıştırılınca doldurur
    "train_ric":       0.024,
    "test_ric":        0.013,
    "degradation_pct": 45.8,
    "verdict":         "⚠️ Biraz bozuldu"
  },

  "backtests": {                   // her pencere için ayrı kayıt
    "TEST":  {"net_return": 62.1, "ir": 1.42, "mdd": 18.3, "annual": 19.8, "at": "..."},
    "TRAIN": {...},
    "TAM":   {...}
  }
}
```

Sıralama (`load_catalog()`):
- Öncelik: `wf.fitness` (varsa)
- Fallback: `|rank_ic|`

### 1.7 Backtest akışı (`engine/backtest_engine.py`)

Kullanıcı tek formül veya ensemble ile `🚀 Backtesti Koştur`:

```python
run_pro_backtest(df, signal_series, top_k=50, n_drop=5,
                  buy_fee=0.0005, sell_fee=0.0015)
```

1. `Period_Ret = Pclose_{t+2} / Pclose_{t+1} - 1`  (ertesi açılıştan al, sonraki kapanıştan sat)
2. Her tarih için sinyali sırala, top_k'yı hedef portföy yap
3. İlk gün: tüm top_k al
4. Sonraki günler: mevcut portföydeki **en düşük n_drop sinyalli hisseyi** çıkar, yerine **yeni ranked top'tan** n_drop tane ekle
5. Günlük getiri = portföy getirisi − alım maliyeti − satım maliyeti
6. `equity = cumprod(1 + ret_daily) × 100000`

Metrikler:
- `IR = mean(ret_daily) / std(ret_daily) × √252`
- `|MDD| = max((equity / cummax(equity) - 1) × -100)`
- `ARR = (equity[-1] / 100000) ^ (252/n) - 1` (yıllık bileşik)

**Özellik:** Bu backtest **zaten long-only**. Negatif sinyalli hisseler top-K'ya giremediği için otomatik elenir. Kısa pozisyon hiç açılmaz.

**Komisyon akışı (≥ v3.8):** `buy_fee` ve `sell_fee` sidebar'da `💰 Komisyon` başlığı altında ayarlanır (varsayılan alış %0.05, satış %0.15 — damga vergisi dahil). Bu parametreler **hem tekil** alpha backtestine **hem de** ensemble backtestine aynı şekilde aktarılır; eskiden bazı çağrılarda kayboluyordu, artık merkezi `_buy_pct`/`_sell_pct` değişkenlerinden tek noktadan besleniyor.

### 1.8 Overfit Testi (expander)

İki mod:

**📍 Tek Split (hızlı):**
- Sidebar split tarihinden önce = train, sonrası = test
- `train_ric` ve `test_ric` Spearman IC
- Karar:
  - Aynı işaret + `|test_ric|/|train_ric| ≥ 0.8` → ✅ Stabil
  - Aynı işaret + ≥ 0.5 → ⚠️ Biraz bozuldu
  - Aynı işaret + < 0.5 → ❌ Overfit
  - Ters işaret → 💀 İşaret döndü

**🔄 Walk-Forward (daha güvenilir):**
- Train tarihleri %40 → %95 arası genişleyen pencere
- K fold (default 5): her biri için
  - Train[0 : t_k] üzerinde eğit (ama biz öğrenmiyoruz, sadece IC ölçüyoruz — "train_ric")
  - Test[t_k : t_{k+1}] üzerinde IC ölç — "test_ric"
- Ortalama test_ric, sign_stable ratio hesapla
- Karar:
  - `sign_stable ≥ 0.8 AND avg_te > 0.005` → ✅ WF-Stabil
  - `sign_stable ≥ 0.6 AND avg_te > 0` → ⚠️ WF-Kararsız
  - `avg_te ≤ 0 OR sign_stable < 0.4` → 💀 WF-Geçersiz
  - Diğer → ❌ WF-Overfit

Sonuçlar `record["overfit"]`'a kaydedilir.

### 1.9 Ensemble

Birden fazla formül birleşimi:

1. Her formülü parse et + evaluate et
2. **Çeşitlilik filtresi (≥ v3.8):** `ens_max_corr` slider'ı (varsayılan 0.7, aralık 0.3–1.0). Greedy seçim — ilk formül her zaman dahil; sonraki her formül, önceden seçilmiş formüllerle maksimum Spearman korelasyonu bu eşiğin altındaysa eklenir, üstündeyse "benzer" gerekçesiyle düşürülür ve kullanıcıya hangi sinyali kırptığı gösterilir.
3. Kalan sinyalleri **günlük rank percentile**'a dönüştür: `signal.groupby("Date").rank(pct=True)` → [0,1]
4. Ağırlıklar:
   - Eşit: `w_i = 1/n`
   - IR-ağırlıklı: TEST backtest IR'lerini normalize et
   - Manuel: kullanıcı girişi
5. Combined = `Σ w_i × rank_i` → tek Series
6. `run_pro_backtest(combined, buy_fee, sell_fee, ...)` ile equity curve

Ayrıca sinyaller arası **korelasyon matrisi** gösterilir — ortalama korelasyon < 0.5 olmalı (düşük = iyi çeşitlilik). `ens_max_corr` zaten bu metriği baştan sınırlıyor.

---

## BÖLÜM 2 — KULLANIM KILAVUZU

### 2.1 Kurulum & ilk çalıştırma

```bash
cd ~/Minerva_v3_Studio
source venv/bin/activate         # veya uygun venv

# Veri güncel değilse:
python fetch_bist_data.py        # artımsal indirme; son 10 yıl garantili

# UI:
streamlit run app.py
```

Tarayıcıda `http://localhost:8501` açılır.

### 2.2 İlk döngü — "golden path"

Bu sıra **temiz bir başlangıç** içindir (önceki deneylerin zehirlemediği hâl):

1. **Buffer'ı sıfırla** (sidebar `🗑️ Buffer sıfırla`) — eski LLM örnekleri karışmasın
2. **Kataloğu temizle** (Alpha Kataloğu expander → `🧹 Kataloğu temizle`) — eski formüller listeyi kirletmesin
3. **Parametreler:**

| Grup | Parametre | Değer |
|---|---|---|
| 📅 Train/Test | Split | **2023-07-01** |
| ⚙️ Madencilik | Popülasyon | **400** |
| ⚙️ Madencilik | Max K | **12** |
| 🔬 WF-Fitness | WF aktif | ✅ |
| 🔬 WF-Fitness | Fold | **5** |
| 🔬 WF-Fitness | λ_std | **0.5** |
| 🔬 WF-Fitness | λ_cx | **0.001** |
| ⚗️ Nötralizasyon | Aktif | ✅ |
| ⚗️ Nötralizasyon | Size-corr hard limit | **0.7** |
| 🎯 Hedef | Mode | **Triple-Barrier** |
| 🎯 Hedef | Horizon | **10** |
| 🎯 Hedef | Çarpan | **1.5** |
| 🎯 Hedef | Long-only | ✅ |
| 🧠 Tree-LSTM | value_fn | ✅ |
| 🧠 Tree-LSTM | MCTS | ❌ (ilk turda kapalı) |

4. **⛏️ Evrimsel Döngüyü Başlat**
   - Beklenen süre: 5–15 dakika (donanıma bağlı)
   - Sonuç: ~5-30 formül (yeni filtrelerle çoğu elenir, kalan gerçek adaylar)

5. Sonuç tablosunda **SizeCorr < 0.3** olan formüllere odaklan — bunlar gerçek alpha adayları

### 2.3 Overfit testi — adayları doğrula

1. Alpha Kataloğu → Overfit Testi expander aç
2. Mod: **🔄 Walk-Forward**
3. Top N: **20**, Fold: **5**
4. `🔬 Validasyon çalıştır`
5. Tabloda **✅ WF-Stabil** veya **⚠️ WF-Kararsız** olan formüller geçerli adaylardır

**Kritik:** Mining-WF (Faz 3 içi) ve gerçek WF (expanding window, overfit testi) farklıdır:
- Mining-WF 5/5 gösterir ama gerçek WF 2/5 çıkabilir — bu, formülün **tarihsel rejimlerde tutarlı** ama **zaman yönünde genellemediği** anlamına gelir (temporal overfit).

### 2.4 Tree-LSTM eğitimi (2. tura geçmeden önce)

İlk döngüden sonra replay buffer dolmuş olmalı (sidebar'da `{N} örnek`).

1. Epoch: **10**
2. Batch: **32**
3. `🏋️ Tree-LSTM eğit`
4. 📈 Tree-LSTM eğitim geçmişi expander'ında val_loss'un düştüğünü doğrula

### 2.5 İkinci döngü — MCTS açık

Tree-LSTM eğitildikten sonra:

| Parametre | Değer |
|---|---|
| Popülasyon | **600** |
| MCTS | ✅ **Açık** |
| MCTS iter | **300** |

Diğer her şey aynı. Bu sefer Tree-LSTM "gelecek vaadi olan AST yapıları"na MCTS'i yönlendirir, rastgele arama yerine eğitilmiş sezgiyle arama yapılır.

**Beklenti:** 10–30 formül, mean_ric daha yüksek, daha fazla `✅ WF-Stabil` aday.

### 2.6 Backtest & ensemble

1. **💎 En İyi Evrimleşmiş Alphalar** bölümünde tekil formüllere backtest çalıştır:
   - Backtest window: **TEST** (adil değerlendirme)
   - `🚀 Backtesti Koştur`
   - IR, MDD, Yıllık kontrol et

2. İyi 3-5 formülü **🎼 Ensemble** kutusuna yapıştır
3. Ağırlık: **IR-ağırlıklı (TEST)**
4. Backtest window: **TEST**
5. `🎼 Ensemble backtest çalıştır`

**Başarı kriterleri (BIST için):**
- Calmar > 2.0 (yıllık/MDD)
- IR > 1.0
- Korelasyon matrisinde ortalama < 0.5

### 2.7 LLM-güdümlü ek formül ekleme

Eğer evrim yeterli formül üretmediyse, harici LLM (ChatGPT/Claude/Gemini)
kullanabilirsin:

1. **🤖 LLM formüllerini Tree-LSTM'e öğret** expander aç
2. Kendi hipotezlerini veya burada önerdiklerimi yapıştır:
   ```
   Mul(-1, CSRank(Div(Delta(Pclose, 5), Delay(Pclose, 5))))
   CSRank(Div(Sub(Pclose, Plow), Sub(Phigh, Plow)))
   Mul(-1, CSRank(Std(Div(Delta(Pclose, 1), Delay(Pclose, 1)), 20)))
   Corr(Div(Delta(Pclose, 1), Delay(Pclose, 1)), Vlot, 10)
   ```
3. `📥 Parse & Değerlendir`
4. Her biri buffer'a eklenir, IC hesaplanır, kataloğa girer
5. Sonra Tree-LSTM'e tekrar eğit — "bu formüller iyiydi" bilgisi ağda kalır

### 2.8 Yaygın hatalar ve çözümleri

| Hata | Sebep | Çözüm |
|---|---|---|
| Döngü 0 alpha bulur | `size_corr_hard_limit` çok sıkı | 0.7 → 0.85'e çık, tekrar dene |
| Her formül SizeCorr ~1.0 | Nötralizasyon kapalı | ⚗️ checkbox'ı açık mı kontrol et |
| WF ratio yüksek ama Yıllık düşük | Market beta değil, gerçek alfa olmayabilir | BIST100 ile karşılaştır |
| `fillna(method=bfill)` hatası | Pandas 2.0+ uyumsuzluğu | `engine/triple_barrier.py` güncel mi |
| LLM formül `parse: ...` hatası | Yanlış operatör adı | `Pow/Log/CSRank/Delta/Delay` gibi canonical isimleri kullan |
| Buffer küçük + MCTS zayıf | LSTM daha eğitilmemiş | Önce random arama + eğitim, sonra MCTS |

### 2.9 Günlük iş akışı (alışkanlık önerisi)

```
Sabah (5 dk):   fetch_bist_data.py çalıştır → yeni gün verisi
                streamlit run app.py
                Overfit Testi (Walk-Forward) → mevcut kataloğu doğrula

Haftada 1 (30 dk):
                ⛏️ Evrim döngüsü çalıştır (600 pop, MCTS açık)
                Yeni adayları WF-test et
                İyi olanları ensemble'a ekle
                Tree-LSTM'i yeniden eğit

Ayda 1 (2 saat):
                Tüm kataloğu tarar, "📅 Stabil" kalmış formülleri
                başka dönem bölmelerinde tekrar test et
                Ensemble'ı yeniden dengelе (IR-ağırlıklı)
```

---

## BÖLÜM 3 — REFERANS

### 3.1 Sidebar parametre haritası

#### 📅 Train / Test Split
- **Split tarihi**: öncesi TRAIN (mining), sonrası TEST (mining'e kapalı)

#### ⚙️ Madencilik
- **Popülasyon**: 100–1000, evrimsel havuz büyüklüğü
- **Max K**: 10–30, AST derinlik üst sınırı

#### 🔬 WF-Fitness
- **WF aktif**: 5-fold iç stabilite testi. Kapalı = tek-train IC (eski mod)
- **Fold**: 3–8
- **λ_std**: 0–4. Fitness'tan `std(fold_rics)`'in ne kadar cezalandırılacağı. BIST için 0.5
- **λ_complexity**: 0–0.01. `|AST|` cezası. 0.001 önerilir

#### ⚗️ Faktör Nötralizasyonu
- **Nötralize aktif**: Her tarih için size/vol/mom'a karşı rank-space OLS artığı
- **λ_size**: sadece nötralize kapalıyken. Ham size_corr > 0.3 → ceza
- **Size-corr hard limit**: `|raw_size_corr|` bu eşiği aşarsa formül direkt reddedilir

#### 🎯 Hedef Değişkeni
- **Next_Ret**: klasik, gürültülü
- **Triple-Barrier**: temiz sinyal
  - **Horizon**: 5–30 gün, bariyerlerin geçerli olduğu pencere
  - **Çarpan**: 0.5–3.0 σ, bariyer uzaklığı
  - **Long-only**: ✅ → BIST gerçeği, IC binary (+1/0)

#### 💰 Komisyon (≥ v3.8)
- **Alış (%)**: 0.00–0.50, varsayılan 0.05 (BBP alış)
- **Satış (%)**: 0.00–0.50, varsayılan 0.15 (alış + damga vergisi)
- Her iki backtest çağrısına (tekil + ensemble) aktarılır

#### ⚡ Performans (≥ v3.8)
- **Paralel Faz 3 (joblib)**: Açıkken `Parallel(prefer="threads")` ile Faz A paralel çalışır. 300+ formülde ~3-5× hızlanma. Hata olursa sessizce seri moda düşer.

#### 🗄️ Veri Tabanı (≥ v3.8)
- **🔄 Veritabanını Yenile**: `db_builder.build_database()` → yfinance artımsal fetch → parke güncellenir, cache temizlenir.

#### 🧠 Tree-LSTM
- **Cihaz**: cpu/cuda
- **value_fn aktif**: MCTS rollout'larında LSTM kullanılır
- **MCTS aktif**: Faz 1'de rastgele yerine yönlendirmeli arama
- **MCTS iter**: 50–1000, her formül için simülasyon sayısı
- **Eğitim Epoch / Batch**: 1–100 / 4–256

#### 🎼 Ensemble (≥ v3.8)
- **Çeşitlilik eşiği (`ens_max_corr`)**: 0.3–1.0, varsayılan 0.7. Formüller arası max Spearman korelasyonu bu değerin üstündeyse redundant sinyal kırpılır.

#### 🤖 LLM Paneli (≥ v3.8)
- **LLM formüller için WF-Fitness hesapla**: Checkbox. Açıkken her LLM formülü için ana mining ayarlarıyla walk-forward fitness da hesaplanır, kataloğa yazılır.

### 3.2 Dosya yapısı

```
Minerva_v3_Studio/
├─ app.py                        # Streamlit entrypoint
├─ config.py                     # load_paths() + MiningConfig dataclass (≥ v3.8)
├─ config.yaml                   # (opsiyonel) özelleştirilmiş yollar & ayarlar
├─ fetch_bist_data.py            # yfinance veri indirici
├─ roni.py                       # deneysel LLM formül script'i (opsiyonel)
├─ data/
│  ├─ market_db.parquet          # long-format OHLCV + Vlot
│  ├─ alpha_catalog.json         # kalıcı alpha kataloğu
│  ├─ replay_buffer.pkl          # Tree-LSTM eğitim örnekleri
│  └─ tree_lstm.pt               # eğitilmiş ağ ağırlıkları
├─ engine/
│  ├─ alpha_cfg.py               # Grammar + Node + evaluate/generate/mutate/crossover
│  ├─ formula_parser.py          # LLM formül → AST
│  ├─ wf_fitness.py              # Walk-Forward fitness + folds + verdict
│  ├─ factor_neutralize.py       # Size/Vol/Mom cross-sectional OLS
│  ├─ triple_barrier.py          # TB label + long_only mod
│  ├─ alpha_catalog.py           # JSON persistence
│  ├─ backtest_engine.py         # TopK-Dropout long-only portfolio
│  ├─ mcts.py                    # Grammar-aware MCTS
│  ├─ tree_lstm.py               # Child-sum Tree-LSTM + heads
│  ├─ trainer.py                 # MSE/CE training loop
│  ├─ replay_buffer.py           # Experience storage + serialization
│  └─ db_builder.py              # yfinance ile BIST evren indirici
├─ tests/                        # (≥ v3.8) 89 birim/entegrasyon/regresyon testi
│  ├─ __init__.py
│  ├─ conftest.py                # make_synthetic_db/idx fixture + pytest hook
│  ├─ test_wf_fitness.py         # _node_complexity, make_date_folds, compute_wf_fitness
│  ├─ test_factor_neutralize.py  # _rank_norm, bin_demean, neutralize_signal
│  ├─ test_triple_barrier.py     # long_only mapping, label_stats
│  ├─ test_integration.py        # uçtan uca mining + nötralize + TB pipeline
│  ├─ test_regression.py         # sentetik + gerçek BIST altın dosya karşılaştırma
│  └─ data/
│     ├─ bist_snapshot.parquet   # 30 likit ticker × 1251 gün — dondurulmuş
│     ├─ golden.json             # sentetik referans fitness değerleri
│     └─ golden_real.json        # gerçek BIST referans fitness değerleri
├─ 2601.22119v1.pdf              # referans makale
├─ 2602.07085v1.pdf              # referans makale
├─ Alpha Mining.md
└─ Minerva_v3_Studio.md
```

### 3.3 Operatör kütüphanesi (`alpha_cfg.AlphaCFG`)

| Kategori | Operatörler |
|---|---|
| Features | `Popen, Phigh, Plow, Pclose, Pvwap, Vlot` |
| Unary | `Abs, Log, Sign` (tek arg) |
| Binary | `Add, Sub, Mul, Div` (2 arg) |
| Binary-Asym | `Greater, Less, Pow` (sıra önemli) |
| Rolling | `Mean, Std, Max, Min, Var, Delay, Delta, Sum, WMA, EMA` (series, window) |
| Paired | `Corr, Cov` (series1, series2, window) |
| Cross-section | `CSRank` (series) — günlük percentile rank |
| Sabitler | `CONSTANTS = {-0.1, -0.05, -0.01, 0.01, 0.05, 0.1}` |
| Pencereler | `NUMS = {10, 20, 30, 40}` |

### 3.4 Verdict kodları

**Mining-WF (Faz 3 içi):**
- `✅ WF-Stabil`: pos_ratio ≥ 0.8 AND mean_ric ≥ 0.005
- `⚠️ Kararsız`: pos_ratio ≥ 0.6 AND mean_ric > 0
- `❌ Overfit`: diğer
- `💀 Geçersiz`: mean_ric ≤ 0 OR pos_ratio < 0.4
- `🏷️ Size-Faktör`: |size_corr| > hard_limit (yeni)

**Overfit Testi (expander):**
- Mode tek-split: `✅ Stabil / ⚠️ Biraz bozuldu / ❌ Overfit / 💀 İşaret döndü`
- Mode WF: `✅ WF-Stabil / ⚠️ WF-Kararsız / ❌ WF-Overfit / 💀 WF-Geçersiz`

### 3.5 Kısa kod arama rehberi

| Ne arıyorsan | Dosya | Fonksiyon |
|---|---|---|
| Formül nasıl değerlendiriliyor | `alpha_cfg.py` | `AlphaCFG.evaluate()` |
| Yeni operatör eklemek | `alpha_cfg.py` | `UNARY_OPS` / `BINARY_OPS` sözlükleri |
| IC hesabı | `wf_fitness.py` | `_ic_on_group` |
| Nötralizasyon matematiği | `factor_neutralize.py` | `neutralize_signal` → rank-space OLS |
| Triple-Barrier etiketleme | `triple_barrier.py` | `compute_triple_barrier_labels` |
| TopK-Dropout portföy mantığı | `backtest_engine.py` | `run_pro_backtest` ana döngü |
| Katalog JSON şeması | `alpha_catalog.py` | `save_alpha` parametreleri |
| MCTS PUCT | `mcts.py` | `GrammarMCTS._puct` |
| Tree-LSTM hücresi | `tree_lstm.py` | `ChildSumTreeLSTMCell.forward` |

### 3.6 Test süiti (≥ v3.8)

89 test — tamamı `python -m pytest tests/ -v` ile yeşil geçer.

| Dosya | Test Sayısı | Odak |
|---|---|---|
| `test_wf_fitness.py` | 24 | `_node_complexity`, `make_date_folds`, `compute_wf_fitness` — fold çakışmazlığı, size_factor reddi, λ_std etkisi |
| `test_factor_neutralize.py` | 27 | `_rank_norm` idempotens, bin-demean sıfır ortalama, `neutralize_signal` size_corr düşürmeli |
| `test_triple_barrier.py` | 16 | long_only → asla -1, horizon lookahead yok, label_stats toplamı 1.0 |
| `test_integration.py` | 11 | Uçtan uca: mining + nötralize + TB_Label pipeline çökmemeli |
| `test_regression.py` | 11 | Altın dosya karşılaştırma — sentetik + gerçek BIST snapshot |

**Altın dosya mekanizması:**
- `tests/data/golden.json` — sentetik veri referansı (her ortamda deterministik)
- `tests/data/golden_real.json` — gerçek BIST (30 likit ticker × 2021-2026) referansı
- `tests/data/bist_snapshot.parquet` — dondurulmuş veri dilimi, test boyunca sabit
- Engine'de hesaplama değişirse yenileme: `pytest tests/test_regression.py --update-golden -q`
- Tolerans: fitness/mean_ric ±%5 sapma kabul edilir; bunun üstü test başarısızlığı

**Sentetik fixture'lar (`conftest.py`):**
- `make_synthetic_db(n_tickers=30, n_days=400, seed=42)` — GBM fiyat süreci, `linspace(5,300)` başlangıç fiyatları (cross-sectional size varyasyonu için)
- Session-scoped `syn_db` / `syn_idx` / `cfg` fixture'ları tüm test modüllerince paylaşılır

### 3.7 Sürüm notları

- **v3.0** — AlphaCFG + Tree-LSTM + MCTS (temel sistem)
- **v3.1** — Alpha kataloğu + Tree-LSTM value fn
- **v3.2** — Walk-Forward validasyon (Overfit Testi)
- **v3.3** — Train/Test split zorlaması
- **v3.4** — 10 yıl veri + artımsal fetch
- **v3.5** — WF-Weighted Fitness + Triple-Barrier labels
- **v3.6** — Factor Neutralization (OLS value-space, ilk versiyon)
- **v3.7** — Rank-space OLS + Size-corr hard filter + Long-only mod
- **v3.8** — Paralel Faz 3 (joblib) + Çeşitlilik filtresi (`ens_max_corr`) +
  LLM WF-Fitness + Komisyon parametreleri her backtest'e +
  `config.py` / `config.yaml` + Sidebar DB yenileme butonu +
  89 birim/entegrasyon/regresyon testi + sentetik & gerçek BIST altın dosyaları (mevcut)

---

*Bu doküman sistem değiştiğinde birlikte güncellenmelidir.*
