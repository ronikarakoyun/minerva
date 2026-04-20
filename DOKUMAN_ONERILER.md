# Minerva v3 Studio — Kod İnceleme: Öneriler & Sıkıntılı Parçalar

Bu doküman, mevcut kodun derin analizinden çıkan **mantıksal hatalar, performans sorunları, mimari riskler ve iyileştirme önerileri**ni içerir. Her madde için:
- **Dosya / Konum**
- **Problem**
- **Etki (neden önemli)**
- **Önerilen Çözüm**
- **Aciliyet**: 🔴 Kritik / 🟠 Yüksek / 🟡 Orta / 🟢 Düşük

---

## İÇİNDEKİLER

1. [🔴 KRİTİK — Fitness & Seçim Hataları](#1-kritik--fitness--seçim-hataları)
2. [🔴 KRİTİK — Walk-Forward Fold Tasarımı](#2-kritik--walk-forward-fold-tasarımı)
3. [🟠 YÜKSEK — Faktör Nötralizasyonu Artıkları](#3-yüksek--faktör-nötralizasyonu-artıkları)
4. [🟠 YÜKSEK — Backtest Problemleri](#4-yüksek--backtest-problemleri)
5. [🟠 YÜKSEK — Pvwap "Proxy" Problemi](#5-yüksek--pvwap-proxy-problemi)
6. [🟡 ORTA — MCTS & Tree-LSTM Eksiklikleri](#6-orta--mcts--tree-lstm-eksiklikleri)
7. [🟡 ORTA — LLM Paneli Tutarsızlığı](#7-orta--llm-paneli-tutarsızlığı)
8. [🟡 ORTA — Performans & Paralel İşleme](#8-orta--performans--paralel-işleme)
9. [🟢 DÜŞÜK — UX / Kalite](#9-düşük--ux--kalite)
10. [📐 Mimari Öneriler](#10-mimari-öneriler)
11. [🧪 Eksik Test Kapsamı](#11-eksik-test-kapsamı)

---

## 1. 🔴 KRİTİK — Fitness & Seçim Hataları

### 1.1 `pool[:20]` Bug — Mutasyon/Çaprazlama Tabanı Yanlış

**Dosya**: `app.py` — Faz 2 ve Faz 3 döngüleri

**Problem**:
```python
# Mevcut (yanlış):
base_pool = pool[:20]  # 'pool' SORTLANMAMIŞ → random subset!
for fml in base_pool:
    mut = mutate(fml)
    ...
```
`pool` listesi **fitness'a göre sıralanmadan** ilk 20 eleman alınıyor. Mining aşaması binlerce formül üretiyor; bunlardan "en iyi 20"yi almak gerekirken, **yaratılış sırasına göre** ilk 20 alınıyor. Yani mutasyon/çaprazlama **zayıf tohumlardan** başlıyor.

**Etki**: Evrim algoritmasının çekirdek avantajı (elitizm) yok. Faz 2 → Faz 3 kalite kazancı neredeyse tesadüf.

**Çözüm**:
```python
pool_sorted = sorted(pool, key=lambda x: x["fitness"], reverse=True)
base_pool = pool_sorted[:20]
```

**Aciliyet**: 🔴 — 1-2 satırlık düzeltme, muazzam kalite farkı.

---

### 1.2 Fitness Size-Penalty Asimetrisi

**Dosya**: `engine/wf_fitness.py` — satır 192-197

**Problem**:
```python
size_penalty = 0.0
if not neutralize and not np.isnan(size_corr):
    size_penalty = lambda_size * max(0.0, abs(size_corr) - 0.3)
```
- `neutralize=True` iken size penalty **TAMAMEN kapatılıyor**.
- Ama rank-space OLS artığı ~0.12 size_corr bırakabiliyor (aşağıda #3'te detay).
- Yani nötralize sonrası hâlâ "biraz size-biased" formül cezasız geçiyor.

**Çözüm**: Nötralizasyon açıkken bile artık-size-corr üzerine küçük penaltı:
```python
if np.isnan(size_corr):
    size_penalty = 0.0
elif neutralize:
    size_penalty = 0.2 * lambda_size * max(0.0, abs(size_corr) - 0.15)
else:
    size_penalty = lambda_size * max(0.0, abs(size_corr) - 0.3)
```

**Aciliyet**: 🟠

---

### 1.3 Karmaşıklık Cezası Çok Küçük

**Dosya**: `engine/wf_fitness.py` — default `lambda_cx=0.003`

**Problem**: 50-nodlu devasa formül 50 × 0.003 = 0.15 ceza. Ama fitness tipik olarak 0.01-0.03 aralığında. 0.15 ceza **ağır cezalar** ama 20-25 node aralığında (0.06-0.075) nötr.

Pratikte AST boyutu 15-25 aralığında "cezasız" formüller tercih ediliyor; ama 5-8 node olan basit formüller avantaj kazanmıyor. **Occam's Razor eksik.**

**Çözüm**: İki opsiyon:
1. `lambda_cx` sqrt-scaled: `fitness -= lambda_cx * sqrt(complexity)` (doğrusal değil)
2. Sabit bonus küçük formüle: `if complexity <= 8: fitness += 0.002`

**Aciliyet**: 🟡

---

## 2. 🔴 KRİTİK — Walk-Forward Fold Tasarımı

### 2.1 Kronolojik Blok Fold'lar != Forward-Stability

**Dosya**: `engine/wf_fitness.py` — `make_date_folds`

**Problem**:
```python
def make_date_folds(dates, n_folds=5, min_fold_days=20):
    unique_dates = np.sort(np.unique(pd.to_datetime(dates)))
    edges = np.linspace(0, n, n_folds + 1, dtype=int)
    folds = [unique_dates[edges[i]:edges[i + 1]]]
```
5 **ardışık, non-overlapping** blok. Her blok ≈ farklı piyasa rejimi:
- Fold 1: 2016-2017 (düşük vol boğa)
- Fold 2: 2018 (TL krizi)
- Fold 3: 2019-2020 (COVID)
- Fold 4: 2021-2022 (enflasyon rallisi)
- Fold 5: 2023 (normalleşme)

`pos_folds = 5/5` dediğinde → "her rejimde pozitif IC". Fakat **gerçek forward-test 2024-2026** bunların hiçbiri değil. Rejim değişirse yine overfit görünebilir.

**Etki**: Mining-UI'da "WF-Stabil" çıkan formüller, gerçek **Overfit Testi** (expanding-window) sekmesinde çöküyor. CSV'lerde gördük: 16:01 export'ta WF mean 0.0329 olan formül, 16:04 expanding-WF'de Test RIC ortalamasi 0.0016 (20× düşüş).

**Çözüm** (ikisini birden):

1. **Purged K-Fold + Embargo** (López de Prado AFML Ch.7):
   ```python
   def make_purged_folds(dates, n_folds=5, embargo_days=10):
       # Her fold sınırında label-leakage önle
       # Triple-Barrier TB_horizon kadar embargo
   ```

2. **Rolling walk-forward**: Fold k = train[0:k], val[k:k+1]
   - Gerçek ileriye dönük test
   - Her fold önceki folddan öğrenilmiş gibi

**Aciliyet**: 🔴 — Bu sistemin en temel güvenilirlik sorunu. Bizim CSV'deki "20× çöküş" tam olarak bunun sonucu.

---

### 2.2 Mining-WF ve Overfit-Test Farklı Fold Yapıları Kullanıyor

**Dosya**: `engine/wf_fitness.py` (kronolojik blok) vs `app.py` Overfit Testi (expanding-window)

**Problem**: İki farklı fold anlayışı:
- **Mining-içi WF**: 5 non-overlapping blok → "tüm rejimlerde tutarlı mı?"
- **Overfit Testi**: Expanding window F1..F5 → "ileriye dönük ne kadar stabil?"

Kullanıcı `+5/5 fold` görüp "süper formül" diyor; Overfit Testi'nde "💀 WF-Geçersiz" çıkıyor. **UI'da iki metrik aynı terimi kullanıyor ama anlamı farklı.**

**Çözüm**:
- Mining-içi WF adını değiştir: "Rejim-Stabilitesi" (Regime Stability)
- Overfit Testi: "Forward-Stability"
- UI rozetlerini ayır: "R:5/5" vs "F:3/5"

**Aciliyet**: 🟠 (UX/güven sorunu)

---

## 3. 🟠 YÜKSEK — Faktör Nötralizasyonu Artıkları

### 3.1 Rank-Space OLS Hâlâ 0.10-0.15 Artık Bırakıyor

**Dosya**: `engine/factor_neutralize.py`

**Problem**: Saf `Pclose` formülü için:
- Nötralizasyondan önce: size_corr = 1.000
- Rank-space OLS sonrası: size_corr = **0.122**

Bunun sebebi: rank-space OLS **doğrusal** bir rank ilişkisi kaldırır; ama Pclose ve log(Pclose) rank-aynıdır, bu artık ise **monotonik-olmayan** gürültü. Yani %12-15 hâlâ "size proxy" kalıyor.

**Etki**: `size_corr_hard_limit=0.7` yakalamazsa, 0.15-0.3 aralığındaki "hafif size-biased" formüller geçiyor.

**Çözüm** (3 seçenek, etki sırasına göre):

1. **Tam rank-nötralizasyon** (en sıkı):
   ```python
   # OLS yerine: rank çoklu-regresyon + quantile binning
   def strict_neutralize(sig, size, n_bins=10):
       bins = pd.qcut(size, n_bins, labels=False)
       return sig.groupby(bins).transform(lambda x: x - x.mean())
   ```
   Size'ı 10 bin'e böl, her bin içinde demean. Monotonik-olmayan bias'ı da öldürür.

2. **Hard limit düşür**: `size_corr_hard_limit=0.3` → daha agresif reddetme.

3. **İki-aşamalı neutralize**: önce rank-OLS, sonra bin-demean → artık < 0.05.

**Aciliyet**: 🟠

---

### 3.2 `_build_factors` Vol Hesabında Potansiyel Bug

**Dosya**: `engine/factor_neutralize.py` — `_build_factors` içinde

**Problem** (tipik yazım):
```python
vol = returns.groupby("Ticker").transform(
    lambda s: s.pct_change().rolling(20).std()
)
```
Eğer `returns` zaten `pct_change()` sonucuysa, bu **çift pct_change** yapar — "getiri değişimi" hesaplar, "vol" değil.

**Çözüm**: Kontrol et ve düzelt:
```python
# Close → returns
rets = close.groupby("Ticker").pct_change()
# Rolling std (vol)
vol = rets.groupby(level="Ticker").transform(lambda s: s.rolling(20).std())
```

**Aciliyet**: 🟠 — kod okumada net görmedim, ama eğer varsa vol faktörünün nötralizasyonu bozuktur.

---

### 3.3 Factor Cache Sadece Bir Kez Hesaplanıyor, Değişmiyor

**Dosya**: `app.py` — Faz-0

**Problem**: `_factor_cache` session state'de tutuluyor. Kullanıcı tarih aralığını değiştirirse cache invalidate olmuyor. Eski tarih aralığının faktörleri yeni mining'te kullanılabilir.

**Çözüm**: Cache key'e (start_date, end_date) hash'i ekle:
```python
cache_key = f"{start}_{end}_{len(tickers)}"
if st.session_state.get("_factor_cache_key") != cache_key:
    st.session_state._factor_cache = build_factors_cache(idx)
    st.session_state._factor_cache_key = cache_key
```

**Aciliyet**: 🟡

---

## 4. 🟠 YÜKSEK — Backtest Problemleri

### 4.1 Benchmark Karşılaştırması Yok

**Dosya**: `engine/backtest_engine.py`

**Problem**: Ensemble %260 getiri rapor ediliyor — ama **BIST100 aynı dönemde ne yaptı?** 2023-2026 arası XU100 yaklaşık %400+ nominal (enflasyon rallisi). Yani "%260 getiri" aslında **benchmarka göre KAYIP** olabilir.

**Etki**: Sistem "kazanıyor gibi" görünüyor ama alfa üretmeyebilir. Size-factor + market-beta karıştırılabilir.

**Çözüm**:
1. `data/bist100_index.parquet` ekle (XU100 kapanış fiyatları)
2. Backtest UI'da:
   - Benchmark getirisi
   - Excess return (Alfa)
   - Information Ratio = mean(alpha) / std(alpha)
   - Beta to BIST100
3. Tüm rapor metrikleri **excess** üzerinden.

**Aciliyet**: 🔴 (kritik olması gerek — şu an "kazanıyoruz" deniyor ama kanıt yok)

---

### 4.2 Naive O(n_dates) Loop — Yavaş

**Dosya**: `engine/backtest_engine.py`

**Problem**: Günlük yeniden dengeleme döngüsü saf Python for-loop. 2500 gün × 700 ticker = 1.75M iterasyon.

**Çözüm**:
1. **Vectorize**: Sinyaller → pandas pivot table (Date × Ticker) → tüm günlerin sıralaması tek seferde
2. Pozisyon değişimleri `np.diff` ile
3. Komisyon `numpy` operasyonu

Beklenen hızlanma: **10-30×**.

**Aciliyet**: 🟡 (şu an çalışıyor, ama ensemble >5 formül olunca saatlerce sürüyor)

---

### 4.3 Hardcoded Komisyon/Slipaj

**Dosya**: `engine/backtest_engine.py`

**Problem**: `buy_cost=0.0005, sell_cost=0.0015` kodun içinde sabit. BIST'te:
- Alış komisyonu ≈ %0.04-0.08 (broker'a göre)
- Satış = komisyon + BSMV + damga → %0.12-0.18
- Slipaj (büyük emir) +%0.05-0.10

Kullanıcı senaryolarını test edemiyor.

**Çözüm**: UI slider'ları ekle, `backtest_engine` sinyatürüne parametre geçir.

**Aciliyet**: 🟡

---

### 4.4 Ensemble'da Korelasyon Kontrolü Yok

**Dosya**: `engine/ensemble.py` (varsa, yoksa app.py içinde)

**Problem**: Top-5 formül seçilirken fitness'a göre sıralanıyor. Ama 5 formülün **birbirine korelasyonu** kontrol edilmiyor. 5 formül hepsi "momentum" türevi olabilir → ensemble diversification vaat ettiği kadar yok.

**Çözüm**:
```python
def select_diverse_ensemble(pool, k=5, max_corr=0.7):
    selected = [pool[0]]
    for candidate in pool[1:]:
        corrs = [signal_corr(candidate, s) for s in selected]
        if max(corrs) < max_corr:
            selected.append(candidate)
        if len(selected) == k: break
    return selected
```

**Aciliyet**: 🟡

---

## 5. 🟠 YÜKSEK — Pvwap "Proxy" Problemi

**Dosya**: `data/db_builder.py` (data pipeline) ve formüllerde `Pvwap` kullanımı

**Problem**: Kod tabanında `Pvwap = (High + Low + Close) / 3` (typical price) formülüyle hesaplanıyor. Ama **GERÇEK VWAP = Σ(Price × Volume) / Σ(Volume)** gün-içi bar'lardan.

Biz end-of-day verisi kullandığımız için gerçek VWAP elimizde yok. Ama `Pvwap` adı, mining algoritmasına "bu bir VWAP" fikri veriyor. Formüller `Corr(Pvwap, Popen, 40)` gibi şeyler üretiyor — aslında `Corr(HLC/3, Open, 40)`.

**Etki**:
- Formül adlandırması yanıltıcı
- Akademik literatürdeki VWAP-tabanlı alfa'ları (Qian et al., Alpha101) taklit edemiyoruz
- Anlamsız "VWAP × Volume" kombinasyonları türemesi zor

**Çözüm**:
1. **En temiz**: `Pvwap` adını `Ptyp` (typical price) olarak değiştir.
2. **Gerçek VWAP** için intraday data bulunduğunda upgrade.
3. Operator library'ye VWAP açıklamalı yorum ekle: "Not: end-of-day proxy".

**Aciliyet**: 🟡 — yanıltıcı ama bozuk değil.

---

## 6. 🟡 ORTA — MCTS & Tree-LSTM Eksiklikleri

### 6.1 Tree-LSTM Policy Head Grammar-Masking Yok

**Dosya**: `engine/tree_lstm.py`

**Problem**: Policy çıktısı tüm operatörler üzerinde softmax. Ama grammar kısıtı var — örneğin Unary yerinde Binary operatörü gelemez. Model geçersiz seçim yapabilir, MCTS bu seçimleri reddetmek zorunda → eğitim sinyali gürültülü.

**Çözüm**:
```python
def masked_policy(logits, legal_action_mask):
    logits = logits.masked_fill(~legal_action_mask, -1e9)
    return F.softmax(logits, dim=-1)
```
Legal actions CFG'den gelir — `cfg.legal_children(parent_symbol, position)`.

**Aciliyet**: 🟡 — Tree-LSTM henüz birincil değil, ama açılırsa lazım.

---

### 6.2 MCTS Rollout Heuristic Crude

**Dosya**: `engine/mcts.py`

**Problem**: Terminal node'a ulaşana kadar rastgele genişletme (default rollout). Bu, "yapay zekasız" klasik MCTS. AlphaZero-stili için Tree-LSTM value head terminal olmayan node'da da çağrılmalı.

**Çözüm**:
```python
if not is_terminal(node):
    value = treelstm.value(node)  # rollout YOK
else:
    value = evaluate_formula(node)
```

**Aciliyet**: 🟡

---

### 6.3 Warm-Start Prior Yok

**Dosya**: `engine/mcts.py`

**Problem**: MCTS her aramada sıfırdan başlıyor. Önceki aramada öğrenilmiş formüllerin alt-yapıları (`Corr(X, Y, 40)` gibi) tekrar keşfedilmeli.

**Çözüm**: Prior olarak **subtree frequency table** — katalogdaki tüm iyi formüllerin alt-ağaçlarının sıklığı. MCTS seçimde PUCT bonus:
```python
puct = Q + c_puct * (prior + subtree_freq_bonus) * sqrt(N) / (1 + n)
```

**Aciliyet**: 🟢

---

## 7. 🟡 ORTA — LLM Paneli Tutarsızlığı

**Dosya**: `app.py` — LLM panel kısmı

**Problem**: Kullanıcı LLM'den formül alıyor → direkt mini-backtest'e gidiyor. **WF-Fitness hesaplanmıyor**. Yani:
- Evrim yolundan gelen formül: mean_ric, std_ric, pos_folds, size_corr hepsi var
- LLM yolundan gelen formül: sadece IC + RankIC

Katalog kolonları boş kalıyor, sıralama karışıyor, "aynı kriterle karşılaştırma" olanaksız.

**Çözüm**: LLM → parse → `compute_wf_fitness(...)` → aynı pipeline. Mini-backtest opsiyonel ek adım.

**Aciliyet**: 🟠 — tutarlılık kritik.

---

## 8. 🟡 ORTA — Performans & Paralel İşleme

### 8.1 Faz 3 Seri İterasyon

**Dosya**: `app.py` — Faz 3 evrim döngüsü

**Problem**: 500 adaym formül → tek tek `compute_wf_fitness` çağrısı. Ortalama 0.8s/formül → 500 × 0.8s = 6.6 dakika.

**Çözüm**: `multiprocessing.Pool` veya `joblib.Parallel`:
```python
from joblib import Parallel, delayed
results = Parallel(n_jobs=-1)(
    delayed(compute_wf_fitness)(tree, cfg.evaluate, idx, folds, ...)
    for tree in pool
)
```
Beklenen hızlanma: **4-8×** (CPU çekirdeğine bağlı).

**Aciliyet**: 🟠 — büyük mining çalıştırırken saatlerce bekliyoruz.

---

### 8.2 `evaluate_fn` Her Çağrıda Yeniden Evaluate Ediyor

**Dosya**: `engine/alpha_cfg.py` — tree evaluate

**Problem**: Aynı alt-formül (`Corr(Pvwap, Popen, 40)`) farklı parent formüllerde tekrar tekrar hesaplanıyor. Yok memoization.

**Çözüm**:
```python
@lru_cache(maxsize=10000)
def evaluate_cached(tree_hash, idx_hash):
    ...
```
Veya hash-based subtree cache.

**Aciliyet**: 🟡 — ciddi hız kazancı potansiyeli (x2-x5).

---

### 8.3 Rank-Norm Her Fold'da Yeniden Hesaplanıyor

**Dosya**: `engine/factor_neutralize.py`

**Problem**: Mining'de 500 formül × 5 fold = 2500 neutralize çağrısı. Her birinde signal'i rank-norm ediyoruz. Ama faktörlerin rank-norm'u fold'tan fold'a değişmiyor.

**Çözüm**: Pre-compute factor ranks in `build_factors_cache`:
```python
cache["size_rank"] = _rank_norm_grouped(cache["size"], by="Date")
cache["vol_rank"]  = ...
cache["mom_rank"]  = ...
```

**Aciliyet**: 🟢

---

## 9. 🟢 DÜŞÜK — UX / Kalite

### 9.1 Hardcoded Dosya Yolları

**Dosya**: `app.py`, `engine/backtest_engine.py`

**Problem**: `"data/market_db.parquet"`, `"data/alpha_catalog.json"` kodun içinde sabit. Kullanıcı başka bir veri seti yüklemek isterse kodu değiştirmek zorunda.

**Çözüm**: `config.yaml`:
```yaml
paths:
  market_db: data/market_db.parquet
  catalog: data/alpha_catalog.json
  factor_cache: data/factor_cache.parquet
```
+ `config.py` loader.

**Aciliyet**: 🟢

---

### 9.2 Katalog Şema Migrasyonu Yok

**Dosya**: `engine/catalog.py` (veya app.py içinde save/load)

**Problem**: Katalog JSON; yeni alan eklediğimizde (örn: `size_corr`) eski kayıtlarda yok → `KeyError` ya da `None`.

**Çözüm**:
```python
CURRENT_SCHEMA = 3
def load_catalog():
    data = json.load(...)
    for rec in data["records"]:
        if rec.get("_schema", 1) < CURRENT_SCHEMA:
            rec = migrate(rec)
    return data
```

**Aciliyet**: 🟢

---

### 9.3 Faz 3 UI'da Train/Test IC Degradation Yok

**Dosya**: `app.py` Faz 3 çıktıları

**Problem**: WF mean, WF std, + fold var. Ama **Train IC vs Test IC** gibi net degradation metriği yok. Overfit Testi'ne girmeden önce bu sinyal görünmeli.

**Çözüm**: Her formülde Faz 3 tablosunda ek kolon:
- **IC Drop**: `(train_ic - cv_mean_ic) / train_ic` → %30 üstü alarm rengi.

**Aciliyet**: 🟡

---

### 9.4 Operator Library Dokümante Değil

**Dosya**: `engine/alpha_cfg.py` — operators

**Problem**: `Corr`, `Delta`, `CSRank`, `TSRank`, `Pow`, `Sign`... hepsi var ama **tek bir dokümanda tanımları yok**. Yeni gelen biri `Div(a, b)` güvenli mi (sıfır bölme)? anlayamıyor.

**Çözüm**: `DOKUMAN_SISTEM.md` içinde var; orayı kanonik yap + kodda docstring ekle:
```python
def Div(a, b):
    """Safe division: b=0 or |b|<eps → return 0."""
```

**Aciliyet**: 🟢 (zaten DOKUMAN_SISTEM.md'de yazılı)

---

### 9.5 `_node_complexity` Fragile

**Dosya**: `engine/wf_fitness.py` — satır 23-33

**Problem**:
```python
try:
    return node.size()
except Exception:
    # Fallback: manuel recursive
    ...
```
`Node.size()` bir kere bug'landı (hep 1 dönüyordu). Fallback var ama fallback da `.children` varsayıyor. Eğer `Node` API'si değişirse sessizce yanlış sonuç verir.

**Çözüm**: Unit test:
```python
def test_node_complexity():
    assert _node_complexity(parse("Delta(Pclose, 20)")) == 3
    assert _node_complexity(parse("Pclose")) == 1
```

**Aciliyet**: 🟢

---

### 9.6 db_builder.py "Dormant"

**Dosya**: `data/db_builder.py`

**Problem**: Dosya var ama aktif pipeline'a bağlı değil (app.py doğrudan `market_db.parquet` okuyor). Kullanıcı veriyi nasıl yeniliyor belirsiz.

**Çözüm** (ikisinden biri):
1. db_builder'ı app.py'de bir "🔄 Veritabanını Yenile" butonuna bağla.
2. db_builder'ı sil, doğrudan Jupyter notebook / CLI script yap.

**Aciliyet**: 🟢

---

## 10. 📐 Mimari Öneriler

### 10.1 Katmanlı Yapı

**Mevcut**:
```
app.py (750 satır, her şey)
engine/
  alpha_cfg.py (AST + evaluate)
  wf_fitness.py
  factor_neutralize.py
  triple_barrier.py
  mcts.py
  tree_lstm.py
  backtest_engine.py
```

**Öneri**:
```
minerva/
  core/          # AST, grammar, evaluate
  features/      # factor_neutralize, triple_barrier
  search/        # evolution, mcts
  learning/      # tree_lstm, training
  evaluation/    # wf_fitness, backtest, overfit_test
  persistence/   # catalog, db_builder
  ui/
    sidebar.py
    faz0.py, faz1.py, faz2.py, faz3.py
    llm_panel.py, backtest_panel.py
  config.py
  run.py  # streamlit run minerva/run.py
```
`app.py` 750 satırdan 150 satıra düşer.

**Aciliyet**: 🟡 (büyüme için gerekli)

---

### 10.2 Config Objesi

Tüm sidebar parametreleri bir `@dataclass MiningConfig` içinde toplanmalı → fonksiyon imzaları temizlenir, deney tekrar edilebilirliği için JSON'a serialize edilebilir.

```python
@dataclass
class MiningConfig:
    n_iter: int = 500
    target_col: str = "Next_Ret"
    use_neutralize: bool = True
    lambda_std: float = 2.0
    lambda_cx: float = 0.003
    lambda_size: float = 0.5
    size_corr_limit: float = 0.7
    tb_horizon: int = 10
    tb_upper_mult: float = 2.0
    tb_lower_mult: float = 2.0
    tb_long_only: bool = True
    ...
```

---

### 10.3 Logging

`print()` ve `st.info()` karışık kullanılıyor. Yapılandırılmış logging:
```python
import structlog
log = structlog.get_logger()
log.info("mining_started", n_iter=500, target="TB_Label")
```
Sonuç: JSON log dosyası → analiz kolay.

---

### 10.4 Deney Takibi

Her mining koşusu için artefact:
- `config.json`
- `catalog.json` snapshot
- `top_10.csv`
- `backtest_equity.png`

Klasör: `experiments/2026-04-18_16-04/`

DVC veya MLflow entegrasyonu.

---

## 11. 🧪 Eksik Test Kapsamı

### 11.1 Unit Tests (Şu an: ~0)

**Kritik testler**:

```python
# test_factor_neutralize.py
def test_neutralize_removes_size():
    # Pclose sinyali → size_corr ~ 1.0
    # Neutralize → size_corr < 0.2
    assert neutralized_size_corr < 0.2

def test_rank_norm_idempotent():
    x = np.array([1, 5, 3, 8, 2])
    assert _rank_norm(_rank_norm(x)) == _rank_norm(x)

# test_wf_fitness.py
def test_size_factor_rejection():
    # Saf Pclose formülü → status="size_factor"
    result = compute_wf_fitness(parse("Pclose"), ...)
    assert result["status"] == "size_factor"

def test_complexity_count():
    assert _node_complexity(parse("Delta(Pclose, 20)")) == 3

# test_triple_barrier.py
def test_long_only_mapping():
    # long_only=True → -1 etiketleri 0 olmalı
    labels = compute_triple_barrier_labels(..., long_only=True)
    assert (labels == -1).sum() == 0

# test_folds.py
def test_folds_non_overlapping():
    folds = make_date_folds(dates, n_folds=5)
    for i in range(len(folds) - 1):
        assert max(folds[i]) < min(folds[i+1])
```

### 11.2 Integration Tests

```python
def test_mining_end_to_end():
    # Synthetic data ile Faz 0-3 çalıştır
    # En az 1 formül "ok" status ile bitmeli
```

### 11.3 Regression Tests

Referans formül listesi (golden file):
- Herhangi bir değişiklik sonrası bu formüllerin fitness'ı ≤ %5 sapmalı.

**Aciliyet**: 🟠 — test olmadan her refactor risk.

---

## 📊 ÖZET: ÖNCELİK SIRASI

| # | Sıkıntı | Aciliyet | Tahmin Emek |
|---|---------|----------|-------------|
| 1.1 | `pool[:20]` sorting bug | 🔴 | 5 dk |
| 2.1 | Purged K-Fold + rolling WF | 🔴 | 4-6 saat |
| 4.1 | BIST100 benchmark eklentisi | 🔴 | 2-3 saat |
| 1.2 | Size penalty asimetrisi | 🟠 | 15 dk |
| 2.2 | Fold tanım ayrımı (R/F rozetleri) | 🟠 | 1 saat |
| 3.1 | Rank-nötralizasyon artıkları | 🟠 | 2-4 saat |
| 3.2 | `_build_factors` vol bug kontrolü | 🟠 | 30 dk |
| 7 | LLM paneli WF-fitness entegrasyonu | 🟠 | 1-2 saat |
| 8.1 | Faz 3 paralel (joblib) | 🟠 | 1-2 saat |
| 11 | Unit test iskeleti | 🟠 | 4-6 saat |
| 4.2 | Backtest vectorize | 🟡 | 3-4 saat |
| 6.1 | Tree-LSTM grammar mask | 🟡 | 2-3 saat |
| 9.3 | IC Drop kolonu Faz 3 | 🟡 | 30 dk |
| 10.1 | Katmanlı refactor | 🟡 | 8-12 saat |
| — | (Düşük öncelik kalemler) | 🟢 | Birikimli |

---

## 🎯 İLK 3 GÜNLÜK SPRINT ÖNERİSİ

**Gün 1** (hızlı kazanımlar, 4-5 saat):
- [x] `pool[:20]` → `sorted(pool, key=fitness)[:20]` — 1.1
- [x] Size penalty asimetrisini düzelt — 1.2
- [x] `_build_factors` vol formülünü doğrula — 3.2
- [x] IC Drop kolonu Faz 3 — 9.3
- [x] Fold tanım rozetleri R/F ayrımı — 2.2

**Gün 2** (güvenilirlik, 6-8 saat):
- [ ] Purged K-Fold / rolling walk-forward implementasyonu — 2.1
- [ ] BIST100 benchmark + excess return metriği — 4.1
- [ ] Unit test iskeleti (5-6 kritik test) — 11

**Gün 3** (performans, 4-6 saat):
- [ ] Faz 3 joblib paralel — 8.1
- [ ] LLM → WF-Fitness entegrasyonu — 7
- [ ] Rank-nötralizasyon iki-aşamalı (OLS + bin-demean) — 3.1

Bu sprint sonunda:
- Mining **güvenilir forward-test** yapıyor
- Alfa / beta ayrımı net
- Performans 3-4× artıyor
- Test kapsamı >%30

---

*Son güncelleme: 2026-04-18*
