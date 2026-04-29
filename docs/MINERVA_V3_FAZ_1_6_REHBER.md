# Minerva v3 — Faz 1-6 Sistem Rehberi

> Otonom BIST alpha-fonu mimarisinin **rejim tespiti → ağırlıklı fitness →
> MCTS+Optuna mining → risk yönetimi → execution layer → kurumsal orkestrasyon**
> uçtan uca rehberi.
>
> Bu dosya: **(1)** her fazın ne yaptığını, **(2)** hangi formülleri kullandığını,
> **(3)** kodun nerede yaşadığını, **(4) sıfırdan nasıl çalıştırılacağını**
> (Sıfır Buffer Kurulum Kılavuzu — bkz. §11) anlatır.

## Kuş Bakışı

| Faz | Kapsam | Anahtar Modül | Yeni Test | Durum |
|---|---|---|---|---|
| 1 | HMM K=6 piyasa rejimi (Forward-only) | `engine/data/regime_detector.py` | tests/test_regime_detector.py | ✓ |
| 2 | Rejim-koşullu ağırlıklı fitness | `engine/validation/weighted_fitness.py` | tests/test_weighted_fitness.py | ✓ |
| 3 | MCTS aktivasyon + Optuna meta-opt | `engine/strategies/{mcts,meta_optimizer}.py` | tests/test_{mcts_integration,meta_optimizer}.py | ✓ |
| 4 | Vol-target + Page-Hinkley + ADV | `engine/risk/*` | 11 test | ✓ |
| 5 | Almgren-Chriss + harmanlama + paper trade | `engine/execution/*` | 12 test | ✓ |
| 6 | Prefect orkestrasyon + adli loglama | `auto_minerva.py`, `engine/execution/forensics.py` | 6 test | ✓ |

**Test toplamı (tüm sistem):** `venv/bin/pytest tests/ -q` → **235/235 passed** (~110 sn).

**Beş prensip (tüm fazlarda korunan):**

1. **Opt-in kalıbı.** Yeni özellikler `*Config(use_xxx=False)` ile default kapalı; Faz 0'daki davranış birebir korunur.
2. **Look-ahead-safe.** Tüm rolling pencereler `shift(1)` ile bir gün geriye kaydırılır.
3. **Modüler bağımsızlık.** Her fazın yan modülü silinebilir/yedek kalabilir; içine bağlı modüller kontrollü import yapar.
4. **Cross-sectional + zaman ağırlığı.** Fitness her zaman cross-sectional RankIC üzerinden, ağırlık ise zaman boyutunda.
5. **Persist edilebilirlik.** Her artifact (`prob_df`, `best_params.json`, `alpha_catalog.json`, `paper_trades.parquet`) crash-resume edebilir.

---

## İçindekiler

1. [Sistem Özeti](#sistem-özeti)
2. [Faz 1 — HMM Rejim Tespiti](#faz-1--hmm-rejim-tespiti)
3. [Faz 2 — Rejim-Koşullu Ağırlıklı Fitness](#faz-2--rejim-koşullu-ağırlıklı-fitness)
4. [Faz 3 — MCTS + Optuna Meta-Optimizer](#faz-3--mcts--optuna-meta-optimizer)
5. [Faz 4 — Risk Yönetimi](#faz-4--risk-yönetimi)
6. [Faz 5 — Execution Layer](#faz-5--execution-layer)
7. [Faz 6 — Kurumsal Orkestrasyon + Adli Loglama](#faz-6--kurumsal-orkestrasyon--adli-loglama)
8. [Uçtan Uca Pipeline](#uçtan-uca-pipeline)
9. [Kullanım Kılavuzu](#kullanım-kılavuzu)
10. [Konfigürasyon Referansı](#konfigürasyon-referansı)
11. **[Sıfır Buffer Kurulum Kılavuzu (Adım Adım)](#sıfır-buffer-kurulum-kılavuzu-adım-adım)** ← BAŞLA BURADAN
12. [Test ve Doğrulama](#test-ve-doğrulama)
13. [Sorun Giderme (Troubleshooting)](#sorun-giderme-troubleshooting)

---

## Sistem Özeti

Minerva v3 altı katmanlı bir otonom alpha keşif, doğrulama ve operasyon sistemidir:

```
┌──────────────────────────────────────────────────────────────────────┐
│  Faz 1: HMM Rejim Tespiti     → prob_df (Date × K=6 olasılık)        │
│         ↓                                                              │
│  Faz 2: Ağırlıklı Fitness     → bugüne benzer geçmiş günlere odaklan │
│         ↓                                                              │
│  Faz 3: MCTS + Optuna         → akıllı formül arama + meta-tuning    │
│         ↓                                                              │
│  Faz 4: Risk Yönetimi         → vol-target + decay + capacity        │
│         ↓                                                              │
│  Faz 5: Execution Layer       → slipaj + harmanlama + paper trade   │
│         ↓                                                              │
│  Faz 6: Orkestrasyon + Adli   → Prefect flow + decisions_log         │
└──────────────────────────────────────────────────────────────────────┘
```

**Veri akışı:** `data/market_db.parquet` (BIST hisseleri, 2016–bugün) → tüm fazlar bunu okur.

**Sonuç ürünleri:**
- `data/regime_segments.csv` — 241 kronolojik rejim segmenti
- `data/regime_prob_df.parquet` — günlük rejim olasılık matrisi (Faz 6)
- `data/best_params.json` — Optuna en iyi MCTS hiperparametreleri
- `data/alpha_catalog.json` — keşfedilen formüller + rejim şampiyonları
- `data/paper_trades.parquet` — paper trading karar logları
- `data/decisions_log.parquet` — adli karar fotoğrafları (Faz 6)
- `data/jobs.db` — API uzun-süreli iş kaydı (SQLite, kalıcı)

---

## Faz 1 — HMM Rejim Tespiti

**Amaç:** BIST100'ün anlık piyasa rejimini soft probability vector olarak çıkarmak.

### Ne yapar?

Gaussian HMM (K=6) kullanarak günlük piyasa özelliklerinden 6 saklı rejim öğrenir:

| Rejim | Ad | Gün Sayısı | Özellik |
|---|---|---|---|
| 0 | Normalize Bull | 516 | Düşük vol + pozitif drift (**bugün burada**) |
| 1 | Kriz / Panik | 227 | Yüksek vol + negatif drift |
| 2 | Sakin Güçlü Bull | 457 | Çok düşük vol + güçlü ralli |
| 3 | Yatay / Belirsiz | 454 | Orta vol + sıfır drift |
| 4 | Hızlı Momentum | 481 | Orta vol + güçlü momentum |
| 5 | Sessiz Konsolidasyon | 343 | Çok düşük vol + sıfır drift |

### Nasıl yapar?

1. **Özellikler** (`compute_features`):
   - Log-getiri, normalize ATR(14), volatility ratio, momentum
2. **HMM Eğitimi** (`fit_hmm`):
   - K=2..8 arası BIC optimizasyonu (BIC=17463.4 @ K=6)
3. **Olasılık Çıkarımı** (`compute_probability_vector`):
   - Forward-backward ile her gün için K-boyutlu olasılık vektörü

### Kod Konumu

- `engine/data/regime_detector.py` — ana modül
- `engine/data/regime.py` — eski kural tabanlı (geriye dönük)

### Kullanım

```python
from engine.data.regime_detector import run_pipeline

prob_df = run_pipeline()                    # data/market_db.parquet'i okur
print(prob_df.tail())
# regime_0  regime_1  ...  regime_5
# 2026-04-25  0.895    0.105    ...  0.000
# 2026-04-26  0.901    0.099    ...  0.000
```

---

## Faz 2 — Rejim-Koşullu Ağırlıklı Fitness

**Amaç:** MCTS, formülleri değerlendirirken "bugüne benzer geçmiş günlere" odaklansın; alakasız ralli dönemleri gürültüsünü baskılasın.

### Ne yapar?

Her tarihin cross-sectional RankIC katkısını, "bugünün rejim vektörüne benzerlik" oranında ağırlıklandırır. Stitching yok — tarih sırası bozulmadan ağırlık uygulanır.

### Matematik

```
sim_t   = cosine(p_t, p_ref)                              ∈ [0, 1]
w_t     = w_min + (w_max - w_min) · exp(T·sim_t) / exp(T)  # üstel
mean_ric_w = Σ_t (w_t · ric_t) / Σ_t w_t                  # ağırlıklı RankIC
```

- `p_ref`: prob_df son günü (bugün)
- `T=2.0`: keskinlik (yüksek → benzer günler dramatik ağırlık alır)
- `w_min=1, w_max=10`

### Sonuç (gerçek BIST)

- Son gün: w = 10.0 (tam benzer)
- COVID dibi (2020-03): w = 2.49 (zayıf benzer)
- Standart sapma: 3.04 (geniş ayrım)

### Kod Konumu

- `engine/validation/weighted_fitness.py` — `WeightConfig`, `compute_regime_weights`, `compute_weighted_wf_fitness`

### Kullanım

```python
from engine.data.regime_detector import run_pipeline
from engine.validation.weighted_fitness import compute_regime_weights, WeightConfig

prob_df = run_pipeline()
weights = compute_regime_weights(prob_df, WeightConfig(temperature=2.0))
print(weights.describe())
# count    2497
# min      1.00
# max     10.00
# std      3.04
```

---

## Faz 3 — MCTS + Optuna Meta-Optimizer

**Amaç:** İç döngüde gerçek MCTS (PUCT/rollout/backprop) ile rehberli formül arama; dış döngüde Optuna ile haftalık Bayesian hiperparametre tuning.

### İki bileşen

#### A) GrammarMCTS Aktivasyonu

Mevcut sistem aslında genetic programming yapıyordu (rastgele crossover/mutate). `engine/strategies/mcts.py` içinde **hazır ama bağlanmamış** `GrammarMCTS` (PUCT formülüyle) duruyordu — Faz 3'te bağlandı.

**MCTS rotası:** `mining_runner.py` → `GrammarMCTS.search(iterations=50)` → her formül için PUCT-rehberli AST araması:

```
PUCT(node) = Q(node) + c_puct · prior(node) · sqrt(N_parent) / (1 + N_node)
```

#### B) Optuna Meta-Optimizer

Haftalık Bayesian arama, **6 kritik parametreyi** optimize eder:

| Parametre | Aralık | Anlam |
|---|---|---|
| `max_K` | 8–20 | AST max derinliği |
| `c_puct` | 0.5–3.0 | MCTS exploration |
| `mcts_rollouts` | 8–32 | Rollout sayısı |
| `lambda_std` | 0.5–4.0 | Fitness'ta std cezası |
| `lambda_cx` | 0.001–0.01 | Karmaşıklık cezası |
| `temperature` | 0.5–5.0 | Faz 2 ağırlık keskinliği |

### Kod Konumu

- `engine/strategies/mining_runner.py` — `MiningConfig` + MCTS branch
- `engine/strategies/meta_optimizer.py` — Optuna CLI
- `engine/strategies/mcts.py` — `GrammarMCTS` (Faz 3'te bağlandı)

### Kullanım

```bash
# CLI: 50 trial Optuna araması
python -m engine.strategies.meta_optimizer --n-trials 50
# Çıktı: data/best_params.json
```

```python
# Round-trip
from engine.strategies.mining_runner import MiningConfig, run_mining_window
mcfg = MiningConfig.from_best_params("data/best_params.json", num_gen=80)
results = run_mining_window(db_window, alpha_cfg, mcfg)
```

### Sonuç (gerçek BIST 3-trial smoke)

- `best_value = 0.0082` (top-10 mean_ric medyanı)
- `best_params`: max_K=14, c_puct=2.1, mcts_rollouts=16, ...

---

## Faz 4 — Risk Yönetimi

**Amaç:** Üç bağımsız risk modülü ile portföyü "bilinçli" hale getir. Hiçbiri default açık değil — opt-in flag ile kontrol edilir.

### 4.1 Volatility Targeting (`position_sizer.py`)

**Sorun:** Backtest_engine eşit ağırlık (1/N) kullanıyor — kriz günü %5 vol ile sakin günü %1 vol aynı boyutta. Risk patlamaları kaçınılmaz.

**Çözüm:** Her ticker için 20-gün rolling realized vol → hedef yıllık fon vol'üne göre pozisyon ölçeklendirmesi:

```
asset_vol_t = std(daily_ret_{t-20..t-1}) · √252
scale_t     = clip(target_vol / asset_vol_t, min_scale=0.1, max_scale=3.0)
position_t  = scale_t · base_position
```

**Look-ahead koruması:** `scale.shift(1)` zorunlu (t-günündeki vol t-1 sonuna kadar olan veriyle hesaplanır).

**BIST sonuç (gerçek smoke):** MDD %26.8 → %13.6 (yarıya iniş).

### 4.2 Alpha Decay Monitor (`decay_monitor.py`)

**Sorun:** Bir formül canlıya verildiğinde performansı düşmeye başladığında erken uyarı yok.

**Çözüm:** Page-Hinkley change-point detection (akademik standart):

```
m_t = max(0, m_{t-1} + (μ_backtest - r_t - δ))
trigger if m_t > λ AND |r_t - μ| > sigma_floor·σ AND consecutive_alarms ≥ N
```

- `δ=5e-4`: minimum drift (gürültü payı)
- `λ=0.01`: kümülatif sapma alarm eşiği
- `sigma_floor=2.0`: σ pre-filter (false-positive azaltır)
- `consecutive_days=5`: 5 gün ardarda kümülatif sapma

**Tetiklendiğinde:** Kill-switch — formül emekli edilir, paper trade durur.

### 4.3 Capacity Estimation (`capacity.py`)

**Sorun:** 10M TL fonda %5 ADV limitini aşan sinyaller mining'den geçiyor → kapasitesi tükenmiş formüller.

**Çözüm:** ADV bazlı maksimum AUM hesabı:

```
ADV_TL_t        = mean(Vlot · Pclose)_{t-20..t-1}
max_position_TL = adv_pct · ADV_TL_t                    # adv_pct=0.05
formula_capacity = min over tickers (max_position_TL · portfolio_size)
```

**Look-ahead koruması:** `compute_adv` içinde `shift(1)` uygulanır.

**BIST sonuç:** ZRGYO Nisan 2026 ADV ~28-30M TL → max pozisyon 1.4-1.5M TL.

### Backtest Entegrasyonu

`run_pro_backtest`'e opsiyonel `risk_cfg` eklendi. None → klasik 1/N. `use_vol_target=True` → her ticker'a scale uygulanır.

```python
from engine.risk.position_sizer import RiskConfig
from engine.core.backtest_engine import run_pro_backtest

_, m1 = run_pro_backtest(db, signal)                                          # klasik
_, m2 = run_pro_backtest(db, signal, risk_cfg=RiskConfig(use_vol_target=True)) # vol-target
```

### Kod Konumu

- `engine/risk/__init__.py`
- `engine/risk/position_sizer.py`
- `engine/risk/decay_monitor.py`
- `engine/risk/capacity.py`

---

## Faz 5 — Execution Layer

**Amaç:** Üç gerçekçilik açığını kapat: dinamik slipaj, rejim harmanı, paper trading.

### 5.1 Almgren-Chriss Slipaj (`slippage.py`)

**Sorun:** Sabit %0.05/%0.15 fee sığ tahta hisselerde "hayalet kâr" yaratıyor — 5M TL'lik THYAO emri fiyatı pek hareket ettirmez ama 5M TL'lik sığ yan tahta alımı kendi fiyatını yukarı çeker.

**Çözüm:** Almgren-Chriss karekök modeli — Faz 4.1'in σ ve Faz 4.3'ün ADV'sini yeniden kullanır:

```
participation = v_traded_TL / ADV_TL_t
slip_bps      = γ · σ_t · sqrt(participation) · 1e4    # γ=0.10 (BIST default)
total_fee     = base_fee + slip_bps/1e4
```

**BIST sonuç:** Sığ hisseler 10 bps fallback'a, likit hisseler ~0 bps slipaja düşüyor.

### 5.2 Rejim Harmanlayıcı (`blender.py`)

**Sorun:** Sert rejim geçişleri whipsaw yaratır — bugün %100 Boğa formülü → yarın %100 Ayı formülü → tüm pozisyonu sat-al → komisyon erozyonu.

**Çözüm:** Faz 1'in soft prob_df'i ile şampiyon harmanı:

```
champion_signals[k] = cfg.evaluate(catalog["regime_champions"][k], df)
target_weights_t    = Σ_k prob_t[k] · champion_signal_k[t]
target_weights_t    = α·new + (1-α)·old                 # EMA smoothing (α=0.3)
```

Bugün %60 Sakin Boğa + %40 Yatay → fonun %60'ı Boğa şampiyonu, %40'ı Yatay şampiyonu pozisyonlarına dağılır.

**EMA smoothing turnover'ı önemli ölçüde azaltır** (test: salınan rejim → smooth versiyon ~%50 daha az turnover).

### 5.3 Paper Trader (`paper_trader.py`)

**Sorun:** Decay monitor `live_returns` parametresi bekliyor ama bu seriyi üretecek mekanizma yoktu.

**Çözüm:** Paper trade pipeline:

1. **Günlük log** — `log_daily_decisions(target_weights, formula_id, date, db)`:
   - `data/paper_trades.parquet`'e append
   - Şema: `date, formula_id, ticker, weight, signal_value, entry_px, exit_px, gross_pnl_pct, slippage_bps, net_pnl_pct`

2. **t+2 fill** — `compute_realized_pnl(db)`:
   - Eski entries için `exit_px = Pclose_{t+2}` doldur
   - `gross_pnl_pct = exit/entry - 1`, `net_pnl_pct = gross - slip/1e4`

3. **Decay feed** — `feed_decay_monitor(formula_id, μ_bt, σ_bt)`:
   - Paper PnL serisini `scan_decay`'e besle
   - Decay tetiklenirse formül emekli edilir

**Promosyon kuralı:** Bir formül yeni paper trade'e başlar; `Sharpe > 1.0` olana kadar (en az 60 gün out-of-sample) sadece sanal para ile koşar.

### Backtest Entegrasyonu

`run_pro_backtest`'e opsiyonel `slippage_cfg` eklendi. None → sabit fee. `use_dynamic_slippage=True` → per-ticker per-day Almgren-Chriss bps cezası eklenir.

```python
from engine.execution.slippage import SlippageConfig
from engine.core.backtest_engine import run_pro_backtest

_, m_flat = run_pro_backtest(db, signal)
_, m_dyn  = run_pro_backtest(db, signal, slippage_cfg=SlippageConfig(use_dynamic_slippage=True))
```

### Kod Konumu

- `engine/execution/__init__.py`
- `engine/execution/slippage.py`
- `engine/execution/blender.py`
- `engine/execution/paper_trader.py`
- `engine/core/alpha_catalog.py` — `save_regime_champion()` helper eklendi

---

## Uçtan Uca Pipeline

Tipik bir haftalık koşum sırası:

```
PAZAR GECESİ                                                          │
─────────────                                                          │
1. ETL: data/market_db.parquet günceleme                              │
                                                                       │
2. Faz 1: prob_df = run_pipeline()                                    │
   → data/regime_segments.csv güncellenir                             │
                                                                       │
3. Faz 3: meta-optimizer 50 trial                                     │
   → data/best_params.json                                            │
                                                                       │
4. Mining: best_params + Faz 2 ağırlıklı fitness                      │
   → her rejim için en iyi formüller alpha_catalog.json'a yazılır    │
   → save_regime_champion(k, formula) ile rejim şampiyonu işaretle   │
                                                                       │
5. Faz 4 doğrulama: capacity check, vol-target backtest              │
                                                                       │
PAZARTESİ 09:45                                                       │
───────────────                                                        │
6. Faz 5 günlük döngü:                                                │
   a. champion_trees = load_champions_from_catalog()                  │
   b. target_weights = blend_regime_signals(champion_trees, prob_df)  │
   c. log_daily_decisions(target_weights, ..., slippage_cfg=...)      │
                                                                       │
GÜN İÇİ                                                                │
───────                                                                │
7. compute_realized_pnl() — eski entries için exit_px doldur          │
                                                                       │
8. feed_decay_monitor(formula_id, ...) — alpha çürüme kontrolü        │
   triggered=True → formül emekli, paper trade durur                  │
```

---

## Kullanım Kılavuzu

### Kurulum

```bash
cd /Users/.../Minerva_v3_Studio
source venv/bin/activate
pip install -r requirements.txt    # optuna 4.8 dahil
```

### Tek seferlik kurulum (yeni rejim modeli)

```python
from engine.data.regime_detector import run_pipeline

prob_df = run_pipeline()
# data/market_db.parquet'i okur, HMM K=2..8 arası BIC optimizasyonu yapar
# Sonuç: prob_df (Date × K=6) — diskte saklanmaz, runtime hesaplanır
```

### Haftalık Optuna meta-tuning

```bash
python -m engine.strategies.meta_optimizer --n-trials 50
# data/best_params.json üretilir; mevcut Optuna SQLite study'sine resume eder
```

### Mining (Faz 1+2+3 birleşik)

```python
import pandas as pd
from engine.core.alpha_cfg import AlphaCFG
from engine.data.regime_detector import run_pipeline
from engine.strategies.mining_runner import MiningConfig, run_mining_window

cfg = AlphaCFG()
db = pd.read_parquet("data/market_db.parquet")
db_window = db[db["Date"] >= "2024-01-01"]    # son 2 yıl
prob_df = run_pipeline()

mcfg = MiningConfig.from_best_params(
    "data/best_params.json",
    num_gen=80,
    use_regime_weighting=True,
    prob_df=prob_df,
)
results = run_mining_window(db_window, cfg, mcfg)

# Top-3 formülü rejim şampiyonu olarak kaydet
from engine.core.alpha_catalog import save_regime_champion
for r in results[:3]:
    save_regime_champion(
        regime_id=0,                    # şampiyon olduğu rejim
        formula=r.formula,
        tree=r.tree,
        ic=r.ic, rank_ic=r.rank_ic, adj_ic=r.adj_ic,
    )
```

### Vol-targeted backtest

```python
from engine.core.backtest_engine import run_pro_backtest
from engine.risk.position_sizer import RiskConfig

curve, met = run_pro_backtest(
    db, signal, top_k=20,
    risk_cfg=RiskConfig(use_vol_target=True, target_annual_vol=0.15),
)
print(f"Sharpe: {met['IR']:.3f}, MDD: {met['MDD']:.1f}%")
```

### Slipaj-aware backtest

```python
from engine.execution.slippage import SlippageConfig

curve, met = run_pro_backtest(
    db, signal, top_k=20,
    slippage_cfg=SlippageConfig(use_dynamic_slippage=True, gamma=0.10),
)
```

### Capacity check (yeni formül için)

```python
from engine.risk.capacity import CapacityConfig, estimate_formula_capacity

signal_mi = cfg.evaluate(tree, db)    # MultiIndex (Ticker, Date)
result = estimate_formula_capacity(signal_mi, db, CapacityConfig(adv_pct_limit=0.05))
print(f"Max AUM: {result['max_aum_TL']:,.0f} TL")
print(f"Bağlayıcı hisse: {result['binding_ticker']}")
```

### Günlük production loop (Faz 5 paper trade)

```python
from datetime import datetime
import pandas as pd
from engine.core.alpha_cfg import AlphaCFG
from engine.data.regime_detector import run_pipeline
from engine.execution.blender import (
    BlenderConfig, blend_regime_signals, load_champions_from_catalog
)
from engine.execution.paper_trader import (
    PaperTraderConfig, log_daily_decisions, compute_realized_pnl, feed_decay_monitor
)
from engine.execution.slippage import SlippageConfig

# 1. Şampiyonları yükle (haftalık mining çıktısı)
cfg = AlphaCFG()
champions = load_champions_from_catalog(alpha_cfg=cfg)

# 2. Bugünün rejim olasılıklarını al
prob_df = run_pipeline()
db = pd.read_parquet("data/market_db.parquet")

# 3. Harmanla → bugünkü target weights
weights_df = blend_regime_signals(
    champions, prob_df, db,
    BlenderConfig(use_blending=True, top_k=20, smoothing_alpha=0.3),
    alpha_cfg=cfg,
)
today = pd.Timestamp.now().normalize()
today_weights = weights_df.loc[today] if today in weights_df.index else weights_df.iloc[-1]

# 4. Karar logu
log_daily_decisions(
    today_weights[today_weights > 0],
    formula_id="champion_blend_v1",
    date=today,
    db=db,
    slippage_cfg=SlippageConfig(use_dynamic_slippage=True),
)

# 5. Eski entries fill (t+2 sonrası)
compute_realized_pnl(db)

# 6. Decay kontrolü (haftalık)
result = feed_decay_monitor(
    "champion_blend_v1",
    backtest_mean=0.001, backtest_std=0.005,
)
if result.get("triggered"):
    print(f"⚠️  ALPHA DECAY: {result['triggered_at']} — formül emekli edildi")
```

---

## Konfigürasyon Referansı

### `WeightConfig` (Faz 2)

```python
@dataclass
class WeightConfig:
    temperature: float = 2.0          # üstel keskinlik
    w_min: float = 1.0
    w_max: float = 10.0
    ref_date: pd.Timestamp | None = None    # None → prob_df.iloc[-1]
```

### `MiningConfig` (Faz 3 — kritik alanlar)

```python
@dataclass
class MiningConfig:
    search_mode: str = "gp"              # {"gp", "mcts"}
    num_gen: int = 200                   # havuz boyutu
    max_K: int = 15                      # AST max derinliği
    c_puct: float = 1.4                  # MCTS exploration
    mcts_rollouts: int = 16
    mcts_iterations_per_root: int = 50
    lambda_std: float = 2.0
    lambda_cx: float = 0.005
    use_regime_weighting: bool = False
    weight_cfg: WeightConfig | None = None
    prob_df: pd.DataFrame | None = None
    # WF + filtre alanları
    use_wf_fitness: bool = True
    wf_n_folds: int = 5
    wf_purge: int = 2
    wf_embargo: int = 5
```

### `RiskConfig` (Faz 4.1)

```python
@dataclass
class RiskConfig:
    use_vol_target: bool = False         # default kapalı
    target_annual_vol: float = 0.15      # %15 yıllık fon vol hedefi
    vol_window: int = 20
    min_scale: float = 0.1               # leverage tabanı
    max_scale: float = 3.0               # leverage tavanı
```

### `DecayConfig` (Faz 4.2)

```python
@dataclass
class DecayConfig:
    delta: float = 5e-4                  # minimum drift
    lambda_threshold: float = 0.01       # kümülatif alarm eşiği
    consecutive_days: int = 5            # ardışık alarm
    sigma_floor: float = 2.0             # σ pre-filter
```

### `CapacityConfig` (Faz 4.3)

```python
@dataclass
class CapacityConfig:
    adv_window: int = 20
    adv_pct_limit: float = 0.05          # ADV'nin %5'i
    min_advs_TL: float = 1_000_000       # 1M TL altı tradable değil
    portfolio_size: int = 20             # top-K eşit ağırlık
```

### `SlippageConfig` (Faz 5.1)

```python
@dataclass
class SlippageConfig:
    use_dynamic_slippage: bool = False
    gamma: float = 0.10                  # BIST likidite katsayısı
    sigma_window: int = 20
    adv_window: int = 20
    min_participation: float = 1e-4
    fallback_bps: float = 10.0           # ADV yok → 10 bps
```

### `BlenderConfig` (Faz 5.2)

```python
@dataclass
class BlenderConfig:
    use_blending: bool = False
    top_k: int = 20
    smoothing_alpha: float = 0.3         # EMA: 1.0 → smoothing yok
    min_weight: float = 0.01
    fallback_to_argmax: bool = True
```

### `PaperTraderConfig` (Faz 5.3)

```python
@dataclass
class PaperTraderConfig:
    output_path: Path = Path("data/paper_trades.parquet")
    log_time: str = "09:45"
    min_sharpe_for_promotion: float = 1.0
    paper_window_days: int = 60
    hold_days: int = 2
    portfolio_capital_TL: float = 1_000_000
```

---

## Faz 6 — Kurumsal Orkestrasyon + Adli Loglama

**Amaç:** Faz 1-5'i her gün **manuel müdahale olmadan** çalıştır + her kararı
adli (post-mortem) loglarla geriye dönülebilir kıl + sessiz çöküşleri alarmla.

### Ne yapar?

**6.1 — `auto_minerva.py` (Prefect 3.x flow)**

Kök dizinde tek dosya, beş `@task` ve bir `@flow` içerir:

```
                    ┌──────────────────┐
        18:30 ───▶  │ task_fetch_data  │ retries=3, delay=300s
                    │ (yfinance)       │
                    └──────────────────┘
                            │
                            ▼
                    ┌──────────────────┐
                    │ task_detect_     │ retries=2
                    │ regime (HMM)     │ → regime_prob_df.parquet
                    └──────────────────┘
                            │
              ┌─────────────┴─────────────┐
              │ (yalnız Cuma)             │ (her gün)
              ▼                           ▼
    ┌──────────────────┐         ┌──────────────────┐
    │ task_nightly_    │         │ task_decay_scan  │
    │ mining (Optuna)  │         │ (şampiyonlar)    │
    │ ~40 dk           │         │ trigger → raise  │
    └──────────────────┘         └──────────────────┘
              │                           │
              └────────────┬──────────────┘
                           ▼
        09:45 ───▶ ┌──────────────────┐
                   │ task_morning_    │ paper_trades + decisions_log +
                   │ execution        │ önceki gün exit fill
                   └──────────────────┘
```

**6.2 — `engine/execution/forensics.py` (Adli kayıt)**

Her aktif pozisyon için "karar fotoğrafı" çeker (`data/decisions_log.parquet`):

| Alan | İçerik |
|---|---|
| `execution_id` | UUID4 — her satırda benzersiz |
| `timestamp` | Karar anı (UTC ISO) |
| `date`, `ticker`, `action` | BUY / SELL / HOLD |
| `target_weight`, `prev_weight` | Önceki güne göre delta (turnover) |
| `hmm_state` | JSON: `{"regime_0": 0.82, ...}` — o anki rejim algısı |
| `hmm_top_regime`, `hmm_top_p` | Argmax rejim ve olasılığı |
| `champion_id`, `champion_formula` | Hangi formül baskındı |
| `adv_TL`, `adv_limit_ratio` | Kapasite kullanımı |
| `expected_slip_bps` | Almgren-Chriss slipaj tahmini |
| `asset_vol` | Annualized realized vol |
| `notes` | Serbest metin (decay alarm, fallback...) |

**Post-mortem örnek:** Fon bir ay sonra %10 zarar etti. `load_decisions(start, end, ticker)` ile adli kayıtları çekersin → "THYAO'ya 4 gün boyunca %4.5 verdik çünkü HMM Regime 0 (boğa) %82 ihtimal verdi, ama gerçekte Regime 1 (kriz) yaşandı" şeklinde root cause yapılır.

**6.3 — Observability (Prefect Automations)**

Kod içinde Telegram/Slack SDK YOK — webhook UI üzerinden bağlanır:

| Trigger | Aksiyon |
|---|---|
| `Minerva_Core_Loop` Failed | 🚨 "Minerva günlük döngü başarısız" |
| `task_morning_execution` Completed | 🟢 "Günaydın, {n} satır loglandı" |
| `task_decay_scan` Failed (DECAY: prefix) | ⚠️ "Şampiyon emekli edilmeli" |

### Hangi formülleri kullanır?

Faz 6 yeni matematik üretmez — yalnızca Faz 1-5'i orkestrelir. Tek istisna:

**ADV limit oranı (forensics.py):**
```
adv_limit_ratio = v_traded_TL / (adv_pct_limit × ADV_TL)
                = (capital × weight) / (0.05 × ADV_TL)
```
1.0 → ADV limitinin tam tepesindeyiz; >1.0 → limit aşıldı.

### Kod konumu

| Dosya | Fonksiyon | Açıklama |
|---|---|---|
| `auto_minerva.py` | 5 task + `run_daily_cycle()` flow | Prefect 3.x orchestrator |
| `engine/execution/forensics.py` | `log_decision_forensics()`, `load_decisions()` | Adli kayıt + sorgu |
| `engine/core/alpha_catalog.py` | `get_active_champions()` | Decay scan için katalog tarayıcı |
| `api/jobs.py` | `JobRegistry` (SQLite) | Crash-safe iş kaydı |

### Test
**`tests/test_auto_minerva.py`** (3 test):
- `test_task_fetch_data_retries_on_yfinance_failure` — yfinance offline → retry mekanizması
- `test_task_decay_scan_raises_when_triggered` — decay tetik → flow Failed
- `test_run_daily_cycle_smoke` — flow uçtan uca veri akışı

**`tests/test_forensics.py`** (3 test):
- `test_log_decision_forensics_appends_parquet` — şema + UUID tekil
- `test_load_decisions_filters_by_date_and_ticker` — sorgu filtreleme
- `test_forensic_hmm_state_json_round_trip` — HMM JSON encode/decode

```bash
venv/bin/pytest tests/test_auto_minerva.py tests/test_forensics.py -v
# Beklenen: 6/6 passed (~9 sn)
```

---

## Test ve Doğrulama

### Tüm fazlar (235 test)

```bash
venv/bin/pytest tests/ -q
# Beklenen: 235/235 passed (~110 sn)
# Faz 4: 11 test, Faz 5: 12 test, Faz 6: 6 test
# Faz 0/regresyon: 206 test (eski mining/backtest/factor pipeline)
```

### Faz bazında çalıştırma

```bash
# Faz 1: HMM rejim
venv/bin/pytest tests/test_regime_detector.py -v

# Faz 2: Ağırlıklı fitness
venv/bin/pytest tests/test_weighted_fitness.py -v

# Faz 3: MCTS + Optuna
venv/bin/pytest tests/test_mcts_integration.py tests/test_meta_optimizer.py -v

# Faz 4: Risk
venv/bin/pytest tests/test_position_sizer.py tests/test_decay_monitor.py tests/test_capacity.py -v

# Faz 5: Execution
venv/bin/pytest tests/test_slippage.py tests/test_blender.py tests/test_paper_trader.py -v

# Faz 6: Orkestrasyon
venv/bin/pytest tests/test_auto_minerva.py tests/test_forensics.py -v
```

### Smoke testleri (gerçek BIST)

```python
# Tek script ile tüm pipeline:
import pandas as pd
from engine.data.regime_detector import run_pipeline
from engine.risk.capacity import CapacityConfig, compute_adv
from engine.execution.slippage import SlippageConfig, build_slippage_matrix

db = pd.read_parquet("data/market_db.parquet")

# Faz 1
prob_df = run_pipeline()
print(f"Bugün rejim olasılıkları: {prob_df.iloc[-1].to_dict()}")

# Faz 4
adv = compute_adv(db, CapacityConfig())
print(f"Son gün ortalama ADV: {adv.groupby('Date').mean().iloc[-1].values[0]:,.0f} TL")

# Faz 5
ret_wide = db.pivot_table(index="Date", columns="Ticker", values="Pclose").pct_change()
slip = build_slippage_matrix(db, ret_wide, SlippageConfig(use_dynamic_slippage=True))
print(f"Son gün slipaj dağılımı: min={slip.iloc[-1].min():.2f} max={slip.iloc[-1].max():.2f} bps")
```

---

## Sıfır Buffer Kurulum Kılavuzu (Adım Adım)

> **Bu bölüm:** Boş bir Mac/Linux makinede başlayıp Minerva v3'ü her sabah
> 09:45'te otomatik çalışan üretim moduna almak için **uygulayacağın her komutu**
> sırayla anlatır. Senin yapacağın 9 adım.

### Ön Koşullar (ilk kurulum, ~30 dk)

#### Adım 0 — Gereksinimler

Tek seferlik kontrol:
```bash
# Python 3.11+ gerekli
python3 --version          # >= 3.11

# Git, brew (macOS) veya apt (Linux)
git --version
which brew                  # macOS için
```

Eksik olan varsa kurulum:
- macOS: `brew install python@3.13 git`
- Linux: `sudo apt install python3.13 python3.13-venv git`

---

#### Adım 1 — Repoyu klonla ve sanal ortam kur

```bash
cd ~                                   # ev dizinine git
git clone <repo-url> Minerva_v3_Studio
cd Minerva_v3_Studio

# Sanal ortam (venv) oluştur
python3 -m venv venv
source venv/bin/activate              # her shell oturumunda gerekli

# Bağımlılıklar (~5 dk)
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Doğrulama:
```bash
venv/bin/pytest tests/ -q
# Beklenen: 235 passed (~110 sn)
# Bir tane bile FAIL varsa kuruluma DEVAM ETME — sorun raporla.
```

---

### Operasyonel Kurulum

#### Adım 2 — BIST verisini ilk kez çek (ilk seferde ~10-15 dk)

```bash
venv/bin/python -m scripts.fetch_bist_data
```

Beklenen çıktı:
```
✓ XU100.IS  10y → 2516 row
✓ THYAO.IS  10y → 2516 row
...
✓ data/market_db.parquet yazıldı (1,150,000+ satır)
```

Doğrula:
```bash
venv/bin/python -c "
import pandas as pd
db = pd.read_parquet('data/market_db.parquet')
print(f'Hisse: {db.Ticker.nunique()}')
print(f'Tarih aralığı: {db.Date.min()} → {db.Date.max()}')
print(f'Toplam satır: {len(db):,}')
"
```

> **Sorun:** yfinance "rate limit" verirse 30 dakika bekle, tekrar dene.

---

#### Adım 3 — HMM rejim modelini eğit (~2-5 dk)

```bash
venv/bin/python -m engine.data.regime_detector
```

Beklenen çıktı (loglarda):
```
=== Minerva Faz 1: Regime Detection ===
yfinance fetch: XU100.IS, period=10y
Veri: 2497 gün, 2016-04-... → 2026-04-...
Features hazır: 2497 satır × 4 kolon (scaled)
K=2  BIC=...  AIC=...
K=6  BIC=17463.4  ...
OPTIMAL K=6 (BIC=17463.4)
=== DONE — K=6, last_day_probs={...} ===
```

Üretilen artifact'lar:
```bash
ls -la data/regime_*
# regime_hmm.pkl       — joblib serileştirilmiş HMM
# regime_metadata.json — eğitim metadata
# regime_plot.png      — fiyat + rejim renkli grafik
```

Görsel kontrol için `data/regime_plot.png` aç — log fiyat üzerine renkli rejim segmentleri çizilmiş olmalı.

---

#### Adım 4 — Optuna meta-tuning (ilk seferde ~30-45 dk)

MCTS hiperparametrelerini BIST'e göre kalibre eder. **Cuma gecesi haftalık** çalışır ama ilk seferde manuel:

```bash
venv/bin/python -m engine.strategies.meta_optimizer --n-trials 50
```

İlk birkaç trial loglarını gör:
```
trial 0: max_K=15 c_puct=1.4 mcts_rollouts=16 ... → score=0.0073
trial 1: max_K=12 c_puct=2.1 mcts_rollouts=24 ... → score=0.0081
...
=== DONE — best_value=0.0089 ===
best_params={'max_K': 14, 'c_puct': 1.83, ...}
Saved: data/best_params.json
```

> **İpucu:** `--storage sqlite:///data/optuna.db` ekle → çalışma kesintiye uğrarsa kaldığı yerden devam eder.

---

#### Adım 5 — İlk mining koşusu (manuel — `data/alpha_catalog.json` üretmek için)

Mining'i programmatic olarak çalıştır + en az 2 rejim için şampiyon yaz:

```bash
venv/bin/python <<'EOF'
import pandas as pd
from pathlib import Path
from engine.core.alpha_cfg import AlphaCFG
from engine.core.alpha_catalog import save_regime_champion
from engine.core.formula_parser import parse_formula
from engine.data.regime_detector import (
    RegimeConfig, run_pipeline, compute_features, compute_filtered_probability_vector,
)
from engine.strategies.mining_runner import MiningConfig, run_mining_window
import joblib

# 1) Veri + rejim
db = pd.read_parquet("data/market_db.parquet")
prob_df = pd.read_parquet("data/regime_prob_df.parquet") \
    if Path("data/regime_prob_df.parquet").exists() else run_pipeline()

# 2) Mining (best_params.json'dan yükle)
cfg = AlphaCFG()
mcfg = MiningConfig.from_best_params(
    "data/best_params.json", num_gen=80, prob_df=prob_df,
)
results = run_mining_window(db, cfg, mcfg)
print(f"Bulunan formül: {len(results)}")

# 3) Top-2'yi rejim 0 ve 1 için şampiyon yap
top = sorted(results, key=lambda r: r.fitness, reverse=True)[:2]
for k, r in enumerate(top):
    tree = parse_formula(r.formula, cfg)
    save_regime_champion(
        regime_id=k, formula=r.formula, tree=tree,
        ic=r.mean_ric, rank_ic=r.mean_ric, adj_ic=r.mean_ric,
    )
    print(f"Rejim {k} şampiyonu: {r.formula}")
EOF
```

Doğrulama:
```bash
venv/bin/python -c "
from engine.core.alpha_catalog import get_active_champions
champs = get_active_champions()
print(f'Aktif şampiyon: {len(champs)}')
for fid, meta in champs:
    print(f'  Rejim {meta[\"regime_id\"]}: {fid[:60]}')
"
```

---

#### Adım 6 — Auto Minerva flow'unu manuel çalıştır (smoke test)

```bash
venv/bin/python -m auto_minerva
```

Beklenen log akışı (Prefect):
```
12:00:00.000 | INFO    | Beginning flow run 'Minerva_Core_Loop'
12:00:01.234 | INFO    | Faz 6.1: BIST veri çekimi başlıyor
12:00:01.456 | INFO    | Veri hazır: data/market_db.parquet
12:00:02.789 | INFO    | Faz 1: HMM rejim tespiti
12:00:05.123 | INFO    | prob_df kaydedildi (... gün, K=6)
12:00:05.456 | INFO    | nightly_mining skip — bugün <gün>, hedef gün 4
12:00:05.567 | INFO    | decay_scan: 2 aktif şampiyon kontrol edilecek
12:00:06.000 | INFO    | compute_realized_pnl tamam
12:00:08.234 | INFO    | paper_trades.parquet: N satır
12:00:08.500 | INFO    | decisions_log.parquet: N satır
12:00:08.789 | INFO    | Finished in state Completed()
```

Üretilen Faz 6 artifact'ları:
```bash
ls -la data/regime_prob_df.parquet data/decisions_log.parquet data/paper_trades.parquet data/jobs.db
```

> **Hata?** Catalog/champion eksikse "no_catalog" / "no_champions" warning verir, çöker değil. Adım 5'i tekrarla.

---

#### Adım 7 — Prefect server'ı başlat (UI için)

İki ayrı terminal aç:

**Terminal A** (Prefect server, sürekli açık kalmalı):
```bash
cd ~/Minerva_v3_Studio
source venv/bin/activate
prefect server start
# UI: http://127.0.0.1:4200
```

**Terminal B** (deployment + agent):
```bash
cd ~/Minerva_v3_Studio
source venv/bin/activate

# 1) Deployment kaydet (her iş günü 18:30 İstanbul)
prefect deployment build auto_minerva.py:run_daily_cycle \
    -n minerva-daily \
    --cron "30 18 * * 1-5" \
    --timezone "Europe/Istanbul" \
    -q default

prefect deployment apply run_daily_cycle-deployment.yaml

# 2) Agent başlat (cron tetikleyicisini dinler)
prefect agent start -q default
```

Tarayıcıdan `http://127.0.0.1:4200` aç → "Deployments" sekmesinde `minerva-daily` görmelisin.

---

#### Adım 8 — Webhook alarm bağla (Slack/Telegram, opsiyonel ama önerilen)

**Slack için:**
1. Slack workspace'inde Incoming Webhook oluştur → URL'yi kopyala (örn `https://hooks.slack.com/services/T.../B.../X...`).
2. Prefect UI → **Blocks** → "Slack Webhook" oluştur, URL'yi yapıştır.
3. **Automations** sekmesi → 3 otomasyon ekle:

| Automation | Trigger | Action |
|---|---|---|
| **Critical** | Flow run "Minerva_Core_Loop" → Failed | Send notification → "🚨 DİKKAT: Minerva günlük döngü başarısız. {{ flow_run.id }}" |
| **Morning OK** | Task "morning_execution" → Completed | "🟢 Günaydın! Portföy harmanlandı" |
| **Decay** | Task "decay_scan" → Failed AND log içerir "DECAY:" | "⚠️ Şampiyon emekli edilmeli — {{ flow_run.logs }}" |

**Telegram için:** Aynı şey, sadece "Telegram Bot" Block'u ile (bot token + chat_id).

> **Güvenlik:** Webhook URL'leri **Prefect Block API**'da saklanır — `.env` veya `git`'e push ETME.

---

#### Adım 9 — Otomatik gece koşusu için sistemi gece bırak

İlk Cuma gecesi (saat 18:30 sonrası) deployment otomatik çalışmaya başlayacak. Sabah:

1. **09:45 itibariyle** `task_morning_execution` tamamlanmış olmalı
2. Slack/Telegram'da 🟢 mesajı gelmeli
3. `data/paper_trades.parquet` ve `data/decisions_log.parquet` o gün satırlarıyla dolmalı

Sabah doğrulama (her gün):
```bash
venv/bin/python -c "
import pandas as pd
from datetime import datetime

# 1) Son paper trade
pt = pd.read_parquet('data/paper_trades.parquet')
today = pt[pt.date == pt.date.max()]
print(f'Bugün ({pt.date.max().date()}): {len(today)} pozisyon, formula={today.formula_id.iloc[0]}')

# 2) Son adli kayıt
fl = pd.read_parquet('data/decisions_log.parquet')
last = fl[fl.date == fl.date.max()]
print(f'Adli log: {len(last)} satır')
print(f'  HMM top regime: {last.hmm_top_regime.mode().iloc[0]} (p={last.hmm_top_p.mean():.2f})')
print(f'  Ortalama ADV oranı: {last.adv_limit_ratio.mean():.2f}')
print(f'  Beklenen slipaj: {last.expected_slip_bps.mean():.1f} bps')
"
```

---

### Günlük Operasyon Çevrimi (her sabah 5 dk)

| Zaman | Eylem | Sen mi sistem mi? |
|---|---|---|
| **18:30** | yfinance veri çek + HMM güncelle | Sistem (Prefect) |
| **18:35** | Decay scan — eski şampiyonlar OK mi | Sistem |
| **Cuma 18:35** | Optuna 50 trial → best_params.json güncelle | Sistem (~40 dk) |
| **09:45** | Önceki gün exit fill + bugünün target weights + paper log + adli log | Sistem |
| **09:50** | 🟢 Slack/Telegram bildirimi | Sistem |
| **10:00** | Sabah doğrulama scripti çalıştır (yukarıdaki) | Sen (1 dk) |
| **Pazar** | Adli logu gözden geçir (haftalık post-mortem) | Sen (5-10 dk) |

### Aylık bakım (sen yaparsın)

```bash
# 1) Test suite hâlâ yeşil mi
venv/bin/pytest tests/ -q

# 2) Disk kullanımı kontrol
du -sh data/*

# 3) Prefect log retention (varsayılan 7 gün; istersen)
prefect work-pool inspect default

# 4) Eski adli kayıtları arşivle (12 ay)
venv/bin/python -c "
import pandas as pd
fl = pd.read_parquet('data/decisions_log.parquet')
cutoff = pd.Timestamp.now() - pd.DateOffset(months=12)
recent = fl[fl.date >= cutoff]
archived = fl[fl.date < cutoff]
recent.to_parquet('data/decisions_log.parquet', index=False)
archived.to_parquet(f'data/archive/decisions_{cutoff.date()}.parquet', index=False)
print(f'Aktif: {len(recent)}, arşiv: {len(archived)}')
"
```

---

## Sorun Giderme (Troubleshooting)

| Belirti | Olası Neden | Çözüm |
|---|---|---|
| `pytest` 1+ test fail | Kurulumda eksik bağımlılık | `pip install -r requirements.txt -r requirements-dev.txt` tekrarla |
| `yfinance` 429 / timeout | Rate limit | 30 dk bekle, `task_fetch_data` 3× retry yapar |
| `auto_minerva` "no_catalog" | İlk mining yapılmamış | Adım 5'i çalıştır |
| `auto_minerva` "no_champions" | Catalog var ama `regime_champion_for` set değil | Adım 5 sonunda `save_regime_champion(...)` çağır |
| Prefect UI açılmıyor | Server başlatılmamış | Terminal A'da `prefect server start` çalışıyor mu kontrol et |
| Cron tetiklenmiyor | Agent çalışmıyor | Terminal B'de `prefect agent start -q default` |
| Decay alarm sürekli geliyor | Şampiyon gerçekten çürüyor | Yeni mining koş, eski şampiyonu yeni formülle değiştir |
| `decisions_log.parquet` yok | Adım 5 + Adım 6'yı sırayla çalıştır | Mining → champion save → auto_minerva |
| `data/jobs.db` lock error | API + manuel çalışıyor aynı anda | API'yi durdur, manuel iş bitsin, tekrar başlat |
| HMM regime sayısı (K) düştü | Yeni veride bir rejim min_samples_per_regime altında | `RegimeConfig(min_samples_per_regime=150)` ile gevşet |

### Acil durum prosedürü

**Sistem 24 saat çalışmıyor:**
1. `tail -100 ~/.prefect/prefect.log`
2. `venv/bin/pytest tests/ -q` — local çalışıyor mu?
3. Terminal A (server) ve B (agent) hâlâ açık mı?
4. `data/market_db.parquet` son tarihi 1 günden eski mi? → manuel `python -m scripts.fetch_bist_data`

**Decay alarm gelirse:**
1. `venv/bin/python -m engine.execution.paper_trader feed_decay_monitor <formula_id>` ile manuel doğrula
2. Doğruysa: yeni mining koş (Adım 5), eski şampiyon JSON'ından sil
3. `data/decisions_log.parquet` ile son 30 gün kararını incele

**Faz 7'ye geçiş için ön koşul:** auto_minerva 60 iş günü (≈3 ay) boyunca her gün başarılı koşmalı + adli log dolmalı + en az 1 decay tetik gözlenip yönetilmeli.

---

## Riskler & Sınırlamalar

1. **γ kalibrasyonu (Faz 5.1):** BIST için literatür değeri yok; `gamma=0.10` tutucu başlangıç. Faz 7'de gerçek paper trade verisiyle backfit yapılacak (≥60 iş günü adli log gerekli).
2. **Vol-target leverage (Faz 4.1):** Düşük vol dönemlerinde scale=3.0'a kadar çıkabilir. Fon mandatesinde kaldıraç yasaksa `max_scale=1.0`'a çekilmeli.
3. **Page-Hinkley false-positive (Faz 4.2):** BIST gibi gürültülü piyasada δ=0.0005 ve λ=0.01 default'u tutucu — production'da Optuna ile tunelar.
4. **Walk-forward referans tarihi (Faz 2):** "prob_df son günü" production'da bugün, ama backtest WF döngüsünde look-ahead yaratır. WF backtest için `WeightConfig.ref_date = fold.train_end` ile fold-bazlı override.
5. **MCTS hızı (Faz 3):** Her formül için iterations=50 MCTS araması GP'ye göre 5-10x yavaş. `num_gen` düşürülür (50-80), `mcts_iterations_per_root` ile kalite-hız trade-off Optuna'da tunelar.

---

## Faz 7+ Yol Haritası

- **Faz 7:** Live broker integration + γ backfit + Tree-LSTM value head + JWT API auth + Docker/CI-CD
  - 7.1 Broker adapter (Matriks, FIX 4.4) — paper → gerçek emir
  - 7.2 Real money risk gates (daily loss limit, position concentration, circuit breaker)
  - 7.3 γ kalibrasyonu — gerçek paper trade verisinden Almgren-Chriss γ fit
  - 7.4 Tree-LSTM value head aktivasyonu — MCTS rollout neural value
  - 7.5 API sertleştirme — JWT, rate limiting, audit log
  - 7.6 Dockerfile + GitHub Actions CI/CD
  - 7.7 SPK uyumluluk raporları + KAP entegrasyonu

- **Faz 8:** Multi-strategy blend — birden fazla şampiyon ailesi (momentum + mean-reversion + factor)
- **Faz 9:** Cross-asset (BIST + dövizler + emtia) genişleme

---

## Dosya Haritası

```
Minerva_v3_Studio/
├── auto_minerva.py                    # FAZ 6: Prefect orchestrator (kök dizin)
├── engine/
│   ├── core/
│   │   ├── alpha_cfg.py               # AlphaCFG grameri
│   │   ├── alpha_catalog.py           # JSON katalog + get_active_champions (Faz 6)
│   │   ├── backtest_engine.py         # run_pro_backtest + risk_cfg + slippage_cfg
│   │   └── formula_parser.py          # AST parser
│   ├── data/
│   │   ├── regime.py                  # eski kural tabanlı (geriye dönük)
│   │   └── regime_detector.py         # FAZ 1: HMM K=6 + Forward-only (look-ahead safe)
│   ├── validation/
│   │   ├── wf_fitness.py              # purged & embargoed K-fold RankIC
│   │   └── weighted_fitness.py        # FAZ 2: rejim ağırlıklı RankIC
│   ├── strategies/
│   │   ├── mining_runner.py           # FAZ 3: MCTS branch + from_best_params
│   │   ├── mcts.py                    # FAZ 3: GrammarMCTS (PUCT, normalize prior)
│   │   └── meta_optimizer.py          # FAZ 3: Optuna CLI + fcntl lock
│   ├── risk/                          # FAZ 4
│   │   ├── position_sizer.py          # vol-target (shift(1) safe)
│   │   ├── decay_monitor.py           # Page-Hinkley
│   │   └── capacity.py                # ADV (per-date n_active)
│   └── execution/                     # FAZ 5+6
│       ├── slippage.py                # Almgren-Chriss
│       ├── blender.py                 # rejim harmanı + EMA smoothing
│       ├── paper_trader.py            # paper trade + decay feed (OOB safe)
│       └── forensics.py               # FAZ 6: adli karar logu
├── api/
│   └── jobs.py                        # SQLite job registry (Faz 6 — crash-safe)
├── scripts/
│   └── fetch_bist_data.py             # ETL — auto_minerva tarafından çağrılır
├── tests/                              # 235 test toplam
│   ├── test_regime_detector.py        (Faz 1)
│   ├── test_weighted_fitness.py       (Faz 2)
│   ├── test_mcts_integration.py       (Faz 3)
│   ├── test_meta_optimizer.py         (Faz 3)
│   ├── test_position_sizer.py         (Faz 4)
│   ├── test_decay_monitor.py          (Faz 4)
│   ├── test_capacity.py               (Faz 4)
│   ├── test_slippage.py               (Faz 5)
│   ├── test_blender.py                (Faz 5)
│   ├── test_paper_trader.py           (Faz 5)
│   ├── test_forensics.py              (Faz 6)
│   └── test_auto_minerva.py           (Faz 6)
└── data/                              # gitignore'da (.gitignore Faz 6'da güncellendi)
    ├── market_db.parquet              # ETL çıktısı (BIST hisseleri)
    ├── regime_segments.csv            # Faz 1 — 241 segment
    ├── regime_prob_df.parquet         # Faz 6 — günlük rejim olasılıkları
    ├── regime_hmm.pkl                 # Faz 1 — HMM model
    ├── regime_metadata.json           # Faz 1 — eğitim metadata
    ├── best_params.json               # Faz 3 — Optuna sonucu (fcntl lock)
    ├── optuna.db                      # Faz 3 — SQLite study (resume)
    ├── alpha_catalog.json             # Faz 3-5 — formüller + şampiyonlar
    ├── paper_trades.parquet           # Faz 5 — günlük loglar
    ├── decisions_log.parquet          # Faz 6 — adli karar fotoğrafları
    └── jobs.db                        # Faz 6 — SQLite job registry
```

---

**Son güncelleme:** 2026-04-29 — Faz 6 tamamlandı, **235/235 test yeşil** (~110 sn).
**Bir sonraki:** Faz 7 (Live broker + γ backfit + JWT auth + Docker/CI).
