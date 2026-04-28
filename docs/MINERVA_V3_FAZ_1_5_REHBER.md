# Minerva v3 — Faz 1-5 Sistem Rehberi

> Otonom BIST alpha-fonu mimarisinin **rejim tespiti → ağırlıklı fitness →
> MCTS+Optuna mining → risk yönetimi → execution layer** uçtan uca rehberi.
>
> Bu dosya: **(1)** her fazın ne yaptığını, **(2)** hangi formülleri kullandığını,
> **(3)** kodun nerede yaşadığını, **(4)** uçtan uca nasıl çalıştırılacağını anlatır.

## Kuş Bakışı

| Faz | Kapsam | Anahtar Modül | Yeni Test | Durum |
|---|---|---|---|---|
| 1 | HMM K=6 piyasa rejimi | `engine/data/regime_detector.py` | tests/test_regime_detector.py | ✓ |
| 2 | Rejim-koşullu ağırlıklı fitness | `engine/validation/weighted_fitness.py` | tests/test_weighted_fitness.py | ✓ |
| 3 | MCTS aktivasyon + Optuna meta-opt | `engine/strategies/{mcts,meta_optimizer}.py` | tests/test_{mcts_integration,meta_optimizer}.py | ✓ |
| 4 | Vol-target + Page-Hinkley + ADV | `engine/risk/*` | 11 test | ✓ |
| 5 | Almgren-Chriss + harmanlama + paper trade | `engine/execution/*` | 12 test | ✓ |

**Test toplamı (tüm sistem):** `pytest tests/ -q` → **229/229 passed** (~102 sn).

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
7. [Uçtan Uca Pipeline](#uçtan-uca-pipeline)
8. [Kullanım Kılavuzu](#kullanım-kılavuzu)
9. [Konfigürasyon Referansı](#konfigürasyon-referansı)
10. [Test ve Doğrulama](#test-ve-doğrulama)

---

## Sistem Özeti

Minerva v3 beş katmanlı bir otonom alpha keşif ve dağıtım sistemidir:

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
└──────────────────────────────────────────────────────────────────────┘
```

**Veri akışı:** `data/market_db.parquet` (BIST hisseleri, 2016–bugün) → tüm fazlar bunu okur.

**Sonuç ürünleri:**
- `data/regime_segments.csv` — 241 kronolojik rejim segmenti
- `data/best_params.json` — Optuna en iyi MCTS hiperparametreleri
- `data/alpha_catalog.json` — keşfedilen formüller + rejim şampiyonları
- `data/paper_trades.parquet` — paper trading karar logları

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

## Test ve Doğrulama

### Tüm fazlar (229 test)

```bash
venv/bin/pytest tests/ -q
# Beklenen: 229/229 passed (~100 sn)
# Faz 1-5 spesifik: 23 yeni test (Faz 4: 11, Faz 5: 12)
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

## Riskler & Sınırlamalar

1. **γ kalibrasyonu (Faz 5.1):** BIST için literatür değeri yok; `gamma=0.10` tutucu başlangıç. Faz 6'da gerçek paper trade verisiyle backfit yapılacak.
2. **Vol-target leverage (Faz 4.1):** Düşük vol dönemlerinde scale=3.0'a kadar çıkabilir. Fon mandatesinde kaldıraç yasaksa `max_scale=1.0`'a çekilmeli.
3. **Page-Hinkley false-positive (Faz 4.2):** BIST gibi gürültülü piyasada δ=0.0005 ve λ=0.01 default'u tutucu — production'da Optuna ile tunelar.
4. **Walk-forward referans tarihi (Faz 2):** "prob_df son günü" production'da bugün, ama backtest WF döngüsünde look-ahead yaratır. WF backtest için `WeightConfig.ref_date = fold.train_end` ile fold-bazlı override.
5. **MCTS hızı (Faz 3):** Her formül için iterations=50 MCTS araması GP'ye göre 5-10x yavaş. `num_gen` düşürülür (50-80), `mcts_iterations_per_root` ile kalite-hız trade-off Optuna'da tunelar.

---

## Faz 6+ Yol Haritası

- **Faz 6:** Tree-LSTM value head — MCTS rollout'larını öğrenen bir değer fonksiyonuyla kısalt; γ backfit
- **Faz 7:** Live broker FIX entegrasyonu — paper → real money geçiş
- **Faz 8:** Multi-strategy blend — birden fazla şampiyon ailesi (momentum + mean-reversion + ...)

---

## Dosya Haritası

```
Minerva_v3_Studio/
├── engine/
│   ├── core/
│   │   ├── alpha_cfg.py           # AlphaCFG grameri
│   │   ├── alpha_catalog.py       # JSON katalog + save_regime_champion (Faz 5)
│   │   ├── backtest_engine.py     # run_pro_backtest + risk_cfg + slippage_cfg
│   │   └── formula_parser.py      # AST parser
│   ├── data/
│   │   ├── regime.py              # eski kural tabanlı (geriye dönük)
│   │   └── regime_detector.py     # FAZ 1: HMM K=6
│   ├── validation/
│   │   ├── wf_fitness.py          # purged & embargoed K-fold RankIC
│   │   └── weighted_fitness.py    # FAZ 2: rejim ağırlıklı RankIC
│   ├── strategies/
│   │   ├── mining_runner.py       # FAZ 3: MCTS branch + from_best_params
│   │   ├── mcts.py                # FAZ 3: GrammarMCTS (PUCT)
│   │   └── meta_optimizer.py      # FAZ 3: Optuna CLI
│   ├── risk/                      # FAZ 4
│   │   ├── position_sizer.py      # vol-target
│   │   ├── decay_monitor.py       # Page-Hinkley
│   │   └── capacity.py            # ADV
│   └── execution/                 # FAZ 5
│       ├── slippage.py            # Almgren-Chriss
│       ├── blender.py             # rejim harmanı
│       └── paper_trader.py        # paper trade + decay feed
├── tests/
│   ├── test_regime_detector.py    (Faz 1)
│   ├── test_weighted_fitness.py   (Faz 2)
│   ├── test_mcts_integration.py   (Faz 3)
│   ├── test_meta_optimizer.py     (Faz 3)
│   ├── test_position_sizer.py     (Faz 4)
│   ├── test_decay_monitor.py      (Faz 4)
│   ├── test_capacity.py           (Faz 4)
│   ├── test_slippage.py           (Faz 5)
│   ├── test_blender.py            (Faz 5)
│   └── test_paper_trader.py       (Faz 5)
└── data/
    ├── market_db.parquet          # ETL çıktısı (BIST hisseleri)
    ├── regime_segments.csv        # Faz 1 — 241 segment
    ├── best_params.json           # Faz 3 — Optuna sonucu
    ├── alpha_catalog.json         # Faz 3-5 — formüller + şampiyonlar
    ├── optuna.db                  # Faz 3 — SQLite study (resume)
    └── paper_trades.parquet       # Faz 5 — günlük loglar
```

---

**Son güncelleme:** 2026-04-28 — Faz 5 tamamlandı, **229/229 test yeşil** (~102 sn).
