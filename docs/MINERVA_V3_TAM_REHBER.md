# Minerva v3 — Tam Sistem Rehberi

**Son güncelleme:** 2026-05-01
**Versiyon:** Sprint 3 sonrası (Q1–Q20 + N1–N60 tüm zafiyetler kapatıldı)
**Hedef:** Bu dosyayı okuduğunda sistem hakkında **bilmediğin tek nokta kalmasın.**

---

## 0. İçindekiler

1. [Sistem Nedir, Ne İçin Var](#1-sistem-nedir-ne-için-var)
2. [Kuş Bakışı Mimari](#2-kuş-bakışı-mimari)
3. [Faz Faz Algoritmik Akış](#3-faz-faz-algoritmik-akış)
4. [`engine/` Modüllerinin Tam Anatomisi](#4-engine-modüllerinin-tam-anatomisi)
5. [`api/` — FastAPI Backend](#5-api--fastapi-backend)
6. [`frontend/` — React/TS SPA](#6-frontend--reactts-spa)
7. [`scripts/` — Yardımcı Komutlar](#7-scripts--yardımcı-komutlar)
8. [`auto_minerva.py` — Prefect Orkestrasyonu](#8-auto_minervapy--prefect-orkestrasyonu)
9. [Veri Akışı Haritası (Data Flow)](#9-veri-akışı-haritası-data-flow)
10. [Test Suite ve Doğrulama](#10-test-suite-ve-doğrulama)
11. [Bu Zamana Ne Yaptık (Sprint Tarihçesi)](#11-bu-zamana-ne-yaptık-sprint-tarihçesi)
12. [Bilinen Bug'lar & Sınırlamalar](#12-bilinen-buglar--sınırlamalar)
13. [Gelecek (LT-1 … LT-20)](#13-gelecek-lt-1--lt-20)
14. [Operasyonel Kılavuz](#14-operasyonel-kılavuz)

---

## 1. Sistem Nedir, Ne İçin Var

**Minerva v3**, Borsa İstanbul (BIST) için **otonom alfa keşfi + paper-trading + risk yönetimi** sistemidir. Tek cümle ile:

> Yahoo Finance'ten BIST verisi çek → HMM ile rejim tespit et → MCTS ile alfa formülleri keşfet → walk-forward + PBO + DSR ile doğrula → rejim olasılığıyla harmanla → paper trade → günlük PnL ölç → decay (Page-Hinkley) tespit et → Telegram'a bildir.

**Akademik kaynaklar**:
- AlphaCFG (arXiv 2601.22119) — α-Sem-k grameri, Tree-LSTM + MCTS
- QuantaAlpha (arXiv 2602.07085) — 6 özellik + operatör kütüphanesi
- López de Prado, *Advances in Financial Machine Learning* — triple-barrier, meta-labeling, PBO/CSCV
- Bailey & López de Prado (2014) — Deflated Sharpe Ratio
- Almgren & Chriss — kareköklü slipaj modeli
- Page-Hinkley — change-point detection (decay alarm)
- Tai et al. 2015 — Child-Sum Tree-LSTM

**Faz 1 → 6** isimli aşamalı geliştirme:
- **Faz 1** — HMM rejim tespiti
- **Faz 2** — Rejim-koşullu ağırlıklı fitness
- **Faz 3** — MCTS + Optuna meta-tuning
- **Faz 4** — Risk yönetimi (vol target, decay, capacity)
- **Faz 5** — Execution (slippage, blender, paper trader)
- **Faz 6** — Prefect orkestrasyon + UI + API + observability

---

## 2. Kuş Bakışı Mimari

```
                    ┌─────────────────────────────────┐
                    │  auto_minerva.py (Prefect Flow) │  ← cron 18:30, Mon-Fri
                    └────────────────┬────────────────┘
                                     │
       ┌──────────────────┬──────────┼─────────────┬─────────────────┐
       ▼                  ▼          ▼             ▼                 ▼
   fetch_data        detect_regime  nightly        decay_scan      morning
   (yfinance)         (HMM)         _mining        (Page-Hinkley)  _execution
       │                  │          │ (Cuma)       │              (blender +
       ▼                  ▼          ▼              ▼              paper trade)
   market_db         regime_prob   alpha_catalog  champion        decisions_log
   .parquet         _df.parquet   .json         status            .parquet
                                                                  paper_trades
                                                                  .parquet

   ┌────────────────────────────────────────────────────────────────┐
   │ FastAPI (api/) ◄─── Frontend SPA (frontend/) ───► Workbench    │
   │   /api/catalog       Catalog / Workbench / LLMTrainer / ...    │
   │   /api/backtest      WS /ws/jobs/{id} progress streaming        │
   │   /api/mining        Zustand store (workbenchStore)            │
   │   /api/training      React Query (useCatalog, useMeta, useJob) │
   │   /api/system                                                   │
   └────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                              Telegram alerts
                              (engine/notifications)
```

**Üç ana çalışma modu:**
1. **Otonom (cron)**: Prefect deployment her akşam 18:30'da `run_daily_cycle()` çağırır.
2. **Manuel (UI)**: Frontend'den `/workbench` ile formül seç, backtest çalıştır, mining başlat, LLM eğit.
3. **CLI**: `python auto_minerva.py`, `python -m engine.strategies.meta_optimizer`, `python scripts/dr_drill.py`.

---

## 3. Faz Faz Algoritmik Akış

### Faz 1 — HMM Rejim Tespiti
Girdi: `XU100.IS` günlük EOD verisi (yfinance).
İşlem:
1. 4 durağan özellik üret: `log_ret`, `realized_vol_20`, `mom_60`, `vol_term_20_60`
2. RobustScaler **train-only** fit (N6 fix), tüm seriye uygulanır
3. Gaussian HMM K∈[2..6] için BIC ile model seçimi (`covariance_type="diag"`, N7 fix)
4. "Min 200 örnek/rejim" kuralı ile aday filtrele
5. Hungarian permutation (Macar algoritması) ile rejim etiketlerini stabilize et

Çıktı:
- `data/regime_hmm.pkl` (joblib)
- `data/regime_metadata.json` (BIC/AIC, per-regime stats, son gün)
- `data/regime_prob_df.parquet` (Date × K filtered probabilities)

### Faz 2 — Rejim-Koşullu Fitness
- Mining'in fitness fonksiyonu: `mean_ric_weighted = Σ_t cosine_sim(prob_t, prob_today) · daily_ric_t`
- Cosine similarity üstel transform ile [w_min=1, w_max=10] aralığına eşlenir
- Sonuç: "bugünün rejimine benzer geçmiş günler 10×, alakasız günler 1×" ağırlıkla katkı yapar
- Look-ahead **yok**: stitching yapılmaz, shift yerinde kalır

### Faz 3 — MCTS + Optuna
**MCTS (engine/strategies/mcts.py):**
- AlphaCFG α-Sem-k grameri → ASR (Abstract Syntax Representation) ağacı
- PUCT formülü: `Q + c·√(b/b_ref)·P_norm·√ΣN/(1+N)`
- Softmax-normalized priors (N11 fix)
- Dead-branch pruning (N12: visit count < 2 olan dallar kırpılır)
- Replay buffer'dan top-20 IC formülü warm-start prior (N15 fix)
- Tree-LSTM enjekte edilebilir (`value_fn`, `policy_fn`)

**Optuna (engine/strategies/meta_optimizer.py):**
- 6 hyperparametre: `c_puct`, `lambda_std`, `lambda_cx`, `lambda_size`, `temp`, `regime_weight_max`
- `n_trials=50` deneme, her trial = kısa MCTS mining
- Objective: top-10 formülün medyan `mean_ric_weighted`
- Cuma gecesi tam kapasite, diğer günler skip
- **N14**: `rolling_tune()` — n_windows örtüşen pencerede ayrı Optuna study + medyan konsensüs params

### Faz 4 — Risk Yönetimi
**4.1 Position Sizer:**
- `target_annual_vol=0.15` (varsayılan)
- `asset_vol = max(EWMA_5, rolling_std_20) · √252` (N21: hızlı + yavaş blend)
- `scale = clip(target/asset_vol, 0.25, 4.0)`

**4.2 Decay Monitor (Page-Hinkley):**
- `m_t = max(0, m_{t-1} + (μ_backtest - r_t - δ))`
- Trigger: `m_t > λ AND consecutive_alarms ≥ N`
- **N20**: Extreme outlier (>3.5σ) freeze — alarm artmaz, sıfırlanmaz
- Trigger → kill_switch dosyası yazılır (`data/.kill_switch`)

**4.3 Capacity:**
- `ADV_TL = mean(Vlot · Pclose, 20)` → look-ahead için `shift(1)`
- **N19**: `min(rolling_20, EWM_10)` blend — likidite şokuna hızlı tepki
- `formula_capacity = min(adv_pct_limit · ADV_TL)` aktif ticker'lar üzerinden

### Faz 5 — Execution
**5.1 Slippage (Almgren-Chriss):**
- `slip_bps = γ · σ_t · √(participation) · 1e4`
- γ=0.10 sabit; **N24**: `scripts/calibrate_slippage.py` ile paper_trades'ten OLS kalibrasyonu

**5.2 Blender:**
- `target_weights_t = Σ_k prob_t[k] · champion_signal_k[t]`
- EMA smoothing turnover'ı bastırır
- `min_weight` filtresinden sonra normalize (N22 fix)

**5.3 Paper Trader:**
- T+0: BUY/SELL/HOLD karar logu
- T+2: gerçek `exit_px` ile `net_pnl_pct` doldurulur
- **N10**: |gross_raw| > 0.5 → dividend/split spike guard
- **N25/26/27**: Kill-switch TTL (24h), daily loss limit (-5%), cumulative drawdown (-10%)
- **N51/N54**: Istanbul TZ-aware timestamp, env-based slippage cap

### Faz 6 — Orkestrasyon
**Prefect 3.x flow:**
- 5 task: `fetch_data → detect_regime → nightly_mining → decay_scan → morning_execution`
- N44: Her run başında `task_cleanup_stale_jobs()` (5dk üstü heartbeat'siz job'ları temizler)
- Telegram hooks (`on_failure`, `on_completion`)
- N48: nightly_mining timeout 7200s (Optuna 50× + tam mining)

**Forensics (5'in altında):**
- Her aktif pozisyon için karar fotoğrafı: HMM state vector, baskın şampiyon, ADV ratio, beklenen slipaj
- N23: `_dominant_champion()` → `signal_mag × prob` ağırlıklı seçim (önceden `argmax(prob)`)

---

## 4. `engine/` Modüllerinin Tam Anatomisi

### 4.1 `engine/core/` — Temel altyapı (8 dosya)

| Dosya | Görev |
|-------|-------|
| **`alpha_cfg.py`** | α-Sem-k grameri + AST `Node` + operatör kütüphanesi (UNARY/BINARY/BINARY_ASYM/ROLLING/PAIRED/CS). `evaluate()` formülü pandas Series'e çevirir. **N1**: 500 kayıtlık LRU cache. **N2**: quantile clip [0.5%, 99.5%]. **N3**: EMA `adjust=True, min_periods=w`. |
| **`alpha_catalog.py`** | Bulunan formülleri `data/alpha_catalog.json`'a yazar. Otomatik şema migrasyonu (`CATALOG_SCHEMA_VERSION`). **N46**: günlük tek snapshot rotation. `save_alpha`, `load_catalog`, `_take_snapshot`. |
| **`backtest_engine.py`** | Tablo 7 uyumlu vektörize backtester. Deal price = `Popen_{t+1}`, getiri = `Pclose_{t+2}/Pclose_{t+1}-1`. Komisyon: %0.05 alım / %0.15 satım. TopkDropout (k=50, drop=5). **5-15× hızlandırma** (pivot_table + numpy argsort). |
| **`api_helpers.py`** | FastAPI ve Streamlit ortak helper'ları. `prepare_eval_idx`, `slice_db_by_window`, `evaluate_ic`, `run_full_evaluate`, `parse_or_raise`. |
| **`formula_parser.py`** | String formül → AST. `Rank(Mul(Pclose, Vlot), 20)` gibi. LLM çıktılarını replay buffer'a aktarır. Aliases: `cs_rank → CSRank`, `ts_mean → Mean` vb. |
| **`logger.py`** | **N40**: structured logging. Loguru varsa JSON, yoksa standart logging. |
| **`metrics.py`** | **N41**: Prometheus counters/gauges. `prometheus_client` yoksa NoOp fallback. Metrics: `mining_jobs_total`, `formula_acceptance_rate`, `decay_alarms_total`, `paper_pnl_daily`, `kill_switch_active`. |

**Önemli gramer detayları (alpha_cfg.py):**
- `FEATURES = ["Popen", "Phigh", "Plow", "Pclose", "Vlot", "Ptyp"]`
- `CONSTANTS = [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]`
- `NUMS = [20, 30, 40]`
- `ROLLING_OPS`: 14 operatör (Rank, WMA, EMA, Ref, Mean, Sum, Std, Var, Skew, Kurt, Max, Min, Med, Mad, Delta)
- `PAIRED_OPS`: Corr, Cov
- `CS_OPS`: CSRank
- `DELTA_K`: üretim kuralı başına uzunluk artımı (Tablo 7)

### 4.2 `engine/data/` — Veri katmanı (6 dosya)

| Dosya | Görev | Çıktı |
|-------|-------|-------|
| **`db_builder.py`** | Tüm BIST evreni için yfinance OHLCV + **N4** `shares_outstanding` (yfinance.info). **N9**: 3× exponential backoff + `missing_tickers.json`. | `data/market_db.parquet` |
| **`regime_detector.py`** | Faz 1 HMM. **N6**: `compute_features()` → `(features_df, scaler)` tuple, RobustScaler train-only fit. **N7**: `covariance_type="diag"`. | `regime_hmm.pkl`, `regime_metadata.json`, `regime_prob_df.parquet`, `regime_plot.png` |
| **`regime.py`** | Kural tabanlı 3-state rejim ("bull"/"chop"/"bear") — fallback. trend × vol bazlı. | (df'e ek kolon) |
| **`factor_neutralize.py`** | İki aşamalı CS neutralization: (1) Rank-Space OLS (size, vol, mom), (2) Quantile bin-demean. **N5**: NaN dropna, np.nan_to_num kaldırıldı. | `compute_size_corr`, `neutralize_signal` |
| **`triple_barrier.py`** | López de Prado AFML Ch.3 — üst/alt/zaman bariyer etiketleme. **N8**: end-window rows `TB_Weight=0.0` ile dahil. | `(labels, weights)` tuple |
| **`meta_label.py`** | López de Prado AFML §3.6 — secondary "bet size / skip" modeli. Logistic regression + TimeSeriesSplit. `apply_meta_filter` → düşük güvenli günler NaN. | (sinyal maskesi) |

### 4.3 `engine/strategies/` — Arama algoritmaları (3 dosya)

| Dosya | Görev |
|-------|-------|
| **`mcts.py`** | Grammar-aware MCTS + PUCT. `_MCTSNode`, `GrammarMCTS`. Tree-LSTM `value_fn` enjekte edilebilir. **N11/N12/N13** fix'leri. |
| **`meta_optimizer.py`** | Optuna haftalık tuning. `MetaOptConfig`, `build_objective`, `run_meta_optimization`. **N14**: `rolling_tune()`, `WindowTuneResult`, `RollingTuneResult`. |
| **`mining_runner.py`** | Pencere bazlı reentrant mining (Mod 3 Full Rolling Discovery). `MiningConfig`, `run_mining_window`. **N15**: replay buffer'dan top-20 IC warm-start. |

### 4.4 `engine/risk/` — Risk yönetimi (3 dosya, Faz 4)

| Dosya | Görev |
|-------|-------|
| **`position_sizer.py`** | Vol targeting. `RiskConfig(target_annual_vol=0.15)`, `apply_vol_target`. **N21**: EWMA + rolling blend. |
| **`decay_monitor.py`** | Page-Hinkley. `DecayConfig(δ, λ, N)`, `update_decay_state`, `scan_decay`. **N20**: extreme_sigma_cap=3.5 freeze. |
| **`capacity.py`** | ADV-tabanlı kapasite. `CapacityConfig(adv_window=20, adv_pct_limit=0.05)`, `compute_adv`, `formula_capacity`. **N19**: rolling+EWM blend. |

### 4.5 `engine/execution/` — Execution layer (4 dosya, Faz 5)

| Dosya | Görev |
|-------|-------|
| **`slippage.py`** | Almgren-Chriss `slip_bps = γ·σ·√participation·1e4`. `SlippageConfig(gamma=0.10)`. |
| **`blender.py`** | Soft regime blend. `BlenderConfig(ema_span, min_weight)`. EMA smoothing + post-filter normalize (**N22**). |
| **`paper_trader.py`** | T+0/T+2 paper trades. **N10/N25/26/27/N51/N54** fix'leri. `is_kill_switch_active()` (TTL), `check_daily_loss_limit()`, `check_cumulative_drawdown()`, `activate_kill_switch()`, `log_daily_decisions()`. |
| **`forensics.py`** | Karar fotoğrafları → `decisions_log.parquet`. **N23**: `_dominant_champion()` weighted. |

### 4.6 `engine/validation/` — Doğrulama (5 dosya)

| Dosya | Görev |
|-------|-------|
| **`wf_fitness.py`** | Walk-forward fitness: `mean(ric) - λ_std·std(ric) - λ_cx·complexity`. **N16**: `size_corr_hard_limit=0.5`. |
| **`weighted_fitness.py`** | Faz 2 rejim-koşullu ağırlıklı fitness. Cosine sim + üstel transform. |
| **`pbo_cscv.py`** | Probability of Backtest Overfitting (Bailey-Borwein-LdP-Zhu 2014). CSCV M-slice. **N17**: `n_slices=8` (default). |
| **`deflated_sharpe.py`** | Bailey-LdP 2014 DSR. `compute_dsr(sr, T, skew, kurt, N_trials)` → p-value. |
| **`ensemble.py`** | Top-K rank-average ensemble + `HallOfFame`. **N18**: `promote_ensemble_champion(catalog, alpha_cfg, db, prob_df, top_k, regime_id)` — şampiyon promotion entegrasyonu. |

### 4.7 `engine/ml/` — Tree-LSTM (3 dosya)

| Dosya | Görev |
|-------|-------|
| **`tree_lstm.py`** | Child-Sum Tree-LSTM (Tai 2015). `PolicyValueNet` — policy head (üretim kuralları softmax) + value head ([-1,1] tanh). |
| **`replay_buffer.py`** | `(Node, IC, [visit_dist])` örnekleri. **JSON** serialize (pickle yerine, RCE riski yok). **N45**: atomic write tempfile+os.replace. **N57**: FileLock fallback to threading.Lock. |
| **`trainer.py`** | `TreeLSTMTrainer`. Value MSE + policy CE. `ic_scale=10` ile tanh range'ine eşle. |

### 4.8 `engine/notifications/` — Bildirimler (1 dosya)

| Dosya | Görev |
|-------|-------|
| **`telegram.py`** | `send_telegram(msg, parse_mode)`. Token + chat_id `.env`'den. `TELEGRAM_DISABLED=1` → no-op. |

---

## 5. `api/` — FastAPI Backend

### 5.1 Dosya yapısı
```
api/
├── main.py          # FastAPI app, CORS, slowapi rate-limit
├── deps.py          # get_market_db, get_brain, verify_api_key
├── jobs.py          # JobRegistry, JobEvent (uzun süren işler)
├── schemas.py       # Pydantic modeller
└── routes/
    ├── catalog.py   # GET/DELETE /api/catalog
    ├── formulas.py  # POST /api/formulas/parse | /evaluate
    ├── backtest.py  # POST /api/backtest/{run, dsr, pbo, ensemble, overfit, rolling-wf, parse-multi}
    ├── mining.py    # POST /api/mining/{start, cancel/{id}, status/{id}}
    ├── training.py  # GET /api/training/buffer + POST /run
    ├── jobs.py      # GET /api/jobs/{id} + WS /ws/jobs/{id}
    └── system.py    # GET /api/system/kill_switch + POST /reset, /activate
```

### 5.2 Kimlik doğrulama modeli
- `verify_api_key`: `X-Api-Key` header okur, env `API_KEY` ile karşılaştırır
- `API_KEY` set değilse **dev-bypass** aktif (development için)
- Uygulanan endpoint'ler: catalog DELETE, mining start, training, system reset
- WS auth: query param `?api_key=...` (N30 fix)

### 5.3 Önemli endpoint detayları

**`POST /api/backtest/run`**
- Request: `{formula: str, window: "test"|"train"|"all"}`
- Response: `{job_id: str}` — async, sonuç WS'ten gelir
- Job WebSocket: `ws://localhost:8000/ws/jobs/{job_id}?api_key=...`
- Mesaj tipi: `{type: "progress"|"log"|"result"|"error", payload}`

**`POST /api/mining/start`**
- Request: `MiningRequest { window, num_gen, max_K, wf_n_folds, wf_embargo, wf_purge, lambda_std, lambda_cx, lambda_size, size_corr_hard_limit, neutralize, save_to_catalog }`
- N32/N60: Module-level `_cancel_events: dict[str, threading.Event]` registry
- Cooperative cancellation: progress callback iterasyonda `event.is_set()` kontrol eder, set ise `InterruptedError`
- `POST /api/mining/cancel/{job_id}` → event.set()

**`DELETE /api/catalog/{formula:path}?confirm=true`**
- N34: `confirm` param yoksa 400 döner ("Silmek için ?confirm=true ekleyin")

**`GET /api/system/kill_switch`**
- Response: `KillSwitchStatus { active: bool, activated_at: str?, reason: str?, ttl_remaining_h: float? }`
- 24 saatlik TTL — sona erdi ise dosya otomatik silinir

### 5.4 Job Registry detayı (`api/jobs.py`)
- `Job` dataclass: id, status (pending/running/done/error), progress, log_lines (deque), result, subscribers (list[asyncio.Queue]), `last_heartbeat`, `is_stale` (5dk eşiği — N31)
- Aktif: in-memory; tamamlananlar: SQLite (`data/jobs.db`)
- N29: `publish()` — QueueFull (subscriber yavaş) → subscriber list'ten silinir
- N44: `cleanup_old()` her flow başında çağrılır

---

## 6. `frontend/` — React/TS SPA

### 6.1 Stack
- **React 18 + TypeScript + Vite**
- **React Query** (`@tanstack/react-query`) — server state
- **Zustand** v5 (N36 sonrası) — client state (`workbenchStore`)
- **react-router-dom** v6
- **Visx** (D3 wrappers) — charts
- **Playwright** E2E (N39 sonrası)

### 6.2 Ekranlar (`src/screens/*`)

| Dosya | Yol | İşlev |
|-------|-----|-------|
| **`Workbench.tsx`** | `/workbench` | Ana ekran. Sol: katalog filtre. Orta: equity/drawdown/heatmap chart. Sağ: backtest + mining param form (Field type=number, N38). 19 useState → Zustand (N36). |
| **`Catalog.tsx`** | `/catalog` | Tüm alfa katalog tablosu, sort, filter, CSV export. Click → workbench. |
| **`BestAlphas.tsx`** | `/best` | En iyi formüller leaderboard. |
| **`BacktestStudio.tsx`** | `/backtest-studio` | Manuel formül girişi + tek tek backtest. |
| **`LLMTrainer.tsx`** | `/llm-trainer` | Tree-LSTM eğitim — buffer durumu, train_epochs slider, async job. |
| **`ResultsReport.tsx`** | `/results/:id` | Tek backtest detay raporu. |
| **`AtomsDemo.tsx`** | `/atoms` | Tasarım sistemi showcase. |

### 6.3 Hooks (`src/hooks/`)
- `useCatalog()` — `/api/catalog` GET, DELETE mutation, exportCsv
- `useFormula()` — `/api/formulas/evaluate` mutation
- `useJob(jobId)` — `/api/jobs/{id}` + WS subscription. State: `{ progress, logs, result, error, done, reconnecting }`. **N30**: WS URL `?api_key=...`
- `useMeta()` — `/api/meta` (split_date, train/test rows, benchmark_days)

### 6.4 Lib (`src/lib/`)
- `api.ts` — `apiFetch<T>(path, init)`, BASE = `VITE_API_URL`
- `ws.ts` — `connectJob(jobId, opts)`. Heartbeat 25s, reconnect retry (varsayılan 3)
- `window.ts` — `WINDOW_MAP`, `windowToLabel`

### 6.5 Store (`src/store/workbenchStore.ts`) — N36
**Zustand v5 `StateCreator<WorkbenchState>`**, 19 state slice:
- View/filter: `backtestWindow`, `filterText`, `sourceFilter`, `neutralize`
- Mining params: `mPopSize`, `mMaxK`, `mFolds`, `mEmbargo`, `mPurge`, `mLambdaStd`, `mLambdaCx`, `mLambdaSize`, `mSizeCorr`
- Backtest job: `jobId`, `isLaunching`, `launchError`
- Mining job: `miningJobId`, `miningLaunching`, `miningError`
+ 19 setter aksiyonu

### 6.6 Components
- `atoms/`: Btn, Pill, Stat, Field (N38: type=number), Check, SegRow, SectionLabel, Logo
- `inputs/`: Input, Select, Stepper, Note
- `chrome/`: CChrome (top/bottom bar layout), Panel, Box
- `charts/`: EquityChart, DrawdownChart, HeatmapRow (N35: NaN guard), MiniSparkline (N35: filter Number.isFinite)

### 6.7 E2E (`frontend/e2e/`) — N39
- `workbench.spec.ts` — 4 test (yükleme, filtre, segment etiketleri, Run Backtest button state)
- `catalog.spec.ts` — 3 test (yükleme, navigate, DELETE backend guard — backend yoksa skip)
- `ws_reconnect.spec.ts` — 2 test (graceful WS handling, reconnect flag UI)
- `playwright.config.ts`: baseURL=localhost:5173, Desktop Chrome, headless

---

## 7. `scripts/` — Yardımcı Komutlar

| Script | Görev |
|--------|-------|
| **`fetch_bist_data.py`** | yfinance EOD veri çekimi (10 yıl). Incremental mode: var olan parquet'e ekleme. |
| **`new_data.py`** | Master pipeline: yfinance + isyatirimhisse + TCMB EVDS (USD/TRY) + TEFAS. Çıktı: `market_db_master.parquet`. |
| **`inspect_data.py`** | Hızlı parquet inceleme — info, head, tail. |
| **`roni.py`** | Train period stats analizi (test split %70). |
| **`config.py`** | Ana config sistem (config.yaml opsiyonel). |
| **`backup_data.py`** | **N43**: restic + 30-snapshot retention + audit log. Cron: `0 3 * * *`. |
| **`dr_drill.py`** | **N42**: DR drill protokolü. RTO < 30dk hedef. `--check` (sadece doğrula), `--report` (son rapor). |
| **`calibrate_slippage.py`** | **N24**: Almgren-Chriss γ OLS kalibrasyonu. Çıktı: `slippage_calibration.json`. |

---

## 8. `auto_minerva.py` — Prefect Orkestrasyonu

### 8.1 Flow yapısı

```python
@flow(name="minerva_daily_cycle",
      on_failure=[_hook_flow_failed],
      on_completion=[_hook_flow_completed])
def run_daily_cycle(only_mining_on_weekday=4, mining_n_trials=50):
    task_cleanup_stale_jobs()                           # N44
    db_path = task_fetch_data()
    prob_path = task_detect_regime(db_path)
    mining_result = task_nightly_mining(db_path, prob_path,
                                         only_on_weekday=only_mining_on_weekday,
                                         n_trials=mining_n_trials)
    decay_result = task_decay_scan(prob_path)
    exec_result = task_morning_execution(db_path, prob_path)
    return {"mining": mining_result, "decay": decay_result, "execution": exec_result}
```

### 8.2 Task detayları

| Task | Retries | Timeout | Tetiklenir mi? |
|------|---------|---------|----------------|
| `task_fetch_data` | 2 | 600s | Her zaman |
| `task_detect_regime` | 2 | 600s | Her zaman |
| `task_nightly_mining` | 0 | **7200s** (N48) | **Sadece Cuma** (`only_on_weekday=4`, Istanbul TZ) |
| `task_decay_scan` | 0 | 300s | Her zaman |
| `task_morning_execution` | 0 | 600s | Her zaman |
| `task_cleanup_stale_jobs` | 0 | 60s | Her run başında (N44) |

### 8.3 Bildirimler (Telegram)
- `_hook_flow_failed`: 🚨 *MINERVA HATA* + flow_run + state + message
- `_hook_flow_completed`: 🟢 Günlük rapor (`_build_portfolio_report`)
- Rapor içeriği: bugünün portföyü, aktif şampiyonlar, net P&L, HMM rejim algısı

### 8.4 Cron deployment
```bash
prefect deployment build auto_minerva.py:run_daily_cycle \
  -n minerva-daily --cron "30 18 * * 1-5" --timezone "Europe/Istanbul"
prefect deployment apply ./run_daily_cycle-deployment.yaml
prefect agent start -q default
```

---

## 9. Veri Akışı Haritası (Data Flow)

```
yfinance ──► fetch_bist_data.py ──► data/market_db.parquet
                                          │
                                          ├──► factor_neutralize.py (signal cleansing)
                                          ├──► triple_barrier.py (labels + weights)
                                          ├──► meta_label.py (secondary model)
                                          │
                                          ├──► regime_detector.py ──► regime_hmm.pkl
                                          │                            regime_metadata.json
                                          │                            regime_prob_df.parquet
                                          │
                                          ├──► mining_runner.py ──► alpha_catalog.json
                                          │     (MCTS+CFG)             snapshots/*.json
                                          │     (replay_buffer.json)
                                          │
                                          ├──► backtest_engine.py ──► (in-memory metrics)
                                          │     wf_fitness, ensemble, pbo_cscv, dsr
                                          │
                                          ├──► meta_optimizer.py ──► best_params.json
                                          │     (Optuna)
                                          │
                                          ├──► decay_monitor.py ──► .kill_switch (trigger)
                                          │
                                          ├──► blender.py ──► (target_weights)
                                          │     + slippage.py
                                          │
                                          └──► paper_trader.py ──► paper_trades.parquet
                                                                   decisions_log.parquet
                                                                   (forensics.py)

api/jobs.py ──► jobs.db (SQLite, completed jobs)
api/jobs.py ──► WebSocket subscribers (in-memory)
backup_data.py ──► restic repository (external)
dr_drill.py ──► dr_drill_report.json
```

### Kritik veri dosyaları (`data/`)

| Dosya | Üreten | Tüketen |
|-------|--------|---------|
| `market_db.parquet` | `fetch_bist_data.py` | tüm engine, api/deps |
| `regime_hmm.pkl` | `regime_detector.py` | `meta_optimizer.py` |
| `regime_metadata.json` | `regime_detector.py` | `auto_minerva` |
| `regime_prob_df.parquet` | `regime_detector.py` | `weighted_fitness`, `blender`, `forensics` |
| `alpha_catalog.json` | `mining_runner.py`, `api/routes/mining` | `api/routes/catalog`, frontend |
| `best_params.json` | `meta_optimizer.py` | `mining_runner.py`, `auto_minerva` |
| `paper_trades.parquet` | `paper_trader.py` | `decay_monitor.py`, `calibrate_slippage.py` |
| `decisions_log.parquet` | `forensics.py` | (rapor + Telegram) |
| `replay_buffer.json` | `mining_runner.py`, `LLMTrainer` | `trainer.py`, `mcts.py` warm-start |
| `jobs.db` | `api/jobs.py` | restart sonrası job sorgu |
| `.kill_switch` | `paper_trader.py`, `api/system` | `paper_trader`, frontend |
| `dr_drill_report.json` | `scripts/dr_drill.py` | (operatör inceleme) |
| `slippage_calibration.json` | `scripts/calibrate_slippage.py` | `SlippageConfig` (manuel update) |
| `bist100.parquet` | `fetch_bist_data.py` | benchmark (`get_benchmark`) |

---

## 10. Test Suite ve Doğrulama

### 10.1 Python testleri (`tests/`)

| Dosya | Kapsam |
|-------|--------|
| `test_api_http.py` | FastAPI endpoint testleri (httpx ASGI) |
| `test_security.py` | **N59**: 20 güvenlik testi — XSS, SQL injection, path traversal, oversized payload, 405, API key enforcement, CORS preflight |
| `test_auto_minerva.py` | Prefect flow integration |
| `test_blender.py`, `test_capacity.py`, `test_decay_monitor.py`, `test_position_sizer.py`, `test_slippage.py` | Faz 4-5 unit testleri |
| `test_factor_neutralize.py`, `test_triple_barrier.py`, `test_meta_label.py` | Faz 1 data testleri |
| `test_mcts_integration.py`, `test_meta_optimizer.py`, `test_mode3.py` | Faz 3 algoritma testleri |
| `test_regime.py`, `test_regime_detector.py` | Rejim tespit |
| `test_pbo_cscv.py`, `test_deflated_sharpe.py`, `test_wf_fitness.py`, `test_weighted_fitness.py` | Doğrulama |
| `test_paper_trader.py`, `test_forensics.py` | Faz 5 execution |
| `test_integration.py`, `test_regression.py` | E2E + regression |

Çalıştırma: `pytest -v` veya `pytest tests/test_security.py -v`.

### 10.2 Frontend E2E (`frontend/e2e/`) — N39

```bash
cd frontend
npx playwright install chromium  # bir kez
npm run dev                       # ayrı terminal
npm run e2e                       # 9/9 geçer (backend açıkken)
```

Sonuç: 9/9 geçti — backend yoksa catalog backend-guard testi sessizce skip.

---

## 11. Bu Zamana Ne Yaptık (Sprint Tarihçesi)

### Sprint 0 — Faz 1-6 İlk İmplementasyon
- HMM rejim tespiti, MCTS+CFG, Tree-LSTM, walk-forward, PBO/CSCV, DSR
- Faz 4 risk modülleri (vol target, decay, capacity)
- Faz 5 execution (slippage, blender, paper trader, forensics)
- Faz 6 Prefect orkestrasyon
- Streamlit UI → React/TS SPA migration
- FastAPI backend + WebSocket job streaming

### Sprint 1 — Q1–Q20 + 15 madde (önceki audit)
İlk derin code review sonucu kapatılan kalite/performans maddeleri.

### Sprint 2 — N1–N60 zafiyet kapatma (57 madde)
Bu sprint'in büyük kısmı. Önemli olanlar:

**Veri & doğruluk:**
- N4: shares_outstanding eklendi
- N5: NaN dropna (factor_neutralize)
- N6: RobustScaler train-only fit
- N7: HMM `covariance_type="diag"`
- N8: triple-barrier end-window weights
- N9: yfinance exponential backoff + missing report
- N10: paper_trader >50% gross PnL guard

**Mining & validation:**
- N11: MCTS softmax PUCT priors
- N12: dead-branch prune
- N13: simulate_value sabit 0.5 (heuristic kaldırıldı)
- N15: replay buffer warm-start prior
- N16: size_corr_hard_limit 0.7 → 0.5
- N17: pbo_cscv n_slices 16 → 8
- N18: ensemble champion promotion entegre

**Risk & execution:**
- N19: capacity rolling+EWM blend
- N20: decay extreme outlier freeze (3.5σ)
- N21: position sizer EWMA+rolling vol
- N22: blender post-filter normalize
- N23: forensics weighted dominant champion

**API & Concurrency:**
- N25/N26/N27: kill-switch TTL + daily loss + cumulative drawdown
- N28: market_db mtime cache (lru_cache yerine)
- N29: dead subscriber removal
- N30: WS api_key auth
- N31: job heartbeat + 5dk stale tespiti
- N32/N60: cooperative mining cancellation
- N34: catalog DELETE confirm guard
- N35: chart NaN guards

**Frontend & DX:**
- N37: Btn loading state
- N38: Field type=number + Workbench numeric state

**Observability & DevOps:**
- N42: DR drill scripti
- N43: restic backup scripti
- N44: stale job cleanup task
- N45: replay buffer atomic write
- N46: alpha_catalog daily snapshot
- N48: nightly_mining timeout 2h
- N51/N54: Istanbul TZ + env-based slippage cap
- N57: replay buffer FileLock fallback
- N58: forensics _make_record TZ aware
- (N40 logger.py + N41 metrics.py yeni dosya)

### Sprint 3 — Son 3 madde
- **N1**: alpha_cfg evaluate cache 500-LRU
- **N14**: meta_optimizer.rolling_tune() + WindowTuneResult/RollingTuneResult
- **N24**: scripts/calibrate_slippage.py (yeni)
- **N36**: Workbench.tsx 19 useState → Zustand store
- **N39**: Playwright E2E (9 test, 3 spec)
- **N59**: tests/test_security.py (20 async test)

**Toplam kapatılan: Q1–Q20 + 60 N-zafiyeti = 80 madde.** 🎉

---

## 12. Bilinen Bug'lar & Sınırlamalar

### 12.1 Sıcak Bug — meta_optimizer compute_features tuple
**Durum:** ✅ Bu oturumda yakalandı ve düzeltildi.
**Sebep:** N6 fix `regime_detector.compute_features()` imzasını `(features, scaler)` tuple'a çevirdi; ancak `engine/strategies/meta_optimizer.py:80` hâlâ tek değer bekliyor → `AttributeError: 'tuple' object has no attribute 'values'`.
**Çözüm:** `features, _scaler = compute_features(df, cfg)` (commit pending).

### 12.2 Mevcut sınırlamalar
- **Multi-worker:** `get_market_db()` mtime cache module-level — Gunicorn 4 worker'da her worker ayrı cache. Düzgün shared memory için Redis (LT-14).
- **SQLite WAL:** `jobs.db` SQLite. 10+ concurrent worker'da kilitlenme riski (LT-14 PostgreSQL).
- **Frontend WS:** Reconnect retry varsayılan 3, kalıcı disconnect'te kullanıcı F5'e mecbur.
- **Real broker integration yok:** paper_trades.parquet'ten gerçek emir gönderim adaptörü yazılmamış (LT-2).
- **Causality-aware neutralization yok:** mevcut sistem rank-OLS + bin-demean. DML/causal forest yok (LT-7).

### 12.3 Test gereksinimleri
- `npx playwright install chromium` E2E için bir kez gerekir
- `restic` brew install — backup_data.py / dr_drill.py için
- `RESTIC_REPOSITORY` + `RESTIC_PASSWORD` env değişkenleri
- Prefect server: `prefect server start` gerekir (`auto_minerva.py` doğrudan çalışır)

---

## 13. Gelecek (LT-1 … LT-20)

| # | Hedef | Durum |
|---|-------|-------|
| **LT-1** | Walk-Forward Online HMM Refit (haftalık, Hungarian permutation) | ⚠️ kısmi (`align_regime_labels` var, cron yok) |
| **LT-2** | Real Order Execution (Garanti / İş Yatırım / AlgoLab) | ❌ |
| **LT-3** | Portfolio Optimization (mean-variance / risk-parity / Black-Litterman) | ❌ |
| **LT-4** | Distributed Mining (Ray/Dask) — 200 formül × 50 iter × 20 worker | ❌ |
| **LT-5** | Bayesian Tree-LSTM (MC Dropout / Variational) — uncertainty-aware | ❌ |
| **LT-6** | Tree Transformer encoder (long-range dependency) | ❌ |
| **LT-7** | Causality-aware factor neutralization (causal forest / DML) | ❌ |
| **LT-8** | Real-time streaming (WebSocket BIST → Kafka → incremental HMM) | ❌ |
| **LT-9** | Multi-universe (S&P 500, crypto, forex) | ❌ |
| **LT-10** | RL Position Sizer (PPO/SAC) — vol×regime×decay×dd state | ❌ |
| **LT-11** | Frontend Zustand state | ✅ N36 |
| **LT-12** | Frontend E2E (Playwright) | ✅ N39 |
| **LT-13** | Observability stack (Prometheus + Grafana + Loki + OTel) | ⚠️ temel altyapı (`logger.py`, `metrics.py`); Grafana/Loki kalan |
| **LT-14** | PostgreSQL migration (connection pool, partitioned tables, PITR) | ❌ |
| **LT-15** | Mining cancellation + checkpointing | ⚠️ cancellation var, checkpoint kalan |
| **LT-16** | Master kill-switch UI butonu | ⚠️ backend ✅, UI butonu kalan |
| **LT-17** | Stale-mining heartbeat | ✅ N31 + N44 |
| **LT-18** | Backup/DR pipeline (S3/B2 + quarterly drill) | ⚠️ temel ✅; external storage cron + auto drill kalan |
| **LT-19** | Catalog restore CLI (`alpha_catalog restore --ts ...`) | ❌ |
| **LT-20** | Slippage γ calibration script | ✅ N24 |

**Sprint 4 Önerisi:**
1. **LT-2** (broker integration) — production'a geçiş için kritik
2. **LT-13** (Grafana + Loki) — observability tam stack
3. **LT-1** (HMM refit cron) — rejim modeli güncel kalsın
4. **LT-19** (catalog restore CLI) — disaster recovery tam pipeline
5. **LT-16** (kill-switch UI butonu) — operatör için tek tıkla acil durum

---

## 14. Operasyonel Kılavuz

### 14.1 İlk kurulum
```bash
git clone <repo>
cd Minerva_v3_Studio
python3.14 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cd frontend && npm install && cd ..

# Veri:
python scripts/fetch_bist_data.py          # ~5 dk

# .env:
cp .env.example .env
# Düzenle: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, API_KEY,
#          RESTIC_REPOSITORY, RESTIC_PASSWORD
```

### 14.2 Manuel günlük koşum
```bash
source venv/bin/activate
prefect server start &              # 4200'de Prefect API
sleep 3
python auto_minerva.py              # tek seferlik flow
```

### 14.3 API + frontend dev
```bash
# Terminal 1
uvicorn api.main:app --reload --port 8000

# Terminal 2
cd frontend && npm run dev          # 5173

# Terminal 3 (opsiyonel)
cd frontend && npm run e2e          # Playwright
```

### 14.4 Yedek + DR
```bash
# Günlük yedek (cron 03:00)
python scripts/backup_data.py

# DR drill (3 ayda bir)
python scripts/dr_drill.py
python scripts/dr_drill.py --report   # son rapor
```

### 14.5 Slipaj kalibrasyonu (paper_trades olduktan sonra)
```bash
python scripts/calibrate_slippage.py
# → data/slippage_calibration.json
# → SlippageConfig.gamma manuel güncelleme
```

### 14.6 Prefect deployment (production)
```bash
prefect deployment build auto_minerva.py:run_daily_cycle \
  -n minerva-daily --cron "30 18 * * 1-5" --timezone "Europe/Istanbul"
prefect deployment apply ./run_daily_cycle-deployment.yaml
prefect agent start -q default
```

### 14.7 Acil durum (kill-switch)
```bash
# Aktive et
curl -X POST -H "X-Api-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"reason":"Manual emergency"}' \
  http://localhost:8000/api/system/kill_switch/activate

# Durum
curl http://localhost:8000/api/system/kill_switch

# Sıfırla (24h TTL'den önce)
curl -X POST -H "X-Api-Key: $API_KEY" \
  http://localhost:8000/api/system/kill_switch/reset
```

### 14.8 Mining iptali
```bash
JOB_ID=...   # /api/mining/start response'undan
curl -X POST http://localhost:8000/api/mining/cancel/$JOB_ID
```

### 14.9 Catalog snapshot kurtarma (manuel)
```bash
# Snapshots klasörüne bak
ls data/snapshots/alpha_catalog_*.json

# Geri yükle
cp data/snapshots/alpha_catalog_20260420_120000.json data/alpha_catalog.json
```

---

## 15. Hızlı Referans (Cheatsheet)

### Önemli dosya yolları
- `data/market_db.parquet` — ana fiyat DB
- `data/alpha_catalog.json` — keşfedilen formüller
- `data/regime_prob_df.parquet` — HMM olasılık matrisi
- `data/paper_trades.parquet` — günlük paper trade kayıtları
- `data/decisions_log.parquet` — forensics karar logları
- `data/.kill_switch` — acil durum bayrak dosyası
- `data/jobs.db` — SQLite job history

### Önemli env değişkenleri
- `API_KEY` — FastAPI X-Api-Key auth (yoksa dev-bypass)
- `MINERVA_API_KEY` — WS auth (yoksa dev-bypass)
- `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` — bildirimler
- `TELEGRAM_DISABLED=1` — test mode
- `RESTIC_REPOSITORY`, `RESTIC_PASSWORD` — yedek
- `SLIPPAGE_CAP_BPS` — paper trader slipaj tavanı
- `VITE_API_URL` — frontend backend URL
- `VITE_API_KEY` — frontend WS api_key

### Önemli sabit/eşik değerler
- HMM K range: 2..6, min 200 örnek/rejim
- MCTS PUCT: c=1.4, b_ref=8, max_K=15
- WF folds: 5, embargo=5, purge=10, size_corr<=0.5
- Vol target: 15% yıllık, scale clip [0.25, 4.0]
- Decay: δ=0.001, λ=0.05, N=5 ardışık, freeze>3.5σ
- Capacity: ADV %5, min 1M TL ADV
- Slippage: γ=0.10 (kalibrasyona göre)
- Kill-switch TTL: 24h
- Daily loss limit: -5%, cumulative DD limit: -10%
- Job stale: 5dk
- Eval cache: 500 entry LRU
- Alpha catalog: günlük snapshot rotation
- nightly_mining timeout: 7200s (2h)
- DR RTO target: <30dk

---

## Doküman Sonu

Bu dokümanı **tek başına** okuyan biri sistemin:
- Ne yaptığını ✅
- Hangi dosya neyi yazdığını ✅
- Akışın nasıl olduğunu ✅
- Bilinen bug'ları & sınırlamaları ✅
- Gelecek hedefleri ✅
- Nasıl çalıştırılacağını ✅

öğrenir.

**Toplam kapsam:** 87 Python dosyası + 30+ TS/TSX dosyası + 25 test dosyası + 9 E2E test + 3 dokümantasyon dosyası (bu dahil).

**Sprint sayacı:** Q1–Q20 (önceki audit) + N1–N60 (60 madde) + LT-11/12/17/20 (long-term tamamlananlar) = **84 kapatılan madde**.

**Hâlâ açık LT hedefleri:** 16 (production'a geçiş için sırasıyla LT-2, LT-13, LT-1 öncelikli).
