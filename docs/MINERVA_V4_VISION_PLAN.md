# Minerva v4 — Vizyon Raporu & Uygulama Planı

> **Versiyon:** 1.0
> **Hazırlayan:** Claude (Baş Sistem Mimarı)
> **Bağlam:** Minerva v3 Faz 1–6 tamamlandı. 10 yıllık BIST maratonu Oracle Cloud'da çalışıyor. Bu doküman, maraton sonrası "v4 Production-Grade" sürümüne geçiş için yol haritasıdır.
> **Bilgi tabanı:** NotebookLM "Minerva v3" notebook (52 akademik/teknik kaynak)

---

## Bölüm 1: STRATEJİK VİZYON

### 1.1 Mevcut Durumun Tespiti

**Güçlü yanlar:**
- HMM rejim tespiti (K=2 BIC-optimal)
- MCTS formül madenciliği + DML faktör nötralizasyonu
- Walk-forward fitness + purged k-fold
- Tree-LSTM + Bahdanau attention + MC dropout
- Risk parity (ERC) + Black-Litterman
- Minimal PPO RL position sizer
- Oracle Cloud 16 OCPU + 64 GB ARM altyapısı

**Sınırlamalar (CRO denetiminden):**
- Survivorship bias (delisting verisi yok)
- MCTS in-sample seçimi → multiple comparison problem
- TRY nominal getiriler reel getiriden kopuk
- BLAS thread oversubscription (OpenBLAS oversubscription)
- ERC dead code'du (sonra yamanmıştı), recent_ic=0.0 hardcoded'ydu
- DML cross-fit per-date loop performans katili
- Slipaj modeli sabit-bps, market-impact-aware değil
- Backtest overfit olasılığı sayısallaştırılmıyor
- RL ajanı pure Sharpe optimize ediyor, mode collapse riski

**Hedef:** v4 sürümünde bu sınırlamaların hepsini kapatmak ve sistemi **canlıya hazır** seviyeye taşımak.

---

### 1.2 Beş Eksen Vizyon

#### 🏗️ Eksen 1 — Mimari Evrim

**Tema:** Oracle 64GB sınıfında "research-grade" performans katmanı.

**Bileşenler:**
- **Apache Arrow memory-mapped DB** — worker'lar arası zero-copy view
- **DuckDB** — wf_fitness ve groupby aggregasyonları için vektörize SQL motoru
- **Polars** — pandas yerine Rust altyapı (5-30× hızlı)
- **Numba JIT** — formül evaluasyon kernel'leri
- **PostgreSQL + asyncpg** — sıcak state (RL agent, champion catalog, audit log)
- **Redis Streams** — günlük sinyal pub/sub

**Beklenen kazanım:** Mining 7 saat → 30 dk; günlük döngü 1.5 dk → 5 sn; 10 yıllık maraton 3-4 gün → 6-8 saat.

#### 🔥 Eksen 2 — İşkence Test Süiti

**Tema:** Hiçbir formül CSCV/PBO sertifikası olmadan canlıya alınamaz.

**Beş test:**
1. **CSCV + PBO** (Bailey, Borwein, Lopez de Prado, Zhu 2015) — backtest overfit olasılığı
2. **Slippage Sensitivity Curve** — break-even slippage
3. **Noise Injection Robustness** — fragile vs robust formül tespiti
4. **Adversarial Reverse Strategy** — anti-skill detection
5. **Block Bootstrap CI** — Sharpe istatistiksel anlamlılığı

**Doğrulama eşikleri:**
- PBO < 0.05 → ACCEPT
- 0.05 ≤ PBO < 0.5 → CAUTION (pozisyon × (1-PBO))
- PBO ≥ 0.5 → REJECT (canlıya alınamaz)

#### 🔬 Eksen 3 — Gölge Fon (Shadow Fund)

**Tema:** Broker'a bağlanmadan canlı piyasayı simüle eden 4 servis.

| Servis | Cron | Görev |
|---|---|---|
| `live_signal_daemon` | 18:15 IST | Gün sonu sinyal üretimi |
| `next_day_executor` | 09:55 IST | T+1 sanal yürütme + slipaj modeli |
| `daily_reconciler` | 18:30 IST | M2M, production parity testi, Telegram raporu |
| `live_dashboard` | always-on | Streamlit canlı izleme |

**Kritik özellik:** Production Parity Test — backtest motoru ile shadow fund günlük sapması ölçülür, > 50 bps aşımda alarm.

#### 🌌 Eksen 4 — Paralel Evrenler

**Tema:** Minerva tek strateji değil, alpha üretim platformu.

**Altı motor:**
1. **Hierarchical Multi-Agent RL** — Macro/Signal/Sizing/Execution ajanları
2. **Composite Reward Shaping** — turbulence + oracle deviation + cost
3. **VİOP Beta Hedger** — XU030 short ile beta-neutral
4. **KAP Event-Driven NLP** — Turkish FinBERT sentiment factor
5. **Crypto Funding Rate Carry** — 7/24 BTC perp/spot arbitrage
6. **Cointegration Pairs Trading** — sektör çiftleri (GARAN-AKBNK, THYAO-PGSUS)

#### 📚 Eksen 5 — Bilgi Katmanı

**Tema:** NotebookLM 52 kaynak ile teorik temellerin kapsamlı tutuluması.

Eklenen son 8 kaynak (kullanıcı tarafından):
- Almgren-Chriss optimal execution (`optliq.pdf`)
- Bryzgalova-Pelger-Zhu factor zoo (`Forest-Through-the-Trees`)
- Cartea-Jaimungal-Penalva HFT book (`9781107091146_frontmatter`)
- Engle-Granger cointegration (`cointegrationandECM`)
- Maillard-Roncalli ERC (`erc.pdf`)
- BIST anomaliler (`1-s2.0-S0957417415005187-main`)
- Time series structural breaks (`time-series.pdf`)
- Pesaran/Türkiye akademik kaynak (`893753483.pdf`)

---

## Bölüm 2: UYGULAMA PLANI (8 Hafta)

### Faz 0 — Maraton Sonu (Hafta 0, mevcut)

**Hedef:** 10 yıllık maraton tamamlanması, sonuçların değerlendirilmesi.

**Çıktılar:**
- `data/historical_paper_trades.parquet`
- `data/historical_equity.parquet`
- `data/historical_summary.json` (Sharpe, MDD, CAGR, IR, Sortino, Calmar, win rate)
- 40 quarterly catalog (`data/historical_catalogs/`)

**Karar noktası:** Eğer toplam Sharpe > 0 ise v4 yatırımı meşru. Değilse temel formula mining yaklaşımını gözden geçir.

---

### Faz 1 — Performans Katmanı (Hafta 1-2)

**Tema:** Oracle ARM üzerinde mevcut kodun 10× hızlanması.

#### 1.1 Apache Arrow Memory-Mapped DB
**Yeni dosya:** `engine/data/arrow_db.py`

```python
import pyarrow as pa, pyarrow.parquet as pq

class MarketDB:
    def __init__(self, parquet_path: str):
        self.table = pq.read_table(parquet_path, memory_map=True)

    def slice(self, start, end):
        mask = (self.table['Date'] >= start) & (self.table['Date'] <= end)
        return self.table.filter(mask)
```

**Değişiklik:** `engine/strategies/mcts_pool.py:167` — `pickle.dumps(db)` yerine Arrow Table referansı geçir.

**Kabul kriteri:** Trial başlatma overhead'i 200 ms → 5 ms, 50 trial × 200 ms = 10 sn kazanç.

#### 1.2 DuckDB Vektörize wf_fitness
**Yeni dosya:** `engine/validation/wf_fitness_duckdb.py`

```python
import duckdb

def compute_fold_ric(signals_df, returns_df, fold_dates):
    con = duckdb.connect(":memory:")
    con.register('sig', signals_df); con.register('ret', returns_df)
    return con.sql("""
        SELECT AVG(corr_spearman) FROM (
            SELECT Date, corr(rank(signal), rank(target)) AS corr_spearman
            FROM sig JOIN ret USING (Ticker, Date)
            WHERE Date IN ($fold_dates)
            GROUP BY Date
        )
    """).fetchone()[0]
```

**Değişiklik:** `engine/validation/wf_fitness.py:259-278` Python loop'u DuckDB'ye taşı.
**Kabul kriteri:** Formül başına fitness eval 2 sn → 70 ms (28× hızlanma).

#### 1.3 Polars DML Neutralization
**Yeni dosya:** `engine/data/factor_neutralize_polars.py`

Per-date Python loop yerine Polars groupby + lazy evaluation.
**Kabul kriteri:** Formül başına neutralization 1.5 sn → 100 ms (15× hızlanma).

#### 1.4 Numba JIT Operatör Kernel'leri
**Değişiklik:** `engine/core/alpha_cfg.py` — `Add`, `Mul`, `Corr`, `Rank`, `WMA`, `EMA` operatörleri `@njit(cache=True, parallel=True)` ile dekore edilir.
**Kabul kriteri:** Formül evaluation 5-15× hızlanma.

#### 1.5 Postgres Hot State + Redis Streams
**Yeni dosya:** `engine/state/pg_state.py`, `engine/state/redis_stream.py`

Tablolar:
```sql
CREATE TABLE rl_agent_state (...);
CREATE TABLE champion_formulas (...);
CREATE TABLE production_parity_log (...);
CREATE TABLE virtual_orders (...);
CREATE TABLE virtual_positions (...);
```

**Kabul kriteri:** Faz 3'teki shadow fund servislerinin tüm state ihtiyacını karşılar.

**Faz 1 toplam kazanım:** Maraton 3-4 gün → 6-8 saat.

---

### Faz 2 — İşkence Test Süiti (Hafta 3)

**Tema:** Backtest'in istatistiksel sertifikasyonu.

#### 2.1 CSCV / PBO Modülü
**Yeni dosya:** `engine/validation/cscv_pbo.py`

Bailey et al. 2015 Algorithm 2.3 birebir uygulanır:
1. Performance matrix M (T × N) kur
2. M'i S=16 alt-matrise böl
3. C(16,8) = 12,780 kombinasyon üret
4. Her kombinasyon için IS-best stratejinin OOS rank'ını ölç
5. Logit dağılımı çıkar, PBO = negatif logit oranı

**Çıktı:** `data/pbo_certification.json`

```json
{
  "pbo": 0.32,
  "logit_mean": 0.45,
  "n_combinations": 12780,
  "verdict": "CAUTION",
  "deflated_sharpe": 0.78,
  "n_trials_evaluated": 1500
}
```

**Karar mekanizması:** PBO < 0.05 → kabul; PBO < 0.5 → kullanıma izin (pozisyon × (1-PBO)); PBO ≥ 0.5 → red.

**Kritik uyarı (Bailey et al. madde 5):** PBO MCTS'in objective function'ı OLAMAZ. Sadece dış sertifikasyon.

#### 2.2 Slippage Sensitivity Curve
**Yeni dosya:** `engine/validation/slippage_sweep.py`

`paper_trades.parquet` üzerinden 0-50 bps arası 5 bps adımlarla CAGR/Sharpe yeniden hesaplanır.

**Çıktı:** `data/slippage_curve.csv`

#### 2.3 Noise Injection Test
**Yeni dosya:** `engine/validation/noise_injection.py`

Geçmiş fiyatlara `r_noise ~ N(0, k × σ_daily)` ekle, k ∈ {0, 0.05, 0.10, 0.20, 0.50}. Robustness skoru = `Sharpe(k=0.10) / Sharpe(k=0)`.

#### 2.4 Adversarial Reverse Test
Sinyalleri ters çevir, ters Sharpe'ı ölç. `forward > 0 AND reverse < -0.5` → real alpha.

#### 2.5 Block Bootstrap CI
30-günlük bloklar, 1000 resample. Sharpe %95 CI çıkar.

**Faz 2 toplam çıktı:** `data/stress_test_report.md` — tüm formülleri 5 testten geçirir, sertifika atar.

---

### Faz 3 — Shadow Fund Mikro-Mimari (Hafta 4-5)

**Tema:** 4 servisli, broker'a bağlı olmayan canlı simülasyon.

#### 3.1 Servis A — `live_signal_daemon`
**Yeni dosya:** `services/live_signal_daemon.py`

Cron: `15 18 * * 1-5` (her iş günü 18:15)

```python
async def daily_signal_pipeline():
    today_ohlcv = await fetch_ise_data()
    await append_to_market_db(today_ohlcv)
    await hmm_refitter.update(market_idx, today())
    champions = await load_certified_champions()  # PBO < 0.5
    weights = blend_regime_signals(champions, hmm_refitter.last_prob_df, market_db)
    weights = apply_erc_with_min_volume(weights, min_vol_TL=1_000_000)
    leverage = rl_agent.act(build_state())
    await persist_virtual_orders(weights * leverage, exec_date=next_trading_day())
    await redis.xadd('signals', {'date': today(), 'n': len(weights), 'lev': leverage})
```

#### 3.2 Servis B — `next_day_executor`
**Yeni dosya:** `services/next_day_executor.py`

Cron: `55 9 * * 1-5` (her iş günü 09:55)

Cont-Stoikov OFI tabanlı slipaj modeli:
```python
beta = c / market_depth ** lambda_param
slippage_pct = beta * order_size / market_depth
entry_px = open_price * (1 + slippage_pct + commission_pct)
```

**Kaynak:** `engine/execution/cont_stoikov_slippage.py` (Cont-Kukanov-Stoikov 2011 formülü).

#### 3.3 Servis C — `daily_reconciler`
**Yeni dosya:** `services/daily_reconciler.py`

Cron: `30 18 * * 1-5`

Fonksiyonlar:
1. Mark-to-market kapanış
2. Production Parity Test (backtest motoru ile sapma)
3. Telegram günlük rapor (PnL, equity, DD, leverage)
4. Postgres `production_parity_log`'a yaz

#### 3.4 Servis D — `live_dashboard`
**Yeni dosya:** `services/live_dashboard.py` (Streamlit)

Real-time Redis Streams consumer + Postgres queries. Equity curve, drawdown, signal heatmap, son 30 gün PBO trendi.

**Faz 3 kabul kriterleri:**
- 5 iş günü kesintisiz çalışma
- Production parity sapması her gün < 50 bps
- Telegram raporu zamanında geliyor

---

### Faz 4 — Multi-Agent RL Yeniden Tasarım (Hafta 6)

**Tema:** Pippas et al. (2025) önerilerini kodla.

#### 4.1 Hierarchical 4-Agent Architecture
**Yeni dosya:** `engine/risk/hierarchical_rl.py`

```
Macro Agent (haftalık)    → risk-on/off karar
    ↓
Signal Agent (günlük)     → top-K hisse seçimi
    ↓
Sizing Agent (günlük)     → kaldıraç (mevcut PPO genişletilir)
    ↓
Execution Agent (sub-day) → limit emir fiyatlama (canlı)
```

**Eğitim:** Lee et al. MAPS framework — her ajan bağımsız Q-learning, paylaşılan experience replay.

#### 4.2 Composite Reward Shaping
**Değişiklik:** `engine/risk/rl_sizer.py:100-120`

```python
def composite_reward(r_t, position_change, fees, oracle_action, action, turbulence):
    if turbulence > threshold:
        return -10.0  # forced cash
    profit = r_t / max(std_recent, 1e-6)
    cost   = -fees * abs(position_change)
    expert = -k * (action - oracle_action)**2
    return 0.6*profit + 0.2*cost + 0.2*expert
```

Oracle action: perfect-foresight optimal kaldıraç (geçmişe bakar, sadece eğitimde kullanılır).

#### 4.3 Turbulence Index State Genişletme
**Değişiklik:** `engine/risk/rl_sizer.py:STATE_DIM = 5 → 7`

Yeni state boyutları:
- `turbulence_index` (Yang et al. 2020) — covariance distance
- `regime_transition_prob` — HMM transition matrix

#### 4.4 Transfer Learning Hazırlık
NotebookLM RL Survey: BIST → Crypto cross-domain transfer için pre-trained encoder.

**Faz 4 kabul kriteri:** Hierarchical sistem mevcut tek-ajan PPO'yu Sharpe ve MDD'de geçer.

---

### Faz 5 — Paralel Sistemler (Hafta 7)

**Tema:** Bağımsız alpha motorları.

#### 5.1 VİOP Beta Hedger
**Yeni dosya:** `services/viop_hedger.py`

```python
def daily_hedge():
    portfolio_beta = compute_beta_vs_xu030(current_portfolio, lookback=60)
    contracts_to_short = -int(portfolio_value * portfolio_beta // contract_size_TL)
    submit_viop_order('XU030', side='SHORT', quantity=contracts_to_short)
```

#### 5.2 KAP Event-Driven NLP
**Yeni dosya:** `services/kap_sentiment.py`

```python
async def kap_listener():
    async for entry in kap_rss_stream():
        sentiment = finbert_turkish.predict(entry.text)
        if sentiment.score < -0.7 and entry.ticker in current_portfolio:
            await emergency_exit(entry.ticker, reason=f"KAP negative: {entry.title}")
```

**Model:** Custom fine-tuned Turkish FinBERT (hugging face base + 5K BIST KAP corpus).

#### 5.3 Crypto Funding Rate Carry
**Yeni dosya:** `services/crypto_carry.py`

Binance + Bybit + OKX cross-exchange. Funding > +0.05%/8h → spot long + perp short. Kelly position sizing.

#### 5.4 Cointegration Pairs
**Yeni dosya:** `services/pairs_trading.py`

Engle-Granger ile günlük kointegre çift taraması. z-score > 2 mean-reversion trade.

**Faz 5 kabul kriteri:** En az 2 paralel sistem 1 ay shadow mode'da çalışır, Minerva ile korelasyonu < 0.3.

---

### Faz 6 — Production Hardening (Hafta 8)

**Tema:** Canlıya gerçek geçiş hazırlığı.

#### 6.1 Comprehensive Test Coverage
- Unit tests: tüm yeni modüller için ≥ 90% coverage
- Integration tests: 4 servisin uçtan uca senaryo testi
- Chaos tests: random worker kill, ağ kesintisi, DB downtime

#### 6.2 Monitoring & Alerting
- Prometheus metrics export
- Grafana dashboard (Sharpe, DD, latency, error rate)
- PagerDuty / Telegram alert kuralları

#### 6.3 Disaster Recovery
- Postgres logical replication (Oracle ↔ backup VPS)
- Daily snapshot to S3-compatible object storage
- Recovery runbook (manuel adımlar dokümante)

#### 6.4 Compliance & Audit
- Tüm sinyal kararları immutable log'da (PostgreSQL append-only)
- Yıllık denetim raporu otomasyonu
- Algoritmik trading regülasyon kontrol listesi (BIST İstanbul)

---

## Bölüm 3: REFERANS KAYNAK HARİTASI

### Akademik Temeller (NotebookLM)

| Kaynak | Faz | Kullanım |
|---|---|---|
| López de Prado AFML (2018) | Tüm fazlar | Genel metodoloji bibliyografi |
| Chernozhukov DML (2018) | F1.3 | Faktör nötralizasyonu |
| Bailey et al. PBO (2015) | F2.1 | CSCV implementasyonu |
| Cont-Kukanov-Stoikov OFI (2011) | F3.2 | Slipaj modeli |
| Pippas RL Survey (2025) | F4 | Multi-agent ve reward shaping |
| Almgren-Chriss optliq | F3.2 | Optimal execution |
| Bryzgalova-Pelger-Zhu Forest | F1.3 | Faktör havuzu genişletme |
| Cartea HFT book | F4.1 | Execution agent teorisi |
| Engle-Granger cointegration | F5.4 | Pairs trading |
| Maillard-Roncalli ERC | F1 (mevcut) | Risk parity sertifikasyonu |
| Gu-Kelly-Xiu Empirical AP ML | F4 | ML-based factor construction |
| AlphaAgent LLM mining | F1 (gelecek) | Yeni nesil mining |
| Time series structural breaks | F4.3 | HMM rejim değişimi tespiti |
| BIST anomaliler | F4 | Calendar feature engineering |

### Mevcut Kod Tabanı Eşlemesi

| v4 Bileşeni | Mevcut Kod Yolu | Değişiklik |
|---|---|---|
| Arrow DB | `engine/strategies/mcts_pool.py:167` | pickle → Arrow Table |
| DuckDB fitness | `engine/validation/wf_fitness.py:259-278` | groupby loop → SQL |
| Polars neutralize | `engine/data/factor_neutralize.py:266` | per-date loop → lazy |
| Numba kernel | `engine/core/alpha_cfg.py` | @njit dekoratörleri |
| PG state | `engine/state/` (yeni dizin) | yeni katman |
| CSCV | `engine/validation/cscv_pbo.py` (yeni) | yeni modül |
| Hierarchical RL | `engine/risk/rl_sizer.py` | 5-dim → 7-dim state, 1 ajan → 4 ajan |
| Slippage | `engine/execution/slippage.py` | fixed-bps → Cont-Stoikov |
| Shadow fund | `services/` (yeni dizin) | 4 servis |

---

## Bölüm 4: BAŞARI METRİKLERİ

### Performans (kalite)
- 10 yıllık backtest annual Sharpe > 1.5
- Max drawdown < 25%
- PBO < 0.05 (en az 1 sertifikalı formül)
- Production parity sapması < 50 bps/gün
- Slippage sensitivity break-even > 30 bps

### Hız (verimlilik)
- Mining süresi: 7 saat → 30 dk (15×)
- Günlük döngü: 1.5 dk → 5 sn (18×)
- Daily reconciler latency < 2 dk

### Sağlamlık (güvenilirlik)
- 30 gün kesintisiz shadow mode
- Disaster recovery test: 4 saatte tam kurtarma
- Test coverage > 85%
- Zero data loss SLA

---

## Bölüm 5: RİSK REGİSTERİ

| Risk | Olasılık | Etki | Azaltma |
|---|---|---|---|
| Survivorship bias çözümsüz | Yüksek | Yüksek | Reel performans %30-50 düşük varsay |
| MCTS overfitting devam | Orta | Yüksek | CSCV/PBO sertifikasyonu zorunlu |
| ARM Linux performans sürprizleri | Orta | Orta | Faz 1 boyunca benchmark |
| Shadow fund canlıdan saparsa | Orta | Yüksek | Production Parity Test günlük |
| RL transfer öğrenme başarısız | Yüksek | Düşük | Tek-ajan PPO'yu fallback olarak tut |
| KAP NLP yanlış sentiment | Orta | Orta | Threshold yükselt (-0.85), human review |
| Crypto exchange downtime | Yüksek | Düşük | 3 exchange yedekli mimari |
| Türk regülasyon değişimi | Düşük | Yüksek | Compliance modülü esnek tasarım |

---

## Bölüm 6: SONRAKİ ADIMLAR

### Bugün (Maraton bitince)
1. `historical_summary.json` analizi (Sharpe, MDD, win rate)
2. CRO denetimi yenileme — gerçek sayılarla
3. Faz 1 başlangıç kararı (mantıklıysa Apache Arrow refactor başlat)

### Bu Hafta
1. Maraton bitiş raporu → NotebookLM'e ekle
2. Faz 1.1 (Arrow DB) prototip — tek branch'te dene
3. Faz 1.2 (DuckDB) benchmark — wf_fitness 50 formül için before/after

### Bu Ay
1. Faz 1 tamamen biter
2. Faz 2 (CSCV/PBO) implementasyonu başlar
3. İlk PBO sertifikasyon raporu çıkar

### 8 Hafta Sonu
1. Tüm 6 faz tamamlanmış
2. Shadow fund 30 gün canlı
3. Canlı broker bağlantısı (Algolab, IsBank Algotrade) için POC

---

## Bölüm 7: PROJE FELSEFESİ

> **"Gerilla Quant" yerine "Cerrahi Quant"**

Eski anayasa M2 Air kısıtlarına optimize edilmişti. Oracle 64GB üzerinde yeni felsefe:

1. **Sürat değil, sertifikasyon.** Hızlı koşan ama PBO > 0.5 olan strateji canlıya alınmaz.
2. **Karmaşıklık değil, kompozisyon.** Tek devasa MCTS yerine 6 küçük uzmanlaşmış motor.
3. **Otomasyon değil, gözetim.** Her servis Telegram raporu yazar, insan ortakta kalır.
4. **Optimizasyon değil, dayanıklılık.** PBO'yu hedef yapma — sadece dış denetim aracı.
5. **Tahmin değil, uyumluluk.** Survivorship bias çözülemiyorsa, gerçek getiriyi %30-50 aşağı varsay.

---

## Ekler

### Ek A: Kod Şablonları
Tüm yeni modüllerin iskelet kodu `docs/v4_templates/` dizininde tutulacak.

### Ek B: NotebookLM Sorguları
Her faz için NotebookLM'e sorulacak standart sorular `docs/v4_notebooklm_queries.md`'de.

### Ek C: Benchmark Sonuçları
Her faz öncesi/sonrası benchmark tabloları `docs/v4_benchmarks/` altında saklanacak.

---

**Son söz:** Bu doküman canlı bir belgedir. Maraton sonuçları geldikten sonra Bölüm 4 (Başarı Metrikleri) güncellenmelidir. Faz 1 başladıktan sonra her haftalık güncellemeyle bu dosya `git commit` edilmeli, evrimi izlenebilir kalmalıdır.

*Hazırlandığı tarih: 2026-05-08*
*Sonraki güncelleme: Maraton bitiş tarihi*
