# Minerva v3 — Sistem Risk Denetimi (Faz 1-5 + ML + API)

> Üç paralel keşif ajanı + doğrudan kod incelemesi sonucu.
> **Amaç:** Üretim hazırlığı öncesi matematiksel, mühendislik ve mimari risk envanteri.

---

## Yönetici Özeti — Risk Skor Tablosu

| # | Risk | Katman | Önem | Onarım Maliyeti |
|---|---|---|---|---|
| 1 | **Slippage double-scaling** (backtest sayıları yanlış) | Faz 5.1 (math) | 🔴 KRİTİK | 1 saat |
| 2 | **Capacity binding-ticker portfolio_size hatası** | Faz 4.3 (math) | 🔴 KRİTİK | 1 saat |
| 3 | **API in-memory job registry** (crash → tüm işler kaybolur) | API | 🔴 KRİTİK | 1-2 gün (Celery+Redis) |
| 4 | **alpha_catalog.json / best_params.json yarış koşulu** | Persist | 🟠 YÜKSEK | 2 saat (fcntl) |
| 5 | **Vol-target NaN→0 fillna leverage şişirmesi** | Faz 4.1 (math) | 🟠 YÜKSEK | 30 dakika |
| 6 | **Tree-LSTM tamamen uyuyan dev** (MCTS heuristic value) | ML | 🟠 YÜKSEK | Faz 6 başlı başına |
| 7 | **MCTS subtree_prior normalize edilmiyor** | Faz 3 (math) | 🟡 ORTA | 30 dakika |
| 8 | **API auth yok** (lokal de olsa) | API | 🟡 ORTA | 2-4 saat (JWT) |
| 9 | **Frontend WS desync server restart sonrası** | UI | 🟡 ORTA | 1 saat |
| 10 | **Paper trader searchsorted business-day uyumsuzluğu** | Faz 5.3 (edge) | 🟢 DÜŞÜK | 30 dakika |

🔴 üretim blokeri, 🟠 kısa vade onarım, 🟡 orta vade, 🟢 cosmetic/edge.

---

## 1. Matematiksel Doğruluk (Faz 1-5)

### 🔴 KRİTİK 1 — Slippage Double-Scaling

**Konum:** [engine/core/backtest_engine.py:223-230](engine/core/backtest_engine.py)

```python
unit_slip = slip_arr[di, portfolio]               # zaten √(1/ADV) ölçeklenmiş
slip_cost = float(np.mean(unit_slip)) * np.sqrt(1.0 / max(size, 1)) * turnover_factor
```

**Sorun:** `build_slippage_matrix` zaten `γ·σ·√(1/ADV)·1e4` döndürüyor (birim hacim ref). Backtest'te ekstra `√(1/size)` çarpılması yanlış agregasyon — Almgren-Chriss'in karekök yapısı portföy seviyesinde toplanır, geometrik çarpılmaz.

**Doğru:** `slip_cost = mean(γ·σ_i·√(v_traded_i / ADV_i))` veya basitleştirilmiş:
```
slip_cost = mean(unit_slip) * sqrt(weight_per_ticker) * turnover_factor
         = mean(unit_slip) * sqrt(equity_TL / size) * turnover_factor   # eşit ağırlık
```

**Etki:** BIST sentetik random sinyalde Δ=-0.08% (küçük çünkü slip zaten küçük), ama **gerçek mining'de büyütülmüş cezalar mantıksız sinyal seçimine sebep olur**.

---

### 🔴 KRİTİK 2 — Capacity Binding-Ticker portfolio_size Hatası

**Konum:** [engine/risk/capacity.py:155](engine/risk/capacity.py)

```python
per_ticker_cap = tradable_adv * cfg.adv_pct_limit * cfg.portfolio_size
max_aum_TL = float(per_ticker_cap.min())
```

**Sorun:** Sinyal sadece M (≠ portfolio_size) hisseyi aktif tutuyorsa, AUM = `adv_pct × ADV × M` olmalı; `portfolio_size` (varsayılan 20) sabitiyle çarpılmak çoğu zaman yanlış.

**Senaryo:** Sinyal 5 hisse aktif, `portfolio_size=20` → kapasite **4× fazla** raporlanır. Production'da bu rakama bakarak fon büyüklüğü ayarlanırsa pozisyon kapasitesini aşar, slipaj patlar.

**Doğru:**
```python
n_active = active_signal.groupby(level="Date").size().median()
per_ticker_cap = tradable_adv * cfg.adv_pct_limit * n_active
```

---

### 🟠 YÜKSEK 3 — Vol-Target NaN→0 fillna Leverage Şişirmesi

**Konum:** [engine/core/backtest_engine.py:103](engine/core/backtest_engine.py)

```python
ret_wide = data.pivot_table(index="Date", columns="Ticker", values="Period_Ret")
scaled_wide = apply_vol_target(ret_wide.fillna(0.0), risk_cfg)
```

**Sorun:** Hisse henüz halka arz olmadığı/kotasyondan kalktığı dönem NaN getiri içerir. `fillna(0.0)` → rolling std düşük → vol-target scale yüksek → o hisseye **sahte yüksek leverage** uygulanır.

**Doğru:** `fillna` öncesi vol hesabı yap, ya da `min_periods=20` zorla:
```python
scaled_wide = apply_vol_target(ret_wide, risk_cfg)   # NaN'ları korumalı
scaled_wide = scaled_wide.fillna(0.0)                 # son aşamada doldur
```

---

### 🟡 ORTA 4 — MCTS subtree_prior Normalize Edilmiyor

**Konum:** [engine/strategies/mcts.py:65-66](engine/strategies/mcts.py)

```python
prior_bonus = self.subtree_prior.get(str(child.state), 0.0)
u = self.c_puct * bal * (child.P + prior_bonus) * math.sqrt(total_N + 1) / (1 + child.N)
```

**Sorun:** `subtree_prior` katalogdan gelen ham skorlar (`{str(tree): 0.5, ...}`) — ölçeklenmemiş. Eğer prior değeri çok büyükse `child.P + prior_bonus` patlar, exploration tamamen prior'a bağımlı hale gelir.

**Doğru:**
```python
max_prior = max(self.subtree_prior.values(), default=1.0) or 1.0
prior_bonus = self.subtree_prior.get(str(child.state), 0.0) / (1.0 + max_prior)
```

---

### 🟢 DÜŞÜK 5 — Paper Trader searchsorted Business-Day

**Konum:** [engine/execution/paper_trader.py:174](engine/execution/paper_trader.py)

```python
exit_date_idx = px_pivot.index.searchsorted(row_date) + cfg.hold_days
```

**Sorun:** `px_pivot.index` zaten BIST iş günleri olduğundan +hold_days iş günü atlar (doğru), AMA tatil günleri/seans yarımları için davranış docstring'de belirsiz. Edge case: `row_date` Cuma + tatil → exit Salı olur (3 gün ileri); test eder.

**Onarım:** `pd.bdate_range` kontrolü veya iş takvimi normalizasyonu.

---

### Faz 1-2 — Sağlam ✓

`regime_detector.py` ve `weighted_fitness.py` — denetimde kritik bulgu yok. Cosine similarity zero-guard'ı, üstel transform overflow korumalı, MultiIndex align doğru.

### Faz 3 — Mostly OK ✓ (sadece prior normalize)

Purged & embargoed K-fold doğru, fold sınırlarında label leakage uyarısı zaten var (`purge_horizon < return_window`).

### Faz 4 (decay_monitor) — Sağlam ✓

Page-Hinkley `μ - r - δ` yön doğru (strateji underperform → m artar), reset semantiği doğru.

### Faz 5 (blender) — Sağlam ✓

EMA smoothing turnover azaltırken signal lag yaratıyor (kabul edilebilir trade-off).

---

## 2. ML Katmanı — Tree-LSTM Uyuyan Dev

**Doğrulandı.** Kullanıcının tespiti birebir doğru:

### Bulgu

| Bileşen | Durum |
|---|---|
| `engine/ml/tree_lstm.py` — `PolicyValueNet` | Sınıf hazır, forward pass OK |
| `engine/ml/trainer.py` — `train_step`, `train_epochs` | MSE + KL loss hazır |
| `engine/ml/replay_buffer.py` — `Sample(Node, ic, visit_dist)` | Schema hazır |
| **Persisted checkpoint** | ❌ Disk'te `.pt` yok |
| **mining_runner.py → GrammarMCTS** | ❌ `value_fn=None` ile çağrılır ([mining_runner.py:191-200](engine/strategies/mining_runner.py)) |
| **replay_buffer populate** | ❌ Sadece `api/routes/backtest.py:73` (LLM Trainer UI) — mining hiç yazmıyor |
| **MCTS rollout fallback** | Operator çeşitliliği heuristic'i (`min(1.0, len(ops)/10.0)`) |

### Sonuç

MCTS şu an saf heuristic + subtree_prior ile koşuyor. Tree-LSTM dosyaları "dünyaya hazır boyamış araba" — yola çıkmadı. Faz 6'nın gerçek hedefi:

1. Mining sırasında her formülün `(tree, fold_mean_ric)` `replay_buffer`'a yazılmalı
2. Haftalık trainer çalışmalı, `models/tree_lstm_vk.pt` oluşturmalı
3. `mining_runner.py` checkpoint varsa `value_fn=lambda n: net.value(n)` olarak GrammarMCTS'e enjekte etmeli
4. AlphaZero loop kapanır: arama → eğitim → daha iyi arama

---

## 3. API & Mühendislik (Backend)

### 🔴 KRİTİK — In-Memory Job Registry

**Konum:** [api/jobs.py:87-93](api/jobs.py)

```python
_jobs: dict[str, dict] = {}            # RAM-only
```

**Sorun:** Worker process restart → 30 dakikalık MCTS mining ve 50 trial Optuna işi geri dönülmez kayıp.

**Çözüm önerisi:**
- Kısa vade: SQLite-backed job registry (dependency-free, atomic)
- Orta vade: Celery + Redis (production standart)
- Uzun vade: TaskIQ veya RQ (modern Python async-native)

### 🟠 YÜKSEK — File Write Race

**Konum:** [engine/core/alpha_catalog.py:77-80](engine/core/alpha_catalog.py), [engine/strategies/meta_optimizer.py:176](engine/strategies/meta_optimizer.py)

İki paralel iş aynı anda `_save_raw(records)` çağırırsa son yazan kazanır → kayıt kaybı.

**Çözüm:**
```python
import fcntl
with open(CATALOG_PATH, "r+", encoding="utf-8") as f:
    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    # read-modify-write
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
```

### 🟠 YÜKSEK — Event Loop Blocking

`asyncio.create_task(mining_runner)` ağır CPU işini event loop'ta tutuyor → diğer endpoint'ler latency artar veya timeout. `asyncio.to_thread` veya `concurrent.futures.ProcessPoolExecutor` gerekli.

### 🟡 ORTA — Auth Yok

[api/main.py](api/main.py) ve [api/deps.py](api/deps.py) hiçbir authentication katmanı içermiyor. Lokalde tehlike sınırlı, ama:

- Workspace paylaşılırsa (Docker-compose, dev tunnel) açık DoS vektörü
- "Heavy mining + optuna tuning" GPU/CPU sömürür
- Tutulan formüller gizli IP — başka kullanıcıdan korunmuyor

**Minimum:** Basic Auth + IP whitelist + rate limit middleware (slowapi).

### 🟡 ORTA — Job Cleanup Yok

[api/jobs.py:cleanup_old()](api/jobs.py) tanımlı ama hiçbir scheduler çağırmıyor — 1000 job sonra subscribers listesinde dangling queue, hafıza sızıntısı.

**Çözüm:** `asyncio.get_event_loop().call_later(600, _cleanup)` kurucu loop.

---

## 4. Frontend

### 🟡 ORTA — WebSocket State Desync

[frontend/src/hooks/useJob.ts](frontend/src/hooks/useJob.ts) — server restart → 404 → WS kapatır → UI `done: true` görür ama `result: null`. Kullanıcı "İş bitti ama sonuç yok" deadlock'una düşer.

**Çözüm:**
- Client tarafında localStorage job history (son 10)
- WS reconnect retry (exponential backoff)
- 404 görüldüğünde "Job kaybolmuş, yeniden başlat?" UX'i

---

## 5. Önerilen Onarım Sırası

### Faz 5.5 (1 hafta) — Üretim Blokerleri

| # | İş | Dosya | Süre |
|---|---|---|---|
| 1 | Slippage double-scaling fix | backtest_engine.py:223 | 1 saat |
| 2 | Capacity binding-ticker fix | capacity.py:155 | 1 saat |
| 3 | Vol-target NaN handling fix | backtest_engine.py:103 | 30 dk |
| 4 | MCTS prior normalize | mcts.py:65 | 30 dk |
| 5 | File locking (fcntl) | alpha_catalog.py + meta_optimizer.py | 2 saat |
| 6 | Job registry SQLite migration | api/jobs.py | 1 gün |
| 7 | Cleanup scheduler | api/jobs.py | 30 dk |
| 8 | Pytest re-run + smoke benchmark | - | 1 saat |

**Beklenen sonuç:** 229/229 test yeşil + math doğrulama smoke + crash-resume edilen API.

### Faz 6 (3-4 hafta) — Tree-LSTM AlphaZero Loop

1. Mining sırasında `replay_buffer.add(tree, mean_ric)` çağrı bağla
2. Haftalık `python -m engine.ml.trainer --epochs 50` cron
3. Checkpoint → `models/tree_lstm_v{N}.pt`
4. `mining_runner.py` checkpoint yüklerse `value_fn` enjekte
5. Optuna 6. parametre olarak Tree-LSTM weight (`alpha_lstm`)
6. A/B benchmark: heuristic vs Tree-LSTM 3 hafta gerçek BIST

### Faz 7 (2-3 hafta) — Production Hardening

- Celery + Redis migration (gerçek queue)
- JWT + rate limit
- Structured logging (loguru/structlog)
- Prometheus metrics + Grafana dashboard
- Frontend WS reconnect + history

---

## 6. Pozitif Bulgular (Güçlü Yönler)

- **Faz 1-2 matematik temiz** — HMM özellik mühendisliği, cosine guard, üstel transform numerik stabil
- **Pydantic input validation** sıkı (regex pattern'lar var)
- **SQL/path injection riski yok** — tüm `data/*.parquet` path'leri hardcoded
- **Test coverage iyi** — 229 test, ~102 sn full pass; her faz için ayrı test dosyaları
- **Look-ahead koruması bilinçli** — `shift(1)` ADV ve scale'de doğru uygulanmış
- **Modüler tasarım** — opt-in flag'ler ile her faz bağımsız test edilebilir
- **Şema versiyonlaması** — `CATALOG_SCHEMA_VERSION` + `_migrate_record` migration pipeline
- **Vol-target leverage clip** — `min_scale=0.1, max_scale=3.0` patolojik durumları engeller
- **Page-Hinkley σ pre-filter** — false-positive azaltır (BIST gibi gürültülü piyasada zorunlu)

---

## 7. Yapılmaması Gerekenler

- ❌ Vol-target ile slippage'ı aynı anda açıp gerçek paraya geçmek (math fix öncesi)
- ❌ `portfolio_capital_TL=10_000_000`+ ile capacity check yapmadan paper trade
- ❌ API'yi 0.0.0.0'a açmak (auth yok)
- ❌ Tek replay buffer ile birden fazla model paralel eğitmek (race)
- ❌ `ref_date=None` (default last day) ile WF backtest koşmak — look-ahead

---

**Son güncelleme:** 2026-04-28 — denetim 3 paralel ajan + doğrudan kod incelemesi.
**Toplam aktif risk:** 10 (2 🔴, 4 🟠, 3 🟡, 1 🟢).
