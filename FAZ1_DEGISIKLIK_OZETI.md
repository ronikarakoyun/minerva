# Minerva v3 — Faz 1 Değişiklik Özeti & Code Review Raporu

**Dal:** `claude/strange-nobel-dca0c9`  
**Tarih:** 2026-05-05  
**Kapsam:** PR-1 → PR-6 (Temel Atma — Altyapı & Darboğaz Kırma)

---

## 1. YENİ DOSYALAR

### `engine/data/fracdiff.py`
**Amaç:** Fractional Differentiation (López de Prado, AFML §5) — log(Pclose) serisini minimum bilgi kaybıyla durağanlaştırır.

**API:**
- `frac_diff_ffd(series, d, thresh=1e-4)` — Sabit-pencere FFD; `sliding_window_view` ile O(n·k) zero-copy.
- `find_min_d(series, p_value_target=0.05)` — ADF grid arama; 500+ hisse için pahalı (bkz. cache).
- `load_fracdiff_cache(path)` / `save_fracdiff_cache(cache, path)` — Atomik JSON cache okuma/yazma.
- `should_recompute_d(ticker, cache, today_close, force)` — 4 kural: 90 gün, %50 fiyat şoku, yeni hisse, force.
- `update_fracdiff_cache_entry(cache, ticker, series, today_close)` — Tek ticker güncelle.

**Cache formatı (`data/fracdiff_d.json`):**
```json
{
  "THYAO.IS": {
    "d_star": 0.45,
    "computed_at": "2026-02-04T00:00:00+00:00",
    "ref_close": 312.50,
    "n_obs": 2487
  }
}
```

---

### `engine/data/db/__init__.py`
Boş paket init; `engine.data.db.postgres.MinervaDB`'yi dışa aktarır.

---

### `engine/data/db/config.py`
**Amaç:** PgBouncer ↔ asyncpg pool boyutları için TEK KAYNAK.

**Kritik sabitler:**
| Sabit | Değer | Açıklama |
|---|---|---|
| `PG_BOUNCER_MAX_CLIENT_CONN` | 100 | Uygulama → PgBouncer max |
| `PG_BOUNCER_DEFAULT_POOL` | 20 | PgBouncer → Postgres max aktif |
| `PG_BOUNCER_MAX_DB_CONNECTIONS` | 25 | default_pool + reserve (tek kaynak) |
| `PG_MAX_CONNECTIONS` | 50 | Postgres max_connections |
| `N_APP_WORKERS` | 5 (env: MINERVA_WORKERS) | uvicorn worker sayısı |
| `ASYNCPG_PER_WORKER_MAX` | 4 | `floor(20/5)` — pool invariant |
| `ASYNCPG_STATEMENT_CACHE` | 0 | **ZORUNLU**: PgBouncer transaction-mode |
| `ASYNCPG_COMMAND_TIMEOUT_S` | 10.0 | Asılı query kesme |

**`assert_pool_invariant()`:** `ASYNCPG_PER_WORKER_MAX × N_APP_WORKERS ≤ PG_BOUNCER_DEFAULT_POOL` ve `ASYNCPG_STATEMENT_CACHE == 0` — startup'ta `RuntimeError` fırlatır.

---

### `engine/data/db/postgres.py`
**Amaç:** PgBouncer-arkalı asyncpg singleton bağlantı havuzu.

**Sınıf: `MinervaDB`**
- `init(dsn, min_size, max_size)` — Pool oluştur; `assert_pool_invariant()` çağırır.
- `close()` — Pool kapat.
- `is_ready()` — Başlatılmış mı?
- `conn()` — `@asynccontextmanager` → `asyncpg.Connection` verir.
- `transaction()` — `@asynccontextmanager` → otomatik COMMIT/ROLLBACK.

**`_redact_dsn(dsn)`:** Log'da şifreyi `***` ile maskeler.

---

### `engine/data/db/schema.sql`
**Tablolar:**
| Tablo | Amaç | PK Tipi |
|---|---|---|
| `jobs` | API job kuyruğu (SQLite yerine) | `TEXT uuid` |
| `alpha_catalog` | Keşfedilen formüller | `TEXT gen_random_uuid()` |
| `decisions_log` | Trading karar logu | `BIGINT IDENTITY` |
| `paper_trades` | Paper trading PnL kaydı | `BIGINT IDENTITY` |

**Özellikler:** JSONB metadata, BRIN-friendly tarih index'leri, FK `ON DELETE SET NULL`, `pgcrypto` uzantısı.

---

### `docker-compose.yml`
- `postgres:16-alpine`: `max_connections=50`, `shared_buffers=128MB`, `wal_level=minimal`
- `bitnami/pgbouncer:1.23.1`: transaction mode, `config.py` değerleriyle eşleşen env var'lar (yorumlarda belirtilmiş)
- Her iki servis için `healthcheck` + `depends_on: condition: service_healthy`

---

### `engine/util/__init__.py`
Boş paket init.

---

### `engine/util/shm_registry.py`
**Amaç:** Açık SharedMemory bloklarını `data/shm_registry.json`'a kaydet; proses ölümünde orphan temizle.

**API:**
- `register(name, path)` — SHM bloğunu PID ile kaydet.
- `unregister(name, path)` — Kayıt sil.
- `cleanup_stale(path, dry_run)` — Ölü PID'lere ait blokları `unlink` et; temizlenenleri döner.
- `_pid_alive(pid)` — `os.kill(pid, 0)` ile PID kontrolü.
- CLI: `python -m engine.util.shm_registry` — manuel orphan temizliği.

---

### `engine/strategies/mcts_pool.py`
**Amaç:** market_db'yi tek `SharedMemory` bloğunda tut; N worker paralel MCTS.

**API:**
- `shared_array(arr_np)` — RAII context manager; 3 katmanlı zombie koruması:
  1. `try/finally`
  2. `atexit.register(_cleanup)`
  3. SIGINT/SIGTERM handler (prev_int restore edildikten sonra `raise_signal`)
- `_worker_init(shm_name, shape, dtype, columns, index)` — Worker process'te SHM'e attach; global `_worker_db` DataFrame.
- `_worker_run_trial(...)` — Worker'da tek MCTS trial.
- `run_parallel_mining(db, alpha_cfg, mining_cfg, prob_df, n_workers=4, n_trials=None)` — Orchestrator; `mp.get_context("spawn")` (macOS fork güvensiz).

---

### `engine/util/checkpoint.py`
**Amaç:** Uzun mining çalışmalarını kaydet/devam ettir.

**Sınıflar:**
- `CheckpointState` (dataclass): `checkpoint_id`, `done_i`, `pool`, `results`, `rng_state`, `np_rng_state`, `saved_at`, `metadata`
- `Checkpoint`:
  - `new(id, dir)` / `load(id, dir)` — Constructor'lar
  - `save(done_i, pool, results, rng_state, np_rng_state)` — Atomik yazma (`.tmp` → `.pkl`)
  - `read()` — Deserialize
  - `exists()`, `delete()`, `list_all(dir)` — Yardımcılar

**Checkpoint dizini:** `data/mining_checkpoints/<id>.pkl`

---

## 2. DEĞİŞTİRİLEN DOSYALAR

### `engine/data/regime_detector.py`
**Eklenen:**
- `RegimeConfig.use_fracdiff: bool = False` — opt-in flag
- `RegimeConfig.fracdiff_cache_path: Path = Path("data/fracdiff_d.json")`
- `compute_features()` içinde: `use_fracdiff=True` ise `data/fracdiff_d.json`'dan `d_star` okur, `frac_diff_ffd(log(Close), d=d_star)` ile `Log_Return` üretir.
- `compute_features()` artık `tuple[pd.DataFrame, RobustScaler]` döner (N6: scaler `train_end_date`'e fit).

### `engine/data/triple_barrier.py`
**Düzeltilen (N8):**
- `return_weights=False` modunda son `horizon` gün artık dışarıda (look-ahead-safe).
- `return label_series[weight_series > 0]` — sadece tamamlanmış bariyerli etiketler döner.
- `return_weights=True` modunda ise son horizon günler `TB_Weight=0` ile dahil — ML eğitiminde `sample_weight` kullanımı için.

### `engine/strategies/mining_runner.py`
**Eklenen:**
- `import logging; logger = logging.getLogger(__name__)`
- `from ..util.checkpoint import Checkpoint`
- `run_mining_window()` parametreleri: `checkpoint_id`, `checkpoint_every=50`, `resume=False`
- `_run_mining_window_impl()` içinde resume bloku: `state = ckpt.read()` → `start_from`, `results`, RNG state geri yükle.
- Döngü her `done_i`'da checkpoint tetikler (`done_i < start_from` ile atlanan iterasyonlar da sayılır).

**Refactor:**
- `continue` kullanımı yerine `stats = None` pattern — checkpoint'in her `done_i`'da çalışması için döngü yapısı yeniden düzenlendi.

### `api/jobs.py`
**Kaldırılan:**
- SQLite bağımlılığı (`sqlite3`, `_DB_PATH`, `_init_db()`, `_db()` context manager)

**Eklenen:**
- `_db()` lazy import helper (circular import koruması)
- `_persist_job(job)` async — PG INSERT ON CONFLICT DO UPDATE
- `_load_job_from_pg(jid)` async — PG'den job yükle
- `_cleanup_old_pg(keep)` async — eski kayıtları temizle
- `JobRegistry.get(jid)` async — önce in-memory, sonra PG
- `JobRegistry.get_sync(jid)` sync — sadece in-memory
- `Job.last_heartbeat`, `Job.touch()`, `Job.is_stale` — N31 stale detection
- `JobRegistry._cleanup_old()` stale job tespiti + PG temizlik

### `api/main.py`
**Eklenen:**
- `from contextlib import asynccontextmanager`
- `lifespan()` async context manager: startup'ta `MinervaDB.init()` (PG yoksa warn + devam), shutdown'da `MinervaDB.close()`
- `app = FastAPI(lifespan=lifespan, ...)`

### `requirements.txt`
**Eklenen:**
- `statsmodels>=0.14`
- `asyncpg>=0.29`

### `tests/test_regime_detector.py`
**Düzeltilen:**
- 6 test: `compute_features()` tuple döndüğü için `feats, _ = compute_features(...)` ile unpack.
- `assert isinstance(scaler, RobustScaler)` eklendi.

### `tests/test_decay_monitor.py`
**Düzeltilen:**
- `test_scan_decay_finds_first_breach_date`: `DecayConfig(extreme_sigma_cap=5.0)` eklendi. Sebep: varsayılan cap=3.5σ, test 4σ kullanıyordu → counter donuyordu.

---

## 3. SİLİNEN DOSYALAR

### `scripts/roni.py`
**Sebep:** 30 satırlık ad-hoc keşif scripti (train split istatistikleri). Üretim akışında hiçbir çağıran yoktu.

---

## 4. YENİ TEST DOSYALARI

| Test Dosyası | Kapsam | Test Sayısı |
|---|---|---|
| `tests/test_fracdiff.py` | d=0 kimlik, d=1 fark, durağanlık, range guard, ağırlık monotonluğu, NaN yayılımı | 6 |
| `tests/test_mcts_pool.py` | shared_array data integrity, cleanup, large float32, register/unregister, stale PID, pid_alive | 7 |
| `tests/test_postgres_layer.py` | Pool invariant, statement_cache=0, bağlantı limiti, DSN redact, init state, conn raises (+ Docker: ping, 50 concurrent insert, rollback, schema) | 7 statik + 4 live |
| `tests/test_checkpoint.py` | save/load round-trip, exists, delete, FileNotFoundError, list_all, RNG restore, start_from aritmetiği | 7 |

**Toplam yeni test: 27**

---

## 5. CODE REVIEW — TESPİT EDİLEN HATALAR VE DÜZELTMELERİ

### BUG-1 (Orta) — Checkpoint yalnızca başarılı formüllerde tetikleniyordu
**Konum:** `engine/strategies/mining_runner.py`  
**Sorun:** `ckpt.save()` çağrısı `results.append()` bloğu içindeydi. Mining'de formüllerin %95'i filtreden geçemezse (düşük IC), 50 başarılı formül asla birikmez → checkpoint hiç alınmaz → hata durumunda tüm iş kaybedilir.  
**Düzeltme:** Döngü yapısı `continue` → `stats = None` olarak refactor edildi. Checkpoint artık her `done_i`'da (başarısız formüller de sayılarak) tetiklenir.

### BUG-2 (Düşük-Orta) — `fracdiff` cache'de `computed_at` eksik entry'ler için recompute tetiklenmiyordu
**Konum:** `engine/data/fracdiff.py::should_recompute_d()`  
**Sorun:** `computed_at_str = entry.get("computed_at")` → `None` ise `if computed_at_str:` bloğu atlanır, `today_close=None` ise fiyat şoku da atlanır → recompute=False döner. Eski format cache entry'leri veya manuel düzenlenen girişler hiç güncellenmez.  
**Düzeltme:** `if not computed_at_str: return True` — `computed_at` alanı yoksa hemen recompute tetiklenir.

### BUG-3 (Düşük) — `_cleanup_old_pg` SQL sorgusunda f-string
**Konum:** `api/jobs.py::_cleanup_old_pg()`  
**Sorun:** `OFFSET {keep}` — `keep` sabit bir `int` olduğu için gerçek SQL injection riski yok, ancak parametre binding standartlarına aykırı; gelecekte değişken bir değer geçilirse güvensiz hale gelebilir.  
**Düzeltme:** `OFFSET $1` parametrize form + `c.fetchval(query, keep)`.

### BUG-4 (Düşük) — `docker-compose.yml` `PGBOUNCER_MAX_DB_CONNECTIONS=25` tek kaynak dışında tanımlıydı
**Konum:** `docker-compose.yml`  
**Sorun:** `config.py` "tek kaynak" vaadini dile getiriyor ama `PGBOUNCER_MAX_DB_CONNECTIONS=25` sabiti `config.py`'de yoktu. Birinin değiştirilmesi diğerinin geride kalmasına yol açardı.  
**Düzeltme:** `config.py`'e `PG_BOUNCER_MAX_DB_CONNECTIONS = PG_BOUNCER_DEFAULT_POOL + 5` eklendi; `docker-compose.yml`'e kaynak referans yorumu eklendi.

### BUG-5 (Düşük) — SIGINT handler'da prev_int restore edilmeden `raise_signal`
**Konum:** `engine/strategies/mcts_pool.py::shared_array()`  
**Sorun:** Lambda `(_cleanup(), signal.raise_signal(SIGINT))` çalışırken `prev_int` henüz restore edilmemişti. CPython sinyal masklama nedeniyle özyinelemeli döngü oluşmuyor, ancak Ctrl+C basıldığında `prev_int` (genellikle `KeyboardInterrupt`) tetiklenmeden önce 2. bir SIGINT gelirse davranış tanımsız.  
**Düzeltme:** Lambda yerine explicit `_handle_sigint()` fonksiyonu; `signal.signal(SIGINT, prev_int)` restore edildikten sonra `raise_signal(SIGINT)` çağrılıyor.

### BUG-6 (Kozmetik) — `mining_runner.py`'de `import logging as _logging` tekrarı
**Konum:** `engine/strategies/mining_runner.py`  
**Sorun:** Modül kapsamında `logger` yoktu; checkpoint resume bloğunda `import logging as _logging` ile geçici logger yaratılıyordu.  
**Düzeltme:** Modül başına `import logging; logger = logging.getLogger(__name__)` eklendi; checkpoint bloğunda `logger` kullanıldı.

---

## 6. KOD KALİTE DEĞERLENDİRMESİ (İncelenip Temiz Bulunanlar)

| Alan | Bulgu |
|---|---|
| `fracdiff._ffd_weights(d=0)` | `d=0` → `w=[1.0]` — köşe durum doğru |
| `fracdiff` SHM adı uzunluğu | `minerva_mcts_<8hex>` = 21 karakter — macOS POSIX 31 sınırının altında ✓ |
| `shm_registry.cleanup_stale(dry_run=True)` | `reg.pop` yapılmaz, `cleaned`'e eklenir — doğru ✓ |
| `checkpoint` + pool deterministik | `mining_cfg.seed` sabit → GP/MCTS aynı pool üretir → `start_from`'dan devam tutarlı ✓ |
| `_load_job_from_pg` asyncpg tipi | `created_at` DOUBLE PRECISION → asyncpg float olarak döner → `job._created_at = row["created_at"]` ✓ |
| `docker-compose.yml` sağlık kontrolleri | Her iki servis `healthcheck` + `depends_on: condition: service_healthy` ✓ |
| `schema.sql` indeksler | `decisions_log`, `paper_trades` üzerinde `(trade_date DESC)` ve `(ticker, trade_date DESC)` — BIST'te çok kullanılacak sorgular için doğru ✓ |
| `MinervaDB.transaction()` | `c.transaction()` otomatik COMMIT/ROLLBACK — asyncpg semantiği doğru ✓ |
| `triple_barrier` `return_weights=False` | `label_series[weight_series > 0]` — look-ahead contamination önlendi ✓ |

---

## 7. KALAN AÇIK KONULAR (Faz 2'ye ertelendi)

| # | Konu | Konum |
|---|---|---|
| A | `scripts/fetch_bist_data.py` ve `engine/data/regime.py` — ölü kod silimi | Faz 2'ye ertelendi (kullanıcı kararı) |
| B | `test_regression.py` flakiness — numpy global state kirliliği (pre-existing) | `tests/test_regression.py` — Faz 2'de conftest ile izolasyon |
| C | `test_security.py` — `pytest_asyncio` bağımlılığı eksik (pre-existing) | venv'e `pytest-asyncio` ekle |
| D | `fracdiff_d.json` refresh scheduled task | `scripts/refresh_fracdiff_d.py` — Faz 2 Prefect task |
| E | `alpha_catalog.py::utcnow()` deprecation uyarısı | `datetime.now(UTC)` ile değiştir |

---

## 8. TEST SONUÇLARI

```
262 passed, 4 skipped, 0 failed
(test_security.py: pytest_asyncio eksik — pre-existing, Faz 1 değişikliklerimizle ilgisiz)
```

**Önceki durum (main branch):** 8 failing test  
**Faz 1 sonrası:** 0 failing test (8 bug fix dahil + 27 yeni test)
