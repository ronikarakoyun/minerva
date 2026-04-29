# Minerva v3 — Sistem Akışı

İki taraf var: **SİSTEM** kendi kendine ne yapıyor, **SEN** ne yapıyorsun.

---

## A. NORMAL GÜN (Pazartesi-Perşembe)

```
SAAT     SİSTEM                         SEN
─────    ──────────────────────────     ────────────────────────
00:00    uyku (cron beklemede)          uyku
06:00    uyku                           uyku
09:00    uyku                           ☕ kahve + telefon kontrol
                                        → 🟢 mesaj geldi mi?
                                        → geldiyse her şey OK, devam
                                        → gelmediyse aşağı bak

10:00    BIST açıldı, sistem atıl        normal hayat
12:00    atıl                           normal hayat
15:00    atıl                           normal hayat
18:00    BIST kapandı, sistem bekliyor   normal hayat

18:30    🚀 OTOMATİK KOŞU BAŞLADI
         ├─ yfinance'den bugünün
         │   verisi çekildi
         ├─ HMM güncellendi
         ├─ Mining skip (Cuma değil)
         ├─ Şampiyonlar kontrol edildi
         └─ Yarın için target weights
            hesaplandı, kaydedildi

18:35    📱 🟢 Günaydın mesajı atıldı   telefon: 🟢 mesaj geldi
                                         → "tamam, sistem çalışıyor"
                                         → uyu
22:00    uyku                            uyku
```

**Senin tek işin:** Sabah telefondaki mesaja bakmak. Mesaj geldiyse o gün hiçbir şey yapma.

---

## B. CUMA — MINING GECESİ

Aynı akış, **TEK FARK:** 18:30'da mining 30-45 dakika sürer.

```
18:30    OTOMATİK KOŞU BAŞLADI
         ├─ veri çekildi (~30 sn)
         ├─ HMM (~3 sn)
         ├─ Mining 50 trial Optuna   ← BU UZUN SÜRER
         │   (yeni formüller aranır)
19:15    Decay scan + execution
19:20    📱 🟢 Günaydın mesajı
```

**Senin işin:** Aynı — sabah telefon kontrol. Mesaj 19:30'a kadar gelmezse panik yapma, mining yavaş.

---

## C. HAFTASONU

```
Cumartesi-Pazar:
SİSTEM: uyku, hiçbir şey yapmaz (cron sadece iş günleri)
SEN:    Pazar 5 dk haftalık özet bak (opsiyonel)
```

---

## D. SİSTEMİN AÇIK KALMASI GEREKEN ŞEYLER

3 terminal **kapanmamalı**, Mac **uyumamalı**:

```
Terminal A: prefect server start          ← UI ve cron motoru
Terminal B: ...serve(...)                 ← gece çalıştıracak işçi
Terminal D: caffeinate -dim               ← Mac uyutmaz
```

Bunlardan biri kapanırsa **gece otomatik koşu olmaz**. Cuma akşam kapatırsan Pazartesi sabah mesaj gelmez.

Kontrol etmek için (10 sn):
```bash
ps aux | grep -E "prefect server|serve|caffeinate" | grep -v grep | wc -l
```
Cevap **3** gelmeli.

---

## E. NE ZAMAN MÜDAHALE EDERSİN?

Sadece 4 durum var. Diğer her şey otomatik.

### 1️⃣ "🟢 Günaydın" mesajı GELMEDİ

```
1. Telefon: hiç mesaj yok
2. Bilgisayar: 3 terminal açık mı?
3. Açıksa Terminal B'ye bak, son 20 satır neyi söylüyor?
4. yfinance hatası varsa: 30 dk bekle, kendiliğinden tekrar dener
5. Başka hata varsa: Telegram'da 🚨 mesajı görmüş olman lazım
```

### 2️⃣ "🚨 MINERVA HATA" mesajı GELDİ

Telegram'daki mesaj nedeni söyler. Sık karşılaşılanlar:

| Mesajda yazıyor | Anlamı | Sen ne yaparsın |
|---|---|---|
| `yfinance ... timeout` | İnternet/yfinance sorunu | 30 dk bekle, sistem otomatik 3 kez tekrar dener |
| `no_catalog` | alpha_catalog.json yok | Mining'i elle koş (aşağıda) |
| `no_champions` | Şampiyon yok | Mining elle koş + şampiyon ata (aşağıda) |
| Diğer | Bilinmeyen | Terminal B logu paylaş |

### 3️⃣ "⚠️ ALPHA DECAY" mesajı GELDİ

Şampiyon formül artık çalışmıyor — yenisini bulmak gerek.

```bash
# 1) Yeni mining koş (~10 dk)
cd ~/Minerva_v3_Studio && source venv/bin/activate
venv/bin/python <<'EOF'
import pandas as pd
from engine.core.alpha_cfg import AlphaCFG
from engine.core.alpha_catalog import save_regime_champion
from engine.core.formula_parser import parse_formula
from engine.strategies.mining_runner import MiningConfig, run_mining_window

db = pd.read_parquet("data/market_db.parquet")
prob_df = pd.read_parquet("data/regime_prob_df.parquet")
unique_dates = sorted(db["Date"].unique())
db_window = db[db["Date"] >= unique_dates[-252]].copy()

cfg = AlphaCFG()
mcfg = MiningConfig.from_best_params("data/best_params.json", num_gen=80, prob_df=prob_df)
results = run_mining_window(db_window, cfg, mcfg)
top = sorted(results, key=lambda r: r.fitness, reverse=True)[:2]
for k, r in enumerate(top):
    tree = parse_formula(r.formula, cfg)
    save_regime_champion(k, r.formula, tree, r.mean_ric, r.mean_ric, r.mean_ric)
    print(f"Rejim {k} → {r.formula[:80]}")
EOF
```

Sonraki 18:30 koşusunda yeni şampiyon kullanılır.

### 4️⃣ Bilgisayarı kapatman gerekti (taşıma, tatil)

```
Çıkmadan:
1. Terminal B'de Ctrl+C
2. Bilgisayarı kapat

Geri döndüğünde:
1. Terminal A: prefect server start
2. Terminal B: serve komutunu tekrar çalıştır
3. Terminal D: caffeinate -dim
4. Eksik gün için elle tetikle:
   prefect deployment run 'Minerva_Core_Loop/minerva-daily'
```

---

## F. SİSTEMİN BUGÜN NE YAPTIĞI

İstediğin zaman bakabilirsin:

```bash
cd ~/Minerva_v3_Studio && source venv/bin/activate

venv/bin/python -c "
import pandas as pd
db = pd.read_parquet('data/market_db.parquet')
dl = pd.read_parquet('data/decisions_log.parquet')
print(f'Son veri: {db.Date.max().date()}')
print(f'Son karar: {dl.date.max().date()}')
print(f'Bugün BUY pozisyon: {len(dl[(dl.date == dl.date.max()) & (dl.action == \"BUY\")])}')
"
```

Veya Prefect UI: tarayıcıdan **http://127.0.0.1:4200**

---

## G. KISA ÖZET — 3 KURAL

1. **Sabah telefon kontrol** — 🟢 geldiyse her şey OK
2. **3 terminal + caffeinate açık kalsın** — yoksa cron çalışmaz
3. **🚨 veya ⚠️ mesajı gelirse Bölüm E'ye bak**

Onun dışında hiçbir şey yapmana gerek yok. Sistem kendi kendine çalışıyor.
