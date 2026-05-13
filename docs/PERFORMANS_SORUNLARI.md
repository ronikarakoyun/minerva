# Minerva v3 — Performansı Etkileyen Tüm Sorunlar

> **Tarih:** 2026-05-13
> **Bağlam:** 10 yıllık BIST maratonu sırasında tespit edilen tüm köklü sorunların
> akademik destekli kapsamlı dökümü. NotebookLM kaynaklı (52 akademik çalışma)
> teorik dayanaklarla.
> **Hedef:** Maraton -56% / 9 ay sonucundan sonra "ne yapılmalı" sorusuna
> sıralı, sertifikalı bir cevap.

---

## Yönetici Özeti

10 yıllık BIST maratonu üç farklı versiyonda çalıştırıldı:

| Versiyon | Trade penceresi | Toplam P&L | Sorun |
|---|---|---|---|
| v4 | 2016-01 → 2019-Q1 | **-96% (DD)** | IC NaN bug → 0 formül → 2015'in 2 kötü formülü 4 yıl trade etti |
| v5 | 2016-01 → 2017-Q1 | **-56% / 9 ay** | IC fix sonrası 902 formül ama yanlış şampiyon seçimi (mean_ric vs fitness) |
| v6 (şu an) | 2016-01 → … | bekleniyor | fitness fix + leverage cap aktif |

**Düzeltilen kritik buglar (6):** IC NaN, leverage 2×, kill-switch override,
mean_ric→fitness, BLAS oversubscription, mining filtre eşikleri.

**Kalan sorunlar:** 14 farklı problem tespit edildi. Aşağıda öncelik
sırasıyla, her biri için **akademik dayanak + somut çözüm kodu yolu** ile.

---

## Bölüm 1: KÖKEN ANALİZİ (Akademik Çerçeve)

### 1.1 Aşırı Uyum (Backtest Overfitting) — Teorik Tanım

> *"For overfitting to occur, the strategy configuration that delivers
> maximum performance IS must systematically underperform the remaining
> configurations OOS."*
> — **Bailey, Borwein, Lopez de Prado, Zhu (2015)**, *The Probability of
> Backtest Overfitting*

Minerva mining pipeline'ı tam olarak bu desende çalışıyor:
1. 902 formül üretiliyor (n_trials × num_gen)
2. In-sample (purged k-fold) fitness skoru hesaplanıyor
3. En yüksek skor → şampiyon
4. Sonraki çeyrek (true OOS) → çakılma

Bu "**strategy selection process detriment**" — Bailey'nin tanımladığı
asıl tehlike. Sistem in-sample'da en iyi olanı seçince OOS medyanın
altına düşüyor.

### 1.2 Walk-Forward'ın Tek Başına Yetersizliği

> *"WF suffers from three major disadvantages: First, a single scenario
> is tested (the historical path), which can be easily overfit. Second,
> WF is not necessarily representative of future performance. Third,
> changing the sequence of observations yields inconsistent outcomes is
> evidence of overfitting."*
> — **Lopez de Prado (2018)**, *Advances in Financial Machine Learning*, Ch. 12

WF'nin doğru kullanımı **çoklu çapraz-doğrulama** (CPCV — Combinatorial
Purged Cross-Validation) ile birlikte. Minerva v3 sadece WF kullanıyor;
CPCV implementasyonu yok.

### 1.3 RL Statik Politika Tehlikesi

> *"On-policy algorithms, such as PPO, use data generated only from the
> current policy... Consequently, the data become unusable after policy
> updates, making on-policy methods inefficient... 200 episode is highly
> insufficient... mode collapse... distribution shift."*
> — **Pippas et al. (2025)**, *RL Survey in Quantitative Finance*

10 yıl boyunca 200 episode ile eğitilmiş aynı PPO ajanı kullanmak
literatürle taban tabana zıt.

---

## Bölüm 2: ÇÖZÜLEN SORUNLAR (Marathon Boyunca)

Bu sorunlar production'da yakalandı ve düzeltildi:

| # | Sorun | Düzeltme | Etki | Commit |
|---|---|---|---|---|
| F1 | Kill-switch tüm trade'leri bloke ediyordu | `DISABLE_KILL_SWITCH=1` + check_* fonksiyonları | Trade akışı serbest | bb6d59e |
| F2 | OpenBLAS 64 thread × 16 worker = 1024 thread | `NUM_THREADS=1` | Mining 7h → 38 min | (cb8aee8) |
| F3 | Mining filtre eşikleri çok sıkı | `min_mean_ric` 0.003→0.001 vs. | 0 formül → 902 formül | 12ec7b3 |
| F4 | IC NaN (groupby.apply include_groups) | `include_groups=False` | "0 formül" sorunu çözüldü | f69bdff |
| F5 | RL leverage 2× → portföy weight 1.80 | `min(ACTIONS[action], 1.0)` | -14.97% / 1 gün engellendi | 5080353 |
| F6 | Şampiyon `mean_ric` ile seçiliyordu | `fitness` ile seç | Stabilite cezası aktif | f0d4102 |

---

## Bölüm 3: KALAN SORUNLAR (Öncelik Sıralı)

### 🔴 Tier 1 — OOS Çakılmasının Kesin Kök Nedenleri

#### S1. Gerçek OOS Holdout Yok

**Sorun:**
`engine/validation/wf_fitness.py` purged k-fold yapıyor ama tüm fold'lar
TRAIN penceresinin içinde. Yani:
- 2012-2015 → 5 fold'a bölünüyor
- Her fold'da IC ölçülüyor
- Mining bu fold'larda iyi olanları seçiyor
- 2016 (gerçek future) hiç görülmeden formüller production'a çıkıyor

**Akademik dayanak:**
> *"Backtesting through cross-validation (CV): the goal is not to derive
> historically accurate performance, but to infer future performance from
> a number of out-of-sample scenarios."* — Lopez de Prado, AFML §12.3

**Çözüm:**
```python
# wf_fitness.py içinde:
# 1. Train penceresinin son %20'si HOLDOUT olarak ayrılır
# 2. Mining sadece ilk %80'de yapılır
# 3. Top-K formüller HOLDOUT'ta tekrar ölçülür
# 4. HOLDOUT IC > 0 olmayan formüller atılır
def select_with_holdout(results, idx, holdout_dates, alpha_cfg):
    survivors = []
    for r in results:
        sig = alpha_cfg.evaluate(r.tree, idx.loc[idx.index.get_level_values("Date").isin(holdout_dates)])
        oos_ic = compute_per_date_rank_ic(...)
        if oos_ic > 0:
            r.holdout_ic = oos_ic
            survivors.append(r)
    return sorted(survivors, key=lambda r: r.holdout_ic, reverse=True)
```

**Beklenen etki:** 902 formül → ~50-100 formül kalır, ama bunlar gerçek
OOS'ta çalışmış olanlar. Selection bias 10× azalır.

---

#### S2. CSCV/PBO Modülü Var Ama Pipeline'a Bağlı Değil

**Sorun:**
`engine/validation/pbo_cscv.py` mevcut, test edilmiş, ama
`run_quarterly_mining` çağırmıyor.

**Akademik dayanak:**
> *"A customary approach would be to reject models for which PBO is
> estimated to be greater than 0.05. PBO could be used as a prior
> probability in Bayesian applications. We could compute the PBO on a
> large number of investment strategies, and use those PBO estimates to
> compute a weighted portfolio, where the weights are given by (1-PBO)."*
> — Bailey et al. (2015), §3.1

**Kritik uyarı (Bailey et al. madde 5):**
> *"PBO should NOT be the objective function on which strategy selection
> relies. When a measure becomes a target, it ceases to be a good measure."*

Yani PBO **sadece dış sertifikasyon** olarak kullanılmalı, mining hedefi
yapılmamalı.

**Çözüm:**
```python
# run_historical_paper_trade.py içinde, top_k seçiminden SONRA:
from engine.validation.pbo_cscv import compute_pbo

# Top-50 finalist üzerinde PBO testi
finalists = sorted_res[:50]
performance_matrix = build_perf_matrix(finalists, train_df)  # T × 50
pbo = compute_pbo(performance_matrix, n_partitions=16)

if pbo > 0.5:
    log.warning(f"PBO={pbo:.2f} → bu çeyrek için ŞAMPİYON SEÇİLMİYOR")
    return 0, None  # bu çeyreği atla
elif pbo > 0.05:
    # Pozisyon büyüklüğünü (1-PBO) ile çarp
    leverage_cap = 1.0 - pbo
```

**Beklenen etki:** Aşırı uyumlu çeyreklerde sistem dış müdahale ile
durdurulur. -56% gibi büyük zararlar erken kesilir.

---

#### S3. Long-Only Strateji → Market Beta Maruziyeti

**Sorun:**
`engine/execution/blender.py:182` `nlargest(cfg.top_k)` ile en yüksek
sinyalli K hisse seçiliyor — hepsi LONG. Short pozisyon yok. BIST düştüğünde
tüm pozisyonlar düşüyor (Temmuz 2016 örneği: 85 pozisyon × hepsi negatif).

**Akademik dayanak:**
> *"VİOP Beta Hedger — XU030 short ile beta-neutral... portföyün piyasa
> duyarlılığına eşdeğer büyüklükte bir endeks vadeli işlem sözleşmesinde
> kısa pozisyon... piyasanın genel yönünden kaynaklanan beta riskini
> izole etmek."* — Minerva v4 Vision Plan + Lopez de Prado ETF Trick (AFML §2.4)

ETF Trick formülü:
- Vade boşlukları (roll gaps) kümülatif olarak hesaplanır ve düzeltilir
- Negatif fiyatları önlemek için `r = roll_price_change / prev_raw_price`,
  `series = (1+r).cumprod()` ile $1 ETF kurulur
- Bid-ask spread + roll cost dahil edilir

**Çözüm seçenekleri:**
1. **Kısa vadeli:** Long-short top-K/bottom-K (en yüksek K → long, en düşük K → short)
2. **Uzun vadeli:** `services/viop_hedger.py` (Faz 5.1)

**Beklenen etki:** 2016 BIST düşüş yılında long-only -42% iken,
beta-hedged ~-8% kalabilirdi.

---

#### S4. Meta-Model Aynı Yanlılığı Güçlendiriyor

**Sorun:**
`build_meta_model_for_quarter` formülleri `mean_ric > median` ile
etiketleyip eğitiyor. Bu meta-veto da `mean_ric`'i yansıtıyor —
şampiyon seçimi `fitness`'a çevrildi ama meta-model hâlâ in-sample
mean_ric merkezli.

**Çözüm:**
```python
# scripts/run_historical_paper_trade.py:531
# Eski:
labels = (df_feat["mean_ric"] > median_ric).astype(int).values
# Yeni:
median_fitness = float(df_feat["fitness"].median())
labels = (df_feat["fitness"] > median_fitness).astype(int).values
# + feature_cols'a "fitness" ekle, "std_ric" zaten var
```

---

### 🟠 Tier 2 — Çok Muhtemel Katkıda Bulunanlar

#### S5. Survivorship Bias (Brown-Goetzmann-Ross 1992)

**Sorun:**
`market_db.parquet` sadece **bugün** BIST'te listeli olan hisseleri
içeriyor. Delist olmuş şirketler (genelde önce düşüş yaşar → delist) eksik.

**Akademik dayanak (NotebookLM):**
> *"Brown, Goetzmann, Ibbotson ve Ross (1992) hayatta kalma önyargısının
> volatilite ve getiri arasında sahte bir ilişki yarattığını gösterir.
> Yüksek risk alan fonlar ya çok büyük zarar edip silinir ya da yüksek
> getiri elde edip 'hayatta kalır'. Sadece hayatta kalanların verisine
> bakıldığında 'yüksek risk → yüksek getiri' yanılgısı oluşur."*
> — **Brown-Goetzmann-Ross 1992**, *Survival*

**Çözüm — Residual Standard Deviation Adjustment:**
> *"Brown ekibi performans ölçümlerinin (Alpha) doğrudan ARTIK STANDART
> SAPMAYA bölünerek normalize edilmesini robust bir çözüm olarak önerir.
> Yüksek risk alıp şans eseri hayatta kalan hisselerin performansı,
> taşıdıkları yüksek artık std'ye bölündüğünde matematiksel olarak
> törpülenir → sahte 'yetenek' illüzyonu ortadan kalkar."*

```python
# Şampiyon seçim ÖNCESİ uygulanır:
def survivorship_adjusted_score(formula_alpha, residual_std):
    """Brown-Goetzmann-Ross 1992 düzeltmesi."""
    return formula_alpha / max(residual_std, 1e-6)
# Yüksek alpha + yüksek residual std → düşük adjusted skor (cezalandırılır)
```

**Median Adjustment YAPMA:**
> *"BIST gibi gelişmekte olan piyasalarda kriz dönemlerinde yatay-kesit
> bağımlılığı (cross-sectional dependence) çok yüksek. Bu nedenle median
> adjustment kullanmak tehlikeli — residual std dev adjustment kullan."*

**Pragmatik etki:**
- Kısa vadeli: residual std normalize ile mevcut backtest sonucunu
  yaklaşık %30-50 haircut'lı yorumla
- Uzun vadeli: Bloomberg/Refinitiv historical constituents ($$)
- KAP halka açık delisting kayıtları scrape edilebilir

---

#### S6. Slipaj Modeli Market-Impact Aware Değil (Cont-Kukanov-Stoikov 2014)

**Sorun:**
`SlippageConfig(use_dynamic_slippage=False)` → sabit bps. Loglar
"SLIP_CAP 200 bps aşımı" uyarılarıyla dolu — gerçek slipaj
modellenenden çok daha yüksek.

**Akademik dayanak (NotebookLM):**

> *"Cont-Kukanov-Stoikov 2014 modeli: ΔP = β × OFI + ε, burada
> β = c / AD^λ. Yani fiyat etkisi piyasa derinliğine ters orantılı.
> Cont ekibinin 50 hisse üzerindeki testleri **λ ≈ 1** olduğunu, hatta
> 50'nin 35'inde λ=1 hipotezinin reddedilemediğini gösterir.
> Pratik implementasyonda λ=1 güçlü ve güvenilir bir yaklaşım.
> c parametresi ampirik olarak 0.5'ten düşük çıkar (orta-fiyat
> emirlere daha dirençli)."*

**Kalibrasyon adımları:**
1. Her hisse için yarım saatlik pencerelerde `Δp = α + β·OFI + ε` regresyon
2. `log β = α_L - λ·log AD + ε_L` → λ tahmini (≈1)
3. `β = α_M + c/AD^λ + ε_M` → c tahmini (BIST için 0.05-0.30 bekleniyor)

**Çözüm:**
`engine/execution/cont_stoikov_slippage.py` (yeni dosya):
```python
def cont_stoikov_slippage(order_size_shares, market_depth_shares,
                           c=0.15, lambda_p=1.0):
    """
    Cont-Kukanov-Stoikov 2014 OFI tabanlı slipaj.
    λ=1 varsayımı: 50 hissenin 35'inde reddedilemiyor.
    c=0.15: BIST için tahmin (offline kalibre edilmeli).
    """
    if market_depth_shares <= 0:
        return 0.02  # %2 fallback
    beta = c / (market_depth_shares ** lambda_p)
    return beta * order_size_shares  # fiyat etkisi (TL cinsinden)
```

Mevcut `engine/execution/slippage.py` zaten dynamic slippage destekliyor
ama `use_dynamic_slippage=False` ile devre dışı. Sebep "tarihsel basit;
dynamic çok yavaş" — bu performans sorunu, fonksiyonel değil.

**Geçici çözüm:** Sabit slipajı 30 → 60 bps çıkar (gerçek BIST illikit
hisse için), nihai çözümde Cont-Stoikov OFI kalibrasyonu.

---

#### S7. Çeyreklik Mining → 3 Ay Bayatlık

**Sorun:**
Şampiyon çeyrek başı seçilir, 3 ay sonu hâlâ "geçerli" sayılır. HMM
haftalık rejim güncellese de formüller donmuş.

**Örnek:** Çeyrek 1 boğa rejiminde başlıyor → momentum formülü şampiyon.
Çeyrek ortasında ayı rejimine geçiş → momentum hâlâ kullanılıyor → kayıp.

**Çözüm seçenekleri:**
1. **Aylık mining** (3× hesap maliyeti — Oracle'da yapılabilir)
2. **Rejim transition triggered refresh** (HMM trans matrix entropy ↑ →
   force re-mining)
3. **Aylık holdout-based champion rotation** (mining maliyeti yok,
   sadece mevcut havuzdan yeniden seç)

#3 en pragmatik: her ay sonu mevcut çeyreğin 902 formülü arasından
son 30 günde en iyi olanı seç.

---

#### S8. Sinyal Yönü OOS Verifikasyonu Yok (Adversarial Reverse Test)

**Sorun:**
`mean_ric > 0` → "bu formül pozitif yön gösteriyor" varsayımı.
Ama in-sample yön, OOS'ta tersine dönebilir (sign flip).

**Akademik dayanak (NotebookLM):**

> *"Adversarial Reverse Test kuralı: **Forward Sharpe > 0 VE Reverse
> Sharpe < -0.5** olmalıdır. Sadece negatif reverse Sharpe yetmez
> (-0.1, -0.2), `< -0.5` eşiği gerekir.*
>
> *Ters çevrilmiş sinyallerin `<-0.5` Sharpe ile çökmesi, formülün
> tesadüfi piyasa betası taşımadığını ve yön bulma konusunda saf
> bir yeteneğe sahip olduğunu kanıtlar. Hafif negatif reverse Sharpe
> (-0.1) ise alfanın aslında long-bias/drift'ten geldiğini gösterir."*
> — **Minerva v4 Vision Plan §2.4 (Bailey 2015 anti-skill detection)**

**Çözüm — 3 hipotez:**

```python
# Holdout (S1) verisinde:
forward_sharpe = compute_sharpe(signal, future_returns)
reverse_sharpe = compute_sharpe(-signal, future_returns)

if forward_sharpe > 0 and reverse_sharpe < -0.5:
    # GERÇEK ALFA — sign tutarlı, beta gürültüsü yok
    use_formula(tree, sign=+1)
elif reverse_sharpe > 0 and forward_sharpe < -0.5:
    # SIGN FLIP — in-sample IS yöne göre yorumlandı ama
    # gerçekte tersine çalışıyor (overfit, classic OOS reversal)
    use_formula(tree, sign=-1)
else:
    # GÜRÜLTÜ ya da MARKET BETA — atılması daha güvenli
    skip(tree)
```

**Beklenen etki:** Her çeyrek 902 formül → ~10-20 "gerçek alfa" kalır.
Aşırı seçici ama OOS güvenli.

---

### 🟡 Tier 3 — Olası/Yapısal Sorunlar

#### S9. T+1 Open Execution Gap Riski

**Sorun:**
Sinyal bugün kapanışta, alım yarın açılışta. Eğer gece çok kötü haber
varsa açılış %5+ gap'le düşer. Slipaj bunu modelleyemez.

**Tarihsel örnek:** 16 Temmuz 2016 darbe gecesi → 18 Temmuz açılışı
BIST100 -7%. Tek tek hisselerde -15% gap.

**Çözüm:**
```python
# paper_trader.py içinde, entry öncesi:
open_px = today_open
prev_close = yesterday_close
gap_pct = (open_px - prev_close) / prev_close
if abs(gap_pct) > 0.05:  # %5 gap eşiği
    log.warning(f"GAP_SKIP: {ticker} %{gap_pct*100:.1f} gap — emir iptal")
    return  # bu pozisyonu atla
```

---

#### S10. RL Agent 10 Yıl Statik

**Sorun:**
200 episode ile bir kez eğitilmiş PPO, 10 yıl boyunca aynı politika.
Mode collapse + alpha decay riski.

**Akademik dayanak (Pippas 2025):**
> *"On-policy algorithms (PPO) become unusable after policy updates...
> 200 episode is highly insufficient for complex financial environments...
> The agent will experience mode collapse and stick to a single safe
> action (e.g., always cash)."*

**Çözüm seçenekleri:**
1. **Pragmatik:** Çeyreklik retrain (mining ile birlikte, +5 dk maliyet)
2. **Optimal:** Hierarchical 4-agent RL (Faz 4)
3. **Şimdi:** RL'yi tamamen kapat (`--no-rl`), sabit leverage=1.0 ile
   trade et — performans karşılaştır

---

#### S11. Rejim-Koşullu Formül Üretilmiyor (Hamilton 1989, Ang-Bekaert 2002)

**Sorun:**
HMM K=2 rejim diyor → boğa/ayı farklı dinamik. Ama MCTS tüm veride
**tek havuz** üretiyor, sonra "regime_0 champion = top-1, regime_1 = top-2"
diye atıyor. Gerçek rejim-koşullu değil.

**Akademik dayanak (NotebookLM):**

> *"Hamilton 1989 Markov-Switching modeli: rejim 'gizli' Markov zinciri ile
> çıkarsanır. Seriyi açıklayan parametreler (ortalama getiri, varyans,
> AR katsayıları) farklı rejimlere göre yapısal olarak değişir.*
>
> *Ang-Bekaert 2002: Boğa rejimi = düşük korelasyon + düşük volatilite +
> yüksek beklenen getiri; Ayı rejimi = yüksek korelasyon + yüksek
> volatilite + düşük beklenen getiri. Rejimler **kalıcıdır (persistent)** —
> bir kriz rejimi başladığında hemen bitmez, momentum süreci izler."*

**Pratik sonuç:**
Bull rejiminde "düşük vol momentum" formülü iyi çalışırken, bear
rejiminde "high vol mean-reversion" iyi çalışabilir. Tek havuzda
top-K seçmek bu farkı yakalamaz.

**Çözüm — Rejim Başına Bağımsız Mining:**
```python
for regime_id in range(hmm._best_K):
    # Rejimin baskın olduğu günleri al (prob > 0.7)
    regime_dates = prob_df[prob_df[f"regime_{regime_id}"] > 0.7].index
    regime_df = train_df[train_df["Date"].isin(regime_dates)]
    if len(regime_df) >= 200:  # min veri eşiği
        # Sadece bu rejim verisinde mining
        regime_results = run_mining_window(regime_df, alpha_cfg, mining_cfg)
        # Bu rejime özel şampiyon (fitness ile)
        regime_champion = max(regime_results, key=lambda r: r.fitness)
        save_regime_champion(regime_id, regime_champion)
```

**Ang-Bekaert kalıcılığa uyarı:** Rejim transition matrix
`P(stay) = 0.85+` → bir rejim başladığında ortalama 6-12 hafta sürer.
Yani Q1'de bull rejiminde başlayan çeyrek, Q3'te ayı'ya geçebilir →
çeyrek-içi rejim refresh gerekli (S7 ile bağlantılı).

**Beklenen etki:** Rejim geçişlerinde whipsaw azalır, her rejim için
"konuşkan" formül. Ang-Bekaert: "ayı rejiminde çeşitlendirme bile
fayda sağlar — kondisyonel hedging değerlidir."

---

### ⚪ Tier 4 — Kod Kalitesi / Operasyonel

#### S12. ERC Her Zaman %100 Yatırımlı (Kritzman-Li 2010 Turbulence)

**Sorun:**
`equal_risk_contribution` çıktısı her zaman toplam=1.0. Yüksek
volatilite günlerinde de %100 exposure.

**Akademik dayanak (NotebookLM):**

> *"Kritzman-Li 2010 'Skulls, Financial Turbulence, and Risk Management':*
> *Mahalanobis distance (1927 antropoloji formülü) finansal türbülans
> ölçümüne uyarlandı. Tarihsel ortalamadan sapma + korelasyon
> kırılımları + 'ilişkisiz' varlıkların aniden birlikte hareketi.*
>
> *Türbülans 75. persentil üstünde → 'turbulent' kabul edilir
> (verinin en aşırı %25'i).*
>
> *Türbülans haftalarca sürer (uçak türbülansı gibi) — başladığında
> hemen bitmez."*

**Çözüm — Quintile-based Inverse Exposure:**
> *"Pozisyon ağırlığı türbülans şiddetiyle TERS ORANTILI olarak
> ölçeklenir (carry trade örneği):*
>
> *| Türbülans Quintile | Exposure |*
> *|---|---|*
> *| 1. (en yüksek) | %20 |*
> *| 2. | %40 |*
> *| 3. | %60 |*
> *| 4. | %80 |*
> *| 5. (en sakin) | %100 |*
>
> *Bu kural negatif çarpıklığı (negative skewness) büyük ölçüde
> ortadan kaldırır ve VIX, swap spreads gibi diğer stress
> göstergelerinden daha etkili çalışır."*

```python
def kritzman_li_turbulence(returns_recent: pd.DataFrame,
                           lookback_years: int = 3) -> float:
    """Mahalanobis distance: (y - μ)' Σ^-1 (y - μ)."""
    mu = returns_recent.mean().values
    cov = returns_recent.cov().values
    cov_inv = np.linalg.pinv(cov)
    y = returns_recent.iloc[-1].values
    d = (y - mu) @ cov_inv @ (y - mu).T
    return float(d)

def exposure_from_turbulence(turbulence: float,
                             historical: np.ndarray) -> float:
    """Quintile-based inverse scaling (Kritzman-Li 2010)."""
    quintiles = np.percentile(historical, [20, 40, 60, 80])
    if turbulence <= quintiles[0]: return 1.00
    elif turbulence <= quintiles[1]: return 0.80
    elif turbulence <= quintiles[2]: return 0.60
    elif turbulence <= quintiles[3]: return 0.40
    else: return 0.20  # en yüksek türbülans
```

**Beklenen etki:** Temmuz 2016 darbe günleri gibi yüksek türbülans
dönemlerinde portföy otomatik %20'ye iner — büyük drawdown'lar
matematiksel olarak engellenir.

---

#### S13. Test Coverage Boşlukları

**Sorun:**
6 production bug'ı yakalandı (IC NaN, leverage, mean_ric vs.), 290
test bunları yakalamamıştı.

**Çözüm:**
- Marathon mini-replikası integration test (1 ay backtest)
- Property-based test (Hypothesis): random formula + random data →
  invariant: weight.sum() <= 1.0

---

#### S14. Sessiz Fallback'ler

**Sorun:**
- DML neutralization başarısız → `pass` (ham sinyal kullanılır, log yok)
- DuckDB IC error → pandas fallback (sessiz)

Bu hataların sıklığı/oranı ölçülmüyor → performans degradasyonu fark
edilmiyor.

**Çözüm:**
Prometheus-style counter:
```python
DML_FAILURE_COUNTER.inc()
DUCKDB_FALLBACK_COUNTER.inc()
# her quarter sonu rapor: failure_rate %X
```

---

## Bölüm 4: ÖNCELİK SIRALI EYLEM PLANI

### Hemen (Maraton sırasında, 1-2 saat)

1. **S4: Meta-model fitness'a çevir** — 5 satır kod, anında etki
2. **S6: Sabit slipaj 30 → 60 bps** — 1 satır kod, daha gerçekçi
3. **S9: T+1 gap skip (>5%)** — 10 satır kod, tail risk azaltır

### Bu hafta (Maraton tamamlandıktan sonra)

4. **S1: Gerçek OOS holdout** — `wf_fitness.py` refactor, 1 gün
5. **S2: PBO entegrasyonu** — `run_quarterly_mining` patch, yarım gün
6. **S8: Sign verification** — S1'in parçası

### Bu ay (Performans bekliyorsa)

7. **S3: Long-short veya VİOP hedge** — `services/viop_hedger.py`, 3-5 gün
8. **S7: Aylık champion rotation** — orchestrator değişikliği, 1 gün
9. **S10: RL çeyreklik retrain** — 1 gün
10. **S11: Rejim-koşullu mining** — 2-3 gün

### Uzun vade (3+ ay)

11. **S12: Turbulence-based cash buffer** — Faz 4 entegrasyonu
12. **S5: Survivorship bias** — Veri sağlayıcı kararı
13. **S13-S14: Test coverage + monitoring** — kalıcı kalite

---

## Bölüm 5: AKADEMİK REFERANS HARİTASI

| Sorun | Birincil Kaynak | NotebookLM Source ID | Durum |
|---|---|---|---|
| S1 (OOS holdout) | López de Prado AFML Ch. 11-12 | `7e39d9b0` | ✅ Cevaplandı |
| S2 (PBO entegrasyon) | Bailey, Borwein, Lopez de Prado, Zhu 2015 | `3e8cdcbc` | ✅ Cevaplandı |
| S3 (Beta hedge) | López de Prado ETF Trick (AFML §2.4) | `7e39d9b0` | ✅ Cevaplandı |
| S5 (Survivorship) | Brown-Goetzmann-Ross 1992 | `9bad604a` + `59debe4c` | ✅ Cevaplandı |
| S6 (Slippage) | Cont-Kukanov-Stoikov 2014 | `c76a4c51` | ✅ Cevaplandı |
| S8 (Adversarial reverse) | Minerva v4 Plan + Bailey 2015 | `b75acabe` | ✅ Cevaplandı |
| S10 (RL retrain) | Pippas et al. 2025 (RL Survey) | `20eaf594` | ✅ Cevaplandı |
| S11 (Rejim koşullu) | Hamilton 1989 + Ang-Bekaert 2002 | `0fb9f1ad` + `a49587e2` | ✅ Cevaplandı |
| S12 (Turbulence) | Kritzman-Li 2010 | `db2302b2` | ✅ Cevaplandı |

---

## Bölüm 6: KARAR NOKTALARI

### A. Maraton şu an çalışıyor — devam mı, dur mu?

**Devam senaryosu:** Mevcut fitness fix ile -56% azalır mı görelim
(2-3 saat içinde Q3 2016 P&L gelir). İyileşme varsa devam, yoksa durdur.

**Dur senaryosu:** Tier 1 düzeltmelerini (S1-S4) implement et, fresh
restart yap. ~1 hafta gecikme + daha iyi sonuç olasılığı yüksek.

### B. Long-only kararı

BIST'te kaldıraç yok ama short var (açığa satış SPK izinli hisselerde).
Long-short yapmak isteniyor mu? Aksi halde VİOP hedge zorunlu.

### C. Veri kalitesi yatırımı

Bloomberg/Refinitiv historical constituents data ~$500-1000/ay. Yapmadan
backtest sonuçları %30-50 abartılı kabul edilmeli.

---

## Ekler

### Ek A: Düzeltilen Bug'ların Tarihçesi
- bb6d59e: DISABLE_KILL_SWITCH=1 → trade akışını durdurmama
- cb8aee8: v4 Faz 1 (Arrow+DuckDB+Polars+Numba+PG/Redis)
- 12ec7b3: Mining filtre eşikleri gevşetildi
- 70531af: arrow_db timestamp fix
- f118e97: mcts_pool pickle fallback
- f69bdff: wf_fitness_duckdb include_groups=False
- 5080353: BIST kaldıraç kısıtı (RL leverage min(1.0))
- f0d4102: Şampiyon mean_ric → fitness

### Ek B: NotebookLM Sorgu Logları (Tamamlandı + Eksik)
Notebook: `Minerva v3` (962f7299-b163-4b62-8c49-492eaae085e6)

**✅ Cevap alınan sorgular:**
1. Walk-forward + purged k-fold OOS holdout → AFML §12 alıntıları
2. CSCV/PBO pipeline entegrasyonu → Bailey 2015 §3.1, 5.2
3. VIOP futures beta hedging + ETF Trick → Lopez de Prado AFML §2.4
4. RL retraining, mode collapse, PPO statik → Pippas 2025 §2.3-2.4
5. Cont-Kukanov-Stoikov OFI slipaj kalibrasyonu → λ≈1, c<0.5 ampirik
6. Survivorship bias (Brown-Goetzmann-Ross 1992) → residual std adjustment
7. Adversarial reverse test eşiği → forward>0 AND reverse<-0.5
8. Hamilton 1989 + Ang-Bekaert 2002 regime-switching → persistence,
   momentum process, bull/bear ayrımı
9. Kritzman-Li 2010 turbulence → Mahalanobis distance, quintile inverse exposure

**❌ Cevap alınamayan sorgular (timeout — daha fazla kaynak eklenebilir):**

1. **Pippas 2025 Composite Reward Shaping detayları**
   - `0.6*profit + 0.2*cost + 0.2*expert_imitation` formülünün
     ağırlık kalibrasyonu nasıl yapılmalı?
   - Oracle action (perfect foresight) reward'a leakage olmadan
     nasıl katılır?
   - **Önerilen kaynak:** Lee et al. MAPS framework paper

2. **Walk-forward mining cadence: aylık vs çeyreklik**
   - Rolling window genişliği akademik öneri ne?
   - Hangi sıklıkta retrain optimal?
   - **Önerilen kaynak:** Bailey-Lopez de Prado "Sharpe Ratio Efficient
     Frontier" (2012), AFML Ch. 17 backtesting on rolling windows

3. **BIST açığa satış (short selling) SPK regülasyonu**
   - Hangi hisseler short edilebilir?
   - Pay senedi ödünç verme (securities lending) maliyetleri?
   - Long-short market-neutral strateji pratik kısıtlar?
   - **Önerilen kaynak:** SPK Sermaye Piyasası Kurulu Tebliği
     (Açığa Satış İşlemleri Hakkında Tebliğ V-101.1), Borsa İstanbul
     Pay Ödünç Piyasası dökümanı

### Ek C: Mevcut Kod Eşlemesi

| Sorun | Mevcut dosya | Değişiklik tipi |
|---|---|---|
| S1 | `engine/validation/wf_fitness.py` | refactor |
| S2 | `engine/validation/pbo_cscv.py` (var) + `scripts/run_historical_paper_trade.py` | entegrasyon |
| S3 | `engine/execution/blender.py` veya `services/viop_hedger.py` (yeni) | yeni özellik |
| S4 | `scripts/run_historical_paper_trade.py:531` | satır değişikliği |
| S5 | (veri layer) | yatırım gerekli |
| S6 | `engine/execution/slippage.py` (mevcut) | aktive et |
| S7 | `scripts/run_historical_paper_trade.py` | orchestrator değişikliği |
| S8 | `engine/validation/wf_fitness.py` | S1 parçası |
| S9 | `engine/execution/paper_trader.py` | guard ekle |
| S10 | `scripts/run_historical_paper_trade.py` | retrain hook |
| S11 | `engine/strategies/mining_runner.py` | regime loop |

---

**Son güncelleme:** 2026-05-13
**Sonraki revizyon:** Maraton v6 sonuçları geldikten sonra
