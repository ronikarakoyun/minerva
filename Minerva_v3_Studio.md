---
title: Minerva v3 Studio
tags:
  - proje
  - quant
  - alpha-mining
  - mcts
  - tree-lstm
  - streamlit
created: 2026-04-18
status: aktif
---

# Minerva v3 Studio

> Grammar-aware MCTS + Tree-LSTM tabanlı **alfa-formül madenciliği** stüdyosu.
> BIST verisi üzerinde AST'lere dayalı alfa keşfi, walk-forward doğrulama ve
> profesyonel backtest çalıştırmak için Streamlit arayüzü.

## Özet

Minerva v3 Studio, iki akademik kaynağın birleşimini operasyonel bir araca
dönüştürüyor:

- **AlphaCFG** (α-Sem-k grameri) → [[2601.22119v1]]
- **QuantaAlpha** operatör kütüphanesi → [[2602.07085v1]]

Formüller AST (Abstract Syntax Representation) olarak temsil edilir, MCTS ile
aranır, Tree-LSTM policy/value ağı ile yönlendirilir ve walk-forward fitness
üzerinden filtrelenir.

## Mimari

```
app.py (Streamlit UI)
    │
    ├── engine/alpha_cfg.py        → Gramer + operatör kütüphanesi (AST)
    ├── engine/alpha_catalog.py    → Kayıtlı alfalar kataloğu
    ├── engine/formula_parser.py   → Metin → AST
    ├── engine/mcts.py             → Grammar-aware PUCT MCTS
    ├── engine/tree_lstm.py        → Policy/Value Tree-LSTM ağı
    ├── engine/trainer.py          → Tree-LSTM eğitim döngüsü
    ├── engine/replay_buffer.py    → Deneyim tamponu
    ├── engine/wf_fitness.py       → Walk-forward IC / verdict
    ├── engine/backtest_engine.py  → Profesyonel backtest
    └── engine/db_builder.py       → Piyasa DB üretimi
```

### Veri katmanı
- `data/market_db.parquet` — OHLCV + türetilmiş alanlar (örn. `Pvwap`)
- `data/alpha_catalog.json` — keşfedilmiş formüller + IC metrikleri
- `data/tree_lstm.pt` — eğitilmiş ağ ağırlıkları
- `data/replay_buffer.pkl` — MCTS self-play deneyimleri

### Özellikler (Tablo 4)
`Popen`, `Phigh`, `Plow`, `Pclose`, `Vlot`, `Pvwap`

### Sabitler ve pencereler (Tablo 5)
- **Constants:** `{-0.1, -0.05, -0.01, 0.01, 0.05, 0.1}`
- **Nums (window):** `{20, 30, 40}`

## İş Akışı

1. **Veri yükleme** — [[fetch_bist_data]] ile BIST fiyat/hacim verisi çekilir,
   [[engine/db_builder]] parquet'e dönüştürür.
2. **Arama** — [[engine/mcts|GrammarMCTS]] PUCT ile AST üretir; `policy_fn` ve
   `value_fn` olarak Tree-LSTM kullanılır.
3. **Doğrulama** — [[engine/wf_fitness]] walk-forward IC / RankIC / Adj-IC
   hesaplar; `wf_verdict` ile elenir.
4. **Katalog** — Kabul edilen formüller [[engine/alpha_catalog]] üzerinden
   `alpha_catalog.json`'a yazılır.
5. **Eğitim** — Replay buffer Tree-LSTM trainer'a beslenir; yeni ağırlıklar
   sonraki MCTS turuna girer (AlphaZero benzeri self-play).
6. **Backtest** — Seçilen formül(ler) [[engine/backtest_engine|run_pro_backtest]]
   ile üzerinde portföy seviyesinde test edilir.

## Arayüz (Streamlit)

- **Session state** olarak `alphas`, `trees`, `train_hist` tutulur.
- `load_data()` → parquet cache'li veri, eksikse `Pvwap` türetir.
- `get_brain(device)` → CFG + vocab + net + trainer + replay buffer tek blok
  halinde cache'li.

## Açık Uçlar / Yapılacaklar

- [ ] Katalog şemasını sürümle (IR/metrik alanları genişletirken migrasyon).
- [ ] MCTS `c_puct`, `rollouts`, `max_K` için UI paneli.
- [ ] Walk-forward fold sayısı ve tarih stratejisini UI'dan kontrol.
- [ ] Tree-LSTM checkpoint versiyonlama.
- [ ] Backtest komisyon/slippage parametrelerini UI ile bağla.

## Referanslar

- 📄 [[2601.22119v1]] — AlphaCFG / α-Sem-k grameri (gramer, Δk uzunluk kısıtı)
- 📄 [[2602.07085v1]] — QuantaAlpha operatör kütüphanesi
- 🧠 MCTS + PUCT — AlphaZero tarzı self-play
- 🌳 Tree-LSTM — Tai, Socher, Manning (2015)

## İlgili Notlar

- [[Alpha Mining]]
- [[BIST Veri Hattı]]
- [[Walk-Forward Validation]]
- [[Tree-LSTM Notları]]
