"""v12 hızlı smoke test: enriched_market_db ile direkt MCTS mining.

Marathon orchestration yerine sadece tek bir çeyrek mining çalıştırır.
Amaç: yeni 60 feature ile mining'in sağlıklı çalıştığını doğrulamak ve
champion formüllerde yeni feature'ların görünüp görünmediğini gözlemlemek.
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from engine.core.alpha_cfg import AlphaCFG
from engine.strategies.mining_runner import MiningConfig, run_mining_window

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("smoke_v12")


def main():
    # Enriched DB yükle
    db_path = ROOT / "data" / "enriched_market_db.parquet"
    log.info("Enriched DB yükleniyor: %s", db_path.name)
    db = pd.read_parquet(db_path)
    db["Date"] = pd.to_datetime(db["Date"])

    # Q1 2023 train window (1 yıl geriye, 2022-2023)
    train_end = pd.Timestamp("2023-03-31")
    train_start = pd.Timestamp("2018-01-01")
    train = db[(db["Date"] >= train_start) & (db["Date"] <= train_end)].copy()
    log.info("Train window: %s → %s, %d satır × %d kolon",
             train_start.date(), train_end.date(), len(train), train.shape[1])

    # Next_Ret hesapla (yoksa)
    if "Next_Ret" not in train.columns:
        log.info("Next_Ret hesaplanıyor...")
        train = train.sort_values(["Ticker", "Date"])
        train["Next_Ret"] = train.groupby("Ticker")["Pclose"].shift(-1) / train["Pclose"] - 1.0

    cfg = AlphaCFG()
    log.info("AlphaCFG FEATURES: %d adet", len(cfg.FEATURES))
    log.info("İlk 10 feature: %s", cfg.FEATURES[:10])
    log.info("Son 10 feature: %s", cfg.FEATURES[-10:])

    # Hızlı küçük mining config
    mcfg = MiningConfig(
        num_gen=10,         # küçük üretim
        search_mode="mcts",
        use_wf_fitness=True,
        neutralize=True,
        wf_n_folds=3,
        wf_purge=2,
        wf_embargo=2,
        min_mean_ric=0.005,
        min_pos_ratio=0.3,
        seed=42,
    )

    log.info("Mining başlıyor (single-thread, num_gen=10)...")
    t0 = time.time()
    results = run_mining_window(train, cfg, mcfg)
    dt = time.time() - t0
    log.info("Mining bitti: %d formül, %.1f sn", len(results), dt)

    if not results:
        log.warning("⚠ Hiç formül bulunamadı")
        return 1

    # Top 10 göster
    sorted_res = sorted(results, key=lambda r: r.fitness, reverse=True)
    log.info("\n=== Top 10 Formül ===")
    for i, r in enumerate(sorted_res[:10], 1):
        log.info("  #%d  fitness=%.4f  mean_ric=%.4f  std_ric=%.4f  formula=%s",
                 i, r.fitness, r.mean_ric, r.std_ric, r.formula[:80])

    # Feature usage analizi — hangi feature'lar kullanıldı?
    log.info("\n=== Feature Kullanımı (top 10) ===")
    feature_usage = {f: 0 for f in cfg.FEATURES}
    for r in results:
        for f in cfg.FEATURES:
            if f in r.formula:
                feature_usage[f] += 1
    sorted_usage = sorted(feature_usage.items(), key=lambda kv: kv[1], reverse=True)
    for f, count in sorted_usage[:15]:
        if count > 0:
            log.info("  %-30s × %d", f, count)

    # Yeni feature'lar kazandı mı?
    new_feats = set(cfg.FEATURES) - {"Popen", "Phigh", "Plow", "Pclose", "Vlot", "Ptyp", "Pvwap"}
    used_new = [r for r in sorted_res[:20] if any(f in r.formula for f in new_feats)]
    log.info("\nYeni feature kullanan formül sayısı (top-20): %d", len(used_new))
    return 0


if __name__ == "__main__":
    sys.exit(main())
