"""
auto_minerva.py — Faz 6: Kurumsal orkestrasyon (Prefect 3.x).

Tüm Faz 1-5 bileşenlerini sırayla çalıştırır:
    1. fetch_bist_data        → data/market_db.parquet
    2. detect_regime (HMM)    → data/regime_prob_df.parquet
    3. nightly_mining (Cuma)  → data/best_params.json (Optuna)
    4. decay_scan             → aktif şampiyon decay kontrolü
    5. morning_execution      → blend + paper trade + adli log

Kullanım:
    python -m auto_minerva                    # tek seferlik manuel koşu
    prefect deployment build auto_minerva.py:run_daily_cycle \\
        -n minerva-daily --cron "30 18 * * 1-5" --timezone "Europe/Istanbul"
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import pandas as pd
from filelock import FileLock
from prefect import flow, get_run_logger, task

from engine.notifications import send_telegram

logger = logging.getLogger(__name__)

# Alpha catalog üzerindeki orchestrator-seviyesi lock.
# alpha_catalog.py kendi filelock'unu kullanır ama decay_scan (Task 4) ve
# morning_execution (Task 5) birbirinden bağımsız task'tır; sıralı çalışsalar
# da teorik olarak Prefect concurrency ile çakışabilir.
_CATALOG_FLOW_LOCK = FileLock("data/.alpha_catalog_flow.lock", timeout=60)


# ──────────────────────────────────────────────────────────────────────
# Telegram hooks (Prefect on_completion / on_failure)
# ──────────────────────────────────────────────────────────────────────
def _hook_flow_failed(flow, flow_run, state):
    """Flow Failed → kritik alarm."""
    msg = (
        f"🚨 *MINERVA HATA*\n"
        f"Flow: `{flow.name}`\n"
        f"Run: `{flow_run.name}`\n"
        f"Durum: {state.type.value}\n"
        f"Mesaj: {state.message or '(boş)'}"
    )
    send_telegram(msg, parse_mode="Markdown")


def _build_portfolio_report() -> str:
    """
    Telegram'a yollanacak günlük portföy raporu.

    İçerik:
      1. Bugünün portföyü (tüm BUY pozisyonlar)
      2. Aktif şampiyonlar (hangi formüller çalışıyor)
      3. Net P&L durumu (bugüne kadarki kapanan trade'ler)
      4. HMM rejim algısı
    """
    import json
    import pandas as pd
    from pathlib import Path

    lines: list[str] = []

    # ── Bugünün portföyü ──
    pt_path = Path("data/paper_trades.parquet")
    dl_path = Path("data/decisions_log.parquet")
    if pt_path.exists() and dl_path.exists():
        pt = pd.read_parquet(pt_path)
        dl = pd.read_parquet(dl_path)
        last_date = dl["date"].max()
        today = dl[dl["date"] == last_date]
        buys = today[today["action"] == "BUY"].sort_values("target_weight", ascending=False)

        lines.append(f"🟢 *MINERVA RAPORU — {pd.Timestamp(last_date).date()}*")
        lines.append("")
        lines.append(f"📊 *Portföy* ({len(buys)} pozisyon)")
        if len(buys) > 0:
            for _, r in buys.head(15).iterrows():
                lines.append(
                    f"  • `{r.ticker:<6}` w={r.target_weight:.1%}  "
                    f"slip={r.expected_slip_bps:.1f}bps"
                )
            if len(buys) > 15:
                lines.append(f"  …ve {len(buys) - 15} pozisyon daha")
        else:
            lines.append("  (boş)")
        lines.append("")

        # ── HMM rejim ──
        if len(today) > 0:
            top_regime = int(today["hmm_top_regime"].mode().iloc[0])
            top_p = float(today["hmm_top_p"].mean())
            lines.append(f"🎯 Baskın rejim: `regime_{top_regime}` (p={top_p:.0%})")
            lines.append("")

        # ── P&L (kapanmış trade'ler) ──
        filled = pt[pt["net_pnl_pct"].notna()].copy()
        if len(filled) > 0:
            # Günlük portföy getirisi (ağırlıklı)
            filled["weighted_pnl"] = filled["net_pnl_pct"] * filled["weight"]
            daily = filled.groupby("date")["weighted_pnl"].sum().sort_index()
            cum_growth = (1.0 + daily).prod() - 1.0
            avg_daily = daily.mean()
            n_days = len(daily)
            win_rate = (filled["net_pnl_pct"] > 0).mean()

            lines.append(f"💰 *P&L* ({n_days} kapanmış gün)")
            lines.append(f"  Kümülatif: *{cum_growth:+.2%}*")
            lines.append(f"  Günlük ort: {avg_daily:+.3%}")
            lines.append(f"  Win rate: {win_rate:.0%}")
            lines.append(f"  Son 5 gün:")
            for d, v in daily.tail(5).items():
                emoji = "🟢" if v > 0 else "🔴" if v < 0 else "⚪"
                lines.append(f"    {emoji} {pd.Timestamp(d).date()}: {v:+.3%}")
        else:
            lines.append("💰 *P&L*: henüz kapanmış trade yok (t+2 bekleniyor)")
        lines.append("")
    else:
        lines.append("🟢 *MINERVA RAPORU*")
        lines.append("(paper_trades.parquet veya decisions_log.parquet yok)")
        lines.append("")

    # ── Aktif şampiyonlar ──
    cat_path = Path("data/alpha_catalog.json")
    if cat_path.exists():
        try:
            recs = json.loads(cat_path.read_text())
            champs = sorted(
                [r for r in recs if r.get("regime_champion_for") is not None],
                key=lambda r: r["regime_champion_for"],
            )
            lines.append(f"🏆 *Şampiyon formüller* ({len(champs)})")
            for c in champs:
                rid = c["regime_champion_for"]
                ic = c.get("ic", 0.0)
                formula = c["formula"]
                # Telegram için formülü kısalt
                if len(formula) > 60:
                    formula = formula[:57] + "…"
                lines.append(f"  Rejim {rid}: `{formula}` (ic={ic:.4f})")
        except Exception as e:
            lines.append(f"🏆 Şampiyonlar okunamadı: {e}")
    else:
        lines.append("🏆 alpha_catalog.json yok")

    return "\n".join(lines)


def _hook_morning_completed(task, task_run, state):
    """morning_execution Completed → detaylı portföy raporu."""
    try:
        report = _build_portfolio_report()
        send_telegram(report, parse_mode="Markdown")
    except Exception as e:
        logger.warning("morning_completed hook hatası: %s", e)
        # Fallback: basit mesaj
        try:
            send_telegram(f"🟢 Minerva tamamlandı (rapor üretilemedi: {e})")
        except Exception as e2:
            logger.error("morning_completed fallback Telegram da başarısız: %s", e2)


def _hook_decay_failed(task, task_run, state):
    """decay_scan Failed (DECAY: prefix) → şampiyon emekli alarmı."""
    msg = (
        f"⚠️ *ALPHA DECAY ALARM*\n"
        f"Task: `{task.name}`\n"
        f"Mesaj: {state.message or '(detay yok)'}\n"
        f"Aksiyon: yeni mining koş, eski şampiyonu değiştir"
    )
    send_telegram(msg, parse_mode="Markdown")


# ──────────────────────────────────────────────────────────────────────
# Stale data SAFE_MODE: veri bu kadar günden eskiyse trade engellenir.
_STALE_DATA_MAX_DAYS: int = 3


def _check_data_freshness(db_path: Path) -> None:
    """
    market_db.parquet'teki en son tarih bugünden 3 günden fazla gerideyse
    SAFE_MODE aktif edilir: RuntimeError fırlatılır → morning_execution çalışmaz.
    """
    try:
        db = pd.read_parquet(db_path, columns=["Date"])
        last_date = pd.Timestamp(db["Date"].max())
        today_ist = pd.Timestamp(datetime.now(ZoneInfo("Europe/Istanbul")).date())
        age_days = (today_ist - last_date).days
        if age_days > _STALE_DATA_MAX_DAYS:
            msg = (
                f"SAFE_MODE: market_db son tarih={last_date.date()} "
                f"({age_days} gün eski) — trade engellendi."
            )
            logger.critical(msg)
            try:
                send_telegram(f"🛑 {msg}")
            except Exception:
                pass
            raise RuntimeError(msg)
        logger.info("Veri tazeliği OK: son tarih=%s (%d gün önce)", last_date.date(), age_days)
    except RuntimeError:
        raise
    except Exception as exc:
        logger.warning("Veri tazeliği kontrolü başarısız: %s", exc)


# ──────────────────────────────────────────────────────────────────────
# Task 1 — Veri çekimi
# ──────────────────────────────────────────────────────────────────────
@task(name="fetch_data", retries=3, retry_delay_seconds=300, timeout_seconds=900)
def task_fetch_data() -> Path:
    """yfinance → data/market_db.parquet. 3× retry, 5 dk arayla."""
    log = get_run_logger()
    log.info("Faz 6.1: BIST veri çekimi başlıyor")
    from scripts.fetch_bist_data import main as fetch_main
    fetch_main()
    out = Path("data/market_db.parquet")
    if not out.exists():
        raise RuntimeError(f"Veri dosyası üretilmedi: {out}")
    log.info("Veri hazır: %s", out)
    return out


# ──────────────────────────────────────────────────────────────────────
# Task 2 — HMM rejim tespiti (Forward-only / look-ahead safe)
# ──────────────────────────────────────────────────────────────────────
@task(name="detect_regime", retries=2, retry_delay_seconds=60, timeout_seconds=600)
def task_detect_regime(db_path: Path) -> Path:
    """Faz 1: HMM fit + filtered prob_df. Çıktı: data/regime_prob_df.parquet."""
    log = get_run_logger()
    log.info("Faz 1: HMM rejim tespiti")
    from engine.data.regime_detector import RegimeConfig, run_pipeline
    cfg = RegimeConfig(use_filtered_probs=True)  # backtest güvenli
    prob_df = run_pipeline(cfg)
    out = Path("data/regime_prob_df.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    prob_df.to_parquet(out)
    log.info("prob_df kaydedildi: %s (%d gün, K=%d)",
             out, len(prob_df), prob_df.shape[1])
    return out


# ──────────────────────────────────────────────────────────────────────
# Task 3 — Optuna meta-tuning (yalnız Cuma gecesi)
# ──────────────────────────────────────────────────────────────────────
# N48: 1h timeout, ama meta-opt (50×) + tam mining iki aşama: 2h daha güvenli.
@task(name="nightly_mining", timeout_seconds=7200)
def task_nightly_mining(db_path: Path, prob_path: Path,
                          only_on_weekday: int = 4,
                          n_trials: int = 50) -> dict:
    """
    Faz 3: MCTS+Optuna. Yalnız Cuma gecesi tam kapasite çalışır
    (only_on_weekday=4); diğer günler skip.
    """
    log = get_run_logger()
    # TZ-aware: Prefect UTC cronları İstanbul saatinden farklı olabilir.
    today = datetime.now(ZoneInfo("Europe/Istanbul")).weekday()
    if only_on_weekday is not None and today != only_on_weekday:
        log.info("nightly_mining skip — bugün %d, hedef gün %d",
                 today, only_on_weekday)
        return {"skipped": True, "reason": f"weekday={today}"}

    log.info("Faz 3: Optuna meta-tuning (n_trials=%d)", n_trials)
    from engine.strategies.meta_optimizer import (
        MetaOptConfig, run_meta_optimization,
    )
    from engine.strategies.mining_runner import MiningConfig, run_mining_window
    from engine.core.alpha_catalog import save_alpha
    from engine.core.alpha_cfg import AlphaCFG

    # Meta-optimizasyon
    result = run_meta_optimization(MetaOptConfig(n_trials=n_trials))
    log.info("best_value=%.4f params=%s",
             result.get("best_value", 0.0), result.get("best_params"))

    # best_params.json varsa sıcak mining başlat (temperature dahil)
    best_params_path = Path("data/best_params.json")
    if best_params_path.exists():
        try:
            mining_cfg = MiningConfig.from_best_params(best_params_path, num_gen=200)
            log.info("from_best_params: temperature=%.2f search_mode=%s",
                     mining_cfg.weight_cfg.temperature if mining_cfg.weight_cfg else 2.0,
                     mining_cfg.search_mode)
            db = pd.read_parquet(Path("data/market_db.parquet"))
            cfg_alpha = AlphaCFG()
            accepted = run_mining_window(db, cfg_alpha, mining_cfg)
            log.info("best_params mining: %d formül kabul edildi", len(accepted))
            for r in accepted:
                try:
                    save_alpha(
                        formula=r.formula, tree=r.tree,
                        ic=r.mean_ric, rank_ic=r.mean_ric,
                        adj_ic=r.mean_ric - r.std_ric,
                        source="meta_opt",
                        wf_mean_ric=r.mean_ric, wf_std_ric=r.std_ric,
                        wf_pos_folds=r.pos_folds, wf_n_folds=r.n_folds,
                        wf_fitness=r.fitness,
                    )
                except Exception as save_exc:
                    log.warning("save_alpha failed: %s", save_exc)
            result["meta_mining_accepted"] = len(accepted)
        except Exception as exc:
            log.warning("best_params mining hatası: %s", exc)

    return result


# ──────────────────────────────────────────────────────────────────────
# Task 4 — Decay scan (aktif şampiyonlar için)
# ──────────────────────────────────────────────────────────────────────
@task(name="decay_scan", retries=1, on_failure=[_hook_decay_failed])
def task_decay_scan(prob_path: Path) -> dict:
    """
    Tüm aktif şampiyonları decay_monitor üzerinden geçir; en az bir tetik
    varsa raise → flow Failed → automation tetiklenir.
    """
    log = get_run_logger()
    from engine.core.alpha_catalog import get_active_champions, set_inactive
    from engine.execution.paper_trader import feed_decay_monitor

    champions = get_active_champions()
    log.info("decay_scan: %d aktif şampiyon kontrol edilecek", len(champions))

    triggered = []
    for fid, meta in champions:
        try:
            result = feed_decay_monitor(
                fid, meta["backtest_mean"], meta["backtest_std"],
            )
        except Exception as e:
            log.warning("scan hatası (%s): %s", fid, e)
            continue
        if result.get("triggered"):
            triggered.append((fid, result))
            log.warning("DECAY tetikleyici: %s → %s", fid, result)
            # Catalog lock ile thread-safe deactivation
            with _CATALOG_FLOW_LOCK:
                deactivated = set_inactive(fid)
            if deactivated:
                log.warning("AUTO-DEACTIVATE: %s katalogdan pasife alındı", fid)
            else:
                log.error("AUTO-DEACTIVATE BAŞARISIZ: %s katalogda bulunamadı", fid)

    if triggered:
        raise RuntimeError(
            f"DECAY: {len(triggered)} şampiyon emekli edildi: "
            + ", ".join(fid for fid, _ in triggered)
        )

    return {"checked": len(champions), "triggered": 0}


# ──────────────────────────────────────────────────────────────────────
# Task 5 — Sabah icrası: blend + paper log + forensic
# ──────────────────────────────────────────────────────────────────────
@task(name="morning_execution", retries=1, timeout_seconds=600,
      on_completion=[_hook_morning_completed])
def task_morning_execution(db_path: Path, prob_path: Path) -> dict:
    """
    Faz 5: Önceki günün exit fill + bugünün blend + paper log + adli log.
    """
    log = get_run_logger()
    # Stale data guard — eski verilerle trade yapmayı engelle
    _check_data_freshness(db_path)

    from engine.core.alpha_cfg import AlphaCFG
    from engine.execution.blender import (
        BlenderConfig, blend_regime_signals, load_champions_from_catalog,
    )
    from engine.execution.forensics import (
        ForensicConfig, log_decision_forensics,
    )
    from engine.execution.paper_trader import (
        PaperTraderConfig, compute_realized_pnl, log_daily_decisions,
    )
    from engine.execution.slippage import SlippageConfig

    db = pd.read_parquet(db_path)
    prob_df = pd.read_parquet(prob_path)

    # `today` için prob_df ve db'nin kesişiminin son gününü kullan.
    # prob_df HMM'den (XU100 yfinance) gelir, db market_db'den — son tarihleri farklı olabilir.
    db_last = pd.Timestamp(db["Date"].max())
    prob_last = pd.Timestamp(prob_df.index[-1])
    today = min(db_last, prob_last)
    log.info("today=%s (db_last=%s, prob_last=%s)",
             today.date(), db_last.date(), prob_last.date())

    # 1) Önceki günün exit fill (t+2)
    pt_cfg = PaperTraderConfig()
    compute_realized_pnl(db, pt_cfg)
    log.info("compute_realized_pnl tamam — %s öncesi açık trade'ler kapandı", today.date())

    # 2) Şampiyonlar
    catalog_path = Path("data/alpha_catalog.json")
    if not catalog_path.exists():
        log.warning("alpha_catalog.json yok — morning_execution skip")
        return {"logged": 0, "skipped": True, "reason": "no_catalog"}

    alpha_cfg = AlphaCFG()
    champions = load_champions_from_catalog(catalog_path, alpha_cfg=alpha_cfg)
    if not champions:
        log.warning("Aktif şampiyon yok — morning_execution skip")
        return {"logged": 0, "skipped": True, "reason": "no_champions"}

    # 3) Bugünün target weights
    weights_df = blend_regime_signals(
        champions, prob_df, db,
        BlenderConfig(use_blending=True),
        alpha_cfg=alpha_cfg,
    )
    # Blend çıktısının gerçek son gününü al (rolling pencere kayıpları olabilir)
    if len(weights_df) == 0:
        log.warning("blend çıktısı boş — morning_execution skip")
        return {"logged": 0, "skipped": True, "reason": "empty_blend"}

    weights_last = pd.Timestamp(weights_df.index[-1])
    if weights_last < today:
        log.info("blend son günü %s, today=%s ileride — son günü kullanıyoruz",
                 weights_last.date(), today.date())
        today = weights_last

    today_weights = weights_df.loc[today].dropna()
    today_weights = today_weights[today_weights > 0]

    # 4) Paper trade log
    formula_id = "blend_v6"
    n_logged = log_daily_decisions(
        today_weights, formula_id, today, db,
        slippage_cfg=SlippageConfig(use_dynamic_slippage=True),
        cfg=pt_cfg,
    )
    log.info("paper_trades.parquet: %d satır", n_logged)

    # 5) Adli log
    if today in prob_df.index:
        prob_row = prob_df.loc[today]
        n_forensic = log_decision_forensics(
            date=today, weights=today_weights, prob_row=prob_row,
            champions=champions, db=db, formula_id=formula_id,
            cfg=ForensicConfig(),
        )
        log.info("decisions_log.parquet: %d satır", n_forensic)
    else:
        n_forensic = 0

    return {
        "logged": int(n_logged),
        "forensic": int(n_forensic),
        "date": str(today.date()),
    }


# ──────────────────────────────────────────────────────────────────────
# Task 6 — N44: Stale job heartbeat cleanup
# ──────────────────────────────────────────────────────────────────────
@task(name="cleanup_stale_jobs", timeout_seconds=60)
def task_cleanup_stale_jobs() -> dict:
    """
    N44: In-memory job registry'deki stale (5dk+ heartbeat yok) job'ları
    "stale" statüsüne al ve SQLite'a persist et.
    Bu task flow'a her çalışmada eklenir → restart sonrası zombie jobs temizlenir.
    """
    log = get_run_logger()
    try:
        from api.jobs import registry
        before = len(registry._jobs)
        registry.cleanup_old()
        after = len(registry._jobs)
        log.info("Stale job cleanup: %d → %d in-memory job", before, after)
        return {"before": before, "after": after}
    except Exception as exc:
        log.warning("Stale job cleanup başarısız: %s", exc)
        return {"error": str(exc)}


# ──────────────────────────────────────────────────────────────────────
# FLOW
# ──────────────────────────────────────────────────────────────────────
@flow(name="Minerva_Core_Loop", log_prints=True,
      on_failure=[_hook_flow_failed])
def run_daily_cycle(
    only_mining_on_weekday: Optional[int] = 4,
    mining_n_trials: int = 50,
) -> dict:
    """Akşam: fetch + regime + (Cuma) mining + decay; Sabah: execution."""
    # N44: Her çalışmada önce stale job'ları temizle
    task_cleanup_stale_jobs()

    db_path = task_fetch_data()
    prob_path = task_detect_regime(db_path)
    mining_result = task_nightly_mining(
        db_path, prob_path,
        only_on_weekday=only_mining_on_weekday,
        n_trials=mining_n_trials,
    )
    decay_result = task_decay_scan(prob_path)
    exec_result = task_morning_execution(db_path, prob_path)
    return {
        "mining": mining_result,
        "decay": decay_result,
        "execution": exec_result,
    }


if __name__ == "__main__":
    run_daily_cycle()
