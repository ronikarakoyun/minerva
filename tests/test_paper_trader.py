"""
tests/test_paper_trader.py — Faz 5.3 paper trading karar logları testleri.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from engine.execution.paper_trader import (
    PAPER_TRADE_COLUMNS,
    PaperTraderConfig,
    log_daily_decisions,
    compute_realized_pnl,
    feed_decay_monitor,
)
from engine.execution.slippage import SlippageConfig


# ──────────────────────────────────────────────────────────────────────
# 1. log_daily_decisions parquet'e append eder, şema doğru
# ──────────────────────────────────────────────────────────────────────
def test_log_daily_decisions_appends_parquet(syn_db, tmp_path):
    """İki gün log → parquet 2× büyür, kolon şeması doğru."""
    out_path = tmp_path / "paper_trades.parquet"
    cfg = PaperTraderConfig(output_path=out_path, portfolio_capital_TL=1_000_000)

    dates = sorted(syn_db["Date"].unique())
    d1 = dates[100]
    d2 = dates[101]

    weights = pd.Series({"T000": 0.5, "T001": 0.3, "T002": 0.2})

    n1 = log_daily_decisions(weights, "test_formula", d1, syn_db, cfg=cfg)
    n2 = log_daily_decisions(weights, "test_formula", d2, syn_db, cfg=cfg)

    assert n1 == 3
    assert n2 == 3

    df = pd.read_parquet(out_path)
    assert len(df) == 6
    assert list(df.columns) == PAPER_TRADE_COLUMNS
    # entry_px > 0
    assert (df["entry_px"] > 0).all()
    # exit_px henüz boş
    assert df["exit_px"].isna().all()
    # slippage_bps ≥ 0
    assert (df["slippage_bps"] >= 0).all()


# ──────────────────────────────────────────────────────────────────────
# 2. compute_realized_pnl t+hold sonrası net_pnl_pct doldurur
# ──────────────────────────────────────────────────────────────────────
def test_compute_realized_pnl_fills_exits(syn_db, tmp_path):
    """t+2 fiyatı geldikten sonra net_pnl_pct doldurulur."""
    out_path = tmp_path / "paper_trades.parquet"
    cfg = PaperTraderConfig(output_path=out_path, hold_days=2)

    dates = sorted(syn_db["Date"].unique())
    d1 = dates[50]  # exit için yeterli sonrası var

    weights = pd.Series({"T000": 0.5, "T010": 0.5})
    log_daily_decisions(weights, "f1", d1, syn_db, cfg=cfg)

    # exit fill
    df = compute_realized_pnl(syn_db, cfg=cfg)
    assert len(df) == 2
    # exit_px doldu
    assert df["exit_px"].notna().all()
    # net_pnl_pct doldu
    assert df["net_pnl_pct"].notna().all()
    # gross_pnl_pct hesaplandı: exit/entry - 1
    actual = df["gross_pnl_pct"].astype(float).to_numpy()
    expected = (df["exit_px"].astype(float) / df["entry_px"].astype(float) - 1.0).to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-9)


# ──────────────────────────────────────────────────────────────────────
# 3. feed_decay_monitor — kalıcı kayıp → triggered=True
# ──────────────────────────────────────────────────────────────────────
def test_feed_decay_monitor_triggers_on_persistent_loss(syn_db, tmp_path):
    """30 gün -3σ paper PnL → decay scan triggered."""
    out_path = tmp_path / "paper_trades.parquet"
    cfg = PaperTraderConfig(output_path=out_path)

    # Manuel paper trade tablosu yarat — net_pnl_pct sürekli kötü
    n = 40
    dates = pd.bdate_range("2024-01-01", periods=n)
    backtest_mean = 0.001
    backtest_std  = 0.005
    # Her gün net_pnl_pct = -3σ
    rows = []
    for d in dates:
        rows.append({
            "date": d, "formula_id": "decayed", "ticker": "T000",
            "weight": 1.0, "signal_value": 1.0,
            "entry_px": 10.0, "exit_px": 9.85,
            "gross_pnl_pct": -0.015,
            "slippage_bps": 0.0,
            "net_pnl_pct": backtest_mean - 3 * backtest_std,
        })
    df = pd.DataFrame(rows, columns=PAPER_TRADE_COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    from engine.risk.decay_monitor import DecayConfig
    decay_cfg = DecayConfig(delta=1e-5, lambda_threshold=0.005,
                             consecutive_days=5, sigma_floor=2.0)

    result = feed_decay_monitor(
        "decayed", backtest_mean, backtest_std, cfg=cfg, decay_cfg=decay_cfg,
    )
    assert result["triggered"] is True, f"trigger gelmedi: {result}"
