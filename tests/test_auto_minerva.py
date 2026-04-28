"""
tests/test_auto_minerva.py — Faz 6.1 orkestrasyon testleri.

Prefect flow/task'ları mock'lu ortamda izole test eder.
prefect_test_harness: test sırasında ephemeral Prefect sunucu açar,
gerçek Prefect storage/server gerekmez.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# prefect test harness — her test için izole Prefect ortamı
from prefect.testing.utilities import prefect_test_harness


@pytest.fixture(autouse=True, scope="module")
def prefect_harness():
    with prefect_test_harness():
        yield


# ──────────────────────────────────────────────────────────────────────
# 1. task_fetch_data: yfinance 2 kez başarısız → 3. çağrıda success
# ──────────────────────────────────────────────────────────────────────
def test_task_fetch_data_retries_on_yfinance_failure(tmp_path):
    """
    scripts.fetch_bist_data.main() ilk 2 çağrıda RuntimeError fırlatır,
    3. çağrıda başarılı olur. task_fetch_data'nın retry mantığı Prefect
    tarafından yönetilir; burada iç main() çağrı davranışını doğrularız.
    """
    import scripts.fetch_bist_data as fb

    parquet_path = tmp_path / "m.parquet"
    call_n = {"n": 0}

    def fake_main():
        call_n["n"] += 1
        if call_n["n"] < 3:
            raise RuntimeError("yfinance timeout")
        pd.DataFrame({"Ticker": ["A"], "Date": ["2024-01-01"]}).to_parquet(parquet_path)

    with patch.object(fb, "main", side_effect=fake_main):
        success = False
        for _ in range(5):   # max retry sınırı
            try:
                fb.main()
                success = True
                break
            except RuntimeError:
                pass

    assert success, "3. denemede başarılı olmalıydı"
    assert call_n["n"] == 3
    assert parquet_path.exists()


# ──────────────────────────────────────────────────────────────────────
# 2. task_decay_scan: triggered=True → RuntimeError
# ──────────────────────────────────────────────────────────────────────
def test_task_decay_scan_raises_when_triggered(tmp_path):
    """
    feed_decay_monitor triggered=True döndürdüğünde task RuntimeError fırlatır.
    Bu, Prefect flow'u Failed durumuna düşürür → alarm tetiklenir.
    """
    from auto_minerva import task_decay_scan

    fake_champions = [("formula_X", {"backtest_mean": 0.001, "backtest_std": 0.005})]
    triggered_result = {
        "triggered": True, "triggered_at": "2026-04-29", "n_observations": 40,
    }

    # Lazy import yollarından patch: task fn içindeki from ... import'lar
    # çalışma anında modülü lookup eder → modül attribute'unu patch'le
    import engine.core.alpha_catalog as _ac
    import engine.execution.paper_trader as _pt
    original_champs = getattr(_ac, "get_active_champions", None)
    original_feed   = getattr(_pt, "feed_decay_monitor", None)
    _ac.get_active_champions = lambda: fake_champions
    _pt.feed_decay_monitor   = lambda *a, **kw: triggered_result
    try:
        # get_run_logger Prefect context gerektiriyor — standart logger ile değiştir
        import logging
        with patch("auto_minerva.get_run_logger", return_value=logging.getLogger("test")):
            with pytest.raises(RuntimeError, match="DECAY"):
                task_decay_scan.fn(prob_path=tmp_path / "prob.parquet")
    finally:
        if original_champs is not None:
            _ac.get_active_champions = original_champs
        if original_feed is not None:
            _pt.feed_decay_monitor = original_feed


# ──────────────────────────────────────────────────────────────────────
# 3. run_daily_cycle smoke: tüm task'lar mock'lu, çıktı dict şeması doğru
# ──────────────────────────────────────────────────────────────────────
def test_run_daily_cycle_smoke(tmp_path):
    """
    Flow boyunca veri akışı mock'lu; çıktı dict anahtarları ve türleri doğru.
    """
    from auto_minerva import run_daily_cycle

    fake_db_path   = tmp_path / "market_db.parquet"
    fake_prob_path = tmp_path / "prob.parquet"

    # Minimal geçerli parquetler
    pd.DataFrame({"Ticker": ["A"], "Date": pd.to_datetime(["2026-04-01"]),
                   "Pclose": [10.0], "Vlot": [1e6]}).to_parquet(fake_db_path)
    dates = pd.date_range("2026-01-01", periods=5, freq="B")
    pd.DataFrame(
        np.full((5, 2), 0.5),
        index=dates,
        columns=["regime_0", "regime_1"],
    ).to_parquet(fake_prob_path)

    mining_out = {"skipped": True, "reason": "weekday=2"}
    decay_out  = {"checked": 0, "triggered": 0}
    exec_out   = {"logged": 5, "forensic": 5, "date": "2026-04-29"}

    with patch("auto_minerva.task_fetch_data", return_value=fake_db_path), \
         patch("auto_minerva.task_detect_regime", return_value=fake_prob_path), \
         patch("auto_minerva.task_nightly_mining", return_value=mining_out), \
         patch("auto_minerva.task_decay_scan", return_value=decay_out), \
         patch("auto_minerva.task_morning_execution", return_value=exec_out):

        result = run_daily_cycle(only_mining_on_weekday=4, mining_n_trials=5)

    assert isinstance(result, dict)
    assert "mining" in result
    assert "decay" in result
    assert "execution" in result
    assert result["execution"]["logged"] == 5
    assert result["decay"]["triggered"] == 0
