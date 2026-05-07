"""Birim testler: engine/util/calendar.py — çeyrek-sonu yardımcıları."""
from __future__ import annotations

import pandas as pd

from engine.util.calendar import (
    is_quarter_end_business_day,
    next_quarter_end,
    quarter_of,
)


# ── Testler ───────────────────────────────────────────────────────────────────

def test_quarter_end_business_day_handles_weekend():
    """Mar 31 hafta sonuna denk geldiğinde son iş günü (Cuma) çeyrek-sonu olmalı."""
    # 2024-03-31 Pazar → son iş günü 2024-03-29 Cuma olmalı
    bdays = pd.bdate_range("2024-01-01", "2024-12-31")

    assert is_quarter_end_business_day(pd.Timestamp("2024-03-29"), bdays), \
        "2024-03-29 (Cuma, Mart son iş günü) çeyrek-sonu olmalı"
    assert not is_quarter_end_business_day(pd.Timestamp("2024-03-28"), bdays), \
        "2024-03-28 son iş günü değil"

    # 2024-06-28 Cuma → Haziran son iş günü (29 Cmt, 30 Pzr)
    assert is_quarter_end_business_day(pd.Timestamp("2024-06-28"), bdays), \
        "2024-06-28 Haziran son iş günü, çeyrek-sonu olmalı"

    # 2025-03-31 Pazartesi → çeyrek-sonu olmalı
    bdays_25 = pd.bdate_range("2025-01-01", "2025-12-31")
    assert is_quarter_end_business_day(pd.Timestamp("2025-03-31"), bdays_25), \
        "2025-03-31 (Pzt, Mart son iş günü) çeyrek-sonu olmalı"

    # Çeyrek-içi rastgele bir gün → False
    assert not is_quarter_end_business_day(pd.Timestamp("2024-04-15"), bdays), \
        "2024-04-15 çeyrek-içi, çeyrek-sonu olmamalı"

    # Çeyrek-sonu olmayan ay (Şubat) → False
    assert not is_quarter_end_business_day(pd.Timestamp("2024-02-29"), bdays), \
        "Şubat çeyrek-sonu ayı değil"


def test_next_quarter_end_returns_next_period_end():
    """next_quarter_end(2016-04-15) → 2016-06-30 (Cuma) dönmeli."""
    bdays = pd.bdate_range("2016-01-01", "2016-12-31")

    # 2016-04-15 Cuma → bir sonraki çeyrek-sonu Haziran son iş günü
    nxt = next_quarter_end(pd.Timestamp("2016-04-15"), bdays)
    assert nxt == pd.Timestamp("2016-06-30").normalize(), (
        f"Beklenen 2016-06-30, gelen: {nxt}"
    )

    # 2016-12-31 Cmt → bir sonraki yıl Q1 sonu (Mart 31, 2017)
    bdays_combined = pd.bdate_range("2016-01-01", "2017-12-31")
    nxt2 = next_quarter_end(pd.Timestamp("2016-12-31"), bdays_combined)
    assert nxt2 == pd.Timestamp("2017-03-31").normalize(), (
        f"Beklenen 2017-03-31, gelen: {nxt2}"
    )


def test_quarter_of_basic():
    """quarter_of: ay → çeyrek eşlemesi."""
    assert quarter_of(pd.Timestamp("2024-01-15")) == 1
    assert quarter_of(pd.Timestamp("2024-06-30")) == 2
    assert quarter_of(pd.Timestamp("2024-09-15")) == 3
    assert quarter_of(pd.Timestamp("2024-12-31")) == 4
