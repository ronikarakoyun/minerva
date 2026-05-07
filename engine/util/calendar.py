"""Takvim yardımcıları — çeyrek-sonu iş günü tespiti.

Mar 31 / Haz 30 / Eyl 30 / Ara 31 hafta sonuna denk gelirse o ayın son iş günü
(genellikle Cuma) çeyrek-sonu olarak kabul edilir. BIST iş takvimi varsayılır
(pd.tseries.offsets.BDay — Pazartesi-Cuma).

Kullanım:
    from engine.util.calendar import is_quarter_end_business_day, next_quarter_end
    bdays = pd.bdate_range("2016-01-01", "2025-12-31")
    if is_quarter_end_business_day(date_t, bdays):
        run_mining(...)
"""
from __future__ import annotations

import pandas as pd

QUARTER_END_MONTHS = {3, 6, 9, 12}


def is_quarter_end_business_day(
    date: pd.Timestamp,
    business_days: "pd.DatetimeIndex | None" = None,
) -> bool:
    """date, takvim çeyreğinin son iş günü mü?

    business_days verilirse: o iş takviminin içinden seçim yapılır
    (BIST tatilleri eklenmek istenirse). None ise pandas BDay (Pzt-Cu) varsayılır.
    """
    date = pd.Timestamp(date)
    if date.month not in QUARTER_END_MONTHS:
        return False

    if business_days is None:
        # Aynı çeyreğin son iş günü = ayın son gününe ≤ tarih, BDay'de
        last_of_month = date + pd.offsets.MonthEnd(0)
        last_bday = last_of_month if last_of_month.weekday() < 5 \
            else last_of_month - pd.offsets.BDay(1)
        return date.normalize() == last_bday.normalize()

    # business_days bazlı: aynı yıl-ay'daki en büyük iş günü
    bdays = pd.DatetimeIndex(business_days)
    same_month = bdays[(bdays.year == date.year) & (bdays.month == date.month)]
    if len(same_month) == 0:
        return False
    return date.normalize() == same_month.max().normalize()


def next_quarter_end(
    date: pd.Timestamp,
    business_days: "pd.DatetimeIndex | None" = None,
) -> pd.Timestamp:
    """date'ten sonraki ilk çeyrek-sonu iş günü."""
    date = pd.Timestamp(date)

    if business_days is None:
        # Bir sonraki QuarterEnd ay'ının son iş gününü hesapla
        cur = date + pd.tseries.offsets.QuarterEnd(startingMonth=12, n=1)
        if cur.weekday() >= 5:
            cur = cur - pd.offsets.BDay(1)
        if cur <= date:
            cur = (date + pd.offsets.QuarterEnd(startingMonth=12, n=1)
                   + pd.offsets.QuarterEnd(startingMonth=12, n=1))
            if cur.weekday() >= 5:
                cur = cur - pd.offsets.BDay(1)
        return cur.normalize()

    bdays = pd.DatetimeIndex(business_days)
    candidates = [d for d in bdays if d > date and is_quarter_end_business_day(d, bdays)]
    if not candidates:
        # Aralıkta yoksa son iş günü
        return bdays[-1].normalize() if len(bdays) > 0 else date.normalize()
    return pd.Timestamp(candidates[0]).normalize()


def quarter_of(date: pd.Timestamp) -> int:
    """date'in takvim çeyreği (1..4)."""
    return (pd.Timestamp(date).month - 1) // 3 + 1
