"""
engine/monitoring/counters.py — S14: Sessiz fallback sayaçları.

Production'da sessiz kalan hatalar (DML failure, DuckDB fallback) artık
sayılıp her çeyrek sonunda raporlanır. Bu sayede performans degradasyonu
erken fark edilir.

Kullanım:
    from engine.monitoring.counters import COUNTERS
    COUNTERS.inc("dml_failure")
    COUNTERS.report("Q2 2016")
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Dict

log = logging.getLogger(__name__)


@dataclass
class FallbackCounters:
    """Thread-safe fallback event sayacı."""
    _counts: Dict[str, int] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    _KNOWN_EVENTS = (
        "dml_failure",        # DML nötralizasyon başarısız → ham sinyal
        "duckdb_fallback",    # DuckDB IC hata → pandas fallback
        "holdout_skipped",    # Holdout verisi yetersiz → filtre atlandı
        "pbo_skipped",        # PBO matrisi oluşturulamadı
        "hmm_refit_failure",  # HMM refit başarısız
        "blend_failure",      # Sinyal blend başarısız
        "rl_retrain_failure", # RL retrain başarısız
    )

    def inc(self, event: str, n: int = 1) -> None:
        """Sayacı n artır (thread-safe)."""
        with self._lock:
            self._counts[event] = self._counts.get(event, 0) + n

    def get(self, event: str) -> int:
        with self._lock:
            return self._counts.get(event, 0)

    def reset(self) -> Dict[str, int]:
        """Tüm sayaçları sıfırla ve önceki değerleri döndür."""
        with self._lock:
            prev = dict(self._counts)
            self._counts.clear()
            return prev

    def report(self, label: str = "") -> None:
        """Tüm sıfır-olmayan sayaçları INFO seviyesinde logla."""
        with self._lock:
            counts = dict(self._counts)
        nonzero = {k: v for k, v in counts.items() if v > 0}
        if not nonzero:
            log.info("S14 FallbackCounters [%s]: tüm sistemler nominal", label)
            return
        total_events = sum(nonzero.values())
        log.warning(
            "S14 FallbackCounters [%s]: %d olay tespit edildi — %s",
            label, total_events,
            ", ".join(f"{k}={v}" for k, v in sorted(nonzero.items())),
        )

    def to_dict(self) -> Dict[str, int]:
        with self._lock:
            return dict(self._counts)


# Singleton — import edilen her modül aynı sayacı kullanır
COUNTERS = FallbackCounters()
