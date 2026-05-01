"""Prometheus metrics — N41. Gracefully degraded if prometheus_client not installed."""
try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
    _PROM = True
except ImportError:
    _PROM = False

# Metric definitions (no-ops when prometheus not installed)
class _NoOp:
    def inc(self, *a, **k): pass
    def set(self, *a, **k): pass
    def observe(self, *a, **k): pass
    def labels(self, *a, **k): return self

def _counter(name, help, labels=()):
    if _PROM:
        return Counter(name, help, labels)
    return _NoOp()

def _gauge(name, help, labels=()):
    if _PROM:
        return Gauge(name, help, labels)
    return _NoOp()

def _histogram(name, help, labels=()):
    if _PROM:
        return Histogram(name, help, labels)
    return _NoOp()

# Custom metrics
mining_jobs_total = _counter("minerva_mining_jobs_total", "Total mining jobs", ["status"])
formula_acceptance_rate = _gauge("minerva_formula_acceptance_rate", "Formula acceptance rate")
decay_alarms_total = _counter("minerva_decay_alarms_total", "Decay alarms fired")
paper_pnl_daily = _gauge("minerva_paper_pnl_daily", "Daily paper PnL")
kill_switch_active = _gauge("minerva_kill_switch_active", "Kill switch active (1/0)")

def start_metrics_server(port: int = 9090) -> None:
    if _PROM:
        start_http_server(port)
