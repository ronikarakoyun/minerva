"""Sıcak state katmanı (Faz 1.5).

PostgreSQL: RL agent state, champion catalog, audit log, virtual orders.
Redis Streams: günlük sinyal pub/sub.

Bu paket Phase 3 shadow fund servislerinin state ihtiyacını karşılar.
asyncpg / redis-py yoksa modüller import edilebilir kalır ama
fonksiyonlar NotImplementedError fırlatır.
"""
