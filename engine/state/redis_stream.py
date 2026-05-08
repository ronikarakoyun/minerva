"""Redis Streams — günlük sinyal pub/sub (Faz 1.5).

Stream'ler:
- signals     : signal_daemon → dashboard, executor
- pnl_updates : reconciler → dashboard, alerts
- alerts      : sistem geneli kritik olaylar (Telegram entegrasyonu)

Kullanım:
    from engine.state.redis_stream import RedisStreams

    rs = RedisStreams(url="redis://localhost:6379/0")
    await rs.connect()

    # Producer
    await rs.publish_signal(date=today(), n_positions=15, leverage=1.5,
                             tickers=["THYAO", "GARAN"])

    # Consumer
    async for msg in rs.consume("signals", consumer_group="dashboard"):
        process(msg)
"""
from __future__ import annotations

import json
import logging
from datetime import date as _date
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)


class RedisStreams:
    """redis-py 5.x async client wrapper."""

    def __init__(self, url: str = "redis://localhost:6379/0"):
        self.url = url
        self._client = None

    async def connect(self) -> None:
        try:
            import redis.asyncio as redis
        except ImportError as exc:
            raise ImportError(
                "redis-py gerekli: `pip install redis>=5.0`"
            ) from exc

        self._client = redis.from_url(self.url, decode_responses=True)
        # Bağlantı doğrulaması
        await self._client.ping()
        logger.info("RedisStreams bağlandı: %s", self.url)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ── Producer ──────────────────────────────────────────────────────────────

    async def publish_signal(self, as_of: _date, n_positions: int,
                              leverage: float, tickers: list[str],
                              extra: Optional[dict] = None) -> str:
        payload = {
            "date": str(as_of),
            "n_positions": str(n_positions),
            "leverage": f"{leverage:.4f}",
            "tickers": json.dumps(tickers),
        }
        if extra:
            payload.update({k: json.dumps(v) for k, v in extra.items()})
        msg_id = await self._client.xadd("signals", payload, maxlen=10000)
        return msg_id

    async def publish_pnl(self, as_of: _date, equity: float,
                          daily_pnl_pct: float, drawdown: float) -> str:
        payload = {
            "date": str(as_of),
            "equity": f"{equity:.2f}",
            "daily_pnl_pct": f"{daily_pnl_pct:.6f}",
            "drawdown": f"{drawdown:.6f}",
        }
        return await self._client.xadd("pnl_updates", payload, maxlen=10000)

    async def publish_alert(self, severity: str, source: str, message: str) -> str:
        payload = {
            "severity": severity,  # info | warn | critical
            "source": source,
            "message": message,
        }
        return await self._client.xadd("alerts", payload, maxlen=1000)

    # ── Consumer ──────────────────────────────────────────────────────────────

    async def consume(self, stream: str, consumer_group: str,
                       consumer_name: str = "default",
                       block_ms: int = 5000) -> AsyncIterator[dict]:
        """Consumer group ile akış oku (at-least-once delivery)."""
        # Group oluştur (idempotent)
        try:
            await self._client.xgroup_create(stream, consumer_group, id="$",
                                                mkstream=True)
        except Exception:
            pass  # zaten varsa sessizce devam

        while True:
            entries = await self._client.xreadgroup(
                consumer_group, consumer_name,
                streams={stream: ">"}, count=10, block=block_ms,
            )
            if not entries:
                continue
            for _stream_name, messages in entries:
                for msg_id, fields in messages:
                    yield {"id": msg_id, **fields}
                    # ack
                    await self._client.xack(stream, consumer_group, msg_id)

    async def latest(self, stream: str, count: int = 10) -> list[dict]:
        """Son N mesajı al (consumer group olmadan, dashboard için)."""
        entries = await self._client.xrevrange(stream, "+", "-", count=count)
        return [{"id": mid, **fields} for mid, fields in entries]
