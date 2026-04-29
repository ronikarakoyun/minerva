"""
engine/notifications/telegram.py — Telegram bot mesajı gönderici.

Token + chat_id `.env` dosyasından (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
ya da ortam değişkenlerinden okunur. Eksikse fonksiyon sessizce no-op yapar
(test ve dev ortamı için sorunsuz).

Kullanım:
    from engine.notifications import send_telegram
    send_telegram("🟢 Günaydın! Portföy harmanlandı.")

    # Markdown destekli:
    send_telegram("*kalın* + _italik_", parse_mode="Markdown")

Ortam değişkenleri:
    TELEGRAM_BOT_TOKEN  — BotFather'ın verdiği token
    TELEGRAM_CHAT_ID    — getUpdates'ten alınan chat ID
    TELEGRAM_DISABLED   — "1" → tüm gönderimleri kapat (test için)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# .env dosyasını ilk import'ta yükle (varsa)
_DOTENV_PATH = Path(".env")
if _DOTENV_PATH.exists():
    for line in _DOTENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def telegram_enabled() -> bool:
    """Telegram konfigürasyonu eksiksiz ve aktif mi?"""
    if os.environ.get("TELEGRAM_DISABLED") == "1":
        return False
    return bool(
        os.environ.get("TELEGRAM_BOT_TOKEN")
        and os.environ.get("TELEGRAM_CHAT_ID")
    )


def send_telegram(
    text: str,
    parse_mode: Optional[str] = None,
    disable_web_page_preview: bool = True,
    timeout: float = 10.0,
) -> bool:
    """
    Telegram mesajı gönder. Token/chat_id eksikse veya gönderim başarısızsa
    sessizce False döner (akışı kesmez).

    Parameters
    ----------
    text : str
        Mesaj içeriği (max 4096 karakter Telegram limiti).
    parse_mode : "Markdown" | "MarkdownV2" | "HTML" | None
        Format desteği.
    disable_web_page_preview : bool
        Link önizlemesini kapat (default True).
    timeout : float
        HTTP timeout saniye.

    Returns
    -------
    bool
        Başarılı gönderildiyse True.
    """
    if not telegram_enabled():
        logger.debug("Telegram disabled or unconfigured — mesaj gönderilmedi")
        return False

    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = os.environ["TELEGRAM_CHAT_ID"]

    # Telegram 4096 karakter sınırı — uzunsa kes
    if len(text) > 4000:
        text = text[:3990] + "\n…(kısaltıldı)"

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "disable_web_page_preview": disable_web_page_preview,
    }
    if parse_mode:
        payload["parse_mode"] = parse_mode

    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return True
    except requests.RequestException as e:
        logger.warning("Telegram gönderim hatası: %s", e)
        return False
