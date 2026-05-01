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

# macOS Python 3.14'te SSL CA bundle eksik kalabiliyor → certifi'ye yönlendir.
# Sadece env zaten set değilse müdahale et (kullanıcı override edebilsin).
try:
    import certifi as _certifi
    _CA_BUNDLE = _certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", _CA_BUNDLE)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", _CA_BUNDLE)
except ImportError:
    _CA_BUNDLE = None

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

    # Telegram 4096 karakter sınırı — uzunsa satır sınırlarında parçala
    chunks = _split_for_telegram(text, max_len=3900)

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    all_ok = True
    for i, chunk in enumerate(chunks):
        suffix = ""
        if len(chunks) > 1:
            suffix = f"\n\n— ({i + 1}/{len(chunks)}) —"

        def _post(body_text: str, mode: Optional[str]):
            payload = {
                "chat_id": chat_id,
                "text": body_text,
                "disable_web_page_preview": disable_web_page_preview,
            }
            if mode:
                payload["parse_mode"] = mode
            return requests.post(url, json=payload, timeout=timeout,
                                 verify=_CA_BUNDLE if _CA_BUNDLE else True)

        try:
            r = _post(chunk + suffix, parse_mode)
            # Markdown parse hatası (400) → plain text fallback
            if r.status_code == 400 and parse_mode:
                logger.info("Telegram Markdown reddedildi, plain-text fallback (chunk %d/%d)",
                            i + 1, len(chunks))
                stripped = _strip_markdown(chunk) + suffix
                r = _post(stripped, None)
            r.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Telegram gönderim hatası (chunk %d/%d): %s",
                           i + 1, len(chunks), e)
            all_ok = False
    return all_ok


def _strip_markdown(text: str) -> str:
    """Telegram Markdown reddettiğinde plain'e düşmek için işaretleri kaldır."""
    import re as _re
    text = _re.sub(r"`([^`\n]*)`", r"\1", text)        # `code` → code
    text = _re.sub(r"\*([^\*\n]+)\*", r"\1", text)     # *bold* → bold
    text = _re.sub(r"_([^_\n]+)_", r"\1", text)        # _italic_ → italic
    return text


def _split_for_telegram(text: str, max_len: int = 3900) -> list[str]:
    """
    Mesajı satır sınırlarında parçala (Markdown bütünlüğü için).
    4096 karakter Telegram limitini aşan rapor için kullanılır.
    """
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in text.split("\n"):
        # +1 ekleme: \n için
        if current_len + len(line) + 1 > max_len and current:
            chunks.append("\n".join(current))
            current = [line]
            current_len = len(line) + 1
        else:
            current.append(line)
            current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks
