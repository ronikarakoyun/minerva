"""Shared Memory kayıt defteri — zombie RAM koruması.

Açılan her SharedMemory bloğu buraya kaydedilir. Süreç başında stale
girişler temizlenir. El ile de çalıştırılabilir:

    python -m engine.util.shm_registry

macOS: `ipcs -m` ile orphan'lar listelenir.
Linux: /dev/shm/ altındaki dosyalar.
"""
from __future__ import annotations

import json
import logging
import os
from multiprocessing import shared_memory
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_REGISTRY_PATH = Path("data/shm_registry.json")
_MINERVA_SHM_PREFIX = "minerva_"


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save(registry: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(registry, indent=2))
    tmp.replace(path)


def register(name: str, path: Path | None = None) -> None:
    """Yeni bir SHM bloğunu kayıt defterine ekle."""
    p = Path(path or _DEFAULT_REGISTRY_PATH)
    reg = _load(p)
    reg[name] = {"pid": os.getpid()}
    _save(reg, p)


def unregister(name: str, path: Path | None = None) -> None:
    """SHM bloğunu kayıt defterinden çıkar."""
    p = Path(path or _DEFAULT_REGISTRY_PATH)
    reg = _load(p)
    reg.pop(name, None)
    _save(reg, p)


def cleanup_stale(path: Path | None = None, dry_run: bool = False) -> list[str]:
    """
    Kayıt defterindeki her blok için:
      - PID hâlâ çalışıyorsa → kayıt geçerli, atla.
      - PID ölmüşse → `shm.unlink()` ile belleği serbest bırak.

    Returns cleaned block names.
    """
    p = Path(path or _DEFAULT_REGISTRY_PATH)
    reg = _load(p)
    cleaned: list[str] = []

    for name, meta in list(reg.items()):
        pid = meta.get("pid", 0)
        if _pid_alive(pid):
            continue
        if not dry_run:
            try:
                shm = shared_memory.SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
                logger.info("Orphan SHM temizlendi: %s (PID=%d ölü)", name, pid)
            except FileNotFoundError:
                pass  # zaten silinmiş
            except Exception as exc:
                logger.warning("SHM temizleme hatası %s: %s", name, exc)
            reg.pop(name)
        cleaned.append(name)

    if not dry_run:
        _save(reg, p)
    return cleaned


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cleaned = cleanup_stale()
    if cleaned:
        print(f"Temizlenen orphan bloklar: {cleaned}")
    else:
        print("Orphan SHM bloğu bulunamadı.")
