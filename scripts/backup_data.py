#!/usr/bin/env python3
"""
scripts/backup_data.py — N43: data/ dizini yedekleme.

Kullanım:
    python scripts/backup_data.py          # restic ile yedekle
    python scripts/backup_data.py --check  # son yedek tarihini kontrol et

Gereksinimler:
    - restic CLI kurulu (brew install restic / apt install restic)
    - RESTIC_REPOSITORY env var (örn: /backup/minerva veya s3:bucket/minerva)
    - RESTIC_PASSWORD env var

Cron örneği (günlük 03:00):
    0 3 * * * cd /path/to/Minerva_v3_Studio && python scripts/backup_data.py
"""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("data")
BACKUP_LOG = DATA_DIR / "backup_log.json"

def _restic(*args) -> tuple[int, str]:
    """restic komutunu çalıştır, (returncode, stdout) döndür."""
    repo = os.getenv("RESTIC_REPOSITORY")
    pw = os.getenv("RESTIC_PASSWORD")
    if not repo or not pw:
        return 1, "RESTIC_REPOSITORY ve RESTIC_PASSWORD env var gerekli"
    env = {**os.environ, "RESTIC_REPOSITORY": repo, "RESTIC_PASSWORD": pw}
    result = subprocess.run(
        ["restic", *args],
        env=env, capture_output=True, text=True
    )
    return result.returncode, result.stdout + result.stderr

def backup():
    """data/ dizinini yedekle."""
    print(f"[{datetime.now().isoformat()}] Yedekleme başlıyor: {DATA_DIR}")
    rc, out = _restic("backup", str(DATA_DIR), "--tag", "minerva-auto")
    if rc == 0:
        _log_backup("success")
        print(f"Yedekleme tamamlandı:\n{out}")
    else:
        _log_backup("error", out)
        print(f"HATA: {out}", file=sys.stderr)
        sys.exit(1)
    # Son 30 snapshotu koru, eskilerini sil
    _restic("forget", "--keep-last", "30", "--prune")

def check():
    """Son yedek tarihini göster."""
    if BACKUP_LOG.exists():
        log = json.loads(BACKUP_LOG.read_text())
        print(f"Son yedek: {log.get('last_success', 'hiç yok')}")
        print(f"Durum: {log.get('last_status', 'bilinmiyor')}")
    else:
        print("Henüz yedek alınmamış.")

def _log_backup(status: str, detail: str = "") -> None:
    log = {}
    if BACKUP_LOG.exists():
        try: log = json.loads(BACKUP_LOG.read_text())
        except Exception: pass
    now = datetime.now(timezone.utc).isoformat()
    if status == "success":
        log["last_success"] = now
    log["last_status"] = status
    log["last_run"] = now
    if detail:
        log["last_detail"] = detail[:500]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_LOG.write_text(json.dumps(log, indent=2))

if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        backup()
