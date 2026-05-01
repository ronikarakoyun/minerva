#!/usr/bin/env python3
"""
scripts/dr_drill.py — N42: Disaster Recovery (DR) Drill Protokolü.

Çalıştırma:
    python scripts/dr_drill.py          # tam DR drill
    python scripts/dr_drill.py --check  # sadece kontrol (restore etme)
    python scripts/dr_drill.py --report # son drill raporunu göster

Hedef:
  - RTO (Recovery Time Objective): < 30 dakika
  - RPO (Recovery Point Objective): < 24 saat (günlük backup)

Adımlar:
  1. Son backup'ın varlığını kontrol et
  2. Backup'tan data/ dizinini geçici bir alana restore et
  3. Kritik dosyaların bütünlüğünü doğrula (parquet, json, pkl)
  4. Sonuçları data/dr_drill_report.json'a yaz
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path("data")
DRILL_REPORT = DATA_DIR / "dr_drill_report.json"

CRITICAL_FILES = [
    "data/market_db.parquet",
    "data/alpha_catalog.json",
    "data/regime_metadata.json",
]

OPTIONAL_FILES = [
    "data/paper_trades.parquet",
    "data/replay_buffer.json",
    "data/regime_hmm.pkl",
    "data/jobs.db",
]


def _check_restic_available() -> bool:
    try:
        result = subprocess.run(["restic", "version"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_env() -> dict[str, bool]:
    return {
        "RESTIC_REPOSITORY": bool(os.getenv("RESTIC_REPOSITORY")),
        "RESTIC_PASSWORD": bool(os.getenv("RESTIC_PASSWORD")),
        "restic_binary": _check_restic_available(),
    }


def _run_restic(*args) -> tuple[int, str]:
    env = dict(os.environ)
    result = subprocess.run(
        ["restic", *args],
        env=env, capture_output=True, text=True, timeout=300
    )
    return result.returncode, result.stdout + result.stderr


def _verify_file_integrity(path: str) -> dict:
    """Dosya var mı ve sağlıklı mı kontrol et."""
    p = Path(path)
    result = {"path": path, "exists": p.exists(), "size_bytes": 0, "valid": False, "error": None}
    if not p.exists():
        return result
    result["size_bytes"] = p.stat().st_size
    if result["size_bytes"] == 0:
        result["error"] = "boş dosya"
        return result
    ext = p.suffix.lower()
    try:
        if ext == ".parquet":
            import pandas as pd
            df = pd.read_parquet(p)
            result["valid"] = len(df) > 0
            result["rows"] = len(df)
        elif ext == ".json":
            with open(p) as f:
                data = json.load(f)
            result["valid"] = True
            result["items"] = len(data) if isinstance(data, list) else "dict"
        elif ext == ".pkl":
            import pickle
            with open(p, "rb") as f:
                pickle.load(f)  # noqa: S301
            result["valid"] = True
        elif ext == ".db":
            import sqlite3
            conn = sqlite3.connect(str(p))
            conn.execute("SELECT 1")
            conn.close()
            result["valid"] = True
        else:
            result["valid"] = True
    except Exception as e:
        result["error"] = str(e)[:200]
    return result


def check_mode() -> dict:
    """Sadece mevcut dosyaları kontrol et, restore etme."""
    print("=== DR CHECK MODE ===")
    start = datetime.now(timezone.utc)
    results = []

    for f in CRITICAL_FILES:
        r = _verify_file_integrity(f)
        status = "✅" if r["valid"] else "❌"
        print(f"  {status} {f} ({r.get('size_bytes', 0):,} bytes)")
        results.append(r)

    for f in OPTIONAL_FILES:
        r = _verify_file_integrity(f)
        status = "⚠️" if not r["valid"] else "✅"
        if r["exists"]:
            print(f"  {status} {f} ({r.get('size_bytes', 0):,} bytes) [optional]")

    critical_ok = all(r["valid"] for r in results)
    env = _check_env()

    report = {
        "mode": "check",
        "timestamp": start.isoformat(),
        "critical_files_ok": critical_ok,
        "files": results,
        "env": env,
    }
    _save_report(report)
    print(f"\n{'✅ TÜM KRİTİK DOSYALAR OK' if critical_ok else '❌ KRİTİK DOSYA HATASI'}")
    return report


def full_drill() -> dict:
    """Tam DR drill — backup var mı kontrol et, geçici alana restore et, doğrula."""
    print("=== DR FULL DRILL ===")
    start = datetime.now(timezone.utc)
    report: dict = {"mode": "full_drill", "timestamp": start.isoformat(), "steps": []}

    # Step 1: Env kontrolü
    env = _check_env()
    report["env"] = env
    if not all(env.values()):
        missing = [k for k, v in env.items() if not v]
        report["status"] = "SKIP"
        report["reason"] = f"Eksik yapılandırma: {missing}"
        print(f"⚠️  DR drill atlandı: {missing}")
        print("   restic + RESTIC_REPOSITORY + RESTIC_PASSWORD gerekli")
        _save_report(report)
        return report

    # Step 2: Son snapshot'ı listele
    print("1/4 Son backup snapshot kontrol ediliyor...")
    rc, out = _run_restic("snapshots", "--json", "--last", "1")
    if rc != 0:
        report["status"] = "FAIL"
        report["reason"] = f"restic snapshots hatası: {out[:200]}"
        _save_report(report)
        print(f"❌ {report['reason']}")
        return report

    try:
        snapshots = json.loads(out)
        if not snapshots:
            report["status"] = "FAIL"
            report["reason"] = "Hiç snapshot bulunamadı — önce backup alın"
            _save_report(report)
            print(f"❌ {report['reason']}")
            return report
        last_snap = snapshots[-1]
        snap_id = last_snap.get("id", "?")[:8]
        snap_time = last_snap.get("time", "?")
        print(f"   Son snapshot: {snap_id} @ {snap_time}")
        report["steps"].append({"step": "snapshot_found", "id": snap_id, "time": snap_time})
    except json.JSONDecodeError:
        report["status"] = "FAIL"
        report["reason"] = "restic snapshot JSON parse hatası"
        _save_report(report)
        return report

    # Step 3: Geçici dizine restore
    print("2/4 Geçici alana restore ediliyor...")
    with tempfile.TemporaryDirectory(prefix="minerva_dr_") as tmpdir:
        rc, out = _run_restic("restore", "latest", "--target", tmpdir, "--include", "data/")
        if rc != 0:
            report["status"] = "FAIL"
            report["reason"] = f"restic restore hatası: {out[:200]}"
            _save_report(report)
            print(f"❌ {report['reason']}")
            return report
        print(f"   Restore tamamlandı: {tmpdir}")
        report["steps"].append({"step": "restore_ok", "target": tmpdir})

        # Step 4: Bütünlük doğrulama
        print("3/4 Dosya bütünlüğü doğrulanıyor...")
        file_results = []
        for f in CRITICAL_FILES:
            restored_path = os.path.join(tmpdir, f)
            r = _verify_file_integrity(restored_path)
            r["original_path"] = f
            status = "✅" if r["valid"] else "❌"
            print(f"   {status} {f}")
            file_results.append(r)

        critical_ok = all(r["valid"] for r in file_results)
        report["files"] = file_results
        report["critical_files_ok"] = critical_ok

    # Step 5: RTO ölçümü
    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    print(f"4/4 Tamamlandı. Süre: {elapsed:.0f}s (RTO hedef: <1800s)")
    report["elapsed_seconds"] = elapsed
    report["rto_ok"] = elapsed < 1800
    report["status"] = "PASS" if critical_ok else "FAIL"
    report["steps"].append({"step": "rto_check", "elapsed_s": elapsed, "ok": report["rto_ok"]})

    _save_report(report)
    print(f"\n{'✅ DR DRILL BAŞARILI' if report['status'] == 'PASS' else '❌ DR DRILL BAŞARISIZ'}")
    print(f"   RTO: {elapsed:.0f}s | Hedef: <1800s | {'✅' if report['rto_ok'] else '❌'}")
    return report


def show_report() -> None:
    if not DRILL_REPORT.exists():
        print("Henüz DR drill raporu yok. 'python scripts/dr_drill.py' ile çalıştırın.")
        return
    with open(DRILL_REPORT) as f:
        report = json.load(f)
    print(f"Son DR drill: {report.get('timestamp', '?')}")
    print(f"Mod: {report.get('mode', '?')} | Durum: {report.get('status', 'N/A')}")
    print(f"Kritik dosyalar: {'OK' if report.get('critical_files_ok') else 'HATA'}")
    if "elapsed_seconds" in report:
        print(f"RTO: {report['elapsed_seconds']:.0f}s")
    if report.get("reason"):
        print(f"Not: {report['reason']}")


def _save_report(report: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(DRILL_REPORT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Rapor: {DRILL_REPORT}")


if __name__ == "__main__":
    if "--report" in sys.argv:
        show_report()
    elif "--check" in sys.argv:
        r = check_mode()
        sys.exit(0 if r.get("critical_files_ok") else 1)
    else:
        r = full_drill()
        sys.exit(0 if r.get("status") in ("PASS", "SKIP") else 1)
