"""Veri Terminali (Arkhimedes) → Minerva v3 entegrasyon.

Tek seferlik manual import script'i. /Desktop/Veri/data altındaki parquet/csv/xlsx
dosyalarını Minerva v3'ün data/external/ klasörüne kopyalar.

Kullanım:
    venv/bin/python scripts/import_from_veri_terminali.py
    venv/bin/python scripts/import_from_veri_terminali.py --source /path/to/veri --update

Çıktı:
    data/external/{features_db,custody_features_db,...}.parquet
    data/external/_manifest.json (provenance kaydı)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path("/Users/unalronikarakoyun/Desktop/Veri/data")
EXTERNAL_DIR = ROOT / "data" / "external"

# (source filename, validate_kind) tuples
FILES_TO_COPY = [
    ("features_db.parquet",                  "parquet"),
    ("BIST_Tarihsel_Temel_Analiz.parquet",   "parquet"),
    ("custody_features_db.parquet",          "parquet"),
    ("EVDS_Verileri_2016_2026.xlsx",         "excel"),
    ("sector_map.csv",                       "csv"),
    ("delisted_tickers.txt",                 "text"),
    ("bist_fiili_dolasim_gunluk_2014_2026.csv", "csv"),
    ("mkk_hisse_demografi_2014_2025.csv",    "csv"),
]


def md5sum(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def validate(path: Path, kind: str) -> dict:
    """Smoke validate kopyalanan dosyayı; shape ve kolon bilgisi döner."""
    try:
        if kind == "parquet":
            df = pd.read_parquet(path)
            return {"ok": True, "shape": list(df.shape),
                    "columns": df.columns.tolist()[:10]}
        if kind == "excel":
            df = pd.read_excel(path)
            return {"ok": True, "shape": list(df.shape),
                    "columns": df.columns.tolist()}
        if kind == "csv":
            df = pd.read_csv(path, nrows=5)
            return {"ok": True, "columns": df.columns.tolist()}
        if kind == "text":
            n = sum(1 for _ in path.open("r"))
            return {"ok": True, "line_count": n}
    except Exception as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True}


def copy_file(src: Path, dst: Path, update: bool) -> dict:
    if dst.exists() and not update:
        return {"status": "skipped", "reason": "exists (use --update to overwrite)"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {"status": "copied"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                    help=f"Veri Terminali data dizini (varsayılan: {DEFAULT_SOURCE})")
    ap.add_argument("--dest", type=Path, default=EXTERNAL_DIR,
                    help=f"Hedef dizin (varsayılan: {EXTERNAL_DIR})")
    ap.add_argument("--update", action="store_true",
                    help="Mevcut dosyaları üzerine yaz (incremental sync)")
    args = ap.parse_args()

    if not args.source.exists():
        print(f"HATA: Kaynak dizin bulunamadı: {args.source}", file=sys.stderr)
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source": str(args.source),
        "dest":   str(args.dest),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "files":  [],
    }

    print(f"Kaynak: {args.source}")
    print(f"Hedef:  {args.dest}")
    print(f"Mod:    {'UPDATE' if args.update else 'CREATE'}")
    print()

    for fname, kind in FILES_TO_COPY:
        src = args.source / fname
        dst = args.dest / fname
        if not src.exists():
            print(f"  [SKIP] {fname} (kaynakta yok)")
            manifest["files"].append({"file": fname, "status": "missing_source"})
            continue

        copy_result = copy_file(src, dst, args.update)
        print(f"  [{copy_result['status'].upper()}] {fname} ({src.stat().st_size / 1024 / 1024:.1f} MB)")
        if copy_result["status"] == "skipped":
            manifest["files"].append({"file": fname, **copy_result})
            continue

        # Validate
        v = validate(dst, kind)
        entry = {
            "file": fname,
            "size_mb": round(dst.stat().st_size / 1024 / 1024, 2),
            "md5": md5sum(dst),
            "validate": v,
            **copy_result,
        }
        manifest["files"].append(entry)
        if v.get("ok"):
            shape = v.get("shape", [])
            print(f"    validate: shape={shape} cols={v.get('columns', [])[:5]}")
        else:
            print(f"    VALIDATE FAILED: {v.get('error')}")

    manifest_path = args.dest / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"\nManifest: {manifest_path}")
    print("Tamamlandı.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
