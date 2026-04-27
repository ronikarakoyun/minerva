"""GET / DELETE /api/catalog — alpha_catalog.json üzerinden CRUD."""
from __future__ import annotations

import io

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from api.schemas import CatalogRecord
from engine.core.alpha_catalog import _load_raw, _save_raw, load_catalog

router = APIRouter()


@router.get("", response_model=list[CatalogRecord])
def list_catalog() -> list[CatalogRecord]:
    """Tüm kayıtlar — `engine.alpha_catalog.load_catalog()` sıralı."""
    raw = load_catalog()
    out: list[CatalogRecord] = []
    for r in raw:
        try:
            out.append(CatalogRecord(**r))
        except Exception:
            # Eski şema kayıtları drop etme — extra=allow var, geçmeli
            out.append(CatalogRecord(
                formula=r.get("formula", "?"),
                ic=r.get("ic", 0.0),
                rank_ic=r.get("rank_ic", 0.0),
                adj_ic=r.get("adj_ic", 0.0),
            ))
    return out


@router.get("/export.csv")
def export_csv() -> StreamingResponse:
    """Tüm kayıtları CSV olarak indir."""
    records = load_catalog()
    if not records:
        raise HTTPException(status_code=404, detail="Katalog boş")
    df = pd.json_normalize(records, sep="_")
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=alpha_catalog.csv"},
    )


@router.delete("")
def clear_catalog() -> dict[str, int]:
    """Tüm katalog'u temizle. data/alpha_catalog.json boş array olur."""
    _save_raw([])
    return {"deleted": 1, "remaining": 0}


@router.delete("/{formula:path}")
def delete_one(formula: str) -> dict[str, int]:
    """Tekil formül sil. URL-encoded formül kabul edilir."""
    records = _load_raw()
    before = len(records)
    records = [r for r in records if r.get("formula") != formula]
    after = len(records)
    if before == after:
        raise HTTPException(status_code=404, detail=f"Formül bulunamadı: {formula}")
    _save_raw(records)
    return {"deleted": before - after, "remaining": after}
