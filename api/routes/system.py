"""
api/routes/system.py — Sistem yönetim endpoint'leri.

N25: Kill-switch durumu + manuel reset.
     GET  /api/system/kill_switch       → durum + aktif olduğu zaman + sebep
     POST /api/system/kill_switch/reset → devre dışı bırak (API-key gerekli)
     POST /api/system/kill_switch/activate → etkinleştir (test + acil durum için)
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api.deps import verify_api_key

router = APIRouter(prefix="/api/system", tags=["system"])

_KILL_SWITCH_PATH = Path("data/.kill_switch")


class KillSwitchStatus(BaseModel):
    active: bool
    reason: str | None = None
    activated_at: str | None = None


class ActivateRequest(BaseModel):
    reason: str = "manual"


@router.get("/kill_switch", response_model=KillSwitchStatus)
def get_kill_switch_status() -> KillSwitchStatus:
    """Kill-switch durumu — auth gerektirmez (monitoring için)."""
    from engine.execution.paper_trader import is_kill_switch_active

    if not is_kill_switch_active():
        return KillSwitchStatus(active=False)

    try:
        with open(_KILL_SWITCH_PATH) as f:
            data = json.load(f)
        return KillSwitchStatus(
            active=True,
            reason=data.get("reason"),
            activated_at=data.get("activated_at"),
        )
    except Exception:
        return KillSwitchStatus(active=True)


@router.post(
    "/kill_switch/reset",
    dependencies=[Depends(verify_api_key)],
    summary="Kill-switch'i manuel olarak devre dışı bırak",
)
def reset_kill_switch() -> dict:
    """
    N25: UI veya CLI'dan tek çağrıyla kill-switch sıfırlama.
    Requires API-key auth.
    """
    from engine.execution.paper_trader import deactivate_kill_switch, is_kill_switch_active

    if not is_kill_switch_active():
        return {"status": "already_inactive", "message": "Kill-switch zaten devre dışı"}

    deactivate_kill_switch()
    return {"status": "deactivated", "message": "Kill-switch devre dışı bırakıldı"}


@router.post(
    "/kill_switch/activate",
    dependencies=[Depends(verify_api_key)],
    summary="Kill-switch'i manuel olarak etkinleştir",
)
def activate_kill_switch_endpoint(req: ActivateRequest) -> dict:
    """
    Manuel kill-switch etkinleştirme (test veya acil durum).
    Requires API-key auth.
    """
    from engine.execution.paper_trader import activate_kill_switch, is_kill_switch_active

    if is_kill_switch_active():
        return {"status": "already_active", "message": "Kill-switch zaten aktif"}

    activate_kill_switch(reason=f"manual: {req.reason}")
    return {"status": "activated", "reason": req.reason}
