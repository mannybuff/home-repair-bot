# app/api/orchestrator.py

from __future__ import annotations

from typing import Optional, Dict, Any

from fastapi import (
    APIRouter,
    HTTPException,
    Header,
    UploadFile,
    File,
    Form,
    Depends,
)
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

from app.core.settings import settings, get_settings
from app.services.orchestrators import synth_entry, SessionContext, run_fusion, run_step_dialog

import time 

router = APIRouter(prefix="/api/v1/rag/orchestrator", tags=["orchestrator"])


# ---------------------------------------------------------------------------
# Auth helper (kept in API layer, same behavior as synthesis)
# ---------------------------------------------------------------------------


def _require_key(x_api_key: Optional[str]) -> None:
    """
    Ensure the caller supplied the expected API key.

    This stays at the API layer so auth behavior is co-located with the route.
    """
    want = getattr(settings, "api_key", None)
    if want and (not x_api_key or x_api_key != want):
        raise HTTPException(status_code=401, detail="invalid api key")

# -----------------------------------------------------
# NEW HEALTH PING (NEEDS TO BE WIRED INTO APP FRONTEND
# -----------------------------------------------------

@router.get("/ping")
async def orchestrator_ping(
    x_api_key: Optional[str] = Header(None),
    settings=Depends(get_settings),
) -> JSONResponse:
    """
    Lightweight health / readiness probe for the orchestrator stack.

    This replaces the old /health route; it lives on the orchestrator
    router so we can keep all v2 wiring in one place.
    """
    # Keep auth behavior consistent; remove this if you want open health.
    _require_key(x_api_key)

    body: Dict[str, Any] = {
        "ok": True,
        "service": "orchestrator",
        "version": getattr(settings, "schema_version", "v2"),
        "timestamp": time.time(),
    }
    return JSONResponse(content=jsonable_encoder(body))

# ---------------------------------------------------------------------------
# POST /dialog – v2 front door (currently: synth_entry only)
# ---------------------------------------------------------------------------


@router.post("/dialog")
async def orchestrate_dialog(
    text: str = Form(""),
    dialog_id: str = Form(""),
    image: UploadFile | None = File(None),
    x_api_key: Optional[str] = Header(None),
    settings=Depends(get_settings),
) -> JSONResponse:
    """
    Orchestration v2 entrypoint.

    Current behavior (Phase 1):
    - Enforces API key.
    - Reads optional text and image.
    - Calls `synth_entry` to build a SessionContext.
    - If `emergency_block` is True, returns a minimal refusal envelope.
    - Otherwise, echoes a trimmed SessionContext for debugging / curl probes.

    Next phases:
    - Use SessionContext.session_mode to route to `run_fusion` or `run_step_dialog`.
    """
    _require_key(x_api_key)

    # Read image bytes (if present)
    image_bytes: Optional[bytes] = None
    if image is not None:
        image_bytes = await image.read()

    # 1) Build session context via synth_entry
    ctx: SessionContext = synth_entry(
        dialog_id=dialog_id or None,
        user_text=text or None,
        image_bytes=image_bytes,
        settings=settings,
    )

    # 2) Emergency short-circuit
    if ctx.get("emergency_block"):
        reason = ctx.get("emergency_reason") or (
            "This might be an emergency. Please evacuate if needed and "
            "contact local emergency services immediately."
        )

        body: Dict[str, Any] = {
            "ok": False,
            "blocked": True,          # Hard gate – app shows dialog, no follow-up
            "requires_ack": False,
            "ack_stage": 0,
            "reason": reason,
            "dialog_id": ctx.get("dialog_id"),
            # Let the app reuse the same text in the safety dialog
            "answer": reason,
        }
        return JSONResponse(content=jsonable_encoder(body))
    
    # 3) Intent gate – not home-repair → soft refusal, no heavy fusion.
    # We still return a normal envelope shape so the app can show the text and
    # optionally surface a "New Task" UI without burning tokens on search.
    if not ctx.get("is_home_repair", True):
        msg = (
            "This doesn't look like a home-repair or DIY maintenance question. "
            "Please start a new task with a home-related issue "
            "(plumbing, electrical, walls, floors, etc.)."
        )

        body: Dict[str, Any] = {
            "ok": False,
            "blocked": False,         # Soft refusal – no safety gate UI
            "requires_ack": False,
            "ack_stage": 0,
            "dialog_id": ctx.get("dialog_id"),
            "answer": msg,
            "orchestration": {
                "session_mode": ctx.get("session_mode"),
                "topic_intent": ctx.get("intent") or {},
            },
        }
        return JSONResponse(content=jsonable_encoder(body))

    # 4) Session gate – for now:
    #    - FIRST_TURN / TOPIC_SHIFT → run_fusion
    #    - SAME_TOPIC → placeholder response until run_step_dialog is ready
    mode = ctx.get("session_mode")

    if mode in ("FIRST_TURN", "TOPIC_SHIFT"):
        body = run_fusion(ctx, settings=settings)
        # expose session_mode at top level as well for easier jq probing
        body["session_mode"] = mode
        
    elif mode == "SAME_TOPIC":
        body = run_step_dialog(ctx, settings=settings)
        body["session_mode"] = mode
        
    else:
        # Fallback: treat as first turn
        body = run_fusion(ctx, settings=settings)
        body["session_mode"] = mode or "UNKNOWN"

    return JSONResponse(content=jsonable_encoder(body))

