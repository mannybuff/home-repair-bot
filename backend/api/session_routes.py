# app/api/session_routes.py
"""
SESSION & DIALOG ROUTERS

This module consolidates the legacy session-related routes:

  • /api/v1/rag/session/index
  • /api/v1/rag/session/resume
  • /api/v1/rag/session/memory
  • /api/v1/rag/session/summarize
  • /api/v1/dialog/state

All filesystem paths and response shapes are preserved to remain compatible
with the original app/api/sessions.py and app/api/dialog.py modules.

Core rules:
  • All filesystem/JSON work is delegated to app.utils.session_utils.
  • Session summarization and memory building are delegated to
    app.services.sessions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
from datetime import datetime

from fastapi import APIRouter, HTTPException, Header, Query, Form
import json as _json

# Settings (for sessions_dir, api_key, schema_version)
try:
    from app.core.settings import settings  # type: ignore
except Exception:
    class _Fallback:
        sessions_dir = "./data/sessions"
        api_key: Optional[str] = None
        schema_version: str = "1.0"

    settings = _Fallback()  # type: ignore

# Session utilities & services
from app.utils.session_utils import (
    session_root,
    events_dir,
    list_event_files,
    load_event_payload,
    load_json,
    write_json,
)
from app.services.sessions import (
    build_memory,
    load_memory,
    run_next_step_turn,
)
from app.models.session_class import SessionMemory, DialogueContext, RecentTurn

# Routers: one for rag/session, one for dialog state
router_sessions = APIRouter(prefix="/api/v1/rag/session", tags=["sessions"])
router_dialog = APIRouter(prefix="/api/v1/dialog", tags=["dialog"])


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _require_key(x_api_key: Optional[str]) -> None:
    """
    Enforce X-API-Key header if settings.api_key is configured.
    """
    want = getattr(settings, "api_key", None)
    if want:
        if not x_api_key or x_api_key != want:
            raise HTTPException(status_code=401, detail="invalid api key")


def _sessions_root() -> Path:
    """
    Compatibility wrapper around utils.session_utils.session_root().
    """
    return session_root()


# ---------------------------------------------------------------------------
# Lightweight index
# ---------------------------------------------------------------------------

def _mk_index() -> Dict[str, Any]:
    """
    Lightweight indexer (read-only request path).
    Mirrors the legacy behavior from app/api/sessions.py.
    """
    root = _sessions_root()
    out: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "dialogs": {},
    }

    # Each dialog dir under sessions_root
    for dlg_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        dlg_id = dlg_dir.name
        ev_dir = dlg_dir / "events"
        if not ev_dir.exists():
            continue

        # Use event files for first+last timestamps and hazards
        files = sorted([p for p in ev_dir.glob("*.json") if p.is_file()])
        if not files:
            continue

        first = load_event_payload(files[0]) or {}
        last = load_event_payload(files[-1]) or {}

        def _extract_hazards(payload: Dict[str, Any]) -> List[str]:
            """
            Extract coarse hazard tags from safety payloads or warnings.
            """
            hazards: List[str] = []
            safety = payload.get("safety")
            if isinstance(safety, dict):
                for key in ("stop", "caution"):
                    v = safety.get(key)
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, str) and item:
                                hazards.append(item.lower())
            warns = payload.get("warnings")
            if isinstance(warns, list):
                for w in warns:
                    if isinstance(w, str) and w:
                        hazards.append(w.lower())
            # Deduplicate while preserving order
            seen = set()
            uniq: List[str] = []
            for h in hazards:
                if h not in seen:
                    uniq.append(h)
                    seen.add(h)
            return uniq

        # last_answer_preview: derived from last["answer"] (string or dict)
        last_answer_preview = None
        ans = last.get("answer")
        if isinstance(ans, dict):
            src = ans.get("final_answer") or ans.get("text") or ""
            last_answer_preview = (src or "")[:200].replace("\n", " ").strip() or None
        elif isinstance(ans, str):
            last_answer_preview = ans[:200].replace("\n", " ").strip() or None

        out["dialogs"][dlg_id] = {
            "dialog_id": dlg_id,
            "created_at": first.get("timestamp_utc"),
            "updated_at": last.get("timestamp_utc"),
            "num_events": len(files),
            "last_answer_preview": last_answer_preview,
            "last_ack_stage": (last.get("query") or {}).get("ack_stage"),
            "hazards": _extract_hazards(last),
            # heavy keywording is done by offline scripts; keep field for compatibility
            "top_keywords": None,
        }

    return out


@router_sessions.get("/index")
def get_session_index(x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Returns ./data/sessions/session_index.json if present,
    otherwise generates a lightweight index on the fly.

    Mirrors the legacy /api/v1/rag/session/index route.
    """
    _require_key(x_api_key)
    root = _sessions_root()
    idx_path = root / "session_index.json"

    if idx_path.exists():
        data = load_json(idx_path)
        if isinstance(data, dict):
            data.setdefault(
                "schema_version",
                getattr(settings, "schema_version", "1.0"),
            )
            return data

    # Regenerate lightweight index
    data = _mk_index()
    data["schema_version"] = getattr(settings, "schema_version", "1.0")
    try:
        write_json(idx_path, data)
    except Exception:
        # soft-fail write; still return index
        pass
    return data


# ---------------------------------------------------------------------------
# Resume dialog
# ---------------------------------------------------------------------------

@router_sessions.get("/resume")
def resume_dialog(
    dialog_id: str,
    x_api_key: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """
    Returns a compact resume payload for the given dialog:
      - dialog summary (created/updated/events count)
      - last event payload (latest.json preferred)
      - recent events list (timestamps + small query/answer previews)

    Mirrors the legacy /api/v1/rag/session/resume route.
    """
    _require_key(x_api_key)
    root = _sessions_root()
    dlg_dir = root / dialog_id
    if not dlg_dir.exists():
        raise HTTPException(status_code=404, detail="dialog not found")

    # summary: prefer index; else compute
    idx = get_session_index(x_api_key)  # type: ignore
    dialogs = idx.get("dialogs") or {}
    summary = dialogs.get(dialog_id)
    if not isinstance(summary, dict):
        evdir = dlg_dir / "events"
        files = sorted([p for p in evdir.glob("*.json") if p.is_file()])
        if not files:
            raise HTTPException(status_code=404, detail="no events for dialog")

        first = load_event_payload(files[0]) or {}
        last = load_event_payload(files[-1]) or {}

        def _extract_hazards(payload: Dict[str, Any]) -> List[str]:
            hazards: List[str] = []
            safety = payload.get("safety")
            if isinstance(safety, dict):
                for key in ("stop", "caution"):
                    v = safety.get(key)
                    if isinstance(v, list):
                        for item in v:
                            if isinstance(item, str) and item:
                                hazards.append(item.lower())
            warns = payload.get("warnings")
            if isinstance(warns, list):
                for w in warns:
                    if isinstance(w, str) and w:
                        hazards.append(w.lower())
            seen = set()
            uniq: List[str] = []
            for h in hazards:
                if h not in seen:
                    uniq.append(h)
                    seen.add(h)
            return uniq

        last_answer_preview = None
        ans = last.get("answer")
        if isinstance(ans, dict):
            src = ans.get("final_answer") or ans.get("text") or ""
            last_answer_preview = (src or "")[:200].replace("\n", " ").strip() or None
        elif isinstance(ans, str):
            last_answer_preview = ans[:200].replace("\n", " ").strip() or None

        summary = {
            "dialog_id": dialog_id,
            "created_at": first.get("timestamp_utc"),
            "updated_at": last.get("timestamp_utc"),
            "num_events": len(files),
            "last_answer_preview": last_answer_preview,
            "last_ack_stage": (last.get("query") or {}).get("ack_stage"),
            "hazards": _extract_hazards(last),
            "top_keywords": None,
        }

    # latest / answer payloads
    latest = load_json(dlg_dir / "latest.json") or {}
    answer = load_json(dlg_dir / "answer.json") or {}

    # recent events (up to last N = 10)
    evdir = dlg_dir / "events"
    files = sorted([p for p in evdir.glob("*.json") if p.is_file()])
    recent: List[Dict[str, Any]] = []
    for p in files[-10:]:
        ev = load_json(p) or {}
        q = ev.get("query") or {}
        a = ev.get("answer")

        # handle both dict + string answer for previews
        if isinstance(a, dict):
            a_preview_src = (
                a.get("final_answer") or a.get("text") or ""
            )
        else:
            a_preview_src = a or ""

        recent.append(
            {
                "timestamp_utc": ev.get("timestamp_utc"),
                "text": q.get("text"),
                "caption": q.get("caption"),
                "has_image": bool(q.get("has_image")),
                "answer_preview": (a_preview_src or "")[:200]
                .replace("\n", " ")
                .strip(),
            }
        )

    return {
        "ok": True,
        "schema_version": getattr(settings, "schema_version", "1.0"),
        "summary": summary,
        "latest": latest,
        "answer": answer,
        "recent_events": recent,
    }


# ---------------------------------------------------------------------------
# Memory (get + summarize)
# ---------------------------------------------------------------------------

@router_sessions.get("/memory")
def get_dialog_memory(
    dialog_id: str,
    x_api_key: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """
    Returns saved memory for dialog, or builds it if missing.

    Mirrors the legacy /api/v1/rag/session/memory route.
    """
    _require_key(x_api_key)
    root = _sessions_root()
    dlg_root = root / dialog_id
    if not dlg_root.exists():
        evdir = dlg_root / "events"
        if not evdir.exists():
            raise HTTPException(status_code=404, detail="dialog not found")

    mem = load_memory(dialog_id)
    if not isinstance(mem, dict):
        mem = build_memory(dialog_id)

    # Normalize via SessionMemory model where possible for a stable schema
    try:
        mem_model = SessionMemory(**mem)  # type: ignore[arg-type]
        mem_out: Dict[str, Any] = mem_model.model_dump()
    except Exception:
        # Fall back to raw dict if legacy or partially missing fields
        mem_out = mem

    return {
        "ok": True,
        "schema_version": getattr(settings, "schema_version", "1.0"),
        "memory": mem_out,
    }

@router_sessions.post("/summarize")
def summarize_dialog(
    dialog_id: str,
    refresh: bool = False,
    x_api_key: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """
    Forces a (re)build of memory.json from recent events.
    If refresh==False and memory exists, returns it as-is.

    Mirrors the legacy /api/v1/rag/session/summarize route.
    """
    _require_key(x_api_key)
    root = _sessions_root()
    dlg_root = root / dialog_id
    if not dlg_root.exists():
        evdir = dlg_root / "events"
        if not evdir.exists():
            raise HTTPException(status_code=404, detail="dialog not found")

    mem = load_memory(dialog_id)
    if isinstance(mem, dict) and not refresh:
        return {
            "ok": True,
            "schema_version": getattr(settings, "schema_version", "1.0"),
            "memory": mem,
            "refreshed": False,
        }

    mem = build_memory(dialog_id)
    return {
        "ok": True,
        "schema_version": getattr(settings, "schema_version", "1.0"),
        "memory": mem,
        "refreshed": True,
    }


# ---------------------------------------------------------------------------
# Dialog state (was app/api/dialog.py)
# ---------------------------------------------------------------------------

@router_dialog.get("/state")
def dialog_state(dialog_id: str = Query(...)) -> Dict[str, Any]:
    """
    Return dialog state and latest safety flags for the client.

    Response shape:

      {
        "dialog_id": "<id>",
        "last_query": "<most recent query text>",
        "blocked": false,
        "requires_ack": true,
        "ack_stage": 1,
        "safety_summary": { ... } | null,
        "events": [ ... raw per-turn events ... ]
      }

    - events are taken from ./data/sessions/<dialog_id>/events/*.json
      and returned most-recent-first.
    - safety flags are derived from the latest event.
    """
    base = _sessions_root()
    dialog_dir = base / dialog_id
    evdir = dialog_dir / "events"

    # Initial defaults
    last_query: str = ""
    events: List[Dict[str, Any]] = []

    # ---- Collect events (most-recent-first) ----
    try:
        if evdir.is_dir():
            event_files = sorted(
                [p for p in evdir.glob("*.json") if p.is_file()]
            )
            for path in event_files:
                try:
                    payload = load_event_payload(path)
                    if not isinstance(payload, dict):
                        continue
                    events.append(payload)
                    # Track last_query; later files override earlier ones
                    last_query = (
                        payload.get("query", {}).get("text")
                        or payload.get("inputs", {}).get("text")
                        or last_query
                    )
                except Exception:
                    # soft-fail individual event
                    continue
            # newest first for the client
            events.reverse()
    except Exception:
        # soft-fail; keep last_query/events as-is
        pass

    # ---- Derive flags from the latest event ----
    blocked = False
    requires_ack = False
    ack_stage = 0
    safety_summary: Any = None

    if events:
        latest = events[0]
        blocked = bool(latest.get("blocked", False))
        requires_ack = bool(latest.get("requires_ack", False))
        try:
            ack_stage = int(latest.get("ack_stage", 0))
        except Exception:
            ack_stage = 0

        synth = latest.get("synthesis")
        if isinstance(synth, dict):
            safety_summary = synth.get("safety_summary")

    return {
        "dialog_id": dialog_id,
        "last_query": last_query or "",
        "blocked": blocked,
        "requires_ack": requires_ack,
        "ack_stage": ack_stage,
        "safety_summary": safety_summary,
        "events": events,
    }

@router_dialog.post("/next-step")
def dialog_next_step(
    dialog_id: str = Form(...),
    text: str = Form(""),
    caption: str = Form(""),
    ack_stage: int = Form(0),
    x_api_key: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """
    Generate and persist the NEXT step for an existing dialog.

    - Does NOT run search again; reuses the dialog's rolling context.
    - Writes a lightweight event + latest/answer + memory update.
    - Accepts an optional image caption to mark this turn as image-based.
    - Returns the model-backed next_step payload.

    Intended for multi-turn guidance after the initial fusion/synthesis pass.
    """
    _require_key(x_api_key)

    if not dialog_id:
        raise HTTPException(status_code=400, detail="dialog_id required")

    # Ensure the dialog exists and has at least one event
    root = _sessions_root()
    dlg_dir = root / dialog_id
    evdir = dlg_dir / "events"
    if not dlg_dir.exists() or not evdir.exists():
        raise HTTPException(status_code=404, detail="dialog not found")

    try:
        step = run_next_step_turn(
            settings=settings,
            dialog_id=dialog_id,
            user_text=text,
            image_caption=(caption or None),
            ack_stage=ack_stage,
        )
    except HTTPException:
        # Propagate FastAPI-style errors unchanged
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"next_step_failed:{type(e).__name__}",
        )

    return {
        "ok": True,
        "schema_version": getattr(settings, "schema_version", "1.0"),
        "dialog_id": dialog_id,
        "next_step": step,
    }
