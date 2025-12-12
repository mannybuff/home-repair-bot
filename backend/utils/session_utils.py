# app/utils/session_utils.py
"""
SESSION UTILITY HELPERS

Pure helper functions for session persistence and manipulation:

    - Path resolution for dialog/session files
    - Safe JSON load/save wrappers
    - Lightweight helpers for events and text summarization
    - Generic timestamp and character-budget utilities

Design rules:
    • No FastAPI imports.
    • No business logic or routing.
    • Only filesystem + JSON + small data-structure utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import re as _re

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

"""
Root folder:
  data/sessions/<dialog_id>/
is controlled by app.core.settings.sessions_dir when available so all writers
agree on a single location (fusion, synthesis, and tools).
"""

try:
    from app.core.settings import settings as _settings  # type: ignore
    DATA_ROOT = Path(getattr(_settings, "sessions_dir", "./data/sessions")).expanduser()
except Exception:
    DATA_ROOT = Path("./data/sessions").expanduser()

# Back-compat constant; prefer using session_root() in new code.
SESS_ROOT = DATA_ROOT


# Explicit public API for other modules
__all__ = [
    "SESS_ROOT",
    "session_root",
    "now_iso",
    "ensure_dir",
    "session_path",
    "memory_path",
    "events_dir",
    "ensure_session_dirs",
    "load_json",
    "write_json",
    "list_event_files",
    "load_event_payload",
    "estimate_chars",
    "shorten",
    "sentences",
    "build_dialogue_context",
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def now_iso() -> str:
    """
    UTC timestamp in ISO-8601, used across session metadata.
    """
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def ensure_dir(path: Path) -> Path:
    """
    Ensure the parent directory of path exists.
    Returns the path unchanged for chaining.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def session_root() -> Path:
    """
    Return the resolved root directory for all sessions, ensuring it exists.
    """
    SESS_ROOT.mkdir(parents=True, exist_ok=True)
    return SESS_ROOT


# ---------------------------------------------------------------------------
# Paths for specific artifacts
# ---------------------------------------------------------------------------

def session_path(dialog_id: str) -> Path:
    """
    ./data/sessions/<dialog_id>/chat_info.json
    """
    return SESS_ROOT / dialog_id / "chat_info.json"


def memory_path(dialog_id: str) -> Path:
    """
    ./data/sessions/<dialog_id>/memory.json
    """
    return SESS_ROOT / dialog_id / "memory.json"


def events_dir(dialog_id: str) -> Path:
    """
    ./data/sessions/<dialog_id>/events/
    """
    return SESS_ROOT / dialog_id / "events"


def ensure_session_dirs(dialog_id: str) -> Dict[str, Path]:
    """
    Ensure per-dialog directories exist and return their paths.

    Structure:
        data/sessions/<dialog_id>/events/
    """
    base = SESS_ROOT / dialog_id
    ev_dir = base / "events"
    base.mkdir(parents=True, exist_ok=True)
    ev_dir.mkdir(parents=True, exist_ok=True)
    return {"base": base, "events": ev_dir}


# ---------------------------------------------------------------------------
# JSON load/save helpers
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a JSON dict from disk, returning None on error.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    """
    Persist a JSON dict to disk with UTF-8 encoding.
    """
    ensure_dir(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Event-level helpers
# ---------------------------------------------------------------------------

def list_event_files(dialog_id: str) -> List[Path]:
    """
    List all event JSON files for a dialog in ascending lexical order.
    """
    evdir = events_dir(dialog_id)
    if not evdir.exists():
        return []
    return sorted([p for p in evdir.glob("*.json") if p.is_file()])


def load_event_payload(path: Path) -> Optional[Dict[str, Any]]:
    """
    Load a single event JSON file; return None on error.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def estimate_chars(s: Optional[str]) -> int:
    """
    Cheap budget guard: approximate tokens via chars; good enough for gating.
    """
    if not s:
        return 0
    return len(s)


def shorten(txt: str, maxlen: int = 280) -> str:
    """
    Shorten a string to maxlen characters with a single Unicode ellipsis.
    """
    txt = (txt or "").strip()
    if len(txt) <= maxlen:
        return txt
    return txt[: maxlen - 1].rstrip() + "…"


def sentences(txt: str) -> List[str]:
    """
    Very light sentence split using simple punctuation rules.
    """
    txt = (txt or "").strip()
    if not txt:
        return []
    raw = _re.split(r'(?<=[\.\?!])\s+', txt)
    return [s.strip() for s in raw if s.strip()]


# ---------------------------------------------------------------------------
# Dialogue context snapshot (for multi-turn LLM usage)
# ---------------------------------------------------------------------------

def build_dialogue_context(dialog_id: str, max_events: int = 4) -> Dict[str, Any]:
    """
    Build a compact, multi-turn context view for a given dialog_id.

    This helper does NOT call any LLMs. It simply reads:

      - events/latest.json         (most recent fused event)
      - memory.json                (rolling project summary)
      - up to max_events events/*  (most recent timestamped events)

    Returns a dict shaped for downstream LLM helpers, e.g.:

        {
          "dialog_id": "...",
          "topic": "plumbing",
          "topic_intent": {...},
          "summary": "The latest query was ... Initial guidance: ...",
          "hazards_seen": ["fire", "live_electric"],
          "recent_turns": [
             {
               "timestamp_utc": "...",
               "user_text": "...",
               "caption": "...",
               "has_image": true,
               "answer_brief": "...",
               "first_step": "..."
             },
             ...
          ],
        }

    Fails soft: on any IO / parse error it returns the most useful
    partial information it could gather.
    """
    ctx: Dict[str, Any] = {
        "dialog_id": dialog_id,
        "topic": None,
        "topic_intent": None,
        "summary": "",
        "hazards_seen": [],
        "recent_turns": [],
    }

    try:
        evdir = events_dir(dialog_id)
        session_dir = evdir.parent

        # --- Latest event (canonical current turn) ---
        latest_path = evdir / "latest.json"
        latest: Dict[str, Any] = {}
        if latest_path.is_file():
            latest = load_event_payload(latest_path) or {}
            if not isinstance(latest, dict):
                latest = {}

        # Topic + intent from synthesis or orchestration
        topic = None
        topic_intent: Optional[Dict[str, Any]] = None

        if latest:
            # Prefer synthesis.topic, fall back to latest["topic"]
            syn = latest.get("synthesis") or {}
            if isinstance(syn, dict):
                topic = syn.get("topic") or latest.get("topic")

            orch = latest.get("orchestration") or {}
            if isinstance(orch, dict) and not topic_intent:
                ti = orch.get("topic_intent") or orch.get("topic_intent_pre")
                if isinstance(ti, dict):
                    topic_intent = ti

            # Fallback: topic_intent directly on latest
            if not topic_intent:
                ti = latest.get("topic_intent") or latest.get("topic_intent_pre")
                if isinstance(ti, dict):
                    topic_intent = ti

        ctx["topic"] = topic
        ctx["topic_intent"] = topic_intent

        # --- Memory summary (if present) ---
        mem_path = session_dir / "memory.json"
        memory = load_json(mem_path) or {}
        if isinstance(memory, dict):
            summary = memory.get("summary") or memory.get("short_summary") or ""
            ctx["summary"] = (summary or "").strip()

            hz = memory.get("hazards_seen") or memory.get("hazards") or []
            if isinstance(hz, list):
                # keep as lowercased strings, deduped
                seen: set[str] = set()
                hz_norm: List[str] = []
                for h in hz:
                    if not isinstance(h, str):
                        continue
                    s = h.strip().lower()
                    if s and s not in seen:
                        seen.add(s)
                        hz_norm.append(s)
                ctx["hazards_seen"] = hz_norm

        # --- Recent turn list (up to max_events) ---
        files = list_event_files(dialog_id)
        if files:
            # newest first
            files = sorted(files, reverse=True)[:max_events]

            recent_turns: List[Dict[str, Any]] = []

            for p in files:
                try:
                    ev = load_event_payload(p) or {}
                    if not isinstance(ev, dict):
                        continue

                    q = ev.get("query") or {}
                    a = ev.get("answer")

                    ts = ev.get("timestamp_utc") or ev.get("ts")
                    user_text = q.get("text")
                    caption = q.get("caption")
                    has_image = bool(q.get("has_image"))

                    # Normalize answer to a short preview
                    ans_preview = ""
                    if isinstance(a, dict):
                        ans_preview = (
                            a.get("final_answer")
                            or a.get("text")
                            or a.get("answer")
                            or ""
                        )
                    elif isinstance(a, str):
                        ans_preview = a

                    ans_preview = (ans_preview or "").strip()
                    if len(ans_preview) > 280:
                        ans_preview = ans_preview[:279].rstrip() + "…"

                    # Optional 'first_step' hint from synthesis, if present
                    syn = ev.get("synthesis") or {}
                    first_step = None
                    if isinstance(syn, dict):
                        first_step = syn.get("first_step")

                    recent_turns.append(
                        {
                            "timestamp_utc": ts,
                            "user_text": user_text,
                            "caption": caption,
                            "has_image": has_image,
                            "answer_brief": ans_preview or None,
                            "first_step": (first_step or "").strip() or None,
                        }
                    )
                except Exception:
                    # ignore broken event file
                    continue

            # Newest first as built; caller can reverse if needed
            ctx["recent_turns"] = recent_turns

    except Exception:
        # keep whatever partial ctx we have
        pass

    return ctx

