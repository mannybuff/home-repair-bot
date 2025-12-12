# app/services/sessions.py
"""
SESSION SERVICE LAYER

Orchestrates read/write and updates for dialog sessions:

    - Load/save full session state for a dialog_id (chat_info.json)
    - Append context snippets and citations
    - Track query + progress history
    - Read recent events to build a lightweight "memory" summary
    - Persist latest event and a light answer.json

Design rules:
    • Uses filesystem helpers from utils/session_utils.py
    • Contains business logic (how we summarize / structure memory)
    • No FastAPI routers here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
from fastapi.encoders import jsonable_encoder
import json

from app.utils.session_utils import (
    now_iso,
    session_path,
    memory_path,
    events_dir,
    ensure_session_dirs,
    list_event_files,
    load_event_payload,
    estimate_chars,
    shorten as _shorten,
    sentences as _sentences,
)
from app.services.qwen_use import generate_next_step_llm


# ---------------------------------------------------------------------------
# Core session dict load/save (chat_info.json)
# ---------------------------------------------------------------------------

def load_session(dialog_id: str) -> Dict[str, Any]:
    """
    Load or initialize a session dict from chat_info.json.
    Ported from legacy app/services/session.py.
    """
    sp = session_path(dialog_id)
    if sp.exists():
        try:
            return json.loads(sp.read_text(encoding="utf-8"))
        except Exception:
            # Corrupt file: fall back to fresh
            pass
    # Fresh seed
    return {
        "dialog_id": dialog_id,
        "created_at": now_iso(),
        "ack_stage": 0,
        "progress": [],
        "citations": [],
        "last_query": "",
        "last_snippets": [],
        "context_used_chars": 0,
        "context_budget_chars": 64000,  # ~50% of 32k tokens
    }


def save_session(dialog_id: str, data: Dict[str, Any]) -> None:
    """
    Persist a session dict back to chat_info.json.
    """
    sp = session_path(dialog_id)
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(data, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Context & citations
# ---------------------------------------------------------------------------

def add_citations(sess: Dict[str, Any], items: List[Dict[str, Any]]) -> None:
    """
    Extract normalized citations from fusion results and append,
    avoiding obvious duplicates.

    Schema:
      - web:   {"type":"web","url":..., "title":..., "snippet":...}
      - text:  {"type":"text","pdf_path":..., "page_index":..., "snippet":...}
      - image: {"type":"image","pdf_path":..., "page_index":..., "image_path":...}
    """
    seen = {json.dumps(c, sort_keys=True) for c in sess.get("citations", [])}
    new: List[Dict[str, Any]] = []
    for it in items:
        t = it.get("type")
        if t == "web":
            c = {
                "type": "web",
                "url": it.get("url", ""),
                "title": it.get("title", ""),
                "snippet": it.get("snippet", ""),
            }
        elif t == "text":
            c = {
                "type": "text",
                "pdf_path": it.get("pdf_path", ""),
                "page_index": it.get("page_index") or it.get("page"),
                "snippet": it.get("snippet", ""),
            }
        elif t == "image":
            c = {
                "type": "image",
                "pdf_path": it.get("pdf_path", ""),
                "page_index": it.get("page_index"),
                "image_path": it.get("image_path", ""),
            }
        else:
            continue
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            new.append(c)
    if new:
        sess.setdefault("citations", []).extend(new)


def add_context_snippets(
    sess: Dict[str, Any],
    snippets: List[str],
    max_chars: Optional[int] = None,
) -> None:
    """
    Append snippets into rolling context until budget (~50% of 32k tokens) is hit.
    Ported from legacy app/services/session.py.
    """
    if max_chars is None:
        max_chars = sess.get("context_budget_chars", 64000)
    used = sess.get("context_used_chars", 0)
    keep: List[str] = []
    for s in snippets:
        c = estimate_chars(s)
        if used + c > max_chars:
            break
        keep.append(s)
        used += c
    if keep:
        sess.setdefault("last_snippets", []).extend(keep)
        sess["context_used_chars"] = used

# ---------------------------------------------------------------------------
# Event & memory summarizer
# ---------------------------------------------------------------------------

def load_recent_events(dialog_id: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Load up to 'limit' recent events from disk for a dialog.
    """
    files = list_event_files(dialog_id)
    if not files:
        return []
    out: List[Dict[str, Any]] = []
    # files are sorted ascending; take tail for "recent"
    for p in files[-limit:]:
        payload = load_event_payload(p)
        if isinstance(payload, dict):
            out.append(payload)
    return out

def _extract_answer_text(ev: Dict[str, Any]) -> str:
    """
    Normalize the 'answer' field from an event into a plain string.

    Supports both:
      - legacy shape: answer: "<final answer text>"
      - new shape:    answer: {"dialog_id": "...", "title": "...",
                               "text": "...", "final_answer": "..."}
    """
    raw = ev.get("answer")

    # New schema: dict with final_answer/text fields
    if isinstance(raw, dict):
        for key in ("final_answer", "text", "message", "body"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        # Nothing usable inside the dict
        return ""

    # Legacy schema: string or None
    if raw is None:
        return ""

    # Anything else: be defensive but try to stringify
    try:
        return str(raw).strip()
    except Exception:
        return ""


def _extract_points(evs: List[Dict[str, Any]]) -> List[str]:
    """
    Build a list of short key points from recent events.

    - Pulls text from answer (string or dict-backed).
    - If answer is empty but synthesis exists, fall back to first_step/scope.
    - Truncates each point to a reasonable length for memory.
    - Deduplicates points case-insensitively to avoid spammy repeats.
    """
    points: List[str] = []

    for ev in evs:
        if not isinstance(ev, dict):
            continue

        # 1) Prefer explicit answer text
        text = _extract_answer_text(ev)

        # Prefer top-level synthesis, but fall back to orchestration.synthesis
        synth = ev.get("synthesis") or {}
        if not (isinstance(synth, dict) and synth):
            orch = ev.get("orchestration") or {}
            if isinstance(orch, dict):
                osynth = orch.get("synthesis") or {}
                if isinstance(osynth, dict) and osynth:
                    synth = osynth

        # 2) Fallback to synthesis bits if answer empty
        if not text and isinstance(synth, dict):
            bits: List[str] = []
            for key in ("first_step", "scope_overview"):
                val = synth.get(key)
                if isinstance(val, str) and val.strip():
                    bits.append(val.strip())
            if bits:
                text = " ".join(bits)

        text = (text or "").strip()
        if not text:
            continue

        # 3) Clamp to keep memory compact
        if len(text) > 400:
            text = text[:400]

        points.append(text)

    # 4) Deduplicate (case-insensitive) so repeated lines from the model
    # do not flood memory.key_points.
    seen = set()
    uniq: List[str] = []
    for p in points:
        k = p.strip().lower()
        if not k or k in seen:
            continue
        seen.add(k)
        uniq.append(p)

    return uniq

def _collect_hazards(events: List[Dict[str, Any]]) -> List[str]:
    """
    Collect coarse hazard keywords from the safety/warnings content in events.
    """
    bag = set()
    for ev in events:
        safety = ev.get("safety")
        if isinstance(safety, dict):
            # new-style safety payload (stop/caution lists)
            for k in ("stop", "caution"):
                v = safety.get(k)
                if isinstance(v, list):
                    for h in v:
                        if isinstance(h, str) and h:
                            bag.add(h.lower())
        warns = ev.get("warnings")
        if isinstance(warns, list):
            for w in warns:
                if isinstance(w, str) and w:
                    wl = w.lower()
                    # simple hazard word sniff (coarse)
                    for tok in (
                        "gas",
                        "electrical",
                        "asbestos",
                        "mold",
                        "lead",
                        "silica",
                        "fall",
                        "shock",
                        "flood",
                        "fire",
                        "structural",
                    ):
                        if tok in wl:
                            bag.add(tok)
    return sorted(bag)


def _best_ack(events: List[Dict[str, Any]]) -> Optional[int]:
    """
    Pick the best/latest acknowledgement stage from events.
    """
    for ev in reversed(events):
        q = ev.get("query") or {}
        st = q.get("ack_stage")
        if isinstance(st, int):
            return st
        idec = ev.get("intent_decision")
        if isinstance(idec, dict) and isinstance(idec.get("ack_stage"), int):
            return idec["ack_stage"]
    return None


def _compose_summary(events: List[Dict[str, Any]], hazards: List[str]) -> str:
    """
    Build 3–7 sentences summarizing intent, context, findings, and safety.

    Uses the same answer-normalization logic as _extract_points so that
    both legacy string answers and the new dict-backed answers are supported.
    Falls back to synthesis snippets if no direct answer text is available.
    """
    if not events:
        return "No prior context available for this dialog."

    latest = events[-1]
    q = latest.get("query") or {}
    qtxt = (q.get("text") or "").strip()
    has_image = bool(q.get("has_image"))

    # Normalize the latest answer text (string or dict-backed)
    ans = _extract_answer_text(latest)

    # If there's still no direct answer, fall back to synthesis snippets.
    # Prefer top-level synthesis, but fall back to orchestration.synthesis.
    if not ans:
        synth = latest.get("synthesis") or {}
        if not (isinstance(synth, dict) and synth):
            orch = latest.get("orchestration") or {}
            if isinstance(orch, dict):
                osynth = orch.get("synthesis") or {}
                if isinstance(osynth, dict) and osynth:
                    synth = osynth

        if isinstance(synth, dict):
            bits: List[str] = []
            for key in ("first_step", "scope_overview"):
                val = synth.get(key)
                if isinstance(val, str) and val.strip():
                    bits.append(val.strip())
            if bits:
                ans = " ".join(bits)

    pieces: List[str] = []

    # Describe the latest query + image status
    if qtxt and has_image:
        pieces.append(
            f"The latest turn included a photo and the query “{_shorten(qtxt, 140)}”."
        )
    elif qtxt:
        pieces.append(f"The latest query was “{_shorten(qtxt, 160)}”.")
    elif has_image:
        pieces.append("The latest turn included a photo without a text query.")

    # Summarize the latest answer
    if ans:
        sents = _sentences(ans)
        if sents:
            pieces.append(f"Initial guidance: {_shorten(sents[0], 240)}")
            if len(sents) > 1:
                pieces.append(_shorten(sents[1], 240))

    # Pull a couple of earlier highlights
    earlier = events[-3:-1]
    for ev in earlier:
        ea = _extract_answer_text(ev)
        if not ea:
            synth = ev.get("synthesis") or {}
            if isinstance(synth, dict):
                for key in ("first_step", "scope_overview"):
                    val = synth.get(key)
                    if isinstance(val, str) and val.strip():
                        ea = val.strip()
                        break
        ea = (ea or "").strip()
        if ea:
            s = _sentences(ea)
            if s:
                pieces.append(_shorten(s[0], 220))

    # Hazard summary
    if hazards:
        pieces.append(
            f"Observed hazards across the dialog: {', '.join(hazards)}."
        )

    return " ".join(pieces) or "Context summarized."

def _derive_projects(
    events: List[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Infer an 'active_project' and a simple 'projects' list from recent events.

    v1 behavior:
      - Treat all events as a single project.
      - Use the oldest event's query as first_query.
      - Use the latest event's topic_intent + synthesis.scope_overview as summary.
    v2 can later refine this to segment by TOPIC_SHIFT / FIRST_TURN markers.
    """
    if not events:
        return None, []

    # Oldest + newest events in the recent window
    first_ev = events[0]
    latest_ev = events[-1]

    # Core query info from earliest event
    fq = (first_ev.get("query") or {})
    first_query = (fq.get("text") or "").strip() or None
    created_at = (first_ev.get("timestamp_utc") or "").strip() or None

    # Latest topic + synthesis bits
    notes = latest_ev.get("notes") or {}
    topic_intent = notes.get("topic_intent") or {}
    topic = (topic_intent.get("topic") or "").strip() or None

    # Prefer top-level synthesis, but fall back to orchestration.synthesis
    synth = latest_ev.get("synthesis") or {}
    if not (isinstance(synth, dict) and synth):
        orch = latest_ev.get("orchestration") or {}
        if isinstance(orch, dict):
            osynth = orch.get("synthesis") or {}
            if isinstance(osynth, dict) and osynth:
                synth = osynth

    scope_overview = ""
    if isinstance(synth, dict):
        val = synth.get("scope_overview")
        if isinstance(val, str) and val.strip():
            scope_overview = val.strip()

    # Fallback summary if scope_overview is empty
    if not scope_overview:
        # We'll let _compose_summary generate a more generic description later.
        # For now, leave it empty; build_memory will fill summary if needed.
        scope_overview = ""

    # Issue string: short problem description
    issue = None
    if first_query:
        issue = first_query
    elif scope_overview:
        issue = scope_overview.split(".")[0].strip()

    project_id = "proj-1"
    last_ts = (latest_ev.get("timestamp_utc") or "").strip() or None

    active_project: Dict[str, Any] = {
        "project_id": project_id,
        "topic": topic,
        "issue": issue or first_query or "",
        "first_query": first_query or "",
        "created_at": created_at or now_iso(),
        "last_event_ts": last_ts or created_at or now_iso(),
        "summary": scope_overview or "",
        "topic_intent": topic_intent,
    }

    projects = [active_project]
    return active_project, projects

def build_memory(dialog_id: str, recent_limit: int = 32) -> Dict[str, Any]:
    """
    Build and persist a lightweight memory.json from recent events.
    """
    evs = load_recent_events(dialog_id, limit=recent_limit)
    if not evs:
        mem = {
            "dialog_id": dialog_id,
            "updated_at": now_iso(),
            "last_ack_stage": None,
            "hazards_seen": [],
            "key_points": [],
            "summary": "No prior context available for this dialog.",
            "active_project": None,
            "projects": [],
            "source": {"recent_events_used": 0},
        }
    else:
        hazards = _collect_hazards(evs)
        key_points = _extract_points(evs)
        last_ack = _best_ack(evs)
        summary = _compose_summary(evs, hazards)

        # NEW: derive project structure from recent events
        active_project, projects = _derive_projects(evs)

        # Prefer project-level summary when available; otherwise fall back
        # to the existing last-turn-centric summary.
        project_summary = (
            (active_project or {}).get("summary") or summary
        )

        mem = {
            "dialog_id": dialog_id,
            "updated_at": now_iso(),
            "last_ack_stage": last_ack,
            "hazards_seen": hazards,
            "key_points": key_points,
            "summary": project_summary,
            "active_project": active_project,
            "projects": projects,
            "source": {"recent_events_used": min(len(evs), recent_limit)},
        }

    mem_path = memory_path(dialog_id)
    mem_path.parent.mkdir(parents=True, exist_ok=True)
    mem_path.write_text(json.dumps(mem, indent=2, ensure_ascii=False))
    return mem

def load_memory(dialog_id: str) -> Optional[Dict[str, Any]]:
    """
    Load memory.json for a dialog, or None if missing/invalid.
    """
    p = memory_path(dialog_id)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None

def run_next_step_turn(
    settings: Any,
    dialog_id: str,
    user_text: str,
    image_caption: Optional[str] = None,
    ack_stage: Optional[int] = None,
) -> Dict[str, Any]:
    """
    High-level helper for a NEXT-STEP dialog turn.

    Responsibilities:
      - Call generate_next_step_llm() using the rolling dialog context.
      - Persist a lightweight event + latest/answer for this turn.
      - Trigger memory.json maintenance via record_session_event().
      - Return the step payload for the API layer to send back.

    This turn does NOT run search again; it is purely context + Qwen text.
    """
    dlg_id = (dialog_id or "anon").strip() or "anon"
    text = (user_text or "").strip()
    has_image = bool(image_caption)

    # 1) Ask the model for the next step, using the rolling context.
    step = generate_next_step_llm(
        dialog_id=dlg_id,
        user_text=text,
        image_caption=image_caption,
    )

    # 2) Build minimal notes payload for this "next step" mode.
    notes: Dict[str, Any] = {
        "mode": "next_step",
        "counts": {"fused": 0},
        "timings_ms": {},
    }
    warnings: List[str] = []

    # Normalize ack_stage
    try:
        ack_val = int(ack_stage) if ack_stage is not None else 0
    except Exception:
        ack_val = 0

    # 3) Persist as an event with an answer that matches the next-step text.
    answer_text = (step.get("next_step_text") or "").strip()
    try:
        record_session_event(
            settings=settings,
            dialog_id=dlg_id,
            query_text=text or None,
            has_image=has_image,
            query_caption=(image_caption or None),
            notes=notes,
            warnings=warnings,
            results=[],          # no new search results in next-step mode
            max_results=0,
            answer=answer_text,  # will populate answer.json + latest.json
            evidence=[],
            blocked=False,
            requires_ack=False,
            ack_stage=ack_val,
        )
    except Exception as e:
        warnings.append(f"next_step_event_failed:{type(e).__name__}")

    # Attach any warnings for debug/clients
    if warnings:
        step["warnings"] = warnings

    # Echo effective ack stage for clients/UI
    step["ack_stage"] = ack_val

    return step

# ---------------------------------------------------------------------------
# Latest event persistence (latest.json + answer.json)
# ---------------------------------------------------------------------------

def persist_latest_event(dialog_id: str, payload: Dict[str, Any]) -> Dict[str, str]:
    """
    Persist the full payload to latest.json and a light answer.json
    next to it, both under the dialog's events folder.
    Returns absolute paths for diagnostics.

    Behavior:
      - If payload["answer"] is a dict and has a non-empty final_answer/text/message/body,
        that value becomes the final answer.
      - If payload["answer"] is a non-empty string, that string becomes the final answer.
      - Otherwise, derive final_answer from synthesis.scope_overview / first_step.
    """
    dirs = ensure_session_dirs(dialog_id)
    latest_path = (dirs["events"] / "latest.json").resolve()
    answer_path = (dirs["events"] / "answer.json").resolve()

    # Make a shallow copy so we don't mutate caller data in-place.
    payload = dict(payload)

    # Ensure the orchestration block exists and carries the dialog id.
    orch = payload.setdefault("orchestration", {})
    orch["dialog_id"] = dialog_id

    # ---- Inspect any existing answer on the payload ----
    raw_answer = payload.get("answer")
    existing_title: Optional[str] = None
    existing_final: Optional[str] = None

    if isinstance(raw_answer, dict):
        existing_title = raw_answer.get("title")
        for key in ("final_answer", "text", "message", "body"):
            val = raw_answer.get(key)
            if isinstance(val, str) and val.strip():
                existing_final = val.strip()
                break
    elif isinstance(raw_answer, str):
        if raw_answer.strip():
            existing_final = raw_answer.strip()

    # ---- Derive fields from query + synthesis as needed ----
    query = payload.get("query") or {}
    synthesis = payload.get("synthesis") or {}

    query_text = (query.get("text") or "").strip()
    scope_overview = (synthesis.get("scope_overview") or "").strip()
    first_step = (synthesis.get("first_step") or "").strip()

    # Title:
    # 1) Prefer an explicit title from synthesis
    # 2) Otherwise, existing title from answer dict (if any)
    # 3) Otherwise, "How do I fix a <query>?"
    if isinstance(synthesis, dict) and synthesis.get("title"):
        title = str(synthesis["title"]).strip()
    elif isinstance(existing_title, str) and existing_title.strip():
        title = existing_title.strip()
    elif query_text:
        title = f"How do I fix a {query_text}?"
    else:
        title = ""

    # Final answer text:
    # 1) Prefer existing_final (string or dict-backed)
    # 2) Otherwise derive from synthesis bits
    if isinstance(existing_final, str) and existing_final.strip():
        final_answer = existing_final.strip()
    else:
        parts: List[str] = []
        if scope_overview:
            parts.append(scope_overview)
        if first_step:
            parts.append(f"First step: {first_step}")
        final_answer = "  ".join(parts).strip()

    # Attach a normalized "answer" block into the full payload for future APIs.
    # Include both text + final_answer for compatibility with older readers.
    payload["answer"] = {
        "dialog_id": dialog_id,
        "title": title,
        "text": final_answer,
        "final_answer": final_answer,
    }

    # ---- Write latest.json ----
    latest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # ---- Write answer.json (lightweight view) ----
    with answer_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "dialog_id": dialog_id,
                "title": title,
                "text": final_answer,
                "final_answer": final_answer,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {"latest": str(latest_path), "answer": str(answer_path)}

def record_session_event(
    settings: Any,
    dialog_id: str,
    query_text: Optional[str],
    has_image: bool,
    query_caption: Optional[str],
    notes: Dict[str, Any],
    warnings: List[str],
    results: List[Dict[str, Any]],
    max_results: int = 8,
    answer: Optional[str] = None,
    evidence: Optional[List[Dict[str, Any]]] = None,
    blocked: Optional[bool] = None,
    requires_ack: Optional[bool] = None,
    ack_stage: Optional[int] = None,
) -> None:
    """
    Canonical fusion/synthesis event recorder.

    Called by:
      - fusion_search(...) in app/api/fusion.py
      - run_next_step_turn(...) for lightweight follow-up guidance

    Steps:
      1) Write timestamped immutable event JSON into events/<timestamp>.json.
      2) Update latest.json + answer.json via persist_latest_event().
      3) Refresh memory.json via build_memory() (fail-soft).
    """
    
    try:
        dlg_id = dialog_id or "anon"

        # Ensure standard dialog directories exist
        dirs = ensure_session_dirs(dlg_id)
        evdir = dirs["events"]

        ts = now_iso().replace(":", "-")
        outp = evdir / f"{ts}.json"

        # -------------------------------------------
        def _count_type(seq: List[Dict[str, Any]], t: str) -> int:
            try:
                return sum(
                    1
                    for x in (seq or [])
                    if isinstance(x, dict) and x.get("type") == t
                )
            except Exception:
                return 0

        source_counts = {
            "image": _count_type(results, "image"),
            "text": _count_type(results, "text"),
            "web": _count_type(results, "web"),
            "total": int(notes.get("counts", {}).get("fused", len(results or []))),
        }

        # Normalize flags
        b = bool(blocked) if blocked is not None else False
        ra = bool(requires_ack) if requires_ack is not None else False
        try:
            ack_val = int(ack_stage) if ack_stage is not None else 0
        except Exception:
            ack_val = 0

        payload: Dict[str, Any] = {
            "timestamp_utc": ts,
            "dialog_id": dlg_id,
            "query": {
                "text": (query_text or "").strip() or None,
                "has_image": bool(has_image),
                "caption": (query_caption or "").strip() or None,
            },
            "notes": notes,
            "warnings": warnings,
            "counts": notes.get("counts", {}),
            "source_counts": source_counts,
            "timings_ms": notes.get("timings_ms", {}),
            "answer": (answer or "").strip() or None,
            "evidence": (evidence or [])[:max_results]
            if isinstance(evidence, list)
            else [],
            "results": list(results or [])[:max_results],
            # Safety/intent flags at the event level
            "blocked": b,
            "requires_ack": ra,
            "ack_stage": ack_val,
        }

        payload = jsonable_encoder(payload)

        # 1) Write immutable timestamped event for history
        outp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # 2) Delegate rolling latest.json + answer.json to the shared helper
        try:
            persist_latest_event(dlg_id, payload)
        except Exception:
            # Never fail the request because we couldn't update latest.json;
            # the timestamped record is still available for offline inspection.
            pass

        # 3) Memory update hook (non-blocking, now AFTER event write)
        try:
            build_memory(dlg_id)
        except Exception:
            # Never fail the request because memory maintenance had an issue
            pass

    except Exception as e:
        try:
            warnings.append(f"session_write_failed:{type(e).__name__}")
        except Exception:
            # If even warnings can't be updated, just swallow
            pass
