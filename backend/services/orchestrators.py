# app/services/orchestrators.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal, Optional, TypedDict, List

import json
import logging
import uuid
import time 
import re

from app.utils.session_utils import (
    events_dir,
    ensure_session_dirs,
)

from app.services.sessions import (
    load_session,
    load_memory,
    run_next_step_turn,
    record_session_event,  
    build_memory,       
    persist_latest_event,
)
from app.services.images import (
    caption_image,
    search_similar_images, 
)
from app.utils.image_utils import load_image_from_bytes
from app.services.qwen_use import (
    infer_topic_intent,
    generate_queries_llm,   
    summarize_longform_llm, 
    answer_from_summary_llm,
    fallback_sections,      
)
from app.services.rag import (
    run_rag_search,             
    fuse_image_hits_with_text,  
)
from app.services.search import (
    run_web_search_service,     
    pack_and_persist_results,   
    inflate_longform_pack,      
)
from app.utils.rag_utils import (
    collect_snippets,           
    build_evidence,             
    enrich_with_page_images,    
)
from app.utils.search_utils import (
    normalize_image_item,       
)
from app.services.safety import (
    analyze_text_blocks,
    safety_payload,
    decide_intent,
    _detect_emergency,
)

log = logging.getLogger(__name__)


SessionMode = Literal["FIRST_TURN", "SAME_TOPIC", "TOPIC_SHIFT"]


class SessionContext(TypedDict, total=False):
    """
    Canonical orchestration context for a single user turn.

    Built by `synth_entry` and consumed by `run_fusion` / `run_step_dialog`.
    """

    # core
    dialog_id: str
    user_text: str

    # image-related
    has_image: bool
    image_bytes: Optional[bytes]
    image_caption: Optional[str]

    # intent / topic
    is_home_repair: bool
    intent: Dict[str, Any]
    topic_state: Dict[str, Any]
    session_mode: SessionMode
    has_memory: bool

    # safety (lightweight emergency gate only)
    emergency_block: bool
    emergency_reason: Optional[str]

    # paths / raw storage pointers
    latest_event_path: Optional[Path]
    session_dir: Optional[Path]
    summary_path: Optional[Path]
    memory_path: Optional[Path]


# ---------------------------------------------------------------------------
# Helper: topic comparison (lifted from fusion, kept here for reuse)
# ---------------------------------------------------------------------------

def _is_same_topic(
    prev_state: Optional[Dict[str, Any]],
    current_intent: Optional[Dict[str, Any]],
    current_query: Optional[str] = None,
) -> bool:
    """
    Coarse topic-match detector between previous session topic_state and
    the latest topic_intent output from Qwen.

    Rules (any one → "same topic"):
      - non-empty topics that match case-insensitively
      - non-empty keyword sets with at least one overlapping term
      - non-trivial overlap between issue/first_query words and current query
    """
    if not prev_state or not current_intent:
        return False

    prev_topic = str(prev_state.get("topic") or "").strip().lower()
    new_topic = str(current_intent.get("topic") or "").strip().lower()

    # 1) Simple topic equality
    if prev_topic and new_topic and prev_topic == new_topic:
        return True

    # 2) Keyword overlap
    prev_kw = {
        str(k).strip().lower()
        for k in (prev_state.get("keywords") or [])
        if isinstance(k, str) and k.strip()
    }
    new_kw = {
        str(k).strip().lower()
        for k in (current_intent.get("keywords") or [])
        if isinstance(k, str) and k.strip()
    }
    if prev_kw and new_kw and prev_kw.intersection(new_kw):
        return True

    # 3) Issue / query overlap as a fallback
    def _word_set(text: str) -> set[str]:
        if not text:
            return set()
        tokens = re.split(r"[^a-z0-9]+", text.lower())
        stop = {
            "the", "a", "an", "my", "is", "it", "to", "of", "in", "on",
            "at", "and", "or", "for", "with", "about", "this", "that",
        }
        return {t for t in tokens if t and t not in stop}

    prev_issue = str(
        prev_state.get("issue")
        or prev_state.get("summary")
        or prev_state.get("first_query")
        or ""
    ).strip()

    cur_issue = str(
        current_intent.get("issue")
        or current_intent.get("summary")
        or ""
    ).strip()

    if not cur_issue and current_query:
        cur_issue = current_query

    prev_ws = _word_set(prev_issue)
    cur_ws = _word_set(cur_issue)

    # Require at least a couple overlapping content words to reduce false positives
    if prev_ws and cur_ws and len(prev_ws.intersection(cur_ws)) >= 2:
        return True

    return False


# ---------------------------------------------------------------------------
# Helper: lightweight emergency check (pre-fusion hard stop)
# ---------------------------------------------------------------------------


def _emergency_check(text: str, caption: Optional[str]) -> tuple[bool, Optional[str]]:
    """
    Lightweight, regex-based emergency detector for the *user's* request
    only (text + caption).

    This is intentionally conservative and should only catch obvious
    life-threatening scenarios like an active fire, strong gas leak,
    or structural collapse. Full safety gating on search results
    remains in run_fusion().
    """
    blob = " ".join(
        [
            (text or "").lower(),
            (caption or "").lower() if caption else "",
        ]
    )

    # Reuse the central EMERGENCY_PATTERNS from app.services.safety
    emergency = _detect_emergency(blob)
    if emergency.get("detected"):
        message = emergency.get("message") or (
            "Emergency detected! Please contact local emergency services immediately."
        )
        return True, message

    return False, None


# ---------------------------------------------------------------------------
# Front-door orchestration: synth_entry
# ---------------------------------------------------------------------------


def synth_entry(
    *,
    dialog_id: Optional[str],
    user_text: Optional[str],
    image_bytes: Optional[bytes],
    settings: Any,
) -> SessionContext:
    """
    v2 front-door orchestrator.

    Responsibilities:
    - Normalize inputs (dialog_id, user_text, image_bytes).
    - If needed, create/resolve dialog_id and session directories.
    - If image is present:
        - Run captioner (Qwen-VL or other).
    - Run a lightweight emergency safety check:
        - Decide `emergency_block` + optional `emergency_reason`.
        - This is a hard stop only; no multi-stage ack here.
    - Run intent/topic gate:
        - `is_home_repair` boolean.
        - `intent` metadata for downstream.
    - Run memory gate:
        - Inspect whether latest.json, summary.json, memory.json exist.
        - Load minimal topic_state from previous session if present.
        - Decide `session_mode` in {FIRST_TURN, SAME_TOPIC, TOPIC_SHIFT}.
    - Return a `SessionContext` bundle with all of the above.
    """

    # --- 0. Normalize text ---
    user_text_norm = (user_text or "").strip()
    if not user_text_norm and not image_bytes:
        # No usable input; synth_entry should not proceed
        raise ValueError("synth_entry requires at least text or an image")

    # --- 1. Resolve / create dialog_id and session paths ---
    dlg_id = (dialog_id or "").strip()
    if not dlg_id:
        # Simple UUID-based dialog id for new sessions
        dlg_id = uuid.uuid4().hex

    # Ensure standard directories; this is idempotent
    dirs = ensure_session_dirs(dlg_id)
    session_dir = dirs["base"]
    evdir = dirs["events"]
    latest_event_path = evdir / "latest.json"
    summary_path = session_dir / "summary.json"
    memory_path = session_dir / "memory.json"

    has_memory = memory_path.exists()

    # --- 2. Caption phase (image → caption) ---
    has_image = bool(image_bytes)
    image_caption: Optional[str] = None

    if has_image and image_bytes:
        try:
            img = load_image_from_bytes(image_bytes)
            cap = caption_image(img)
            image_caption = (cap or "").strip() or None
        except Exception as e:
            log.warning("synth_entry: caption_exception:%s", type(e).__name__)

    # --- 3. Lightweight emergency check (text + caption) ---
    emergency_block, emergency_reason = _emergency_check(user_text_norm, image_caption)

    # --- 4. Intent / topic gate (Qwen) ---
    intent_payload: Dict[str, Any] = {}
    is_home_repair = True

    try:
        ti = infer_topic_intent(
            user_text=user_text_norm,
            caption=image_caption,
        )
        if hasattr(ti, "model_dump"):
            intent_payload = ti.model_dump()
        elif isinstance(ti, dict):
            intent_payload = ti
        else:
            intent_payload = {}
    except Exception as e:
        log.warning("synth_entry: infer_topic_intent_failed:%s", type(e).__name__)
        intent_payload = {}

    if intent_payload:
        is_home_repair = bool(intent_payload.get("home_repair", True))
    else:
        # Default to True if the router fails; fusion will still have full safety
        is_home_repair = True

    # --- 5. Memory gate + topic_state / session_mode ---
    prev_topic_state: Dict[str, Any] = {}
    has_prev_session = False

    # 5a. Try chat_info.json topic_state (legacy v1/v2 state)
    try:
        sess = load_session(dlg_id)
        if isinstance(sess, dict):
            ts = sess.get("topic_state")
            if isinstance(ts, dict):
                prev_topic_state = ts
            has_prev_session = True
    except Exception as e:
        log.debug("synth_entry: load_session_failed:%s", type(e).__name__)

    # 5b. Overlay topic/issue from memory.active_project, if available.
    #     Memory is now the primary source of "what project are we in?"
    try:
        mem = load_memory(dlg_id)
    except Exception as e:
        log.debug("synth_entry: load_memory_failed:%s", type(e).__name__)
        mem = None

    if isinstance(mem, dict):
        ap = mem.get("active_project") or {}
        ap_intent = ap.get("topic_intent") or {}

        merged: Dict[str, Any] = dict(prev_topic_state) if isinstance(prev_topic_state, dict) else {}

        # Prefer topic/keywords from active_project.topic_intent when present.
        if isinstance(ap_intent, dict):
            topic_val = ap_intent.get("topic")
            if isinstance(topic_val, str) and topic_val.strip():
                merged["topic"] = topic_val
            kw_val = ap_intent.get("keywords")
            if isinstance(kw_val, (list, tuple)) and kw_val:
                merged["keywords"] = kw_val

        # Also bring in issue / first_query from active_project.
        issue_val = ap.get("issue")
        if isinstance(issue_val, str) and issue_val.strip():
            merged["issue"] = issue_val
        fq_val = ap.get("first_query")
        if isinstance(fq_val, str) and fq_val.strip():
            merged["first_query"] = fq_val

        prev_topic_state = merged
        # Presence of memory implies some prior context even if chat_info is missing.
        has_prev_session = has_prev_session or True
        
    # Decide session_mode
    mode: SessionMode

    # No memory, no prior session/topic_state → first turn
    if not has_memory and not has_prev_session and not latest_event_path.exists():
        mode = "FIRST_TURN"
    else:
        # We have some history; decide same-topic vs topic shift
        same_topic = _is_same_topic(
            prev_topic_state or None,
            intent_payload or None,
            user_text_norm,  # NEW: use current query text in gating
        )
        if same_topic:
            mode = "SAME_TOPIC"
        else:
            # Either a genuine topic shift or we couldn't compare properly
            mode = "TOPIC_SHIFT"

    ctx: SessionContext = {
        "dialog_id": dlg_id,
        "user_text": user_text_norm,
        "has_image": has_image,
        "image_bytes": image_bytes,
        "image_caption": image_caption,
        "is_home_repair": is_home_repair,
        "intent": intent_payload,
        "topic_state": prev_topic_state,
        "session_mode": mode,
        "has_memory": has_memory,
        "emergency_block": emergency_block,
        "emergency_reason": emergency_reason,
        "latest_event_path": latest_event_path if latest_event_path.exists() else None,
        "session_dir": session_dir,
        "summary_path": summary_path,
        "memory_path": memory_path,
    }

    return ctx


# ---------------------------------------------------------------------------
# Heavy first-turn engine: run_fusion (to be implemented next)
# ---------------------------------------------------------------------------


def run_fusion(
    ctx: SessionContext,
    *,
    settings: Any,
) -> Dict[str, Any]:
    """
    Heavy search + first-turn synthesis engine.

    Preconditions:
    - ctx["session_mode"] is either "FIRST_TURN" or "TOPIC_SHIFT".
    - ctx["emergency_block"] is False.

    This is the pure orchestrator version of the v1 fusion + synthesis
    path, built on top of existing service helpers.
    """
    dialog_id = ctx.get("dialog_id") or "anon"
    user_text = (ctx.get("user_text") or "").strip()
    image_bytes = ctx.get("image_bytes")
    image_caption = (ctx.get("image_caption") or None) or None

    has_image = bool(ctx.get("has_image"))

    # --- Metadata containers (for debug + memory) ---
    notes: Dict[str, Any] = {
        "session_mode": ctx.get("session_mode"),
        "topic_intent": ctx.get("intent") or {},
        "counts": {},
        "timings_ms": {},
        "queries": [],
    }
    warnings: list[str] = []

    # ------------------------------------------------
    # 1) Query generation (Qwen)
    # ------------------------------------------------
    t0 = time.perf_counter()
    try:
        query_items = generate_queries_llm(
            user_text=user_text,
            caption=image_caption,
            prefer_rag=True,
            max_queries=6,
        )
    except Exception as e:
        log.warning("run_fusion: generate_queries_llm failed: %s", e, exc_info=True)
        query_items = []

    notes["queries"] = query_items or []

    # Canonical query_text: prefer first RAG query, then WEB, then fallback
    query_text = ""
    for qi in query_items or []:
        if not isinstance(qi, dict):
            continue
        src = (qi.get("source") or "").lower()
        q = (qi.get("query") or "").strip()
        if not q:
            continue
        if src == "rag":
            query_text = q
            break
        if not query_text and src == "web":
            query_text = q

    if not query_text:
        # Ultimate fallback if LLM did not produce anything
        query_text = user_text or (image_caption or "")

    notes["timings_ms"]["query_gen"] = int((time.perf_counter() - t0) * 1000)
    notes["counts"]["query_tokens"] = len(query_text.split()) if query_text else 0

    # ------------------------------------------------
    # 2) RAG search
    # ------------------------------------------------
    rag_hits: list[dict] = []
    t1 = time.perf_counter()
    if query_text:
        try:
            rag_out = run_rag_search(query=query_text, top_k=5)
            if rag_out.get("ok"):
                rag_hits = list(rag_out.get("results") or [])
            else:
                warnings.append("rag_not_ok")
        except Exception as e:
            log.warning("run_fusion: run_rag_search failed: %s", e, exc_info=True)
            warnings.append("rag_exception")
    else:
        warnings.append("rag_empty_query")

    notes["timings_ms"]["rag"] = int((time.perf_counter() - t1) * 1000)
    notes["counts"]["rag_hits"] = len(rag_hits)

    # ------------------------------------------------
    # 3) Image NN search
    # ------------------------------------------------
    image_hits: list[dict] = []
    t2 = time.perf_counter()
    if has_image and image_bytes:
        try:
            img = load_image_from_bytes(image_bytes)
            neighbors = search_similar_images(img, top_k=5)
            image_hits = [normalize_image_item(n) for n in (neighbors or [])]
        except Exception as e:
            log.warning("run_fusion: search_similar_images failed: %s", e, exc_info=True)
            warnings.append("image_search_exception")

    notes["timings_ms"]["image"] = int((time.perf_counter() - t2) * 1000)
    notes["counts"]["image_hits"] = len(image_hits)

    # ------------------------------------------------
    # 4) Web search (also merges hits into latest event)
    # ------------------------------------------------
    web_hits: list[dict] = []
    t3 = time.perf_counter()
    if query_text:
        try:
            web_out = run_web_search_service(
                dialog_id=dialog_id,
                queries=[query_text],
                max_results=6,
                apply_filter=False,
            )
            if web_out.get("ok"):
                web_hits = list(web_out.get("results") or [])
            else:
                warnings.append("web_not_ok")
        except Exception as e:
            log.warning("run_fusion: run_web_search_service failed: %s", e, exc_info=True)
            warnings.append("web_exception")
    else:
        warnings.append("web_empty_query")

    notes["timings_ms"]["web"] = int((time.perf_counter() - t3) * 1000)
    notes["counts"]["web_hits"] = len(web_hits)

    # ------------------------------------------------
    # 5) Fuse results + enrich with page images
    # ------------------------------------------------
    fused: list[dict] = []
    fused.extend(rag_hits)
    fused.extend(web_hits)
    fused.extend(image_hits)

    try:
        fused = enrich_with_page_images(fused)
    except Exception as e:
        log.warning("run_fusion: enrich_with_page_images failed: %s", e, exc_info=True)
        warnings.append("enrich_with_page_images_exception")

    notes["counts"]["fused_hits"] = len(fused)

    # Build evidence buckets (text/image/web)
    try:
        evidence = build_evidence(fused, limit=9)
    except Exception as e:
        log.warning("run_fusion: build_evidence failed: %s", e, exc_info=True)
        warnings.append("build_evidence_exception")
        evidence = []

    # ------------------------------------------------
    # 6) Safety (advisory + staged gating)
    # ------------------------------------------------
    snippets = collect_snippets(fused, limit=24)
    safety_report = analyze_text_blocks(snippets)
    safety_dict = safety_report.model_dump()

    # Start from a neutral, advisory-only baseline. We will override with the
    # staged intent decision from safety_payload/decide_intent.
    blocked = False
    requires_ack = False
    ack_stage_effective = 0

    intent_decision: Dict[str, Any] = {}
    try:
        payload = safety_payload(snippets, query_text=user_text)
        intent_decision = decide_intent(payload, ack_stage=0)

        # Keep the interesting bits around for debugging / UI, but do not
        # force the UI shape to know the full payload.
        safety_dict["payload"] = {
            "emergency": payload.get("emergency"),
            "work_types": payload.get("work_types"),
            "staged": payload.get("staged"),
        }
        safety_dict["intent_decision"] = intent_decision
    except Exception as e:
        log.warning("run_fusion: safety_payload/decide_intent failed: %s", e, exc_info=True)
        warnings.append("safety_payload_exception")
        payload = {}

    gate_text = ""
    gate_type = "none"

    if intent_decision:
        decision = (intent_decision.get("decision") or "").strip()
        msg = (intent_decision.get("message") or "").strip()

        if decision == "emergency_override":
            # Late emergency detection – treat as hard block, no ack.
            blocked = True
            requires_ack = False
            ack_stage_effective = 0
            gate_type = "emergency"
            gate_text = (
                msg
                or "Emergency detected! Please contact local emergency services immediately."
            )

        elif decision in ("stage_1_required", "stage_2_required"):
            # Hazard gate: soft refusal + one-tap acknowledgement in the app.
            # The model work still runs and the answer is ready once the user
            # taps through the gate.
            blocked = False
            requires_ack = True
            ack_stage_effective = 0  # UX: treat as a single-step ack for now
            gate_type = "hazard_s2" if decision == "stage_2_required" else "hazard_s1"
            gate_text = (
                msg
                or "This repair involves higher-risk work. Please review the safety warning before proceeding."
            )

        else:
            # "proceed" or unknown tag → no gating from this layer.
            blocked = False
            requires_ack = False
            ack_stage_effective = 0
            gate_type = "none"
    else:
        # No staged payload → default to advisory-only
        blocked = False
        requires_ack = False
        ack_stage_effective = 0
        gate_type = "none"

    # Normalize the dict we send to the app
    safety_dict["blocked"] = blocked
    safety_dict["requires_ack"] = requires_ack
    safety_dict["ack_stage"] = ack_stage_effective
    safety_dict["gate_type"] = gate_type
    if gate_text:
        safety_dict["gate"] = gate_text

    # ------------------------------------------------
    # 7) Record session event (search phase)
    # ------------------------------------------------
    try:
        record_session_event(
            settings=settings,
            dialog_id=dialog_id,
            query_text=query_text,
            has_image=has_image,
            query_caption=image_caption,
            notes=notes,
            warnings=warnings,
            results=fused,
            max_results=8,
            answer=None,
            evidence=evidence,
            blocked=blocked,
            requires_ack=requires_ack,
            ack_stage=ack_stage_effective,
        )
    except Exception as e:
        log.warning("run_fusion: record_session_event failed: %s", e, exc_info=True)
        warnings.append("record_session_event_exception")

    # ------------------------------------------------
    # 8) Pack + longform + summary
    # ------------------------------------------------
    try:
        pack_info = pack_and_persist_results(
            dialog_id=dialog_id,
            k_text=3,
            k_web=3,
            k_image=3,
        )
        pack_path = Path(pack_info.get("path") or "")
        longform_pack: Dict[str, Any] = {}
        if pack_path.is_file():
            longform_pack = json.loads(pack_path.read_text(encoding="utf-8"))
    except Exception as e:
        log.warning("run_fusion: pack_and_persist_results failed: %s", e, exc_info=True)
        warnings.append("pack_and_persist_results_exception")
        longform_pack = {}

    longform: Dict[str, Any] = {}
    if longform_pack:
        try:
            longform = inflate_longform_pack(
                longform_pack,
                rag_window=1,
                rag_max_chunks=3,
                web_max_paragraphs=6,
                web_max_chars=2000,
                max_sources=3,
            )
        except Exception as e:
            log.warning("run_fusion: inflate_longform_pack failed: %s", e, exc_info=True)
            warnings.append("inflate_longform_pack_exception")
            longform = {}

    # Summarize with LLM; keep a conservative fallback
    summary: Dict[str, Any] = {}
    if longform:
        try:
            summary = summarize_longform_llm(
                issue_text=user_text,
                longform_pack=longform,
                topic=(ctx.get("intent") or {}).get("topic"),
                topic_intent=ctx.get("intent") or {},
                image_caption=image_caption,
                safety=safety_dict,
                dialog_id=dialog_id,
            )
        except Exception as e:
            log.warning("run_fusion: summarize_longform_llm failed: %s", e, exc_info=True)
            warnings.append("summarize_longform_llm_exception")
            summary = {}
    if not summary:
        summary = {
            "title": "Home repair guidance",
            "summary": "I could not build a full project summary from the available evidence.",
            "steps": [],
        }

    # Persist summary.json under the session directory
    try:
        session_dir = ctx.get("session_dir") or ensure_session_dirs(dialog_id)["base"]
        summary_path = Path(session_dir) / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception as e:
        log.warning("run_fusion: summary write failed: %s", e, exc_info=True)
        warnings.append("summary_write_exception")

    # ------------------------------------------------
    # 9) Build memory + answer from summary
    # ------------------------------------------------
    try:
        mem = build_memory(dialog_id)
    except Exception as e:
        log.warning("run_fusion: build_memory failed: %s", e, exc_info=True)
        warnings.append("build_memory_exception")
        mem = {"key_points": []}

    history = list(mem.get("key_points") or [])

    user_bundle = {
        "user_text": user_text,
        "image_caption": image_caption,
        "topic_intent": ctx.get("intent") or {},
        "safety": safety_dict,
        "ack_stage": ack_stage_effective,
    }

    try:
        sections = answer_from_summary_llm(
            summary,
            history,
            user_bundle,
            dialog_id=dialog_id,
        )
    except Exception as e:
        log.warning("run_fusion: answer_from_summary_llm failed: %s", e, exc_info=True)
        warnings.append("answer_from_summary_llm_exception")
        sections = fallback_sections(user_text)

    # Synthesis object: normalized keys
    synthesis_obj: Dict[str, Any] = {
        "title": sections.get("title"),
        "scope_overview": sections.get("scope_overview"),
        "potential_causes": sections.get("potential_causes"),
        "tools_required": sections.get("tools_required"),
        "first_step": sections.get("first_step"),
        "longform_refs": sections.get("longform_refs"),
        "image_citations": sections.get("image_citations"),
        "topic": (ctx.get("intent") or {}).get("topic"),
        "topic_intent": ctx.get("intent") or {},
        "image_caption": image_caption,
        "safety_summary": safety_dict,
    }

    # Answer preview: overview + first step if available
    preview_parts: list[str] = []
    if synthesis_obj.get("scope_overview"):
        preview_parts.append(str(synthesis_obj["scope_overview"]))
    if synthesis_obj.get("first_step"):
        preview_parts.append(f"First step: {synthesis_obj['first_step']}")
    answer_preview = "\n\n".join(preview_parts) if preview_parts else ""

    # ------------------------------------------------
    # 10) Attach synthesis to latest event
    # ------------------------------------------------
    try:
        latest_path = ctx.get("latest_event_path") or (events_dir(dialog_id) / "latest.json")
        latest_path = Path(latest_path)
        if latest_path.is_file():
            ev = json.loads(latest_path.read_text(encoding="utf-8"))
        else:
            ev = {}

        orch = ev.get("orchestration") or {}
        orch["synthesis"] = {
            "summary": summary,
            "safety": safety_dict,
        }
        orch.setdefault("topic_intent", ctx.get("intent") or {})
        orch.setdefault("notes", notes)
        orch.setdefault("warnings", warnings)

        ev["orchestration"] = orch
        ev["synthesis"] = synthesis_obj
        ev["answer"] = answer_preview
        ev["blocked"] = blocked
        ev["requires_ack"] = requires_ack
        ev["ack_stage"] = ack_stage_effective

        persist_latest_event(dialog_id, ev)
    except Exception as e:
        log.warning("run_fusion: persist_latest_event failed: %s", e, exc_info=True)
        warnings.append("persist_latest_event_exception")

    # ------------------------------------------------
    # 11) Response envelope
    # ------------------------------------------------
    body: Dict[str, Any] = {
        "ok": True,
        "dialog_id": dialog_id,
        "blocked": blocked,
        "requires_ack": requires_ack,
        "ack_stage": ack_stage_effective,
        "orchestration": {
            "session_mode": ctx.get("session_mode"),
            "topic_intent": ctx.get("intent") or {},
            "safety": safety_dict,
            "notes": notes,
            "warnings": warnings,
        },
        "synthesis": synthesis_obj,
        "answer": answer_preview,
    }
    return body


# ---------------------------------------------------------------------------
# Multi-turn engine: run_step_dialog (to be implemented later)
# ---------------------------------------------------------------------------


def run_step_dialog(
    ctx: SessionContext,
    *,
    settings: Any,
) -> Dict[str, Any]:
    """
    Multi-turn SAME_TOPIC engine.

    Preconditions:
    - ctx["session_mode"] == "SAME_TOPIC"
    - ctx["has_memory"] == True
    - ctx["emergency_block"] == False

    Responsibilities:
    - Use existing multi-turn helper run_next_step_turn() to:
        - Call generate_next_step_llm() with rolling dialog context.
        - Persist a lightweight "next step" event and refresh memory.json.
    - Reload latest.json to attach STEP_DIALOG orchestration metadata.
    - Preserve / extend previous synthesis where available.
    - Return an envelope shaped like run_fusion(), but focused on next step.
    """
    dialog_id = (ctx.get("dialog_id") or "anon").strip() or "anon"
    user_text = (ctx.get("user_text") or "").strip()
    image_caption = (ctx.get("image_caption") or "").strip() or None

    # --- 0. Resolve paths & load previous latest for synthesis reuse ---
    latest_path = ctx.get("latest_event_path")
    if latest_path:
        latest_path = Path(latest_path)
    else:
        latest_path = events_dir(dialog_id) / "latest.json"

    prev_latest: Dict[str, Any] = {}
    prev_synthesis: Dict[str, Any] = {}
    prev_orch: Dict[str, Any] = {}

    try:
        if latest_path.is_file():
            prev_latest = json.loads(latest_path.read_text(encoding="utf-8"))
            if isinstance(prev_latest, dict):
                prev_synthesis = prev_latest.get("synthesis") or {}
                prev_orch = prev_latest.get("orchestration") or {}
    except Exception as e:
        log.warning("run_step_dialog: failed to load previous latest: %s", e, exc_info=True)
        prev_latest = {}
        prev_synthesis = {}
        prev_orch = {}

    # --- 1. Ask the multi-turn helper for the next step ---
    # Ack semantics are minimal for now; we treat next-step as post-ack.
    step = run_next_step_turn(
        settings=settings,
        dialog_id=dialog_id,
        user_text=user_text,
        image_caption=image_caption,
        ack_stage=None,
    )

    next_step_text = (step.get("next_step_text") or "").strip()
    if not next_step_text:
        # Hard fallback: use the conservative message from generate_next_step_llm
        next_step_text = (
            "I couldn't clearly determine the next step. "
            "Please briefly restate what you've done so far and what "
            "you would like to do next (include a photo if helpful)."
        )

    topic = (step.get("topic") or "").strip() or None
    topic_intent = step.get("topic_intent") or ctx.get("intent") or {}
    hazards_seen = step.get("hazards_seen") or []
    step_warnings: List[str] = list(step.get("warnings") or [])
    ack_stage_effective = int(step.get("ack_stage") or 0)

    # --- 2. Reload latest after run_next_step_turn wrote the new event ---
    latest: Dict[str, Any] = {}
    try:
        if latest_path.is_file():
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
        else:
            latest = {}
    except Exception as e:
        log.warning("run_step_dialog: reload latest.json failed: %s", e, exc_info=True)
        latest = {}

    if not isinstance(latest, dict):
        latest = {}

    # Safety flags for step-dialog: we reuse whatever latest has, but
    # note that run_next_step_turn currently records blocked=False/requires_ack=False.
    blocked = bool(latest.get("blocked")) if "blocked" in latest else False
    requires_ack = bool(latest.get("requires_ack")) if "requires_ack" in latest else False

    # Prior safety summary, if we had one from the first-turn fusion
    orch = latest.get("orchestration") or prev_orch or {}
    safety_dict = (orch.get("safety") or {})

    # --- 3. Build / extend synthesis snapshot ---
    syn: Dict[str, Any] = latest.get("synthesis") or prev_synthesis or {}

    if not syn:
        # Minimal synthesis constructed from summary + topic + next step.
        summary_text = (step.get("summary") or "").strip()
        syn = {
            "title": f"Home repair guidance – {topic or 'project'}",
            "scope_overview": summary_text or "Ongoing repair project.",
            "potential_causes": [],
            "tools_required": [],
            "first_step": next_step_text,
            "longform_refs": [],
            "image_citations": [],
            "image_caption": image_caption,
            "safety_summary": safety_dict,
            "source_counts": {},
            "topic": topic or (topic_intent.get("topic") if isinstance(topic_intent, dict) else None),
            "topic_intent": topic_intent,
        }

    # Append to step history on synthesis
    step_history = syn.get("step_history") or []
    if not isinstance(step_history, list):
        step_history = [str(step_history)]
    step_history = list(step_history)
    step_history.append(next_step_text)
    syn["step_history"] = step_history

    # Keep first_step as "current" step in synthesis for convenience
    syn["first_step"] = next_step_text

    # --- 4. Orchestration notes / warnings for this turn ---
    notes = orch.get("notes") or {}
    warnings = list(orch.get("warnings") or [])
    warnings.extend(step_warnings)

    notes.setdefault("step_dialog", {})
    notes["step_dialog"]["source"] = step.get("_source")
    if hazards_seen:
        notes["step_dialog"]["hazards_seen"] = hazards_seen

    orchestration = {
        "session_mode": "SAME_TOPIC",
        "mode": "STEP_DIALOG",
        "topic_intent": topic_intent,
        "safety": safety_dict,
        "notes": notes,
        "warnings": warnings,
    }

    # --- 5. Persist updated latest with synthesis + answer attached ---
    try:
        latest["orchestration"] = orchestration
        latest["synthesis"] = syn
        latest["answer"] = next_step_text
        latest["blocked"] = blocked
        latest["requires_ack"] = requires_ack
        latest["ack_stage"] = ack_stage_effective
        persist_latest_event(dialog_id, latest)
    except Exception as e:
        log.warning("run_step_dialog: persist_latest_event failed: %s", e, exc_info=True)

    # --- 6. Response envelope (API-facing) ---
    body: Dict[str, Any] = {
        "ok": True,
        "dialog_id": dialog_id,
        "blocked": blocked,
        "requires_ack": requires_ack,
        "ack_stage": ack_stage_effective,
        "session_mode": "SAME_TOPIC",
        "orchestration": orchestration,
        "synthesis": syn,
        "answer": next_step_text,
    }
    return body

