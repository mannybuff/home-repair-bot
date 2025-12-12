# app/services/qwen_use.py
"""
High-level Qwen usage helpers.

Centralizes:
  - Topic/intent inference
  - Query generation (LLM side)
  - Section synthesis (LLM-backed)
  - Captioner access (via qwen_captioner)

All Qwen model calls should flow through this module so that:
  - api/fusion.py, synthesis, query_gen, etc. do not talk to transformers directly.
  - Future model swaps happen in one place.
"""

from __future__ import annotations

from typing import Any, Tuple, Dict, List, Optional
import json
import os
import re

from app.utils.qwen_utils import (
    _llm_chat,
    _LLMQ_DEFAULT_TEMP,
    get_image_captioner,
    build_step_dialogue_messages,
)
from app.utils.session_utils import build_dialogue_context

from app.core.settings import settings



# ---------------------------------------------------------------------------
# Captioner access
# ---------------------------------------------------------------------------

def get_qwen_captioner():
    """
    Service-level accessor for the Qwen captioner.

    All caption usage (fusion, image services, legacy /vlm router) should
    import THIS function instead of touching vlm-specific modules.
    """
    return get_image_captioner()


# ---------------------------------------------------------------------------
# Shared JSON helper
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a single JSON object from text by locating the outermost {...}.
    """
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Topic / intent inference
# ---------------------------------------------------------------------------

def infer_topic_intent(user_text: str, caption: Optional[str] = None) -> Dict[str, Any]:
    """
    Classify a request into a coarse home-repair topic and intent, with hazards
    and keywords, and decide if it is actually about home repair.

    Returns:
      {
        "topic": "drywall|plumbing|electrical|hvac|appliance|painting|flooring|structural|general|non_repair",
        "intent": "diagnose|repair|replace|install|maintain|unknown",
        "hazards": [str, ...],
        "keywords": [str, ...],
        "home_repair": bool,
      }
    """
    txt = (user_text or "").strip()
    cap = (caption or "").strip()

    # Combined description for the model: user text + caption
    full_input = txt
    if cap:
        if full_input:
            full_input = f"{full_input}  {cap}"
        else:
            full_input = cap
    full_input = full_input or ""

    sys = (
        "You are a classifier for a home-repair assistant.\n"
        "\n"
        "Given a short description (user text plus optional image caption), your job is to:\n"
        "- Decide if the user is asking about a home repair, maintenance, or\n"
        "  home improvement task (home_repair = true), or something unrelated\n"
        "  like pets, people, vehicles, stories, general life questions, work,\n"
        "  school, or finances (home_repair = false).\n"
        "- If home_repair is true, choose a coarse TOPIC from:\n"
        "    drywall, plumbing, electrical, hvac, appliance, painting, flooring,\n"
        "    structural, general\n"
        "- If home_repair is false, set topic to \"non_repair\".\n"
        "- Choose INTENT from: diagnose, repair, replace, install, maintain, unknown.\n"
        "- List any obvious HAZARDS such as: fire, live_electric, gas_leak, flooding,\n"
        "  fall_risk, structural_collapse, asbestos, lead_paint, mold.\n"
        "- Provide 5-10 strong KEYWORDS summarizing the situation.\n"
        "\n"
        "Be slightly conservative: if the description could easily be general life advice\n"
        "or entertainment instead of a project on the home itself, prefer\n"
        "home_repair: false and topic: non_repair.\n"
        "\n"
        "Output format (use EXACTLY these labeled lines, and nothing else):\n"
        "text_caption: <your one-sentence rephrasing of the situation>\n"
        "topic: <one word topic>\n"
        "intent: <one word intent>\n"
        "hazards: <comma-separated hazard words or \"none\">\n"
        "keywords: <comma-separated keyword phrases>\n"
        "home_repair: <true or false>\n"
    )

    msgs = [
        {"role": "system", "content": sys},
        {"role": "user", "content": f"Input:\n{full_input}"},
    ]

    # Slightly lower max_tokens to discourage rambling / extra examples
    raw = _llm_chat(msgs, temperature=_LLMQ_DEFAULT_TEMP, max_tokens=128) or ""
    text = raw.strip()

    topic: str = ""
    intent: str = ""
    hazards: List[str] = []
    keywords: List[str] = []
    home_repair: Optional[bool] = None

    if text:
        for line in text.splitlines():
            line_strip = line.strip()
            if not line_strip:
                continue
            lower = line_strip.lower()

            def _extract(label: str) -> Optional[str]:
                if lower.startswith(label + ":"):
                    return line_strip.split(":", 1)[1].strip()
                return None

            v = _extract("topic")
            if v is not None:
                topic = v
                continue

            v = _extract("intent")
            if v is not None:
                intent = v
                continue

            v = _extract("hazards")
            if v is not None:
                items = [x.strip() for x in v.split(",") if x.strip()]
                if len(items) == 1 and items[0].lower() in ("none", "no", "n/a"):
                    hazards = []
                else:
                    hazards = items
                continue

            v = _extract("keywords")
            if v is not None:
                keywords = [x.strip() for x in v.split(",") if x.strip()]
                continue

            v = _extract("home_repair")
            if v is not None:
                v_low = v.lower()
                if v_low in ("true", "yes", "y", "1"):
                    home_repair = True
                elif v_low in ("false", "no", "n", "0"):
                    home_repair = False
                continue

    # Fallback: if we got nothing usable, treat as generic home repair
    # (non-repair override below can still flip this).
    topic = (topic or "").strip().lower()
    intent = (intent or "").strip().lower()

    valid_topics = {
        "drywall", "plumbing", "electrical", "hvac", "appliance",
        "painting", "flooring", "structural", "general", "non_repair",
    }
    if topic not in valid_topics:
        topic = "general"

    valid_intents = {"diagnose", "repair", "replace", "install", "maintain", "unknown"}
    if intent not in valid_intents:
        intent = "diagnose"

    if not isinstance(hazards, list):
        hazards = []
    if not isinstance(keywords, list):
        keywords = []

    if home_repair is None:
        home_repair = topic != "non_repair"

    # Heuristic override for obvious non-repair pet/people content
    text_low = full_input.lower()
    pet_terms = ["cat", "kitten", "dog", "puppy", "pet", "hamster", "rabbit", "bird"]
    home_terms = [
        "wall", "ceiling", "floor", "roof", "window", "door", "sink", "toilet",
        "faucet", "pipe", "outlet", "socket", "switch", "heater", "furnace",
        "water heater", "boiler", "dryer", "washer", "stove", "oven", "garage",
        "basement", "attic", "deck", "porch", "tile", "grout",
    ]

    has_pet = any(term in text_low for term in pet_terms)
    has_home = any(term in text_low for term in home_terms)

    if has_pet and not has_home:
        topic = "non_repair"
        home_repair = False

    # Heuristic nudge for appliance-related case
    appliance_terms = [
        "dryer", "dishwasher", "fridge", "refrigerator", "microwave", "oven",
        "stove", "range", "washer", "washing machine", "laundry machine",
        "water heater", "boiler",
    ]
    smell_terms = ["smell", "odor", "odour", "smells", "stinks", "stinky"]
    burn_terms = ["burning", "burn", "smoke", "smoky"]
    has_appliance = any(term in text_low for term in appliance_terms)
    has_smell = any(term in text_low for term in smell_terms)
    has_burn = any(term in text_low for term in burn_terms)
    has_gas = "gas" in text_low

    if has_appliance:
        # If Qwen was unsure, treat it as appliance home repair.
        if topic in ("general", "non_repair"):
            topic = "appliance"
        if home_repair is False:
            home_repair = True

        # Promote obvious hazard signals for smells around appliances.
        if has_smell or has_burn:
            if "fire" not in hazards:
                hazards.append("fire")
        if has_gas:
            if "gas_leak" not in hazards:
                hazards.append("gas_leak")

    return {
        "topic": topic,
        "intent": intent,
        "hazards": hazards,
        "keywords": keywords,
        "home_repair": home_repair,
    }
    
# ---------------------------------------------------------------------------
# Query generation (LLM side)
# ---------------------------------------------------------------------------
def _heuristic_queries_from_text(
    user_text: str,
    caption: Optional[str],
    max_queries: int = 4,
) -> List[Dict[str, str]]:
    """
    Very simple fallback when Qwen output is unusable.
    """
    base = (user_text or "").strip()
    extra = (caption or "").strip()
    if extra and extra.lower() not in base.lower():
        base = f"{base} {extra}".strip()
    base = base or "home repair problem"

    # crude keyword compression
    words = re.findall(r"[A-Za-z0-9]+", base.lower())
    uniq = []
    for w in words:
        if w not in uniq:
            uniq.append(w)
    short = " ".join(uniq[:12]) or base

    out: List[Dict[str, str]] = [
        {"source": "rag", "query": short},
    ]

    if len(uniq) > 6:
        mid = " ".join(uniq[:8])
        out.append({"source": "web", "query": mid})

    return out[:max_queries]

def generate_queries_llm(
    user_text: str,
    caption: Optional[str] = None,
    prefer_rag: bool = True,
    max_queries: int = 6,
) -> List[Dict[str, str]]:
    """
    Use Qwen2-VL (text mode) to generate search queries for RAG + Web.

    Input may be:
      - a user problem description, or
      - a short instruction / step such as "Shut off the water under the sink".

    Returns a list of dicts:
        { "source": "rag" | "web", "query": "<short search query>" }
    """
    user_text = (user_text or "").strip()
    caption = (caption or "").strip() or None

    if not user_text and not caption:
        return []

    # Build combined input text
    if user_text and caption:
        full_input = f"{user_text}\n\nImage/caption: {caption}"
    elif user_text:
        full_input = user_text
    else:
        full_input = caption

    full_input = full_input.strip()
    if not full_input:
        return []

    # ---- 1) System prompt ----
    #
    # We explicitly support both problem descriptions and step text,
    # and we ask for slightly longer, more descriptive web queries
    # (7–12 words) for better semantic + image search.
    sys_prompt = (
        "You are a helpful AI assistant. Your job is to generate home repair "
        "search queries from a short input.\n"
        "\n"
        "The input may be either:\n"
        "- a user describing a problem, or\n"
        "- a short instruction or step in the repair process.\n"
        "\n"
        "You must:\n"
        "- Extract important information such as materials, location, room, "
        "  surfaces, tools, and symptoms.\n"
        "- Create TWO queries:\n"
        "  1) rag: a compact keyword-style search phrase (about 5–10 words) "
        "     that is good for a dense vector search index.\n"
        "  2) web: a natural-language search query (about 7–12 words) that is "
        "     good for modern web search and finding clear photos or diagrams.\n"
        "\n"
        "Both queries should include specific objects and context when possible, "
        "such as the room (kitchen, bathroom), fixture (sink, GFCI outlet), "
        "and action (shut off water, remove trim, replace P-trap).\n"
        "\n"
        "Do NOT include explanations or any extra text. Output must be EXACTLY "
        "two lines in this format:\n"
        "rag: <short keyword phrase>\n"
        "web: <natural-language query>\n"
        "\n"
        "Examples (do not repeat these in your answer):\n"
        "rag: peeling bathroom ceiling paint above shower moisture\n"
        "web: how to fix peeling paint on bathroom ceiling above shower\n"
        "\n"
        "rag: shut off valve under kitchen sink supply lines\n"
        "web: how to find and turn off water shut off valve under kitchen sink\n"
    )

    msgs = [
        {"role": "system", "content": sys_prompt},
        {
            "role": "user",
            "content": f"Generate search queries for this home repair context:\n\n{full_input}",
        },
    ]

    # Slightly higher temperature than defaults to allow a bit of elaboration,
    # but still deterministic enough for stable parsing.
    raw = _llm_chat(msgs, temperature=0.3, max_tokens=96) or ""
    text = raw.strip()
    if not text:
        return _heuristic_queries_from_text(user_text, caption, max_queries=max_queries)

    rag_raw: Optional[str] = None
    web_raw: Optional[str] = None

    # ---- 2) Parse the two labeled lines ----
    for line in text.splitlines():
        line_strip = line.strip()
        if not line_strip:
            continue
        lower = line_strip.lower()
        if lower.startswith("rag:"):
            rag_raw = line_strip.split(":", 1)[1].strip()
        elif lower.startswith("web:"):
            web_raw = line_strip.split(":", 1)[1].strip()

    # If tags missing, try to recover from the first two non-empty lines
    if rag_raw is None or web_raw is None:
        non_empty = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if rag_raw is None and non_empty:
            rag_raw = non_empty[0]
            if ":" in rag_raw:
                rag_raw = rag_raw.split(":", 1)[1].strip()
        if web_raw is None and len(non_empty) > 1:
            web_raw = non_empty[1]
            if ":" in web_raw:
                web_raw = web_raw.split(":", 1)[1].strip()

    # If still unusable, fall back to simple heuristic keyword queries
    if not rag_raw or not web_raw:
        return _heuristic_queries_from_text(user_text, caption, max_queries=max_queries)

    rag_q = rag_raw.strip()
    web_q = web_raw.strip()
    if not rag_q or not web_q:
        return _heuristic_queries_from_text(user_text, caption, max_queries=max_queries)

    # ---- 3) Build the final list in preferred order ----
    queries: List[Dict[str, str]] = []

    if prefer_rag:
        queries.append({"source": "rag", "query": rag_q})
        queries.append({"source": "web", "query": web_q})
    else:
        queries.append({"source": "web", "query": web_q})
        queries.append({"source": "rag", "query": rag_q})

    # Clip to max_queries (we only generate 2 here, but keep the parameter)
    return queries[:max_queries]    

def fallback_sections(user_text: str) -> Dict[str, Any]:
    """
    Simple, safe baseline if no evidence or LLM failure.

    This is intentionally conservative and does not assume any specific
    diagnosis. It gives the app a consistent JSON shape even when
    evidence or LLM synthesis is unavailable.
    """
    return {
        "scope_overview": (
            "I couldn’t find usable evidence for this dialog yet, so here’s a "
            "conservative starting plan based on the request."
        ),
        "potential_causes": [],
        "tools_required": [],
        "first_step": (
            "Confirm the area is safe (water and power hazards). Take a clear "
            "photo of the problem and provide any make/model details. Then "
            "re-run search to gather references."
        ),
        "longform_refs": [],
        "image_citations": [],
        "_source": "fallback",
    }
    
# ---------------------------------------------------------------------------
# Multi-turn "next step" generation (LLM-backed)
# ---------------------------------------------------------------------------

def generate_next_step_llm(
    dialog_id: str,
    user_text: str,
    image_caption: Optional[str] = None,
    *,
    temperature: float = _LLMQ_DEFAULT_TEMP,
    max_tokens: Optional[int] = 256,
) -> Dict[str, Any]:
    """
    Generate the NEXT step for an existing dialog using the rolling context.

    This version follows a simple labeled-line format similar in spirit to
    `answer_from_summary_llm`, but focused only on the next actionable step.

    It:

      - Loads a compact context snapshot for dialog_id
      - Builds a single system+user message with:
          * topic / intent / hazards
          * a short repair-guide style summary
          * a brief view of recent turns
          * the latest user question
      - Calls the shared Qwen text LLM via _llm_chat
      - Parses two labeled lines:

            Next step: <3–6 sentences of concrete instructions>
            Follow-up question: <one short question, or "none">

      - Normalizes into a dict suitable for event/answer writers

    It does NOT write any files; callers remain responsible for recording
    events and updating memory.
    """
    dialog_id = (dialog_id or "").strip()
    user_text = (user_text or "").strip()

    # Build context snapshot (topic, summary, recent_turns, hazards, etc.)
    ctx = build_dialogue_context(dialog_id, max_events=4)

    topic = ctx.get("topic") or "general"
    topic_intent = ctx.get("topic_intent") or "followup"
    summary_text = ctx.get("summary") or ""
    hazards_seen = ctx.get("hazards_seen") or []

    # Try to build a compact view of recent turns from whatever the context provides.
    recent = (
        ctx.get("recent_turns")
        or ctx.get("recent_events")
        or ctx.get("turn_summaries")
        or []
    )

    recent_lines: List[str] = []
    if isinstance(recent, list):
        for ev in recent:
            if isinstance(ev, dict):
                role = ev.get("role") or ev.get("speaker") or "event"
                text = (
                    ev.get("text")
                    or ev.get("content")
                    or ev.get("summary")
                    or ""
                )
                text = str(text).strip()
                if text:
                    recent_lines.append(f"- {role}: {text}")
            else:
                recent_lines.append(f"- {str(ev).strip()}")
    elif recent:
        recent_lines.append(str(recent).strip())

    recent_block = "\n".join(recent_lines[:6]) if recent_lines else "[no recent turns recorded]"

    # -----------------------------
    # Slim, direct system prompt
    # -----------------------------
    system_msg = """
You are HomeRepairBot, an AI assistant that helps users safely continue a home repair project.

You will be given:
- a short description of what the user just did or wants to do next
- the overall repair topic and intent
- a concise repair-guide style summary of the project so far
- a short log of recent conversation turns

Your job is to decide the SINGLE NEXT STEP the user should take, not the whole plan.

Output format (exactly two lines, in this order):

Next step: <3–6 sentences giving very concrete, actionable instructions for the very next thing the user should do>
Follow-up question: <one short, specific question you want the user to answer after finishing this step, or "none">

Important:
- Stay on the same repair topic; do NOT change the project.
- Use only safe, practical actions. When there is any risk (electricity, gas, water leaks, structural issues), include making the area safe as part of the next step.
- Use the repair summary and the recent turns to avoid repeating steps that have already been done.
- Focus only on the next step, not the entire repair from start to finish.

Additional rules:
- Put the entire content for each label on a SINGLE LINE. Do not insert line breaks inside a value.
- Do NOT add any extra lines, headings, bullet lists, or labels beyond the two specified.
- Do NOT talk about “the guide”, “the log”, “the conversation”, or yourself. Just write the two labeled lines.
""".strip()

    # -----------------------------
    # Minimal, data-only user message
    # -----------------------------
    hazards_str = ", ".join(map(str, hazards_seen)) if hazards_seen else "none"

    user_msg = f"""
Latest user message:
{user_text or "[not provided]"}

Project topic: {topic}
Project intent: {topic_intent}
Known hazards so far: {hazards_str}

Concise repair summary (what this project is trying to do overall):
{summary_text or "[no summary text]"}

Recent turns (most recent last):
{recent_block}

Using ONLY this information, produce the two labeled lines described in the system message.
""".strip()

    # -----------------------------
    # Execute model call
    # -----------------------------
    try:
        raw = _llm_chat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.38,   # a bit lower for consistency
            max_tokens=max_tokens or 256,
        ) or ""
    except Exception:
        raw = ""

    full_text = str(raw).strip()

    # If available, strip transcript scaffolding so we only parse the assistant part.
    try:
        assistant_text = _extract_assistant_block(full_text)  # type: ignore[name-defined]
    except Exception:
        assistant_text = full_text

    lines_out = [ln.strip() for ln in assistant_text.splitlines() if ln.strip()]

    def pull(label: str) -> str:
        prefix = f"{label}:"
        for ln in lines_out:
            if ln.lower().startswith(prefix.lower()):
                return ln[len(prefix) :].strip()
        return ""

    next_step_line = pull("Next step")
    followup_line = pull("Follow-up question")

    # -----------------------------
    # Fallbacks if parsing failed or was partial
    # -----------------------------
    if not next_step_line:
        # Very conservative fallback if the model ignored the format
        fallback = (
            "I couldn't determine a precise next step from the current context. "
            "Please briefly describe what you have already done and what you "
            "want to do next (include a photo if helpful), and I will guide "
            "you to the next part of the repair."
        )
        next_step_line = fallback

    # Build final next_step_text: next-step instructions + optional follow-up question
    if followup_line and followup_line.lower() != "none":
        next_step_text = f"{next_step_line}\n\nFollow-up question: {followup_line}"
    else:
        next_step_text = next_step_line

    return {
        "dialog_id": dialog_id,
        "topic": ctx.get("topic"),
        "topic_intent": ctx.get("topic_intent"),
        "summary": ctx.get("summary"),
        "hazards_seen": hazards_seen,
        "next_step_text": next_step_text,
        "raw": raw,
        "_source": "next_step_llm" if raw else "fallback",
    }

# ------------------------------------------------------------
# 1. Project Summary Generator  (longform_pack → summary.json)
# ------------------------------------------------------------

def _extract_assistant_block(raw: str) -> str:
    """
    Qwen sometimes returns a transcript-style blob that includes markers like
    'system', 'user', and 'assistant'. This helper tries to pull out just the
    final assistant message so our downstream logic sees clean content.

    Strategy:
      - If we find a line that is exactly 'assistant' (case-insensitive),
        keep everything *after* the last such line.
      - Otherwise, if we see the token '\\nassistant\\n' anywhere, split on
        the last occurrence and keep the tail.
      - As a last resort, return the original raw string.
    """
    if not raw:
        return ""

    text = str(raw)

    # Case 1: explicit 'assistant' line(s)
    lines = text.splitlines()
    idxs = [i for i, ln in enumerate(lines) if ln.strip().lower() == "assistant"]
    if idxs:
        last = idxs[-1]
        # Everything AFTER the 'assistant' line
        tail = "\n".join(lines[last + 1 :]).strip()
        if tail:
            return tail

    # Case 2: inline marker
    marker = "\nassistant\n"
    if marker in text:
        tail = text.split(marker)[-1].strip()
        if tail:
            return tail

    # Fallback: nothing recognizable, just return as-is
    return text.strip()
    
def _summ_longform_evidence_blocks(longform_pack: Dict[str, Any]) -> Tuple[str, str]:
    """Compact text + web evidence into short bullet blocks for the prompt."""
    text_items = (longform_pack.get("longform") or {}).get("text") or []
    web_items = (longform_pack.get("longform") or {}).get("web") or []

    def _fmt(items, max_n=5):
        lines = []
        for item in items[:max_n]:
            src = item.get("source") or item.get("type") or "text"
            snip = (item.get("snippet") or "").strip()
            if not snip:
                continue
            lines.append(f"- [{src}] {snip}")
        return "\n".join(lines) if lines else "<none>"

    return _fmt(text_items), _fmt(web_items)

def _parse_comma_list(line: str) -> List[str]:
    if not line:
        return []
    # Allow comma or semicolon separated lists
    parts = re.split(r"[;,]", line)
    return [p.strip() for p in parts if p.strip()]

def summarize_longform_llm(
    issue_text: str,
    longform_pack: Dict[str, Any],
    *,
    topic: Optional[str] = None,
    topic_intent: Optional[Dict[str, Any]] = None,
    image_caption: Optional[str] = None,
    safety: Optional[Dict[str, Any]] = None,
    dialog_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Produce a single, detailed repair guide from longform search results.

    INPUT:
      - issue_text: latest user description of the problem
      - longform_pack: packed search results (RAG + web + image). We assume this
        contains the best longform repair writeups for the current topic.

    OUTPUT (stored as summary.json):
      {
        "dialog_id": str,
        "topic": str,
        "text": "<combined, detailed repair guide>",
        "image_caption": str|None,
        "evidence_refs": [...],
        "_source": "summary_longform_v2",
        "_raw": "<raw model output>",
      }

    Notes:
      - We treat the entire model output as a single long text plan.
    """
    topic_label = topic or (topic_intent or {}).get("topic") or "general"
    _ = safety  # currently unused, kept for signature compatibility

    # Build compact evidence blocks from the longform pack.
    try:
        text_block, web_block = _summ_longform_evidence_blocks(longform_pack)
    except Exception:
        # Very simple concatenation fallback if helper ever fails
        text_items = []
        longform = longform_pack.get("longform") or {}
        for item in longform.get("text", []) or []:
            txt = item.get("text") or item.get("snippet")
            if txt:
                text_items.append(str(txt))
        text_block = "\n\n".join(text_items) or "[no local/manual evidence]"

        web_items = []
        for item in longform.get("web", []) or []:
            txt = item.get("text") or item.get("snippet")
            if txt:
                web_items.append(str(txt))
        web_block = "\n\n".join(web_items) or "[no web evidence]"

    # Merge blocks into a single evidence bundle string
    pieces: List[str] = []
    if text_block and text_block.strip() and text_block.strip() != "<none>":
        pieces.append("LOCAL / MANUAL EVIDENCE:\n" + text_block.strip())
    if web_block and web_block.strip() and web_block.strip() != "<none>":
        pieces.append("WEB EVIDENCE:\n" + web_block.strip())
    joined_evidence_passages = "\n\n".join(pieces).strip()
    if not joined_evidence_passages:
        joined_evidence_passages = "[no detailed evidence passages available]"

    # -----------------------------
    # System + user messages
    # -----------------------------
    system_msg = """
You are HomeRepairBot, an AI assistant that creates clear, detailed step-by-step home repair guides.

You will receive:
- a short description of the user's problem
- the repair topic (such as plumbing, drywall, electrical, etc.)
- several evidence passages taken from manuals and trustworthy web articles about ONE type of project

Your task is to write ONE self-contained repair guide the user can follow from start to finish using the provided material.
Use roughly 200 – 450 words in total.

Output requirements:
- Write in plain, friendly, professional language.
- First, write a very detailed between one paragraph to five paragraph description of the repair in plain text from start to finish (no heading).
- It’s good to be very descriptive. Give individual, specific, thoroughly detailed steps that cover the repair from start to finish. Do NOT write multiple distinct steps into any single step. Do keep all steps separate.
- Write a sequence of numbered steps, using the format:
    Step 1: ...
    Step 2: ...
- Do NOT restart the numbering; do NOT repeat the step numbers. Do NOT repeat advice in multiple steps.
- In the steps, focus on tools, materials needed, and hazards that may be encountered (for example: “Use an adjustable wrench to loosen…”, “Wear gloves and be careful of sharp edges…”, "Pipe tape and plumbers putty...").
- Focus ONLY on the current repair topic. Do NOT introduce other projects or off-topic advice.
- Stay grounded in the evidence. Avoid specific product brands, exact dimensions, or building-code requirements that are not supported by the passages.
- Avoid repeating the same sentence or phrase.
- Avoid using markdown headings or section titles such as “Overview”, “Step-by-Step Guide”, or “Final Checks”.
- Avoid talking about “evidence”, “passages”, or yourself. Just write the overview paragraph and all of the numbered steps.
""".strip()

    # User message: DATA ONLY, no instructions.
    issue_line = (issue_text or "").strip() or "Home repair issue."
    user_msg = f"""
Here is an evidence bundle about ONE home repair problem.

User issue description:
{issue_line}

Each item below is a passage from a manual or a trusted web article about the same project. Use them together to write a detailed repair guide that covers the entire project from preparation to final checks.

EVIDENCE PASSAGES:
{joined_evidence_passages}
""".strip()

    # -----------------------------
    # Execute model call
    # -----------------------------
    try:
        raw = _llm_chat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.42,   # slightly higher to avoid “stuck” token patterns
            max_tokens=512,    # keep room for a full guide
        ) or ""
    except Exception:
        raw = ""

    # Strip off any transcript scaffolding so we only keep the actual guide
    full_text = str(raw).strip()
    guide_text = _extract_assistant_block(full_text)

    summary: Dict[str, Any] = {
        "dialog_id": dialog_id,
        "topic": topic_label,
        "image_caption": image_caption,
        "text": guide_text,
        "evidence_refs": longform_pack.get("evidence_refs", []),
        "_source": "summary_longform_v2",
        "_raw": full_text,
    }

    return summary

# ------------------------------------------------------------
# 2. Step Answer Generator (summary.json → actionable step)
# ------------------------------------------------------------

def answer_from_summary_llm(
    summary: Dict[str, Any],
    history: List[Any],
    user_bundle: Dict[str, Any],
    *nargs,
    dialog_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the initial ANSWER block for this dialog turn using the
    canonical project summary.

    INPUT:
      - summary: canonical project summary (from summarize_longform_llm)
      - history: prior Q&A / steps (currently not heavily used here)
      - user_bundle: current turn (text, caption, topic, safety, etc.)

    OUTPUT (merged into synthesis.synthesis and answer.json):
      {
        "title": str,
        "scope_overview": str,
        "first_step": str,
        "potential_causes": [str],
        "tools_required": [str],
        "materials": [str],
        "hazards": [str],
        "next_question": str,
        "answer_preview": str,
        "longform_refs": [str],
        "image_citations": [],
        "_source": "answer_llm_lines",
        "_raw": "<assistant-only model output>",
      }
    """
    # Basic fields from current turn and summary
    issue_text = user_bundle.get("user_text") or user_bundle.get("text") or ""
    topic = summary.get("topic") or user_bundle.get("topic") or "general"
    topic_intent = (
        (summary.get("topic_intent") or {}).get("intent")
        or (user_bundle.get("topic_intent") or {}).get("intent")
        or "unknown"
    )

    guide_text = summary.get("text") or ""
    if not isinstance(guide_text, str):
        guide_text = str(guide_text)

    project_overview = summary.get("project_overview") or ""
    project_hazards = ", ".join(summary.get("project_hazards", []))
    project_tools = ", ".join(summary.get("project_tools", []))
    project_materials = ", ".join(summary.get("project_materials", []))

    # -----------------------------
    # Slim, direct system prompt
    # -----------------------------
    system_msg = """
You are HomeRepairBot, an AI assistant that helps users safely start a home repair project.

You will be given:
- a short description of the user's problem
- the repair topic and intent
- a detailed repair guide that was already generated from trusted sources

Your job is to create a concise STARTING PLAN for the user, using five labeled lines.

Output format (exactly five lines, in this order):

Overview: <2–4 sentences describing what you are going to help the user do and the general approach, from start to finish>
Hazards: <short comma-separated list of the most important safety concerns, or "none">
Tools: <short comma-separated list of tools needed, or "none">
Materials: <short comma-separated list of materials/parts needed, or "none">
First Step: <3–6 sentences giving very concrete, actionable instructions for the very first part of the repair>

Important:
- Use ONLY the information from the repair guide text and the user description.
- Do NOT invent new steps, dimensions, building-code rules, or product details that are not in the guide.
- Keep the focus on what the user should do FIRST to the repair started.

Additional rules:
- Put the entire content for each label on a SINGLE LINE. Do not insert line breaks inside a value.
- For Hazards, include only the key risks (for example: “live electric, fire, flooding, sharp edges”) or "none".
- For Tools and Materials, list what is needed to complete the entire project.
- In the First Step, focus on preparation and initial inspection: making the area safe, turning off utilities, protecting the space, and doing the first checks. Do NOT describe later steps such as replacing parts, full reassembly, or final testing.
- In the First Step, it is OK to end by asking the user to report back what they see so you can guide the next step.
- Do NOT add any extra lines, headings, bullet lists, or labels beyond the five specified.
- Do NOT talk about “the guide”, “the passages”, or yourself. Just write the five labeled lines.
""".strip()
    # -----------------------------
    # Minimal, data-only user message
    # -----------------------------
    user_msg = f"""
User problem description:
{issue_text or "[not provided]"}

Project topic: {topic}
Project intent: {topic_intent}

Here is the repair guide you MUST base your answer on:
=== REPAIR GUIDE START ===
{guide_text or "[no guide text]"}
=== REPAIR GUIDE END ===
Using this information, produce the five labeled lines described in the system message.
""".strip()

    # -----------------------------
    # Execute model call
    # -----------------------------
    try:
        raw = _llm_chat(
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.41,   # good balance: not too flat, not too wild
            max_tokens=448,    # enough for 5 dense lines, but less room to ramble
        ) or ""
    except Exception:
        raw = ""

    full_text = str(raw).strip()

    # If available, strip transcript scaffolding so we only parse the assistant part.
    try:
        assistant_text = _extract_assistant_block(full_text)  # type: ignore[name-defined]
    except Exception:
        assistant_text = full_text

    lines_out = [ln.strip() for ln in assistant_text.splitlines() if ln.strip()]

    def pull(label: str) -> str:
        prefix = f"{label}:"
        for ln in lines_out:
            if ln.lower().startswith(prefix.lower()):
                return ln[len(prefix) :].strip()
        return ""

    # We only asked for Overview/Hazards/Tools/Materials/First Step
    # but we also gracefully support "Scope Overview" if the model echoes older prompts.
    scope_overview = pull("Scope Overview")
    if not scope_overview:
        scope_overview = pull("Overview")

    first_step = pull("First Step")
    tools_line = pull("Tools")
    materials_line = pull("Materials")
    hazards_line = pull("Hazards")

    # Legacy / optional labels – may remain empty and be filled by fallbacks
    title = pull("Title")
    next_question = pull("Next Question")
    answer_preview = pull("Answer Preview")
    longform_refs_line = pull("LongformRefs")

    tools = _parse_comma_list(tools_line)
    materials = _parse_comma_list(materials_line)
    hazards = _parse_comma_list(hazards_line)
    longform_refs = _parse_comma_list(longform_refs_line)

    # If the model left some lists empty, fall back to summary-level metadata.
    if not hazards and project_hazards:
        hazards = _parse_comma_list(project_hazards)

    if not tools and project_tools:
        tools = _parse_comma_list(project_tools)[:4]

    if not materials and project_materials:
        materials = _parse_comma_list(project_materials)[:4]

    # -----------------------------
    # Fallbacks if parsing failed or was partial
    # -----------------------------
    if not title:
        title = issue_text or f"Start the {topic} project"

    if not scope_overview:
        scope_overview = (
            project_overview
            or "We will begin this repair with a safe, simple starting step and then continue based on what you observe."
        )

    if not first_step:
        lower_text = f"{issue_text} {project_overview}".lower()
        if "faucet" in lower_text or ("sink" in lower_text and "leak" in lower_text):
            first_step = (
                "Turn off the hot and cold shutoff valves under the sink, then open "
                "the faucet to relieve pressure and see exactly where the water is leaking."
            )
        else:
            first_step = (
                "Make the area safe and ready: turn off any relevant utilities "
                "(such as power, water, or gas) and clear and protect the work area "
                "so you can see and reach the problem clearly."
            )

    if not next_question:
        next_question = (
            "After you complete this step, tell me what you observed so I can guide you to the next part of the repair."
        )

    if not answer_preview:
        answer_preview = f"{scope_overview}\n\nFirst step: {first_step}"

    sections: Dict[str, Any] = {
        "title": title,
        "scope_overview": scope_overview,
        "first_step": first_step,
        "potential_causes": [],   # optional; can be expanded later
        "tools_required": tools,
        "materials": materials,
        "hazards": hazards,
        "next_question": next_question,
        "answer_preview": answer_preview,
        "longform_refs": longform_refs,
        "image_citations": [],
        "_source": "answer_llm_lines",
        "_raw": assistant_text,
    }

    return sections

