# app/services/safety.py

"""
SAFETY SERVICE LAYER

Purpose:
- Post-process RAG/search results for safety-sensitive topics and attach warnings.
- Lightweight keyword/regex rules for GAS, APPLIANCE, ELECTRICAL, STRUCTURAL, etc.
- Non-blocking in general: we rarely hide results; we annotate with flags + advisories.

How it's used:
- analyze_text_blocks(blocks) → compact SafetyReport for UI/logs.
- safety_payload(snippets, query_text) → structured safety bundle for fusion + gating:
    - emergency (detected/message/signals)
    - work_types (for staged warnings)
    - staged (stage 1 / stage 2 text)
    - report (warnings + counts)
- decide_intent(safety, ack_stage) → "proceed"/"stage_1_required"/"stage_2_required"/"emergency_override"
- add_staged_messages_text(intent) → back-compat helper for flat text UIs.
"""

from __future__ import annotations

from typing import List, Dict, Tuple, Any, Optional
import re
import json

from app.models.safety_class import SafetyWarning, SafetyReport


# ============================================================================
# 1) CATEGORY-LEVEL ADVISORY (used by analyze_text_blocks)
# ============================================================================

CATEGORY_RULES: Dict[str, Dict[str, List[str]]] = {
    "electrical": {
        "stop": [
            r"\bexposed (lugs|bus|conductors?)\b",
            r"\b(main )?service panel\b.*\bopen\b",
            r"\b(live|energized)\s+(wire|conductor)\b",
        ],
        "caution": [
            r"\boutlet\b.*\bscorch(ed)?\b",
            r"\bgfci\b.*\btrip(ping)?\b",
            r"\btripp?ing breaker\b",
        ],
        "info": [r"\bturn off power\b", r"\btest with (meter|tester)\b"],
    },
    "gas": {
        "stop": [
            r"\bstrong\s+gas\s+odor\b",
            r"\bsmell(s|ed)?\s+gas\b",
            r"\bcarbon monoxide\b|\bCO\s+alarm\b",
        ],
        "caution": [
            r"\bgas\s+(line|pipe|valve|meter)\b",
            r"\bpilot (light)?\b",
        ],
        "info": [
            r"\bshut\s*off\s+gas\b",
            r"\bcall\s+(utility|gas company)\b",
        ],
    },
    "appliance": {
        "stop": [
            r"\bscorch(ed)?\b.*\b(outlet|plug)\b",
            r"\bmelting\b.*\b(insulation|cord)\b",
        ],
        "caution": [
            r"\bwater\b.*\binside\b.*\bpanel\b",
            r"\boverheating\b",
        ],
        "info": [
            r"\bunplug\b.*\b(appliance|device)\b",
        ],
    },
    "fire_smoke": {
        "stop": [
            r"\b(active )?fire\b",
            r"\bvisible smoke\b",
            r"\bsmoke\b.*\bin (house|home|room)\b",
        ],
        "caution": [r"\bscorch(ed)?\b", r"\bsoot\b"],
        "info": [r"\bsmoke (alarm|detector)\b"],
    },
    "water_structural": {
        "stop": [
            r"\b(load[- ]bearing|structural)\b.*\b(remove|cut)\b",
            r"\broof\b.*\b(sag|collapse)\b",
        ],
        "caution": [
            r"\bleak(ing)?\b",
            r"\bactive leak\b",
            r"\bdrywall\b.*\bsoft\b",
            r"\bsubfloor\b.*\brot\b",
        ],
        "info": [r"\bmoisture\b", r"\bdrip\b"],
    },
}

LEVEL_ORDER = ["info", "caution", "stop"]
LEVEL_SCORE = {"info": 1, "caution": 2, "stop": 3}


def _scan_text_for_category(text: str, patterns: Dict[str, List[str]]) -> Tuple[str, List[str]]:
    best_level, hits = "", []
    for level in LEVEL_ORDER:
        for pat in patterns.get(level, []):
            if re.search(pat, text, flags=re.IGNORECASE):
                hits.append(pat)
                if LEVEL_SCORE[level] > LEVEL_SCORE.get(best_level, 0):
                    best_level = level
    return best_level, hits


def analyze_text_blocks(blocks: List[str]) -> SafetyReport:
    """
    Build a compact category-level SafetyReport across text blocks.

    Used by:
      - RAG `/api/v1/rag/search` over result snippets.
      - Any caller that wants a high-level advisory summary.

    NOTE:
    - This function is advisory only. Hard gating (emergency / staged ACK) is
      handled by safety_payload + decide_intent.
    """
    joined = "\n".join([b for b in blocks if b])[:200_000]

    warnings: List[SafetyWarning] = []
    for category, patterns in CATEGORY_RULES.items():
        level, hits = _scan_text_for_category(joined, patterns)
        if level:
            reason = {
                "stop": "Multiple high-risk indicators detected.",
                "caution": "Potentially hazardous condition detected.",
                "info": "General safety consideration applicable.",
            }[level]
            warnings.append(
                SafetyWarning(
                    category=category.replace("_", "/"),
                    level=level,
                    reason=reason,
                    signals=hits[:5],
                )
            )

    # Simple compounding: gas + fire → stop for both
    cats = {w.category for w in warnings}
    if "gas" in cats and "fire/smoke" in cats:
        for w in warnings:
            if w.category in ("gas", "fire/smoke"):
                w.level = "stop"
                w.reason = "Compound risk: gas + fire/smoke indicators."

    blocked = any(w.level == "stop" for w in warnings)
    advisory: List[str] = [f"{w.category}: {w.reason}" for w in warnings]

    notes = ""
    if warnings:
        levels = sorted(
            {w.level for w in warnings},
            key=lambda lvl: LEVEL_SCORE.get(lvl, 0),
            reverse=True,
        )
        notes = (
            f"Detected safety-relevant categories: {', '.join(sorted(cats))}. "
            f"Highest level: {levels[0]}."
        )

    return SafetyReport(
        blocked=blocked,
        requires_ack=False,   # fusion/intent own staged gating
        ack_stage=0,
        advisory=advisory,
        per_item_flags=[],
        notes=notes,
        warnings=warnings,
    )


# ============================================================================
# 2) STAGED WARNINGS + EMERGENCY (used for gating)
# ============================================================================

# "Work type" detection for staged messages
STAGE_TYPES: Dict[str, List[str]] = {
    "Gas line work": [
        r"\bgas\s+(line|pipe|valve|fitting|meter)\b",
        r"\bgas\s+leak\b",
        r"\bshut\s*off\s+gas\b",
    ],
    "Electrical (beyond low-risk)": [
        r"\b(service|breaker)\s+panel\b",
        r"\blive\s+wire\b",
        r"\bwiring\b",
        r"\bjunction\s+box\b",
        r"\b240\s*v\b",
        r"\bline\s+voltage\b",
    ],
    "Hazardous materials": [
        r"\bchemical spill\b",
        r"\basbestos\b",
        r"\blead paint\b",
        r"\bpcbs?\b",
    ],
    "Appliance service": [
        r"\b(water heater|furnace|boiler|dryer|oven|range|dishwasher|refrigerator)\b",
        r"\b(igniter|thermocouple|pilot|flue|vent|combustion)\b",
    ],
}


def _find_types(joined: str) -> List[str]:
    types = []
    for t, pats in STAGE_TYPES.items():
        if any(re.search(p, joined, flags=re.IGNORECASE) for p in pats):
            types.append(t)
    return types


def build_staged_messages(types: List[str]) -> List[Dict[str, str]]:
    """
    Stage 1 then Stage 2 messages. UI can show them sequentially as the
    user acknowledges each stage.
    """
    if not types:
        return []
    kind = ", ".join(types)
    return [
        {
            "stage": "1",
            "text": (
                f"Warning: This repair involves {kind}. An AI assistant is not "
                f"qualified to help this type of repair. Please contact a "
                f"professional in your area."
            ),
        },
        {
            "stage": "2",
            "text": (
                "As an AI repair assistant, I am not qualified to help you with "
                "this repair. Please contact a professional. If you continue to "
                "seek my help to work on this repair, you are agreeing not to "
                "hold DIY-AI HomeRepairBot, or its creators, responsible for any "
                "additional issues that may arise."
            ),
        },
    ]


# ----- Emergency detection (centralized) -----------------------------------

# We keep emergency detection separate and *conservative*: it should only
# fire for clear, "call 911" scenarios. Everything else becomes a hazard
# handled via staged warnings.

FIRE_EMERGENCY_PATTERNS = [
    r"\b(on\s+fire)\b",
    r"\bhouse\s+fire\b",
    r"\bkitchen\s+is\s+on\s+fire\b",
    r"\bvisible\s+flames?\b",
    r"\bflames?\s+(coming\s+from|in)\b",
    r"\bsmoke\b.*\b(cabinet|ceiling|wall|room|house|home)\b",
    r"\babout\s+to\s+explode\b",
]

GAS_EMERGENCY_PATTERNS = [
    r"\bsmell\s+gas\b",
    r"\bsmelling\s+gas\b",
    r"\bstrong\s+gas\s+(smell|odor|odour)\b",
    r"\bstrong\s+(propane|natural\s+gas)\s+(smell|odor|odour)\b",
    r"\bgas\s+leak\b",
    r"\bleaking\s+gas\b",
]

STRUCTURAL_EMERGENCY_PATTERNS = [
    r"\bceiling\s+(collapsed|caving\s+in)\b",
    r"\bwall\s+(collapsed|caving\s+in)\b",
    r"\broof\b.*\bcollapse\b",
    r"\b(load[-\s]?bearing)\b.*\bfailed\b",
    r"\bcarbon\s+monoxide\b",
]


def _detect_emergency(text: str) -> Dict[str, Any]:
    """
    Detect *active* emergencies from the user's text.

    - Fires for: active fire, strong gas smell/leak, structural collapse,
      carbon monoxide alarms, etc.
    - Does NOT fire for: planned work like "replace gas line" with
      explicit "no leak" / "no gas leak" context.
    """
    blob = (text or "").lower().strip()
    if not blob:
        return {"detected": False, "message": "", "signals": []}

    def any_match(patterns: List[str]) -> bool:
        return any(re.search(p, blob) for p in patterns)

    # Fire / flames / heavy smoke
    fire = any_match(FIRE_EMERGENCY_PATTERNS)

    # Gas: must imply active leak/smell, not just "gas line" or "gas valve"
    has_gas = any(x in blob for x in ["gas", "propane", "natural gas"])
    # Negating phrases we *don't* want to treat as emergencies
    negating = ("no gas leak" in blob) or ("no leak" in blob and has_gas)

    gas_phrase = any_match(GAS_EMERGENCY_PATTERNS) and not negating

    gas_context_words = [
        "smell",
        "smells",
        "odor",
        "odour",
        "leak",
        "leaking",
        "hissing",
        "strong",
        "fumes",
    ]
    gas_context = (
        has_gas
        and any(w in blob for w in gas_context_words)
        and not negating
    )

    gas = gas_phrase or gas_context

    # Structural / CO emergencies
    structural = any_match(STRUCTURAL_EMERGENCY_PATTERNS)

    detected = fire or gas or structural
    if not detected:
        return {"detected": False, "message": "", "signals": []}

    signals: List[str] = []
    if fire:
        signals.append("fire")
    if gas:
        signals.append("gas_leak_or_smell")
    if structural:
        signals.append("structural_failure_or_CO")

    message = (
        "Emergency detected! Please dial 911 or contact local/regional emergency "
        "services immediately!"
    )
    return {"detected": True, "message": message, "signals": signals[:5]}


# ============================================================================
# 3) HAZARD LEXICON FOR RAG SNIPPETS (used inside safety_payload)
# ============================================================================

_STOP_PATTERNS: Dict[str, List[re.Pattern]] = {
    "gas": [
        re.compile(r"\brotten\s+egg\b", re.I),
        re.compile(r"\b(?:gas|propane|natural\s+gas)\s+(?:leak|smell|odor)\b", re.I),
        re.compile(r"\bcarbon\s+monoxide\b|\bCO\s+alarm\b", re.I),
    ],
    "electrical": [
        re.compile(r"\bsparking\b|\barcing\b", re.I),
        re.compile(r"\bburning\s+smell\b", re.I),
        re.compile(r"\bexposed\s+(?:hot\s+)?wire\b", re.I),
        re.compile(r"\b(outlet|receptacle)\s+(is\s+)?(?:hot|smoking|buzzing)\b", re.I),
        re.compile(r"\bbreaker\s+(keeps\s+)?tripp?ing\b", re.I),
        re.compile(r"\blive\s+wire\b", re.I),
    ],
    "structural": [
        re.compile(r"\bsagging\s+(roof|ceiling|floor)\b", re.I),
        re.compile(r"\b(load[-\s]?bearing)\b", re.I),
        re.compile(r"\bmajor\s+foundation\s+crack\b|\bbowed\s+wall\b", re.I),
        re.compile(r"\bceil(ing)?\s+collapse\b", re.I),
    ],
    "asbestos": [
        re.compile(r"\basbestos\b", re.I),
        re.compile(r"\bvermiculite\b", re.I),
        re.compile(r"\b(popcorn|acoustic)\s+ceiling\b.*\b(?:pre|before)\s*-?\s*1980", re.I),
    ],
    "lead": [
        re.compile(r"\blead\s+paint\b", re.I),
        re.compile(r"\b(pre|before)\s*-?\s*1978\b.*\bpaint\b", re.I),
    ],
}

_CAUTION_PATTERNS: Dict[str, List[re.Pattern]] = {
    "mold": [
        re.compile(r"\bblack\s+mold\b", re.I),
        re.compile(r"\bvisible\s+mold\b|\bmusty\b|\bwater\s+damage\b", re.I),
    ],
    "chemicals": [
        re.compile(r"\bsolvent\b|\bstripper\b|\bpaint\s+fumes\b|\bVOC(s)?\b", re.I),
    ],
    "ladder": [
        re.compile(r"\bladder\b", re.I),
        re.compile(r"\bwork\s+at\s+height\b|\broof\s+edge\b", re.I),
    ],
    "dust": [
        re.compile(r"\b(?:silica|drywall)\s+dust\b", re.I),
        re.compile(r"\bsanding\b.*\bmask\b", re.I),
    ],
}

_SIGNATURE_MESSAGES: Dict[str, Dict[str, str]] = {
    "gas": {
        "stop": "Possible gas/CO hazard. Evacuate, ventilate if safe, and contact your utility or emergency services.",
    },
    "electrical": {
        "stop": "Electrical hazard. De-energize the circuit at the breaker and call a qualified electrician.",
    },
    "structural": {
        "stop": "Possible structural instability. Avoid load and consult a licensed contractor/engineer.",
    },
    "asbestos": {
        "stop": "Potential asbestos. Do not disturb material; seek certified abatement guidance.",
    },
    "lead": {
        "caution": "Lead paint risk. Use EPA RRP-safe methods; avoid dry sanding; contain dust.",
    },
    "mold": {
        "caution": "Mold/moisture issue. Wear PPE (N95/gloves), contain area, and fix moisture source first.",
    },
    "chemicals": {
        "caution": "Chemical exposure risk. Ensure ventilation, use PPE, and follow product SDS.",
    },
    "ladder": {
        "caution": "Fall risk. Use proper ladder angle, stable footing, and spotter when possible.",
    },
    "dust": {
        "caution": "Dust inhalation risk. Use wet methods or dust extraction and appropriate respirator.",
    },
}


def _scan_texts(texts: List[str], q: Optional[str]) -> List[Dict[str, Any]]:
    warnings: List[Dict[str, Any]] = []

    def _emit(level: str, category: str, msg: str, evidence: str) -> None:
        warnings.append(
            {
                "level": level,
                "category": category,
                "message": msg,
                "evidence": evidence[:240],
            }
        )

    # Aggregate list to scan: query first (highest priority), then snippets
    scan: List[str] = []
    if q:
        scan.append(q)
    scan.extend([t for t in (texts or []) if t])

    # STOP first
    for cat, pats in _STOP_PATTERNS.items():
        for t in scan:
            for rx in pats:
                m = rx.search(t)
                if m:
                    _emit(
                        "stop",
                        cat,
                        _SIGNATURE_MESSAGES.get(cat, {}).get(
                            "stop", "Immediate hazard detected."
                        ),
                        m.group(0),
                    )

    # CAUTION next
    for cat, pats in _CAUTION_PATTERNS.items():
        for t in scan:
            for rx in pats:
                m = rx.search(t)
                if m:
                    _emit(
                        "caution",
                        cat,
                        _SIGNATURE_MESSAGES.get(cat, {}).get(
                            "caution", "Use extra care and PPE."
                        ),
                        m.group(0),
                    )
    return warnings


def safety_payload(snippets: List[str], query_text: Optional[str] = None) -> Dict[str, Any]:
    """
    Rule-based safety tagging from user query + fused snippets.

    Returns:
      {
        "emergency": {detected, message, signals},
        "work_types": [...],
        "staged": [...],
        "report": {
          "warnings": [...],
          "counts": {"stop": n, "caution": m, "total": k}
        }
      }
    """
    try:
        warns = _scan_texts(snippets or [], query_text)
        joined = " ".join([query_text or ""] + [t for t in (snippets or []) if t])[:200_000]

        # Emergency + staged detection
        work_types = _find_types(joined)
        # IMPORTANT: run emergency detection only on *user query*, not on fused
        # snippets, so generic safety disclaimers in search results don't cause
        # full emergency overrides.
        emergency_source = query_text or ""
        emergency = _detect_emergency(emergency_source)
        staged = build_staged_messages(work_types)

        report = {
            "warnings": warns,
            "counts": {
                "stop": sum(1 for w in warns if w.get("level") == "stop"),
                "caution": sum(1 for w in warns if w.get("level") == "caution"),
                "total": len(warns),
            },
        }
        return {
            "emergency": emergency,
            "work_types": work_types,
            "staged": staged,
            "report": report,
        }
    except Exception:
        return {
            "emergency": {"detected": False, "message": "", "signals": []},
            "work_types": [],
            "staged": [],
            "report": {
                "warnings": [],
                "counts": {"stop": 0, "caution": 0, "total": 0},
            },
        }


# ============================================================================
# 4) BACK-COMPAT SHIMS
# ============================================================================

def analyze_snippets(snippets: List[str]) -> SafetyReport:
    """
    Backward-compatible alias used by legacy modules.
    """
    return analyze_text_blocks(snippets)


def analyze_snippets_json(snippets: List[str]) -> Dict[str, Any]:
    """
    JSON-friendly variant if any caller expects a plain dict.
    """
    return analyze_text_blocks(snippets).model_dump()


# ============================================================================
# 5) INTENT DECISION + UI HELPERS
# ============================================================================

def decide_intent(
    safety: Dict[str, Any],
    ack_stage: int,
) -> Dict[str, Any]:
    """
    Determine how the system should proceed based on:
        - safety: output of safety_payload(...)
        - ack_stage: user's current acknowledgement stage:
            0 = no ack,
            1 = stage 1 acknowledged,
            2+ = stage 2 acknowledged

    Decision values:
        - "emergency_override" -> hard stop, show emergency message only
        - "stage_1_required"   -> show staged[0] and stop (until ack)
        - "stage_2_required"   -> show staged[1] and stop (until ack)
        - "proceed"            -> okay to show fused guidance/results
    """
    safety = safety or {}
    try:
        stage_val = int(ack_stage)
    except Exception:
        stage_val = 0

    emergency = safety.get("emergency") or {}
    work_types = safety.get("work_types") or []
    staged = safety.get("staged") or []

    # 1) Emergency override
    if emergency.get("detected"):
        msg = emergency.get("message") or (
            "Emergency detected! Please contact local emergency services immediately."
        )
        signals = emergency.get("signals") or []
        return {
            "decision": "emergency_override",
            "message": msg,
            "signals": signals,
            "require_ack_stage": None,
            "work_types": work_types,
        }

    # 2) Stage-gated warnings for pro-level categories
    if work_types:
        if stage_val < 1:
            return {
                "decision": "stage_1_required",
                "message": staged[0]["text"] if staged else "",
                "require_ack_stage": 1,
                "work_types": work_types,
            }
        if stage_val < 2:
            return {
                "decision": "stage_2_required",
                "message": staged[1]["text"] if len(staged) > 1 else "",
                "require_ack_stage": 2,
                "work_types": work_types,
            }

    # 3) Otherwise proceed
    return {
        "decision": "proceed",
        "message": "",
        "signals": [],
        "require_ack_stage": None,
        "work_types": work_types,
    }


def add_staged_messages_text(intent: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure intent['staged_messages_text'] is a list[str] even if
    intent['staged_messages'] contains dicts like {'stage': '1','text':'...'}.

    This is a UI/back-compat helper for clients that only understand a flat
    list of strings, while the safety/intent services keep richer structured
    staged messages.

    Never raises; falls back to intent['staged_messages_text'] = [] on error.
    """
    try:
        msgs = intent.get("staged_messages", None)
        if isinstance(msgs, list):
            out: List[str] = []
            for m in msgs:
                if isinstance(m, str):
                    out.append(m)
                elif isinstance(m, dict):
                    txt = (
                        m.get("text")
                        or m.get("message")
                        or m.get("desc")
                    )
                    out.append(
                        txt
                        if isinstance(txt, str)
                        else json.dumps(m, ensure_ascii=False)
                    )
                else:
                    out.append(str(m))
            intent["staged_messages_text"] = out
    except Exception:
        intent["staged_messages_text"] = []
    return intent
