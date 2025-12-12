# app/models/session_class.py
"""
SESSION DATA MODELS & EVENT SCHEMA

This module centralizes the data shapes used by the session layer:

Disk artifacts under data/sessions/<dialog_id>/:

  1) chat_info.json        -> SessionState
  2) events/<timestamp>.json
                           -> EventRecord (immutable per-turn history)
  3) events/latest.json    -> LatestEvent (overlay of last EventRecord)
  4) events/answer.json    -> AnswerBlock (lightweight "current answer" view)
  5) memory.json           -> SessionMemory (rolling multi-turn summary)

Derived in-memory context (not persisted):

  - DialogueContext        -> output of utils.session_utils.build_dialogue_context()

These models are primarily documentation / type hints. The service layer and
API can progressively adopt them where it is convenient, but existing dict-based
code does not depend on them yet.

FILE SCHEMA OVERVIEW
--------------------

1) chat_info.json  (SessionState)
   {
     "dialog_id": "abc123",
     "created_at": "2025-11-30T17:18:00Z",
     "ack_stage": 0,
     "progress": [...],
     "citations": [...],
     "last_query": "How do I fix a leaky faucet?",
     "last_snippets": ["..."],
     "context_used_chars": 1234,
     "context_budget_chars": 64000
   }

2) events/<timestamp>.json  (EventRecord)
   {
     "timestamp_utc": "2025-11-30T17:19:42Z",
     "dialog_id": "abc123",
     "query": {
       "text": "How do I fix a leaky faucet?",
       "has_image": true,
       "caption": "Close-up of a faucet dripping under the sink"
     },
     "notes": {...},                # internal diagnostics, counts, phases
     "warnings": ["..."],
     "counts": {...},
     "source_counts": {
       "image": 1,
       "text": 4,
       "web": 2,
       "total": 7
     },
     "timings_ms": {...},
     "answer": "Short text answer for this turn"  # or a structured dict
     "evidence": [...],            # compact evidence bundle
     "results": [...],             # fused RAG + web + image hits
     "blocked": false,
     "requires_ack": false,
     "ack_stage": 0,

     # Optional, added later by synthesis / fusion:
     "synthesis": {...},
     "orchestration": {...},
     "safety": {...},
     "intent_decision": {...}
   }

3) events/latest.json        (LatestEvent)
   - Same shape as EventRecord, but with the last event's payload and any
     overlays from search/synthesis.
   - The "answer" field is normalized by persist_latest_event() into a dict:

     "answer": {
       "dialog_id": "abc123",
       "title": "How do I fix a leaky faucet?",
       "text": "<final answer text>",
       "final_answer": "<final answer text>"
     }

4) events/answer.json        (AnswerBlock)
   {
     "dialog_id": "abc123",
     "title": "How do I fix a leaky faucet?",
     "text": "<final answer text>",
     "final_answer": "<final answer text>"
   }

5) memory.json               (SessionMemory)
   {
     "dialog_id": "abc123",
     "updated_at": "2025-11-30T17:20:10Z",
     "last_ack_stage": 1,
     "hazards_seen": ["live_electric", "water_damage"],
     "key_points": [
       "Shut off water to the faucet before disassembling.",
       "Do not work on wet circuits or energized outlets."
     ],
     "summary": "Short, multi-sentence recap of the dialog so far.",
     "source": {
       "recent_events_used": 8
     }
   }

Derived context (DialogueContext; in-memory only):

   {
     "dialog_id": "abc123",
     "topic": "plumbing",
     "topic_intent": {...},
     "summary": "Latest short memory summary string...",
     "hazards_seen": ["live_electric", "water_damage"],
     "recent_turns": [
       {
         "timestamp_utc": "2025-11-30T17:19:42Z",
         "user_text": "How do I fix a leaky faucet?",
         "caption": "Close-up of the dripping faucet",
         "has_image": true,
         "answer_brief": "Start by shutting off the water...",
         "first_step": "Locate the shutoff valves under the sink."
       },
       ...
     ]
   }
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# SessionState (chat_info.json)
# ---------------------------------------------------------------------------

class SessionState(BaseModel):
    """
    Rolling per-dialog state stored in chat_info.json.

    Primarily maintained by app/services/sessions.py via load_session() /
    save_session(). The fields here mirror the default seed in load_session().
    """
    dialog_id: str
    created_at: str

    # Acknowledgement stage for safety gating
    ack_stage: int = 0

    # Bread-crumbs of the backend pipeline; useful for debugging.
    progress: List[Dict[str, Any]] = Field(default_factory=list)

    # Aggregated citations from RAG / web / image results.
    citations: List[Dict[str, Any]] = Field(default_factory=list)

    # Last seen user query text, and context snippets fed to the LLM.
    last_query: str = ""
    last_snippets: List[str] = Field(default_factory=list)

    # Approximate character budget for rolling context (50% of 32k tokens).
    context_used_chars: int = 0
    context_budget_chars: int = 64000


# ---------------------------------------------------------------------------
# Event and answer views
# ---------------------------------------------------------------------------

class QuerySnapshot(BaseModel):
    """
    Query info attached to each event.

    - text:      The raw user query (possibly None for image-only turns).
    - has_image: True if the turn included an image.
    - caption:   Caption produced by the VLM for the image, if any.
    """
    text: Optional[str] = None
    has_image: bool = False
    caption: Optional[str] = None


class AnswerBlock(BaseModel):
    """
    Normalized answer view exposed in events/latest.json and events/answer.json.

    This is the output of persist_latest_event(). Legacy code may sometimes
    store a string answer in EventRecord.answer; new code should prefer this
    structured form.
    """
    dialog_id: str
    title: str = ""
    text: str = ""
    final_answer: str = ""


class EventRecord(BaseModel):
    """
    Immutable per-turn event as written to events/<timestamp>.json.

    Fields:
      - timestamp_utc: ISO-8601 UTC timestamp when the event was recorded.
      - dialog_id:     Session identifier.
      - query:         QuerySnapshot describing this turn's user input.
      - notes:         Internal diagnostics / pipeline notes (arbitrary dict).
      - warnings:      Human-readable warnings (strings) for this turn.
      - counts:        Per-stage result counts (RAG, web, image, etc.).
      - source_counts: Per-source result counts (image/text/web/total).
      - timings_ms:    Timing info for major pipeline stages.
      - answer:        Either a raw string answer (legacy) or an AnswerBlock.
      - evidence:      Compact evidence bundle threaded to synthesis.
      - results:       Fused search results (text, web, image, etc.).
      - blocked:       True if the turn was blocked by safety.
      - requires_ack:  True if this turn requires user acknowledgement.
      - ack_stage:     0 (none), 1 or 2 depending on safety gating.

    Optional overlays (added by synthesis / orchestration):
      - synthesis:     dict with scope_overview, first_step, tools, etc.
      - orchestration: dict with topic, topic_intent, and routing decisions.
      - safety:        SafetyReport-like structure from safety service.
      - intent_decision:
                       Intent decision block from intent routing.
    """
    timestamp_utc: str
    dialog_id: str

    query: QuerySnapshot
    notes: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    counts: Dict[str, Any] = Field(default_factory=dict)
    source_counts: Dict[str, int] = Field(default_factory=dict)
    timings_ms: Dict[str, int] = Field(default_factory=dict)

    # "answer" may still be string in raw event history, or an AnswerBlock
    answer: AnswerBlock | str | None = None

    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    results: List[Dict[str, Any]] = Field(default_factory=list)

    blocked: bool = False
    requires_ack: bool = False
    ack_stage: int = 0

    # Overlays
    synthesis: Optional[Dict[str, Any]] = None
    orchestration: Optional[Dict[str, Any]] = None
    safety: Optional[Dict[str, Any]] = None
    intent_decision: Optional[Dict[str, Any]] = None

    # Allow extra keys if we add fields later without breaking clients.
    model_config = {"extra": "allow"}


class LatestEvent(EventRecord):
    """
    LatestEvent is the same shape as EventRecord but persisted to
    events/latest.json and updated by:

      - record_session_event() via persist_latest_event()
      - synthesis.generate() when it enriches the last event.

    The "answer" field is normalized to an AnswerBlock by persist_latest_event().
    """
    # No extra fields; this subclass is primarily a semantic alias.
    pass


# ---------------------------------------------------------------------------
# Memory summary (memory.json)
# ---------------------------------------------------------------------------

class MemorySourceMeta(BaseModel):
    """
    Metadata attached to memory.json to describe how it was built.

    - recent_events_used: number of events actually considered (capped by
                          build_memory(..., recent_limit)).
    """
    recent_events_used: int = 0


class SessionMemory(BaseModel):
    """
    Rolling multi-turn summary stored in memory.json.

    Built and maintained by app/services/sessions.build_memory().
    """
    dialog_id: str
    updated_at: str

    # Last known acknowledgement stage from the event history.
    last_ack_stage: Optional[int] = None

    # Unique hazard tags encountered across the dialog.
    hazards_seen: List[str] = Field(default_factory=list)

    # Short bullet-like phrases extracted from recent answers.
    key_points: List[str] = Field(default_factory=list)

    # Multi-sentence, human-readable recap suitable for feeding into the LLM.
    summary: str

    source: MemorySourceMeta


# ---------------------------------------------------------------------------
# Derived dialogue context (build_dialogue_context)
# ---------------------------------------------------------------------------

class RecentTurn(BaseModel):
    """
    Lightweight representation of a recent turn used by the LLM helper
    build_dialogue_context() in app/utils/session_utils.py.
    """
    timestamp_utc: Optional[str] = None
    user_text: Optional[str] = None
    caption: Optional[str] = None
    has_image: bool = False
    answer_brief: Optional[str] = None
    first_step: Optional[str] = None


class DialogueContext(BaseModel):
    """
    Multi-turn context snapshot used to build follow-up LLM prompts.

    Not stored on disk; built on the fly by build_dialogue_context(dialog_id).
    """
    dialog_id: str

    topic: Optional[str] = None
    topic_intent: Optional[Dict[str, Any]] = None

    summary: str = ""
    hazards_seen: List[str] = Field(default_factory=list)

    recent_turns: List[RecentTurn] = Field(default_factory=list)
