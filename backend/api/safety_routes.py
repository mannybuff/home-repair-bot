# app.api.safety_routes.py

"""
SAFETY ROUTER

Thin API layer exposing the safety service:

    - Accepts a user query and/or text snippets.
    - Runs keyword/regex based safety analysis over the snippets.
    - Returns:
        * A structured SafetyReport
        * The staged/emergency payload used by fusion + intent gating

This endpoint is intentionally simple:
    - No session writes.
    - No VLM/LLM calls.
    - Same core logic used by fusion_search().
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.safety import analyze_text_blocks, safety_payload
from app.models.safety_class import SafetyReport

router = APIRouter(
    prefix="/api/v1/safety",
    tags=["safety"],
)


# ---------------------------------------------------------------------------
# REQUEST / RESPONSE MODELS
# ---------------------------------------------------------------------------

class SafetyRequest(BaseModel):
    """
    Request body for safety analysis.

    query_text:
        Optional user query or description. This is passed into safety_payload()
        so emergency detection can consider the query text itself in addition to
        the snippets.
    snippets:
        Text blocks (e.g. fused RAG/Web snippets) to analyze.
    """
    query_text: Optional[str] = None
    snippets: List[str] = []


class SafetyResponse(BaseModel):
    """
    High-level safety analysis response.

    ok:
        Indicates the analysis completed successfully.
    query_text, snippets:
        Echo back the inputs for tracing/debug.
    report:
        Structured SafetyReport (blocked, advisory, warnings, etc.).
    payload:
        Dict containing staged messages, emergency info, and raw warning counts,
        matching what fusion_search() uses internally.
    """
    ok: bool
    query_text: Optional[str]
    snippets: List[str]
    report: SafetyReport
    payload: Dict[str, Any]


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------

@router.post("/analyze", response_model=SafetyResponse)
def analyze_safety(req: SafetyRequest) -> SafetyResponse:
    """
    Run safety analysis over provided snippets (and optional query_text).

    This is the same logic used inside fusion_search(), but exposed as a
    standalone endpoint for:

        - Testing and debugging safety behavior
        - Tooling / admin UIs
        - Future per-session safety audits

    It does NOT perform intent gating; that remains the job of
    app.services.intent.decide_intent + fusion.
    """
    # Core analysis used by RAG and fusion
    report = analyze_text_blocks(req.snippets)

    # Staged + emergency payload used by fusion intent gate
    payload = safety_payload(req.snippets, query_text=req.query_text or None)

    return SafetyResponse(
        ok=True,
        query_text=req.query_text,
        snippets=req.snippets,
        report=report,
        payload=payload,
    )
