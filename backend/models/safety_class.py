# app/models/safety_class.py

"""
SAFETY DATA MODELS

Pydantic models for typed safety responses, used by the
service layer and fusion layer.

This module centralizes the SafetyWarning and SafetyReport
models that were previously defined in app.services.safety.
"""

from __future__ import annotations

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class SafetyWarning(BaseModel):
    """
    Structured warning for a specific safety-relevant category.

    Fields:
        category: e.g. "electrical", "gas", "hazmat", "appliance"
        level:    "info" | "caution" | "stop"
        reason:   brief human-readable explanation
        signals:  matched keywords/phrases (patterns) that triggered the warning
    """
    category: str
    level: str
    reason: str
    signals: List[str] = Field(default_factory=list)


class SafetyReport(BaseModel):
    """
    v2 safety report used across the backend.

    - blocked:      True if we detect any "stop"-level signal in the analyzed text.
                    In RAG, this is *advisory-only* (no hard gate); fusion will
                    implement hard blocks using its own safety_payload + intent gate.
    - requires_ack: True when a staged workflow is required (set by fusion/intent).
    - ack_stage:    0 = none, 1/2 = which stage of acknowledgement is needed.
    - advisory:     High-level, human-readable warnings suitable for UI banners.
    - per_item_flags:
                    Row-level flags; RAG uses this to attach flags to individual
                    result items. For text-block analysis, we leave this empty.
    - notes:        Free-form text notes (e.g., which categories were detected).
    - warnings:     Structured warnings (category + level + reason + signals).
    """
    blocked: bool = False
    requires_ack: bool = False
    ack_stage: int = 0

    advisory: List[str] = Field(default_factory=list)
    per_item_flags: List[List[str]] = Field(default_factory=list)
    notes: str = ""

    # Back-compat: original field used by earlier code paths
    warnings: List[SafetyWarning] = Field(default_factory=list)
