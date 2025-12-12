# app/models/search_class.py
"""
SEARCH DATA MODELS

Pydantic models for search input/output so the API and service
layers have a stable, self-documenting contract.

Extracted from:
    - app/search/schema.py

Used by:
    - app/services/search.py
    - api layers that expose search endpoints
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class SearchQuery(BaseModel):
    """
    Search request payload.

    Fields:
        - query: user search text
        - max_results: upper bound on result count (1..25)
        - urls: optional explicit URLs to fetch, in addition to query-based search
    """
    query: str = Field(..., min_length=2, description="User's search text")
    max_results: int = Field(
        5,
        ge=1,
        le=25,
        description="Max results to return",
    )
    # Optional direct URLs to fetch (must be whitelisted) in addition to query-based results
    urls: Optional[List[HttpUrl]] = None


class SearchResult(BaseModel):
    """
    A single search result, typically derived from Google CSE or direct URL fetch.
    """
    title: str
    url: HttpUrl
    snippet: str = ""
    score: float = 0.0


class SearchResponse(BaseModel):
    """
    Standardized search response wrapper.
    """
    ok: bool = True
    query: str
    results: List[SearchResult] = []
    allowed_domains: List[str] = []
    notes: Optional[str] = None
