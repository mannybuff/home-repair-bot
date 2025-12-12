# app/utils/search_utils.py

"""
 - This file manages all of the search function helpers.
 - The first section of helpers are related to Web search.
 - The next section of helpers are related to RAG search.

"""

from __future__ import annotations

from typing import List, Dict, Tuple, Set, Optional, Any
from pathlib import Path
from urllib.parse import urlparse
from fastapi.encoders import jsonable_encoder

import requests
import httpx
import os
import re

from app.core.settings import settings
from app.models.search_class import SearchResult

# ---------------------------------------------------------------------------
# WEB IMAGE FETCH LINK HELPER (migrated from fusion._web_fetch_link)
# ---------------------------------------------------------------------------

def web_fetch_link(img_url: str, thumb_w: int) -> Dict[str, Any]:
    """
    Build a small descriptor for a web image and its thumbnail endpoint.

    This is used when we want to:
      - keep the original image URL
      - expose a backend thumbnail endpoint that can proxy/resize it

    """
    if not img_url:
        return {}

    return {
        "type": "web_image",
        "url": img_url,
        "thumbnail": {
            "url": f"/api/v1/rag/images/web-thumb?url={img_url}&w={thumb_w}",
            "width": thumb_w,
        },
    }

# ---------------------------------------------------------------------------
# WHITELIST HELPERS (migrated from app/search/whitelist.py)
# ---------------------------------------------------------------------------

def _read_lines(path: Path) -> List[str]:
    """Read non-empty, stripped lines from a text file."""
    try:
        if not path.exists():
            return []
        text = path.read_text(encoding="utf-8", errors="ignore")
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
    except Exception:
        return []

def _extract_domain(url: str) -> str:
    """Return domain portion (lowercased) without leading www."""
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""

def load_whitelists(
    file_path: Path,
    defaults_domains: List[str],
    defaults_repos: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Merge whitelist file + defaults. 
    Returns (domains, repos) lists.
    """
    lines = _read_lines(file_path)
    file_domains: List[str] = []
    file_repos: List[str] = []

    for ln in lines:
        if "github.com" in ln or ln.endswith(".git"):
            file_repos.append(ln)
        else:
            dom = _extract_domain(ln)
            if dom:
                file_domains.append(dom)

    domains = list(dict.fromkeys(
        [d.lower() for d in defaults_domains] +
        [d.lower() for d in file_domains]
    ))
    repos = list(dict.fromkeys(defaults_repos + file_repos))

    return domains, repos

def is_url_allowed(url: str, allowed_domains: List[str]) -> bool:
    """True if URL's normalized domain is in allowed_domains."""
    dom = _extract_domain(url)
    return dom in {d.lower() for d in allowed_domains}

# ---------------------------------------------------------------------------
# GOOGLE CSE HELPERS (migrated from app/search/google_cse.py)
# ---------------------------------------------------------------------------

GOOGLE_ENDPOINT = "https://www.googleapis.com/customsearch/v1"

def _domain_of(url: str) -> str:
    """Extract normalized domain for safety checks."""
    try:
        d = urlparse(url).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""

def _api_params(
    query: str,
    api_key: str,
    cx: str,
    domain: str,
    num: int,
) -> Dict[str, str]:
    """Build Google CSE request parameters for a single domain."""
    return {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": str(min(num, 10)),
        "safe": "active",
        "siteSearch": domain,
        "siteSearchFilter": "i",
    }

def google_search_text(
    query: str,
    api_key: str,
    cx: str,
    allowed_domains: list[str] | None = None,
    max_results: int = 10,
) -> list[SearchResult]:
    """
    Call Google Custom Search JSON API and return filtered SearchResult list.

    STRICT WHITELIST, DOMAIN-SCOPED MODE

    - If allowed_domains is provided, we iterate over those domains and
      call Google CSE with `siteSearch=<domain>` for each one.
    - We dedupe across domains by URL.
    - We stop once we have max_results total.
    - No fallback to unfiltered, non-whitelisted domains.
    - If there are no allowed_domains, we just return [].
    """

    allowed_domains = [d.strip().lower() for d in (allowed_domains or []) if d.strip()]

    if not api_key or not cx:
        print(
            f"[search] google_search_text SKIP: missing api_key or cx "
            f"(api={bool(api_key)} cx={bool(cx)})"
        )
        return []

    if not allowed_domains:
        print("[search] google_search_text: no allowed_domains; returning empty list.")
        return []

    max_results = max_results or 10
    url = GOOGLE_ENDPOINT

    results: list[SearchResult] = []
    seen: set[str] = set()
    total_raw = 0

    # Decide how many we *try* to pull per domain.
    # Simple approach: give each domain up to max_results, but we'll stop
    # once the global max_results is reached.
    per_domain_num = max_results

    print(
        f"[search] google_search_text START query={query!r} "
        f"domains={len(allowed_domains)} max_results={max_results}"
    )

    for domain in allowed_domains:
        if len(results) >= max_results:
            break

        params = _api_params(query, api_key, cx, domain, per_domain_num)

        try:
            print(
                f"[search] google_search_text calling CSE for domain={domain!r} "
                f"q={query!r}"
            )
            resp = httpx.get(url, params=params, timeout=10.0)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(
                f"[search] google_search_text ERROR for domain={domain!r}: "
                f"{type(e).__name__}: {e}"
            )
            continue

        items = data.get("items") or []
        total_raw += len(items)

        for item in items:
            link = (item.get("link") or "").strip()
            if not link:
                continue
            if link in seen:
                continue
            seen.add(link)

            snippet = item.get("snippet") or ""
            title = item.get("title") or ""

            thumb = None
            try:
                pagemap = item.get("pagemap") or {}
                cse_images = pagemap.get("cse_image") or []
                if isinstance(cse_images, list) and cse_images:
                    thumb = cse_images[0].get("src")
            except Exception:
                thumb = None

            results.append(
                SearchResult(
                    title=title,
                    url=link,
                    snippet=snippet,
                    thumbnail_url=thumb,
                    source="web",
                )
            )

            if len(results) >= max_results:
                break

    print(
        f"[search] google_search_text DONE total_raw={total_raw} "
        f"unique_kept={len(results)} domains={len(allowed_domains)}"
    )

    return results[:max_results]  
    
# ---------------------------------------------------------------------------
# URL Helpers
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    try:
        return url.lower().strip()
    except Exception:
        return url

def normalize_domain(url: str) -> str:
    d = _domain_of(url)
    return d.lower() if d else ""

def dedupe_results(results: List[Dict[str, Any]], key: str = "url") -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for r in results:
        val = r.get(key)
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(r)
    return out

    
# ---------------------------------------------------------------------------
# SIMPLE CROSS-MODAL RERANKING
# ---------------------------------------------------------------------------

def _fused_text_blob(item: dict) -> str:
    """
    Extract a plain-text blob from a normalized fused item for scoring.

    We reuse the same logic as _text_of but assume the normalized
    fusion schema: type in {"image","text","web"}.
    """
    t: list[str] = []
    itype = item.get("type")

    if itype == "image":
        for s in item.get("snippets", []):
            t.append(str(s))
        # also include any caption if present
        cap = item.get("caption")
        if cap:
            t.append(str(cap))
    elif itype == "text":
        t.append(str(item.get("snippet", "")))
        t.append(str(item.get("title", "")))
    elif itype == "web":
        t.append(str(item.get("title", "")))
        t.append(str(item.get("snippet", "")))
    return " ".join(t)


def _simple_overlap_score(text: str, query: str) -> float:
    """
    Very lightweight similarity score: Jaccard overlap of token sets.

    This avoids embeddings and keeps the rerank deterministic and cheap.
    """
    if not text or not query:
        return 0.0

    # lowercase + dumb token split
    t_tokens = set(text.lower().split())
    q_tokens = set(query.lower().split())
    if not t_tokens or not q_tokens:
        return 0.0

    inter = t_tokens & q_tokens
    if not inter:
        return 0.0
    union = t_tokens | q_tokens
    return len(inter) / len(union)


def rerank_fused_results(
    fused: list[dict],
    query_text: str | None = None,
) -> list[dict]:
    """
    Rerank the fused result list ONCE, across image/text/web, without dropping items.

    Heuristics:
      - Add a type-based weight (e.g., prioritize text + web over pure image neighbors).
      - Add a simple token-overlap score between the item's text and the query_text.
      - Preserve original order among items with the same score.

    This is intentionally simple; it gives a gentle nudge toward more relevant
    evidence while keeping the overall pool intact.
    """

    if not fused:
        return fused

    # type weights: tweak as desired
    type_weight = {
        "text": 1.2,
        "web": 1.1,
        "image": 1.0,
    }

    # Attach scores with stable index for tie-breaking
    scored: list[tuple[float, int, dict]] = []
    for idx, item in enumerate(fused):
        if not isinstance(item, dict):
            scored.append((0.0, idx, item))
            continue

        itype = item.get("type") or "text"
        base_w = type_weight.get(itype, 1.0)
        txt = _fused_text_blob(item)
        sim = _simple_overlap_score(txt, query_text or "") if query_text else 0.0

        score = base_w + sim
        scored.append((score, idx, item))

    # Sort by score DESC, then by original index ASC (stable for ties)
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [item for (_score, _idx, item) in scored]

# ---------------------------------------------------------------------------
# RESULT BUCKETING / TOP-K HELPERS
# ---------------------------------------------------------------------------

def top_k(results: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    """
    Return the top-k results by 'score' (descending),
    but preserve existing ordering if scores are equal or missing.
    """

    if not results or k <= 0:
        return []
    # Keep exact original behavior = preserve existing order
    return results[:k]


def bucketize_results(
    results: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group results by 'type' into text / web / image buckets.

    This matches how pack_and_persist_results(...) calls it:
        buckets = bucketize_results(results)

    It preserves original order within each bucket and falls back
    conservatively when 'type' is missing.
    """

    out: Dict[str, List[Dict[str, Any]]] = {"text": [], "web": [], "image": []}

    for r in results or []:
        if not isinstance(r, dict):
            continue

        t = r.get("type")
        if t in out:
            out[t].append(r)
        else:
            # Conservative fallback:
            #   - If it looks URL-ish or image-ish, treat as web
            #   - Otherwise treat as text
            if r.get("url") or r.get("image_url"):
                out["web"].append(r)
            else:
                out["text"].append(r)

    return out

# ---------------------------------------------------------------------------
# FUSION NORMALIZATION HELPERS (migrated from api/fusion.py)
# ---------------------------------------------------------------------------

def normalize_image_item(it: Any) -> Dict[str, Any]:
    """
    Normalize an image result item to a dict and ensure type='image'.

    Supports:
      - plain dicts (from legacy code)
      - ImgEntry-like objects with .meta (doc_hash, page_index, pdf_path, image_path)
        and an optional .score attribute.
    """
    if isinstance(it, dict):
        d: Dict[str, Any] = dict(it)
    else:
        d = {}
        meta = getattr(it, "meta", None)

        # Pull fields from meta if present
        if meta is not None:
            d["doc_hash"] = getattr(meta, "doc_hash", None)
            d["page_index"] = getattr(meta, "page_index", None)

            pdf_path = getattr(meta, "pdf_path", None)
            img_path = getattr(meta, "image_path", None)

            if pdf_path is not None:
                try:
                    d["pdf_path"] = str(pdf_path)
                except Exception:
                    d["pdf_path"] = None

            if img_path is not None:
                try:
                    d["image_path"] = str(img_path)
                except Exception:
                    d["image_path"] = None

        # Propagate score if available
        if hasattr(it, "score"):
            try:
                d["score"] = float(getattr(it, "score"))
            except Exception:
                pass

    d.setdefault("type", "image")
    return d


def normalize_text_item(it: Any) -> Dict[str, Any]:
    """
    Normalize a text (RAG) result item to a dict and ensure type='text'.
    """
    d = dict(it) if isinstance(it, dict) else {}
    d.setdefault("type", "text")
    return d


def normalize_web_item(it: Any) -> Dict[str, Any]:
    """
    Normalize a web result item:

    - Accepts pydantic models, dataclasses, or dict-like objects.
    - Converts to dict and ensures type='web'.
    """
    if hasattr(it, "model_dump"):
        d = it.model_dump()
    elif hasattr(it, "dict"):
        d = it.dict()
    else:
        d = dict(it) if isinstance(it, dict) else {}
    d = jsonable_encoder(d)
    d.setdefault("type", "web")
    return d
