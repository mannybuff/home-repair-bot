# app/services/search.py

"""
 - This provides the search functions explicitly.
 - Relies on app/utils/ {search_utils.py , common.py}
 - The first functions are Web related.
 - The next several are RAG related.

"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.core.settings import settings
from selectolax.parser import HTMLParser

from app.utils.common import load_json as _load_json, jsonable as _jsonable

from app.utils.search_utils import (
    load_whitelists,
    is_url_allowed,
    google_search_text,
    normalize_url,
    normalize_domain,
    dedupe_results,
)

from app.services.sessions import (
    load_recent_events,
    persist_latest_event,
    record_session_event,
)

from app.models.search_class import SearchQuery, SearchResponse, SearchResult
from app.utils.search_utils import top_k as _top_k
from app.utils.search_utils import bucketize_results as _bucketize_results
from app.utils.session_utils import session_root, list_event_files

import json

# ---------------------------------------------------------------------------
# Dialog path helper (replaces legacy _dialog_paths from old layout)
# ---------------------------------------------------------------------------

def _dialog_paths(dialog_id: str) -> tuple[Path, Path]:
    """
    Resolve the per-dialog root and the latest event JSON path.

    dlg_root:  data/sessions/<dialog_id>/
    ev_latest: last event file in data/sessions/<dialog_id>/events/*.json
    """
    root = session_root() / dialog_id
    ev_files = list_event_files(dialog_id)

    if ev_files:
        ev_latest = ev_files[-1]  # newest by lexical sort of filename
    else:
        ev_latest = root / "events" / "latest.json"

    return root, ev_latest

# ---------------------------------------------------------------------------
# PACK THE SEARCH RESULTS
# ---------------------------------------------------------------------------

def pack_and_persist_results(
    dialog_id: str,
    k_text: int = 3,
    k_web: int = 3,
    k_image: int = 3,
) -> Dict[str, Any]:
    """
    Read the latest event for a dialog, partition full results by type,
    keep the top K per type (no snippet truncation), and persist to
    ./data/sessions/<dialog_id>/searchres-<timestamp>.json

    This is used primarily for debugging / traceability. It should be:
      - strictly per-dialog
      - faithful to the latest event's query
      - simple and non-duplicative in its counts
    """
    dlg_root, ev_latest = _dialog_paths(dialog_id)
    if not ev_latest.exists():
        raise FileNotFoundError("latest_event_missing")

    latest = _load_json(ev_latest) or {}

    # --- Query metadata: prefer the top-level query block ---
    q = latest.get("query") or {}
    q_text = q.get("text") or None
    q_caption = q.get("caption") or None
    q_has_image = bool(q.get("has_image"))

    # --- Results and evidence from the latest event ---
    results = latest.get("results") or []
    evidence = latest.get("evidence") or []

    # Partition by type
    buckets = _bucketize_results(results)

    # Simple, non-duplicated counts derived from the buckets
    counts = {
        "results_total": len(results),
        "text": len(buckets["text"]),
        "web": len(buckets["web"]),
        "image": len(buckets["image"]),
    }

    def _top_k(seq: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
        if not isinstance(seq, list) or not seq:
            return []
        if k <= 0:
            return []
        # Keep original shapes; just truncate
        return seq[:k]

    pack = {
        "schema_version": getattr(settings, "schema_version", "v1.1"),
        "dialog_id": dialog_id,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "source_event": ev_latest.name,
        "query": {
            "text": q_text,
            "caption": q_caption,
            "has_image": q_has_image,
        },
        "counts": counts,
        # keep top-K long-form items without snipping; rely on original shapes
        "longform": {
            "text": _top_k(buckets["text"], k_text),
            "web": _top_k(buckets["web"], k_web),
            "image": _top_k(buckets["image"], k_image),
        },
        # record what the event surfaced as best-evidence at that time
        "evidence_refs": evidence[: min(6, len(evidence))],
    }

    outp = dlg_root / f"searchres-{datetime.utcnow().isoformat(timespec='milliseconds').replace(':','-')}.json"
    outp.write_text(json.dumps(pack, indent=2, ensure_ascii=False))

    return {
        "ok": True,
        "file": outp.name,
        "path": str(outp),
        "counts": pack["counts"],
        "k_used": {"text": k_text, "web": k_web, "image": k_image},
    }
    
# ---------------------------
# 1) RAG LONGFORM INFLATION
# ---------------------------

def _inflate_rag_items(
    items: List[Dict[str, Any]],
    *,
    rag_window: int,
    rag_max_chunks: int,
    web_max_chars: int,
) -> List[Dict[str, Any]]:
    """
    For each RAG item:
      - Try to pull nearby chunks from the same PDF page(s)
        using the RAG meta index.
      - Combine snippet + neighbors into a multi-paragraph 'text'.
      - Sane fallbacks if anything fails.
    """
    if not items:
        return items

    # Detect whether any item even references a PDF; if not, just
    # ensure we have reasonable 'text' from snippet and bail early.
    has_pdf = any((itm.get("pdf_path") or "").strip() for itm in items)
    if not has_pdf:
        for itm in items:
            snip = (itm.get("snippet") or "").strip()
            if snip and not itm.get("text"):
                itm["text"] = snip
        return items

    page_map = None
    try:
        # Lazy-import inside function to avoid any import cycles
        from app.services.rag import load_index_and_meta
        from app.utils.rag_utils import build_page_map, nearby_chunks, extract_page_index

        # Load meta only; index is unused here
        _index, meta, _norm = load_index_and_meta(Path(settings.data_dir))
        if meta:
            # meta.chunks is typically a list of Pydantic models / records
            meta_d = {"chunks": [vars(c) for c in meta.chunks]}
            page_map = build_page_map(meta_d)
    except Exception:
        page_map = None

    for itm in items:
        # Start from the existing snippet if present
        snippets: List[str] = []
        base_snip = (itm.get("snippet") or "").strip()
        if base_snip:
            snippets.append(base_snip)

        # Try to extend with nearby chunks if we have page_map and pdf_path
        if page_map is not None:
            try:
                pdf_path = (itm.get("pdf_path") or "").strip()
                if pdf_path:
                    # Prefer explicit page_index/page_from; otherwise infer
                    page_idx = itm.get("page_index")
                    if page_idx is None:
                        page_idx = itm.get("page_from")
                    if page_idx is None:
                        try:
                            page_idx = extract_page_index(itm)  # type: ignore[name-defined]
                        except Exception:
                            page_idx = 0
                    page_idx = int(page_idx or 0)

                    neighbor_snips = nearby_chunks(  # type: ignore[name-defined]
                        page_map,
                        pdf_path,
                        page_idx,
                        window=rag_window,
                        max_return=rag_max_chunks,
                    ) or []

                    for s in neighbor_snips:
                        s = (s or "").strip()
                        if s and s not in snippets:
                            snippets.append(s)
            except Exception:
                # If anything goes wrong here, we just fall back to base snippet
                pass

        if snippets:
            combined = "\n\n".join(snippets)
            # Trim to a manageable size
            if len(combined) > web_max_chars:
                combined = combined[:web_max_chars]
                # avoid cutting mid-word
                last_space = combined.rfind(" ")
                if last_space > 0:
                    combined = combined[:last_space]
            itm["text"] = combined
        elif base_snip:
            itm["text"] = base_snip

        # Ensure we have some sort of human-readable title
        if not itm.get("title"):
            pdf_path = (itm.get("pdf_path") or "").replace("\\", "/")
            pdf_name = pdf_path.split("/")[-1] if pdf_path else "Manual"
            page_from = itm.get("page_from")
            try:
                page_num = int(page_from) + 1 if page_from is not None else None
            except Exception:
                page_num = None
            if page_num is not None:
                itm["title"] = f"{pdf_name} — p.{page_num}"
            else:
                itm["title"] = pdf_name
    return items


def _extract_paragraphs_from_html(html: str) -> List[str]:
    """
    Very simple HTML → paragraph extractor using selectolax.
    We:
      - collect non-empty <p> tags
      - strip and keep those with some length
    """
    try:
        parser = HTMLParser(html)
    except Exception:
        return []

    paras: List[str] = []
    for node in parser.css("p"):
        txt = (node.text() or "").strip()
        if len(txt) >= 40:
            paras.append(txt)
    return paras


def _inflate_web_items(
    items: List[Dict[str, Any]],
    *,
    web_max_paragraphs: int,
    web_max_chars: int,
) -> List[Dict[str, Any]]:
    """
    For each web item:
      - Fetch the HTML for its URL (best-effort).
      - Extract up to `web_max_paragraphs` paragraphs.
      - Combine those into a multi-paragraph 'text'.
      - Fallback to snippet if fetching or parsing fails.
    """
    if not items:
        return items

    # Lazy import to avoid forcing httpx if unused elsewhere
    try:
        import httpx  # type: ignore
    except Exception:
        httpx = None  # type: ignore

    for itm in items:
        url = (itm.get("url") or itm.get("link") or "").strip()
        base_snip = (itm.get("snippet") or "").strip()

        body_text: Optional[str] = None
        if url and httpx is not None:
            try:
                with httpx.Client(timeout=5.0, follow_redirects=True) as client:
                    resp = client.get(url)
                    ctype = resp.headers.get("content-type", "")
                    if resp.status_code == 200 and "text/html" in ctype:
                        html = resp.text
                        paras = _extract_paragraphs_from_html(html)
                        if paras:
                            # Limit number of paragraphs and total char length
                            selected: List[str] = []
                            total_chars = 0
                            for p in paras:
                                if len(selected) >= web_max_paragraphs:
                                    break
                                if total_chars + len(p) > web_max_chars:
                                    break
                                selected.append(p)
                                total_chars += len(p)
                            if selected:
                                body_text = "\n\n".join(selected)
            except Exception:
                # Any network / parsing failure → no body_text
                body_text = None

        # Fallback ordering for 'text' field
        if body_text:
            itm["text"] = body_text
        elif base_snip:
            itm["text"] = base_snip

        # Ensure a decent title
        if not itm.get("title"):
            if body_text:
                # Take first line as a pseudo-title if nothing else
                first_line = body_text.splitlines()[0].strip()
                itm["title"] = first_line[:80] if first_line else (url or "Web Source")
            else:
                itm["title"] = url or "Web Source"

    return items


def inflate_longform_pack(
    pack: Dict[str, Any],
    *,
    rag_window: int = 1,
    rag_max_chunks: int = 3,
    web_max_paragraphs: int = 6,
    web_max_chars: int = 2000,
    max_sources: int = 3,
) -> Dict[str, Any]:
    """
    Enrich a searchres pack's longform section with multi-paragraph text.

    INPUT SHAPE (current pack_and_persist_results output):

        {
          "longform": {
            "text":  [ { ...rag_hit... }, ... ],
            "web":   [ { ...web_hit... }, ... ],
            "image": [ ... ]
          },
          ...
        }

    OUTPUT GUARANTEES:

      - Returns a *new* dict (does not mutate input).
      - Preserves overall pack structure and counts.
      - For each selected text/web item:
          - Ensures a 'title' field is present.
          - Adds a 'text' field containing 1–6 paragraphs
            (up to `web_max_chars` characters).
      - Ignores image items for now.
      - Limits the *total* number of longform sources to `max_sources`
        across text + web combined.

    The function is designed to *never throw*:
      - On any failure, it returns the original pack structure.
    """
    if not isinstance(pack, dict):
        return pack

    longform = (pack.get("longform") or {}) if isinstance(pack.get("longform"), dict) else {}
    text_items = list(longform.get("text") or [])
    web_items = list(longform.get("web") or [])

    # If there's nothing to enrich, just return original pack
    if not text_items and not web_items:
        return pack

    # Deep-ish copy so we don't mutate caller's dict
    out = json.loads(json.dumps(pack))
    lf = out.setdefault("longform", {})
    lf_text = list(lf.get("text") or [])
    lf_web = list(lf.get("web") or [])

    # Apply inflators to the copies we pulled into lf_text/lf_web
    try:
        lf_text = _inflate_rag_items(
            lf_text,
            rag_window=rag_window,
            rag_max_chunks=rag_max_chunks,
            web_max_chars=web_max_chars,
        )
    except Exception:
        # Graceful fail: keep items as-is
        pass

    try:
        lf_web = _inflate_web_items(
            lf_web,
            web_max_paragraphs=web_max_paragraphs,
            web_max_chars=web_max_chars,
        )
    except Exception:
        # Graceful fail: keep items as-is
        pass

    # ---------------------------
    # LIMIT TOTAL SOURCES TO max_sources
    # ---------------------------

    # Preserve original ordering: text items first, then web items, but
    # enforce a global cap on total sources.
    combined: List[tuple[str, Dict[str, Any]]] = []
    for itm in lf_text:
        combined.append(("text", itm))
    for itm in lf_web:
        combined.append(("web", itm))

    selected = combined[: max_sources if max_sources > 0 else len(combined)]

    new_text: List[Dict[str, Any]] = []
    new_web: List[Dict[str, Any]] = []

    for src_type, itm in selected:
        if src_type == "text":
            new_text.append(itm)
        elif src_type == "web":
            new_web.append(itm)

    # Update the longform section; leave image list untouched for now,
    # but it's fine if downstream ignores it.
    lf["text"] = new_text
    lf["web"] = new_web
    # Explicitly keep 'image' as a list (may be empty)
    if not isinstance(lf.get("image"), list):
        lf["image"] = list(lf.get("image") or [])

    return out

# --------------------------------------------------------------------------------

def search_service(payload: SearchQuery) -> SearchResponse:
    # Load whitelist (merges file + defaults)
    allowed_domains, _repos = load_whitelists(
        settings.search_whitelist_file, settings.allowed_domains, settings.allowed_repos
    )

    urls = [str(u) for u in (payload.urls or [])]
    bad_urls = [u for u in urls if not is_url_allowed(u, allowed_domains)]
    if bad_urls:
        return SearchResponse(
            ok=False,
            query=payload.query,
            results=[],
            allowed_domains=allowed_domains,
            notes=f"Blocked URLs outside whitelist: {', '.join(bad_urls)}",
        )

    results: List[SearchResult] = []
    notes: List[str] = []

    if urls:
        # Direct fetch of explicit URLs (already validated against whitelist)
        fetched = fetch_many(urls, max_results=payload.max_results)
        results.extend(fetched)
        notes.append("Fetched direct URLs.")
    else:
        # Whitelist-first Google CSE: single broad call, Python-side domain filter
        api_key = settings.google_api_key
        cx = settings.google_text_cse_id

        # google_search_text handles missing api_key/cx internally and returns []
        g = google_search_text(
            payload.query,
            api_key,
            cx,
            allowed_domains=allowed_domains,
            max_results=payload.max_results,
        )
        results.extend(g)
        if g:
            notes.append("Google CSE + whitelist filter.")
        else:
            notes.append("No whitelisted hits from Google CSE.")

    return SearchResponse(
        ok=True,
        query=payload.query,
        results=results,
        allowed_domains=allowed_domains,
        notes=" ".join(notes),
    )

def _collect_web_hits(
    queries: List[str],
    max_results: int,
) -> List[Dict[str, Any]]:
    """
    Collect normalized Web hits for all queries.
    Pure logic: no session writes, no fusion coupling.
    """
    all_hits: List[Dict[str, Any]] = []

    for q in (queries or []):
        resp = search_service(SearchQuery(query=q, max_results=max_results))
        if not getattr(resp, "ok", False):
            continue

        hits = getattr(resp, "results", []) or []
        norm_list: List[Dict[str, Any]] = []

        for h in hits:
            if hasattr(h, "model_dump"):
                d = h.model_dump()
            elif hasattr(h, "dict"):
                d = h.dict()
            else:
                d = dict(h)

            d = _jsonable(d)
            d.setdefault("type", "web")
            norm_list.append(d)

        # Filtering logic (filters will move into search_utils next wave)
        # For now, keep behavior identical by not altering here

        all_hits.extend(norm_list)

    return all_hits

# ---------------------------------------------------------------------------
# FUNCTIONS FOR FETCHING
# ---------------------------------------------------------------------------

def fetch_url(url: str) -> Optional[SearchResult]:
    """
    Fetch a single URL (must be whitelisted) and parse basic info.
    Returns SearchResult or None if disallowed/failure.
    """
    # Load whitelist fresh (merges file + defaults)
    domains, _repos = load_whitelists(
        settings.search_whitelist_file, settings.allowed_domains, settings.allowed_repos
    )
    if not is_url_allowed(url, domains):
        return None

    try:
        resp = httpx.get(url, timeout=10.0, follow_redirects=True)
        resp.raise_for_status()
    except Exception:
        return None

    parser = HTMLParser(resp.text)

    title = (parser.css_first("title").text(strip=True) if parser.css_first("title") else url)
    snippet_node = parser.css_first("p") or parser.css_first("div")
    snippet = snippet_node.text(strip=True)[:300] if snippet_node else ""

    return SearchResult(title=title, url=url, snippet=snippet, score=1.0)

def fetch_many(urls: List[str], max_results: int = 5) -> List[SearchResult]:
    out: List[SearchResult] = []
    for u in urls[:max_results]:
        res = fetch_url(u)
        if res:
            out.append(res)
    return out

# ---------------------------------------------------------------------------
# CURRENT SEARCH THROUGH WEB
# ---------------------------------------------------------------------------

def run_web_search_service(
    dialog_id: str,
    queries: List[str],
    max_results: int = 6,
    apply_filter: bool = False,
) -> Dict[str, Any]:
    """
    Run web search for the given queries, attach hits to the latest session
    event for this dialog, persist, and return a summary + raw hits.

    This function is called by fusion_search and must be robust even when
    there are no prior events.

    IMPORTANT CHANGE:
    - We no longer call pack_and_persist_results() here.
      The canonical search pack is now created later (in synthesis.generate),
      after fusion has added RAG + image hits and reranked everything.
    """

    # 1) gather hits
    all_hits = _collect_web_hits(queries, max_results)

    if not all_hits:
        return {"ok": False, "results": [], "n": 0, "warnings": "No web hits"}

    # 2) merge into latest event
    events = load_recent_events(dialog_id) or []

    # load_recent_events returns a list[dict]; we want the most recent one
    if isinstance(events, list) and events:
        base = events[-1]
    elif isinstance(events, dict):
        # extremely defensive: if some caller ever returned a dict
        base = events
    else:
        base = {}

    # Make a shallow copy, then make it JSON-able
    ev = _jsonable(dict(base) if isinstance(base, dict) else {}) or {}

    # Ensure "results" and "query" blocks exist
    ev.setdefault("results", [])
    q_block = ev.setdefault("query", {})
    # Preserve has_image flag if present; default False if unknown
    if "has_image" not in q_block:
        q_block["has_image"] = bool(q_block.get("has_image"))

    # Extend with new web hits
    ev["results"].extend(all_hits)

    # 3) debug breadcrumbs
    dbg = ev.setdefault("debug", {}).setdefault("web_search", {})
    dbg["queries"] = list(queries)
    dbg["added"] = len(all_hits)

    # 4) save back
    persist_latest_event(dialog_id, ev)

    # 5) return hits only; pack is now created later in synthesis
    out: Dict[str, Any] = {
        "ok": True,
        "results": all_hits,
        "n": len(all_hits),
    }

    return out


