# app/utils/rag_utils.py

"""
RAG UTILITY HELPERS

This module holds low-level helpers used by services/rag.py,
fusion.py, and synthesis.

This is where we migrate:
    - pdf_path resolution from RAG hits
    - page-index extraction
    - snippet/metadata normalization
    - RAG hit → fused item shaping (non-UI)

Design notes:
    - keep pure functions here
    - no session writes
    - no external network IO
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
from pathlib import Path
from urllib.parse import urlencode

from app.core.settings import settings
from app.utils.common import jsonable as _jsonable
import json

from app.core.settings import settings


def resolve_pdf_path(hit: Dict[str, Any]) -> str:
    """
    Resolve the PDF path for a single RAG hit.

    RAG hits from app.rag.indexer.search(...) expose 'file_path' which
    may be relative or absolute. We normalize to an absolute path string.

    If anything goes wrong, we return the original string best-effort.
    """
    raw = (hit or {}).get("file_path") or ""
    if not raw:
        return ""
    try:
        return str(Path(raw).resolve())
    except Exception:
        # fall back to the raw value if resolution fails
        try:
            return str(raw)
        except Exception:
            return ""


def extract_page_index(hit: Dict[str, Any]) -> int:
    """
    Extract a representative page index from a RAG hit.

    RAG hits expose:
        - page_from: int (0-based)
        - page_to:   int (0-based, inclusive or equal)

    For most purposes, page_from is an adequate representative index.
    """
    if not hit:
        return 0
    try:
        return int(hit.get("page_from", 0))
    except Exception:
        return 0


def normalize_rag_hit(hit: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize a single RAG hit into the common text evidence schema.

    Input (from app.rag.indexer.search):
        {
            "score": float,
            "file_path": str,
            "page_from": int,
            "page_to": int,
            "snippet": str,
        }

    Output (normalized):
        {
            "type": "text",             # treated as text in fusion
            "source": "rag",            # provenance for downstream logic
            "score": float,
            "pdf_path": "<abs path>",
            "page_from": int,
            "page_to": int,
            "page_index": int,          # alias of page_from
            "snippet": str,
            "raw": {...},               # original hit (JSON-safe)
        }
    """
    hit = _jsonable(hit or {})

    score = hit.get("score", 0.0)
    pdf_path = resolve_pdf_path(hit)
    page_from = hit.get("page_from", 0)
    page_to = hit.get("page_to", page_from)
    page_idx = extract_page_index(hit)
    snippet = (hit.get("snippet") or "").strip()

    normalized = {
        # IMPORTANT: treat RAG hits as text for fusion/safety/evidence flows
        "type": "text",
        "source": "rag",
        "score": float(score) if isinstance(score, (int, float)) else 0.0,
        "pdf_path": pdf_path,
        "page_from": int(page_from) if isinstance(page_from, (int, float)) else 0,
        "page_to": int(page_to)
        if isinstance(page_to, (int, float))
        else int(page_from)
        if isinstance(page_from, (int, float))
        else 0,
        "page_index": int(page_idx),
        "snippet": snippet,
        "raw": hit,
    }
    return normalized

def build_page_map(meta_d: Dict[str, Any]) -> Dict[str, List[Tuple[int, Dict[str, Any]]]]:
    out: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for c in meta_d.get("chunks", []):
        pdf = c.get("file_path","")
        page = int(c.get("page_from", 0))
        out.setdefault(pdf, []).append((page, c))
    for pdf, lst in out.items():
        lst.sort(key=lambda x: x[0])
    return out

def nearby_chunks(page_map: Dict[str, List[Tuple[int, Dict[str, Any]]]],
                   pdf_path: str, page_index: int, window: int = 1, max_return: int = 3) -> List[str]:
    lst = page_map.get(pdf_path, [])
    if not lst:
        return []
    targets = set(range(max(0, page_index - window), page_index + window + 1))
    hits: List[str] = []
    for p, c in lst:
        if p in targets:
            snip = (c.get("text","") or "").replace("\n"," ").strip()
            if snip:
                hits.append(snip[:400])
            if len(hits) >= max_return:
                break
    return hits

def build_registry_from_pdfs(pdf_dir: Path) -> dict[str, str]:
    reg = {}
    for pdf in sorted(Path(pdf_dir).glob("**/*.pdf")):
        reg[_hash_path(pdf)] = str(pdf.resolve())
    return reg

# ---------------------------------------------------------------------------
# PDF ROOT SAFETY + PAGE-IMAGE REGISTRY (from legacy api/sources.py)
# ---------------------------------------------------------------------------

def resolved_allowed_roots() -> List[Path]:
    """
    Resolve the allowed roots from settings into absolute Paths.

    Mirrors the ALLOWED_ROOTS logic from api/sources.py but exposes it as
    a reusable helper for services.rag / services.images.
    """
    roots: List[Path] = []
    for p in settings.resolved_allowed_roots():
        try:
            roots.append(Path(p).resolve())
        except Exception:
            continue
    return roots


def safe_under_allowed_roots(p: Path) -> bool:
    """
    True if the given path is under any allowed root.

    This is the canonical replacement for _safe_under_allowed_roots() in
    api/sources.py and should be used anywhere we gate direct filesystem access
    (serving PDFs, resolving page images from arbitrary paths, etc.).
    """
    try:
        rp = p.resolve()
    except Exception:
        return False

    for root in resolved_allowed_roots():
        if rp == root or root in rp.parents:
            return True
    return False


def load_pdf_to_doc_hash_registry(page_images_root: Path) -> dict[str, str]:
    """
    Load a mapping: absolute pdf_path -> doc_hash.

    This mirrors the behavior of _load_registry() in api/sources.py, but is
    generalized as a utility that can be reused by services.images and fusion.

    Expected layout under page_images_root:

        page_images_root/
            registry.jsonl      # lines: {"doc_hash": "...", "pdf_path": "..."}
            <doc_hash>/
                page_0001.jpg
                page_0002.jpg
                ...

    Returns:
        dict mapping absolute PDF paths (str) to doc_hash (str).
    """
    mapping: dict[str, str] = {}
    reg_path = page_images_root / "registry.jsonl"
    if not reg_path.exists():
        return mapping

    try:
        with reg_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                pdf_path = str(obj.get("pdf_path", "")).strip()
                doc_hash = str(obj.get("doc_hash", "")).strip()
                if not (pdf_path and doc_hash):
                    continue
                try:
                    pdf_abs = Path(pdf_path).resolve()
                except Exception:
                    continue
                mapping[str(pdf_abs)] = doc_hash
    except Exception:
        # defensive: never let registry issues crash callers
        return {}

    return mapping


def lookup_doc_hash_for_pdf(
    pdf_path: Path,
    page_images_root: Path,
) -> str | None:
    """
    Convenience helper: pdf_path (absolute) -> doc_hash using the registry.

    Returns:
        doc_hash string if found, else None.
    """
    try:
        pdf_abs = pdf_path.resolve()
    except Exception:
        return None

    reg = load_pdf_to_doc_hash_registry(page_images_root)
    return reg.get(str(pdf_abs))

# ---------------------------------------------------------------------------
# PAGE-IMAGE ENRICHMENT HELPERS (migrated from api/fusion.py)
# ---------------------------------------------------------------------------


def load_page_registry(page_images_root: Path | None = None) -> Tuple[dict[str, str], dict[str, str]]:
    """
    Load mapping between doc_hash and pdf_path from page_images/registry.json[l].

    Returns:
        (doc2pdf, pdf2doc) where:
            doc2pdf: doc_hash -> pdf_path
            pdf2doc: pdf_path -> doc_hash
    """
    root = Path(page_images_root or settings.page_images_dir)
    json_path = root / "registry.json"
    jsonl_path = root / "registry.jsonl"
    doc2pdf: dict[str, str] = {}
    pdf2doc: dict[str, str] = {}

    try:
        if json_path.exists():
            data = json.loads(json_path.read_text(encoding="utf-8"))
            pages = data.get("pages", [])
            for p in pages:
                dh = str(p.get("doc_hash", "")).strip()
                pp = str(p.get("pdf_path", "")).strip()
                if dh and pp:
                    doc2pdf[dh] = pp
                    pdf2doc[pp] = dh
        elif jsonl_path.exists():
            for line in jsonl_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                dh = str(obj.get("doc_hash", "")).strip()
                pp = str(obj.get("pdf_path", "")).strip()
                if dh and pp:
                    doc2pdf[dh] = pp
                    pdf2doc[pp] = dh
    except Exception:
        # silent fallback; citation enrichment is best-effort
        pass

    return doc2pdf, pdf2doc


def page_thumb_url(doc_hash: str, page_index: int, w: int = 320, h: int | None = None) -> str:
    """
    Build a thumbnail URL for the page-image router.
    """
    q: dict[str, int] = {"doc": int(doc_hash) if doc_hash.isdigit() else doc_hash, "page": int(page_index)}
    # NOTE: doc is kept as string; only page must be int
    q = {"doc": doc_hash, "page": int(page_index)}
    if w:
        q["w"] = int(w)
    if h is not None:
        q["h"] = int(h)
    return f"/api/v1/rag/pages/get-image?{urlencode(q)}"


def region_url(doc_hash: str, page_index: int, bbox: dict, out_w: int = 512, out_h: int | None = None) -> str:
    """
    Build a region-crop URL for an image ROI on a page.
    """
    q = {
        "doc": doc_hash,
        "page": int(page_index),
        "x": int(bbox.get("x", 0)),
        "y": int(bbox.get("y", 0)),
        "w": int(bbox.get("w", 1)),
        "h": int(bbox.get("h", 1)),
        "out_w": int(out_w),
    }
    if out_h is not None:
        q["out_h"] = int(out_h)
    return f"/api/v1/rag/pages/get-image-region?{urlencode(q)}"


def derive_doc_and_page(
    row: dict,
    pdf2doc: dict[str, str],
) -> tuple[str | None, int | None, str | None]:
    """
    Extract (doc_hash, page_index, pdf_path) from a fused row.

    Supports:
      - Image hits: doc_hash, page_index already present.
      - Text hits (RAG): file_path + page_from/page_to (prefer page_from).
      - Any row exposing pdf_path or file_path (mapped to doc_hash via registry).
    """
    # 1) Direct fields first
    doc_hash = row.get("doc_hash")
    page_index = row.get("page_index")
    if page_index is None:
        page_index = row.get("page_idx", row.get("page"))

    # 2) RAG text provenance
    pdf_path = row.get("pdf_path") or row.get("file_path")
    if page_index is None:
        pf = row.get("page_from")
        if pf is not None:
            try:
                page_index = int(pf)
            except Exception:
                page_index = None
        if page_index is None:
            pt = row.get("page_to")
            if pt is not None:
                try:
                    page_index = int(pt)
                except Exception:
                    page_index = None

    # 3) Map pdf_path -> doc_hash via registry if needed
    if not doc_hash and pdf_path and pdf2doc:
        try:
            doc_hash = pdf2doc.get(str(pdf_path))
        except Exception:
            doc_hash = None

    # Normalize
    try:
        if page_index is not None:
            page_index = int(page_index)
    except Exception:
        page_index = None

    if doc_hash:
        doc_hash = str(doc_hash)

    return doc_hash, page_index, pdf_path


def enrich_with_page_images(results: list[dict]) -> list[dict]:
    """
    Add 'thumbnail_url' and 'citation{...}' (and optional 'region_url') to image/text items
    when doc/page can be resolved. Leaves web-only items untouched.
    """
    doc2pdf, pdf2doc = load_page_registry()
    out: list[dict] = []

    for r in results:
        rtype = r.get("type")
        if rtype not in ("image", "text"):
            out.append(r)
            continue

        doc_hash, page_index, pdf_path = derive_doc_and_page(r, pdf2doc)
        if doc_hash is None or page_index is None:
            out.append(r)
            continue

        thumb = page_thumb_url(doc_hash, page_index, w=320)
        region = None
        bbox = r.get("bbox") or r.get("region") or r.get("roi")
        if isinstance(bbox, dict):
            region = region_url(doc_hash, page_index, bbox, out_w=640)

        score = r.get("score")

        image_file = None
        try:
            disk_idx = int(page_index) + 1
            candidate = Path(settings.page_images_dir) / doc_hash / f"page_{disk_idx:04d}.jpg"
            if candidate.exists():
                image_file = str(candidate)
        except Exception:
            pass

        r2 = {
            **r,
            "thumbnail_url": thumb,
            "citation": {
                "doc_hash": doc_hash,
                "page_index": page_index,
                "image_file": image_file,
                "pdf_path": pdf_path,
                "score": score,
            },
        }
        if region:
            r2["region_url"] = region

        out.append(r2)

    return out

# ----------------------------------------------

# ----------------------------------------------
def collect_snippets(results: list[dict], limit: int = 24) -> list[str]:
    """Pull short text blocks for safety analysis and quick context."""
    out: list[str] = []
    for r in results or []:
        if not isinstance(r, dict):
            continue
        t = r.get("type")
        if t == "web":
            s = r.get("snippet") or ""
            if r.get("title"):
                s = (r["title"] + " — " + s).strip()
            if r.get("url"):
                s = (s + f" [{r['url']}]").strip()
        elif t == "text":
            s = r.get("snippet") or r.get("text") or ""
        elif t == "image":
            s = r.get("caption") or r.get("alt") or ""
        else:
            s = ""
        s = (s or "").strip()
        if s:
            out.append(s[:320])
        if len(out) >= limit:
            break
    return out


def build_evidence(results: list[dict], limit: int = 9) -> list[dict]:
    """Bucketed evidence so each modality appears (text/image/web)."""
    text_items: list[dict] = []
    image_items: list[dict] = []
    web_items: list[dict] = []

    for r in results or []:
        if not isinstance(r, dict):
            continue
        t = r.get("type")
        if t == "text":
            text_items.append(r)
        elif t == "image":
            image_items.append(r)
        elif t == "web":
            web_items.append(r)

    def to_ev(r: dict) -> dict:
        t = r.get("type")
        if t == "web":
            return {
                "type": "web",
                "title": r.get("title"),
                "url": r.get("url") or r.get("link"),
                "snippet": r.get("snippet"),
                "preview_image_url": r.get("preview_image_url"),
                "thumbnail_url": None,
                "web_image": r.get("web_image"),
                "sim_to_query_image": r.get("sim_to_query_image"),
            }
        if t == "text":
            return {
                "type": "text",
                "title": r.get("title") or r.get("pdf_path"),
                "snippet": r.get("snippet") or r.get("text"),
                "thumbnail_url": r.get("thumbnail_url"),
                "region_url": r.get("region_url"),
                "citation": r.get("citation"),
            }
        if t == "image":
            return {
                "type": "image",
                "title": r.get("title") or r.get("image_path"),
                "snippet": r.get("snippet") or r.get("caption"),
                "thumbnail_url": r.get("thumbnail_url"),
                "region_url": r.get("region_url"),
                "citation": r.get("citation"),
                "score": r.get("score"),
            }
        return {}

    take_text = [to_ev(x) for x in text_items[:3]]
    take_img = [to_ev(x) for x in image_items[:3]]
    take_web = [to_ev(x) for x in web_items[:3]]

    ev: list[dict] = take_text + take_img + take_web

    if len(ev) < limit:
        used_ids = {id(x) for x in text_items[:3] + image_items[:3] + web_items[:3]}
        for r in results or []:
            if len(ev) >= limit:
                break
            if not isinstance(r, dict) or id(r) in used_ids:
                continue
            row = to_ev(r)
            if row:
                ev.append(row)
                used_ids.add(id(r))

    return ev[:limit]
