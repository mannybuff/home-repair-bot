# app/api/image_routes.py
"""
Consolidated image-related API routes.

This file replaces:
    - app/api/web_images.py
    - app/api/images.py (RAG page images)
    - app/api/image_search.py

It exposes three logical groups of endpoints:

    1) Web images (fetch & thumb caching)
       - /api/v1/webimg/fetch
       - /api/v1/webimg/thumb

    2) RAG page images (PDF page rendering & listing)
       - /api/v1/rag/pages/status
       - /api/v1/rag/pages/get-image
       - /api/v1/rag/pages/get-image-region
       - /api/v1/rag/pages/build
       - /api/v1/rag/pages/list

    3) Image→RAG index & search
       - /api/v1/rag/image/index/build
       - /api/v1/rag/image/search

All heavy lifting is delegated to:
    - app.utils.image_utils
    - app.services.images
    - app.services.safety
"""

from __future__ import annotations

from pathlib import Path
from io import BytesIO
from typing import Any, List, Dict, Optional

import glob
import re

import requests
from fastapi import APIRouter, HTTPException, Query, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from PIL import Image

from app.core.settings import settings
from app.utils.image_utils import (
    load_image_from_bytes,
    ensure_rgb,
    resize_preserving_aspect,
    compute_image_sha1,
)
from app.utils.image_utils import (
    safe_url,
    file_id_url,
    img_path,
)
from app.services.images import (
    resolve_page_image_path,
    render_pdf_pages,
    list_page_images,
    search_similar_images,
    build_image_index,
    caption_image,
)
from app.services.safety import analyze_snippets


router = APIRouter(tags=["images"])


# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------

def _pil_to_jpeg_response(img: Image.Image, quality: int = 85) -> StreamingResponse:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/jpeg")

# ---------------------------------------------------------------------------
# INTERNAL WEB-IMAGE CACHING HELPER
# ---------------------------------------------------------------------------

def _ensure_cached_web_image(url: str, max_bytes: int) -> tuple[str, Path]:
    """
    Validate URL, ensure it's downloaded & cached, and return (fid, path).
    Shared by /webimg/fetch and legacy /rag/images/web-thumb.
    """
    if not safe_url(url):
        raise HTTPException(status_code=400, detail="invalid_url")

    fid = file_id_url(url)
    outp = img_path(fid)

    if outp.exists():
        return fid, outp

    # Build headers that pass simple hotlink checks
    try:
        from urllib.parse import urlparse
        u = urlparse(url)
        referer = f"{u.scheme}://{u.netloc}/"
    except Exception:
        referer = None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    if referer:
        headers["Referer"] = referer

    try:
        # Stream with redirect and size guard
        with requests.get(
            url,
            headers=headers,
            timeout=10,
            stream=True,
            allow_redirects=True,
        ) as r:
            r.raise_for_status()
            raw = BytesIO()
            total = 0
            for chunk in r.iter_content(chunk_size=16_384):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(status_code=413, detail="image_too_large")
                raw.write(chunk)
            raw.seek(0)

            im = Image.open(raw).convert("RGB")
            outp.parent.mkdir(parents=True, exist_ok=True)
            im.save(outp, format="JPEG", quality=90)

            # Optional caption sidecar
            try:
                cap = caption_image(im)
                cap = (cap or "").strip()
                if cap:
                    (outp.parent / f"{fid}.caption.txt").write_text(
                        cap, encoding="utf-8"
                    )
            except Exception:
                # Caption failures should not break fetch
                pass

    except HTTPException:
        raise
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", None)
        raise HTTPException(
            status_code=502,
            detail=f"download_failed:HTTPError:{code}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"download_failed:{type(e).__name__}",
        )

    return fid, outp
    
# ---------------------------------------------------------------------------
# 1) WEB IMAGE FETCH & THUMBNAILS  (/api/v1/webimg/...)
# ---------------------------------------------------------------------------

@router.get("/api/v1/webimg/fetch")
def fetch_and_cache_web_image(
    url: str = Query(..., description="HTTP/HTTPS image URL"),
    w: int = Query(320, ge=1),
    max_bytes: int = Query(12_000_000, ge=1),
) -> Dict[str, Any]:
    """
    Download an image from the web and cache it locally.

    Behavior:
        - Validate URL (scheme + netloc).
        - Stream with size cap.
        - Decode via PIL, convert to RGB, save as JPEG.
        - Best-effort caption sidecar via Qwen captioner.
        - Return thumbnail URL pointing to /api/v1/webimg/thumb.
    """
    fid, outp = _ensure_cached_web_image(url=url, max_bytes=max_bytes)
    thumb_url = f"/api/v1/webimg/thumb?id={fid}&w={int(w)}"
    return {"ok": True, "id": fid, "cached_path": str(outp), "thumbnail_url": thumb_url}

@router.get("/api/v1/rag/images/web-thumb")
def legacy_web_thumb(
    url: str = Query(..., description="HTTP/HTTPS image URL"),
    w: int = Query(320, ge=1),
    max_bytes: int = Query(12_000_000, ge=1),
):
    """
    Back-compat thumbnail endpoint used by fusion's web_fetch_link.

    Accepts a raw image URL, ensures it is cached, then streams a JPEG
    thumbnail at approximately width=w (preserving aspect ratio).
    """
    fid, outp = _ensure_cached_web_image(url=url, max_bytes=max_bytes)

    # Reuse same resizing behavior as /api/v1/webimg/thumb, but by URL
    try:
        im = Image.open(outp).convert("RGB")
        if w:
            ar = im.height / im.width
            im = im.resize((int(w), max(1, int(round(w * ar)))))
        return _pil_to_jpeg_response(im, quality=85)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"thumb_failed:{type(e).__name__}")

@router.get("/api/v1/webimg/thumb")
def serve_web_image_thumb(
    id: str = Query(...),
    w: int = Query(320, ge=1),
    h: int | None = Query(default=None, ge=1),
):
    """
    Serve a cached web image by ID, optionally resized.

    Args:
        id: cache key derived from the URL
        w/h: requested size; preserves aspect if one dimension omitted.
    """
    p = img_path(id)
    if not p.exists():
        raise HTTPException(status_code=404, detail="not_found")

    try:
        im = Image.open(p).convert("RGB")
        if w or h:
            if w and h:
                im = im.resize((int(w), int(h)))
            elif w and not h:
                ar = im.height / im.width
                im = im.resize((int(w), max(1, int(round(w * ar)))))
            elif h and not w:
                ar = im.width / im.height
                im = im.resize((max(1, int(round(h * ar))), int(h)))
        return _pil_to_jpeg_response(im, quality=85)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"thumb_failed:{type(e).__name__}")


# ---------------------------------------------------------------------------
# 2) RAG PAGE IMAGES  (/api/v1/rag/pages/...)
# ---------------------------------------------------------------------------

@router.get("/api/v1/rag/pages/status")
def pages_status() -> Dict[str, Any]:
    """
    Lists known doc_hashes and page ranges under page_images_dir.

    This is similar to the old /status endpoint in images.py.
    """
    try:
        page_root = Path(getattr(settings, "page_images_dir", "./data/page_images")).expanduser()
    except Exception:
        page_root = Path("./data/page_images").expanduser()

    out: List[Dict[str, Any]] = []
    if page_root.exists():
        for d in sorted([p for p in page_root.iterdir() if p.is_dir()]):
            doc_hash = d.name
            pages = sorted(glob.glob(str(d / "page_*.jpg")))
            if pages:
                m = re.findall(r"page_(\d+)\.jpg$", pages[-1])
                max_idx = int(m[0]) if m else (len(pages) - 1)
                out.append({"doc_hash": doc_hash, "pages": {"min": 0, "max": max_idx}})

    return {"ok": True, "docs": out, "root": str(page_root)}


@router.get("/api/v1/rag/pages/get-image")
def get_page_image(
    doc: str = Query(..., description="doc_hash directory under page_images/"),
    page: int = Query(..., ge=0, description="0-based page index"),
    w: int | None = Query(default=320, ge=1, le=4096),
    h: int | None = Query(default=None, ge=1, le=4096),
):
    """
    Return a (optionally resized) page image.

    - doc: doc_hash directory under page_images/
    - page: 0-based page index
    - w/h: optional target size; preserves aspect if one dim is omitted
    """
    try:
        root = Path(settings.page_images_dir)
        img_path = resolve_page_image_path(doc, page, root)
        im = Image.open(img_path).convert("RGB")
        if w or h:
            if w and h:
                im = im.resize((int(w), int(h)))
            elif w and not h:
                ar = im.height / im.width
                im = im.resize((int(w), max(1, int(round(w * ar)))))
            elif h and not w:
                ar = im.width / im.height
                im = im.resize((max(1, int(round(h * ar))), int(h)))
        return _pil_to_jpeg_response(im)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"render_failed:{type(e).__name__}")


@router.get("/api/v1/rag/pages/get-image-region")
def get_page_image_region(
    doc: str = Query(...),
    page: int = Query(..., ge=0),
    x: int = Query(ge=0),
    y: int = Query(ge=0),
    w: int = Query(gt=0),
    h: int = Query(gt=0),
    out_w: int | None = Query(default=None, ge=1, le=4096),
    out_h: int | None = Query(default=None, ge=1, le=4096),
):
    """
    Return a cropped region from a page image; optionally scaled to (out_w, out_h).
    All coordinates in pixels on the full page image.
    """
    try:
        root = Path(settings.page_images_dir)
        img_path = resolve_page_image_path(doc, page, root)
        im = Image.open(img_path).convert("RGB")

        # Clamp crop to image bounds
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(im.width, x0 + w), min(im.height, y0 + h)
        if x1 <= x0 or y1 <= y0:
            raise HTTPException(status_code=400, detail="invalid_crop_bounds")

        region = im.crop((x0, y0, x1, y1))
        if out_w and out_h:
            region = region.resize((int(out_w), int(out_h)))
        return _pil_to_jpeg_response(region)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"crop_failed:{type(e).__name__}")


@router.post("/api/v1/rag/pages/build")
def build_pages(
    dpi: int = Query(150, ge=72, le=600),
    limit: int | None = Query(
        default=None,
        ge=1,
        description="Limit # of PDFs for debug builds",
    ),
):
    """
    Render PDFs under settings.pdf_dir to JPEGs under settings.page_images_dir.
    Delegates to services.images.render_pdf_pages.
    """
    return render_pdf_pages(
        pdf_dir=Path(settings.pdf_dir),
        out_dir=Path(settings.page_images_dir),
        dpi=dpi,
        limit=limit,
    )


@router.get("/api/v1/rag/pages/list")
def list_pages():
    """
    List discovered page images (capped for payload safety).

    Delegates to services.images.list_page_images.
    """
    pages = list_page_images(Path(settings.page_images_dir))
    return {"ok": True, "count": len(pages), "pages": pages}


# ---------------------------------------------------------------------------
# 3) IMAGE→RAG INDEX & SEARCH  (/api/v1/rag/image/...)
# ---------------------------------------------------------------------------

@router.post("/api/v1/rag/image/index/build")
def build_image_faiss():
    """
    Build or rebuild the CLIP image index over page images.

    Delegates to services.images.build_image_index.
    """
    return build_image_index(
        page_root=Path(settings.page_images_dir),
        index_dir=Path(settings.image_index_dir),
        model_name=settings.image_embed_model_name,
        batch_size=32,
    )


@router.post("/api/v1/rag/image/search")
async def image_search(
    k: int = Query(default=5, ge=1, le=20),
    image: UploadFile = File(...),
):
    """
    Search the image index by uploaded image and attach safety info.

    Steps:
        - decode uploaded image
        - caption via services.images.caption_image
        - run safety analyzer over caption
        - search image neighbors via services.images.search_similar_images
        - attach caption + warnings to response
    """
    # Decode image
    raw = await image.read()
    try:
        img = load_image_from_bytes(raw)
        img = ensure_rgb(img)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid_image")

    # Caption for safety / UX
    try:
        cap = caption_image(img)
    except Exception:
        cap = ""

    safety = analyze_snippets([cap])
    if hasattr(safety, "model_dump"):
        warnings = safety.model_dump().get("warnings", [])
    elif isinstance(safety, dict):
        warnings = safety.get("warnings", [])
    else:
        warnings = []

    # Search neighbors
    try:
        hits = search_similar_images(img=img, top_k=k)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"image_search_failed:{type(e).__name__}"
        )

    # Adapt hits (List[ImgEntry]) → plain dicts for JSON
    results: List[Dict[str, Any]] = []
    for h in hits or []:
        meta = getattr(h, "meta", None)
        results.append(
            {
                "score": float(getattr(h, "score", 0.0)) if hasattr(h, "score") else None,
                "doc_hash": getattr(meta, "doc_hash", None) if meta else None,
                "page_index": getattr(meta, "page_index", None) if meta else None,
                "pdf_path": str(getattr(meta, "pdf_path", "")) if meta else None,
                "image_path": str(getattr(meta, "image_path", "")) if meta else None,
            }
        )

    return {
        "ok": True,
        "k": k,
        "results": results,
        "query_caption": cap,
        "warnings": warnings,
    }
