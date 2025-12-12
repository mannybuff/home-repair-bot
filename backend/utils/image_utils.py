# app/utils/image_utils.py

"""
IMAGE UTILITY HELPERS

This module contains low-level, reusable helpers for working with images:

    - Loading / decoding from bytes
    - Converting to RGB
    - Resizing / thumbnail generation (preserving aspect)
    - Computing stable image hashes
    - Generic L2-normalization for embedding matrices
    - Scanning page-images directories for existing JPEGs

Design notes:
    - Keep functions here pure or very close to pure (no session writes).
    - No HTTP calls; only local file / PIL / numpy operations.
    - Higher-level orchestration belongs in:
        * services/images.py
        * services/search.py
        * services/rag.py
"""

from __future__ import annotations

from urllib.parse import urlparse
from PIL import Image
import numpy as np

import io, hashlib, requests, re
from pathlib import Path

from app.core.settings import settings

# ---------------------------------------------------------------------------
# BASIC IMAGE LOADING / CONVERSION
# ---------------------------------------------------------------------------

def load_image_from_bytes(data: bytes) -> Image.Image:
    """
    Load a PIL Image from raw bytes.

    - Raises IOError / OSError if the bytes cannot be decoded as an image.
    - Does not modify the original mode; combine with ensure_rgb() if needed.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("load_image_from_bytes expects bytes or bytearray")
    bio = io.BytesIO(data)
    img = Image.open(bio)
    # Defer loading until needed, but ensure the file is closed
    img.load()
    return img


def ensure_rgb(img: Image.Image) -> Image.Image:
    """
    Ensure the image is in RGB mode.

    - If already 'RGB', returns the same instance.
    - Otherwise returns a converted copy.
    """
    if img.mode == "RGB":
        return img
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# MIGRATED UTILS FROM WEB FILES (needs usage and import verification)
# ---------------------------------------------------------------------------

def extract_og_image(html: str) -> str | None:
    """
    Very small OG parser to find <meta property="og:image" content="...">.
    Returns absolute/relative URL string or None if not found.
    """
    # Case-insensitive, tolerant
    m = re.search(r'<meta[^>]+property=["\']og:image["\'][^>]*>', html, re.IGNORECASE)
    if not m:
        return None
    tag = m.group(0)
    m2 = re.search(r'content=["\']([^"\']+)["\']', tag, re.IGNORECASE)
    if not m2:
        return None
    return m2.group(1).strip() or None

def peek_page_image_url(page_url: str, timeout: float = 3.5) -> str | None:
    """
    Downloads the HTML of a page quickly and returns og:image URL if present.
    Does not download the image itself.
    """
    try:
        r = requests.get(page_url, timeout=timeout, headers={"User-Agent": "vision-rag/1.0"})
        r.raise_for_status()
        return extract_og_image(r.text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# RESIZING / THUMBNAILS
# ---------------------------------------------------------------------------

def resize_preserving_aspect(
    img: Image.Image,
    max_size: Tuple[int, int],
) -> Image.Image:
    """
    Resize an image to fit within max_size (width, height) preserving aspect.

    - Does not enlarge images that are already smaller than max_size.
    - Returns a new Image instance.
    """
    img = ensure_rgb(img)
    max_w, max_h = max_size
    w, h = img.size
    if w <= max_w and h <= max_h:
        return img.copy()

    # Compute scale factor
    scale = min(max_w / float(w), max_h / float(h))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


# ---------------------------------------------------------------------------
# HASHING HELPERS
# ---------------------------------------------------------------------------

def compute_image_sha1(img: Image.Image) -> str:
    """
    Compute a simple SHA1 hash over the raw RGB bytes of an image.

    Useful for:
        - detecting duplicates
        - building cache keys for thumbnails
    """
    img = ensure_rgb(img)
    data = img.tobytes()
    return hashlib.sha1(data).hexdigest()

# ---------------------------------------------------------------------------
# WEB-IMAGE HELPERS
# ---------------------------------------------------------------------------

# Base directory for page JPGs (used by some legacy helpers)
DATA = Path(getattr(settings, "page_images_dir", "./data/page_images")).expanduser()


def page_jpg_path(doc_hash: str, page_index: int) -> Path:
    """
    Resolve a page JPG path under the page_images directory.
    """
    # zero-pad like page_0026.jpg (1-based for this helper)
    return DATA / doc_hash / f"page_{page_index + 1:04d}.jpg"


def cache_root() -> Path:
    """
    Root directory for cached web images used by /api/v1/webimg/*.
    """
    p = Path(getattr(settings, "data_dir", "./data")).expanduser() / "web_images"
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_url(u: str) -> bool:
    try:
        p = urlparse(u)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def file_id_url(u: str) -> str:
    """
    Stable ID for a web image URL (SHA1 hex digest).
    """
    return hashlib.sha1(u.encode("utf-8")).hexdigest()


def img_path(fid: str) -> Path:
    """
    Local cache path for a web image.
    """
    return cache_root() / f"{fid}.jpg"

# ---------------------------------------------------------------------------
# EMBEDDING NORMALIZATION
# ---------------------------------------------------------------------------

def l2_normalize(mat: np.ndarray) -> np.ndarray:
    """
    L2-normalize a 2D numpy array along axis=1.

    This is the canonical replacement for the old `_l2_normalize` helper
    used in the image indexer. Call sites can import as:

        from app.utils.image_utils import l2_normalize as _l2_normalize
    """
    if mat.ndim != 2:
        raise ValueError("l2_normalize expects a 2D matrix")
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    return mat / norms


# ---------------------------------------------------------------------------
# PAGE-IMAGE DIRECTORY HELPERS
# ---------------------------------------------------------------------------

def find_page_images(page_root: Path) -> List[Path]:
    """
    Scan the page images directory for JPEGs:

        page_root/<doc_hash>/page_XXXX.jpg

    This is the canonical replacement for the old `_find_page_images`
    helper in the image indexer. Call sites can import as:

        from app.utils.image_utils import find_page_images as _find_page_images
    """
    if not page_root.exists():
        return []
    out: List[Path] = []
    for docdir in sorted(page_root.glob("*")):
        if not docdir.is_dir():
            continue
        for jpg in sorted(docdir.glob("page_*.jpg")):
            out.append(jpg)
    return out

# ---------------------------------------------------------------------------
# WEB PREVIEW RANKING HELPERS (migrated from api/fusion.py)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, Any] = {"name": None, "obj": None}


def get_clip_model(model_name: str):
    """
    Lazy-load a SentenceTransformer CLIP model once per process.
    Fail-soft: returns None on import/load errors.
    """
    try:
        if _MODEL_CACHE["name"] == model_name and _MODEL_CACHE["obj"] is not None:
            return _MODEL_CACHE["obj"]
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model_name)
        _MODEL_CACHE["name"] = model_name
        _MODEL_CACHE["obj"] = m
        return m
    except Exception:
        return None


def embed_pil_with_model(img: Image.Image, model_name: str) -> Optional[np.ndarray]:
    """
    Embed a PIL image using a CLIP-like SentenceTransformer model.
    Returns a 1D numpy array or None on failure.
    """
    try:
        m = get_clip_model(model_name)
        if m is None:
            return None
        v = m.encode([img], convert_to_numpy=True, normalize_embeddings=True)
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[0] >= 1:
            return v[0]
        return None
    except Exception:
        return None


def cosine_similarity(a: np.ndarray | None, b: np.ndarray | None) -> Optional[float]:
    """
    Cosine similarity between two 1D vectors; returns None if invalid.
    """
    try:
        if a is None or b is None:
            return None
        return float(np.clip((a * b).sum(), -1.0, 1.0))
    except Exception:
        return None


def download_preview_image(url: str, timeout: float = 6.0, max_bytes: int = 6_000_000) -> Optional[Image.Image]:
    """
    Small downloader for ranking only (no disk writes).
    Browser-like headers + size guard; returns PIL.Image or None.
    """
    try:
        u = urlparse(url)
        referer = f"{u.scheme}://{u.netloc}/" if u.scheme and u.netloc else None
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        }
        if referer:
            headers["Referer"] = referer

        with requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True) as r:
            r.raise_for_status()
            raw = io.BytesIO()
            total = 0
            for chunk in r.iter_content(16384):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    return None
                raw.write(chunk)
            raw.seek(0)
        return Image.open(raw).convert("RGB")
    except Exception:
        return None


def rank_web_by_image(
    web_items: list[dict],
    query_img: Optional[Image.Image],
    model_name: str,
) -> list[dict]:
    """
    If an uploaded image exists, compute cosine similarity to each web preview image
    (best-effort) and sort web items by similarity desc. Adds 'sim_to_query_image'.
    """
    if not query_img or not web_items:
        return web_items

    qv = embed_pil_with_model(query_img, model_name=model_name)
    if qv is None:
        return web_items

    ranked = []
    for r in web_items:
        sim = None
        try:
            url = r.get("preview_image_url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                pim = download_preview_image(url)
                if pim is not None:
                    rv = embed_pil_with_model(pim, model_name=model_name)
                    sim = cosine_similarity(qv, rv)
        except Exception:
            sim = None

        r2 = dict(r)
        if sim is not None:
            r2["sim_to_query_image"] = sim
        ranked.append(r2)

    with_sim = [x for x in ranked if x.get("sim_to_query_image") is not None]
    without = [x for x in ranked if x.get("sim_to_query_image") is None]
    with_sim.sort(key=lambda x: x["sim_to_query_image"], reverse=True)
    return with_sim + without
