# app/services/images.py
"""
IMAGE SERVICE LAYER

This module centralizes all *service-level* image behavior:

    - Running image captioning via Qwen2-VL
    - Building and querying a local FAISS image index of page images
    - PDF → image pipelines (render, list, fetch)
    - Lightweight helpers for page-image resolution

Heavy lifting is delegated to:
    - utils/image_utils.py
    - models/images_class.py
    - rag_utils/common for registry + safety of paths

Routers that depend on this module:
    - app/api/image_routes.py

Design goals:
    - No FastAPI request/response types here.
    - Accept and return plain Python objects (PIL Images, dicts, dataclasses).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pathlib import Path
from dataclasses import asdict
import json
import hashlib

import numpy as np
from PIL import Image

# Optional deps: we guard these so that import-time errors don't kill the app
try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover
    SentenceTransformer = None  # type: ignore

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None  # type: ignore

try:
    from pdf2image import convert_from_path
except Exception:  # pragma: no cover
    convert_from_path = None  # type: ignore

from app.core.settings import settings
from app.models.images_class import ImgMeta, ImgEntry, PageImage
from app.utils.common import ensure_dir
from app.utils.image_utils import (
    load_image_from_bytes,
    ensure_rgb,
    find_page_images,
    l2_normalize,
)
from app.utils.rag_utils import (
    safe_under_allowed_roots,
    load_pdf_to_doc_hash_registry,
    lookup_doc_hash_for_pdf,
)
from app.utils.image_utils import ensure_rgb
from app.services.qwen_use import get_qwen_captioner as get_captioner

# ---------------------------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------------------------


def _hash_path(p: Path) -> str:
    """
    Small helper to compute a stable document hash from a PDF path.
    """
    return hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:16]


def _index_paths(index_dir: Path) -> tuple[Path, Path, Path]:
    """
    Given an index_dir, return (index_path, meta_path, norm_path).
    """
    index_dir = ensure_dir(index_dir.expanduser())
    index_path = index_dir / "img.index.faiss"
    meta_path = index_dir / "img.index_meta.json"
    norm_path = index_dir / "img.index_norm.npy"
    return index_path, meta_path, norm_path


def _load_image_index(
    index_dir: Path,
) -> tuple[Any | None, Dict[str, Any] | None, bool]:
    """
    Load FAISS index + metadata JSON + norm flag.
    Returns (index | None, meta_dict | None, use_norm: bool).
    """
    index_path, meta_path, norm_path = _index_paths(index_dir)
    if not (index_path.exists() and meta_path.exists()):
        return None, None, False

    if faiss is None:
        return None, None, False

    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    use_norm = False
    if norm_path.exists():
        arr = np.load(norm_path)
        use_norm = bool(arr[0])
    return index, meta, use_norm

# ---------------------------------------------------------------------------
# IMAGE CAPTIONING (Qwen2-VL)
# ---------------------------------------------------------------------------

def caption_image(
    img: Image.Image,
    prompt: Optional[str] = None,
    use_safe_mode: bool = True,
) -> str:
    """
    Run image captioning via the configured VLM model (Qwen2-VL).

    Args:
        img: PIL Image (any mode; converted to RGB here)
        prompt: Optional textual hint
        use_safe_mode: kept for API compatibility; currently unused by the
                       underlying captioner. Safety is handled by separate
                       safety services.

    Returns:
        caption: str (may be empty if captioning fails)
    """
    try:
        img = ensure_rgb(img)
        captioner = get_captioner()
        # NOTE:
        #   The new captioner interface does NOT accept a 'safe' kwarg.
        #   Safety is handled by downstream safety analysis, not here.
        caption = captioner.caption(img, prompt=prompt)
        return (caption or "").strip()
    except Exception:
        # Any caption failure should degrade gracefully to empty string.
        return ""
        
# ---------------------------------------------------------------------------
# IMAGE EMBEDDING / INDEXING
# ---------------------------------------------------------------------------


def build_image_index(
    page_root: Path,
    index_dir: Path,
    model_name: Optional[str] = None,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Build (or rebuild) the FAISS index over rendered page images.

    This implementation matches the expectations of:
        /api/v1/rag/image/index/build in app/api/image_routes.py

    Args:
        page_root: root directory of page images (settings.page_images_dir)
        index_dir: directory where index + meta + norm artifacts are written
        model_name: CLIP-like SentenceTransformer model name to use
        batch_size: number of images per embedding batch

    Returns:
        Summary dict with counts and paths.
    """
    page_root = page_root.expanduser()
    index_dir = index_dir.expanduser()
    index_path, meta_path, norm_path = _index_paths(index_dir)

    if model_name is None:
        model_name = getattr(settings, "image_embed_model_name", "")

    # Discover page images
    imgs = find_page_images(page_root)
    if not imgs:
        # Clean up any stale index
        if index_path.exists():
            index_path.unlink()
        meta = {"embed_model": model_name, "dim": 0, "entries": []}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        np.save(norm_path, np.array([True], dtype=bool))
        return {
            "ok": True,
            "images": 0,
            "dim": 0,
            "note": "No page images found.",
            "index_path": str(index_path),
            "meta_path": str(meta_path),
        }

    # Registry: doc_hash → pdf_path (if available)
    try:
        registry = load_pdf_to_doc_hash_registry(page_root) or {}
    except Exception:
        registry = {}

    # Build metadata list
    metas: List[ImgMeta] = []
    for p in imgs:
        doc_hash = p.parent.name
        try:
            # Expect names like page_0000.jpg (0-based), but tolerate others
            stem = p.stem  # e.g. "page_0000"
            page_str = stem.split("_", 1)[1]
            page_index = int(page_str)
        except Exception:
            page_index = 0

        reg_val = registry.get(doc_hash)
        pdf_path_str = ""
        if isinstance(reg_val, dict):
            pdf_path_str = reg_val.get("pdf_path", "") or ""
        elif isinstance(reg_val, str):
            pdf_path_str = reg_val

        pdf_path = Path(pdf_path_str) if pdf_path_str else Path("")

        metas.append(
            ImgMeta(
                doc_hash=doc_hash,
                page_index=page_index,
                pdf_path=pdf_path,
                image_path=p,
            )
        )

    # Guard: embeddings backend must be available
    if SentenceTransformer is None or faiss is None:
        return {
            "ok": False,
            "reason": "sentence-transformers-or-faiss-not-available",
            "images": len(metas),
        }

    encoder = SentenceTransformer(model_name)
    dim = encoder.get_sentence_embedding_dimension()

    # Encode images in batches
    embs_all: List[np.ndarray] = []
    for i in range(0, len(metas), batch_size):
        batch_metas = metas[i : i + batch_size]
        pil_batch: List[Image.Image] = []
        for m in batch_metas:
            try:
                im = Image.open(m.image_path).convert("RGB")
            except Exception:
                # fallback placeholder
                im = Image.new("RGB", (224, 224), color=(200, 200, 200))
            pil_batch.append(im)
        try:
            vecs = encoder.encode(
                pil_batch,
                batch_size=len(pil_batch),
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as e:
            # Skip this batch if embedding fails; record error
            continue

        if not isinstance(vecs, np.ndarray):
            vecs = np.array(vecs)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        embs_all.append(vecs.astype(np.float32))

    if not embs_all:
        # Nothing embedded successfully
        meta = {"embed_model": model_name, "dim": 0, "entries": []}
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if index_path.exists():
            index_path.unlink()
        np.save(norm_path, np.array([True], dtype=bool))
        return {
            "ok": False,
            "reason": "all-embeddings-failed",
            "images": len(metas),
        }

    embs = np.vstack(embs_all)
    embs = l2_normalize(embs)  # cosine via inner product

    # Build FAISS index
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)
    faiss.write_index(index, str(index_path))

    # Persist metadata & norm flag
    meta_dict = {
        "embed_model": model_name,
        "dim": int(embs.shape[1]),
        "entries": [
            {
                "doc_hash": m.doc_hash,
                "page_index": m.page_index,
                "pdf_path": str(m.pdf_path) if m.pdf_path else "",
                "image_path": str(m.image_path),
            }
            for m in metas
        ],
    }
    meta_path.write_text(json.dumps(meta_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(norm_path, np.array([True], dtype=bool))

    return {
        "ok": True,
        "images": len(metas),
        "dim": meta_dict["dim"],
        "index_path": str(index_path),
        "meta_path": str(meta_path),
    }


def search_similar_images(
    img: Image.Image,
    top_k: int = 5,
) -> List[ImgEntry]:
    """
    Search the image index using a PIL Image query.

    This implementation matches image_routes.image_search, which expects:
        - a list of ImgEntry objects
        - each entry having .meta (ImgMeta)
        - an optional .score attribute used by the router when serializing.

    Returns:
        List[ImgEntry]; empty if the index is missing or embedding fails.
    """
    index_dir = Path(getattr(settings, "image_index_dir", "./data/image_index")).expanduser()
    index, meta_dict, use_norm = _load_image_index(index_dir)
    if index is None or not meta_dict:
        return []

    model_name = meta_dict.get("embed_model") or getattr(
        settings, "image_embed_model_name", ""
    )
    if not model_name or SentenceTransformer is None:
        return []

    encoder = SentenceTransformer(model_name)

    img = ensure_rgb(img)
    q = encoder.encode([img], convert_to_numpy=True)
    if not isinstance(q, np.ndarray):
        q = np.array(q)
    if q.ndim == 1:
        q = q.reshape(1, -1)
    q = q.astype(np.float32)
    if use_norm:
        q = l2_normalize(q)

    D, I = index.search(q, top_k)
    scores = D[0].tolist()
    idxs = I[0].tolist()

    entries_meta = meta_dict.get("entries", [])
    dim = int(meta_dict.get("dim") or 0)

    results: List[ImgEntry] = []
    for score, idx in zip(scores, idxs):
        if idx < 0 or idx >= len(entries_meta):
            continue
        m = entries_meta[idx]
        meta = ImgMeta(
            doc_hash=m.get("doc_hash", ""),
            page_index=int(m.get("page_index", -1)),
            pdf_path=Path(m.get("pdf_path", "")) if m.get("pdf_path") else Path(""),
            image_path=Path(m.get("image_path", "")),
        )
        # We don't actually need the embedding on the Python side after indexing;
        # store a placeholder vector with the correct dimensionality.
        emb = np.zeros((dim,), dtype=np.float32) if dim > 0 else np.zeros((1,), dtype=np.float32)
        entry = ImgEntry(embedding=emb, meta=meta)
        # Attach score as a dynamic attribute so image_routes can read it.
        setattr(entry, "score", float(score))
        results.append(entry)

    return results


# ---------------------------------------------------------------------------
# PDF → IMAGE PIPELINE (rendering)
# ---------------------------------------------------------------------------


def render_pdf_pages(
    pdf_dir: Path,
    out_dir: Path,
    dpi: int = 150,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Render PDFs under pdf_dir to JPEGs under out_dir.

    This implementation matches the /api/v1/rag/pages/build route which
    calls:

        render_pdf_pages(
            pdf_dir=Path(settings.pdf_dir),
            out_dir=Path(settings.page_images_dir),
            dpi=dpi,
            limit=limit,
        )

    A registry.jsonl file is written under out_dir with lines:
        {"doc_hash": "...", "pdf_path": "..."}
    """
    out_dir = ensure_dir(out_dir.expanduser())
    pdf_dir = pdf_dir.expanduser()

    if convert_from_path is None:
        return {"ok": False, "reason": "pdf2image-not-available"}

    pdfs = sorted(pdf_dir.glob("**/*.pdf"))
    if limit is not None:
        pdfs = pdfs[:limit]

    pages: List[PageImage] = []
    skipped: List[str] = []
    errors: List[str] = []

    registry_path = out_dir / "registry.jsonl"
    seen_hashes: set[str] = set()

    with registry_path.open("a", encoding="utf-8") as regf:
        for pdf in pdfs:
            try:
                doc_hash = _hash_path(pdf)
                if doc_hash not in seen_hashes:
                    regf.write(
                        json.dumps(
                            {"doc_hash": doc_hash, "pdf_path": str(pdf.resolve())}
                        )
                        + "\n"
                    )
                    seen_hashes.add(doc_hash)

                doc_dir = ensure_dir(out_dir / doc_hash)
                imgs = convert_from_path(str(pdf), dpi=dpi)
                for i, img in enumerate(imgs):
                    name = f"page_{i:04d}.jpg"  # 0-based index in filename
                    outp = doc_dir / name
                    img.save(str(outp), format="JPEG", quality=90)
                    pages.append(
                        PageImage(
                            doc_hash=doc_hash,
                            page_index=i,
                            image_path=outp.resolve(),
                        )
                    )
            except Exception as e:
                skipped.append(str(pdf.resolve()))
                errors.append(f"{pdf.name}: {e!r}")
                continue

    return {
        "ok": True,
        "count": len(pages),
        "pages": [asdict(p) for p in pages[:200]],
        "skipped": skipped,
        "errors": errors,
        "note": "Preview capped to 200 pages; full set is on disk.",
    }


def list_page_images(root: Path) -> List[Dict[str, Any]]:
    """
    List discovered page images under the given root.

    Matches image_routes.list_pages, which expects a simple list[dict].
    """
    root = root.expanduser()
    if not root.exists():
        return []

    out: List[Dict[str, Any]] = []
    for doc_dir in sorted(root.glob("*")):
        if not doc_dir.is_dir():
            continue
        for jpg in sorted(doc_dir.glob("page_*.jpg")):
            try:
                page_str = jpg.stem.split("_", 1)[1]
                page_idx = int(page_str)
            except Exception:
                page_idx = -1
            out.append(
                {
                    "doc_hash": doc_dir.name,
                    "page_index": page_idx,
                    "image_path": str(jpg.resolve()),
                }
            )
            if len(out) >= 1000:
                return out
    return out


# ---------------------------------------------------------------------------
# PAGE-IMAGE PATH RESOLUTION
# ---------------------------------------------------------------------------


def resolve_page_image_path(doc_hash: str, page_index: int, root: Path) -> Path:
    """
    Resolve absolute path for the image of a specific page.

    Matches image_routes.get_page_image / get_page_image_region expectations.
    """
    root = root.expanduser()
    img_path = root / doc_hash / f"page_{page_index:04d}.jpg"
    if not img_path.exists():
        raise FileNotFoundError(
            f"Page image not found for doc_hash={doc_hash}, page_index={page_index}"
        )
    return img_path


# ---------------------------------------------------------------------------
# OPTIONAL: PDF RESOLUTION HELPERS (mirroring old sources.py behavior)
# ---------------------------------------------------------------------------


def resolve_page_image_by_doc_hash(
    doc_hash: str,
    page: int,
    page_root: Optional[Path] = None,
) -> Path:
    """
    Resolve the page image path when doc_hash is already known.
    """
    if page_root is None:
        page_root = Path(
            getattr(settings, "page_images_dir", "./data/page_images")
        ).expanduser()
    return resolve_page_image_path(doc_hash=doc_hash, page_index=page, root=page_root)


def resolve_page_image_by_pdf_path(
    pdf_path: str,
    page: int,
    page_root: Optional[Path] = None,
) -> Path:
    """
    Resolve the page image path when the caller provides a pdf_path.
    """
    if page_root is None:
        page_root = Path(
            getattr(settings, "page_images_dir", "./data/page_images")
        ).expanduser()

    if not pdf_path:
        raise ValueError("pdf_path is required")

    pdf_abs = Path(pdf_path).resolve()
    if not safe_under_allowed_roots(pdf_abs):
        raise PermissionError("pdf_path is not under the allowed roots")

    doc_hash = lookup_doc_hash_for_pdf(pdf_abs, page_root)
    if not doc_hash:
        raise FileNotFoundError("Document not found in registry")

    return resolve_page_image_path(doc_hash=doc_hash, page_index=page, root=page_root)


def safe_resolve_pdf(pdf_path: str) -> Path:
    """
    Resolve and validate a PDF path for internal use.
    """
    if not pdf_path:
        raise ValueError("pdf_path is required")

    pdf_abs = Path(pdf_path).resolve()
    if not safe_under_allowed_roots(pdf_abs):
        raise PermissionError("pdf_path is not under the allowed roots")

    if not pdf_abs.exists() or pdf_abs.suffix.lower() != ".pdf":
        raise FileNotFoundError("PDF not found")

    return pdf_abs

