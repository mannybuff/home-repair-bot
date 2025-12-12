# app/services/rag.py
"""
RAG SERVICE LAYER

This module centralizes all local document-retrieval behavior.

It provides:
    - Text-based RAG query execution
    - Page-level metadata normalization
    - Result shaping (score, snippet, provenance)
    - Integration hooks for evidence building and fusion

High-level design:
    - utils/rag_utils.py  → parsing, provenance, normalization helpers
    - services/rag.py     → orchestration of RAG operations
"""

from __future__ import annotations

from typing import Any, Dict, List, Union
from pathlib import Path
from datetime import datetime
import json

import numpy as np

from app.models.rag_class import RagPaths, RagMeta, ChunkRecord
from app.utils.image_utils import l2_normalize

from app.services.safety import analyze_snippets

from app.core.settings import settings
from app.utils.common import jsonable as _jsonable, load_registry
from app.utils.rag_utils import (
    normalize_rag_hit,
    build_page_map,
    nearby_chunks,
    build_registry_from_pdfs,
)

# ---------------------------------------------------------------------------
# HIGH-LEVEL RAG ENTRYPOINTS (for fusion, synthesis, etc.)
# ---------------------------------------------------------------------------


def run_rag_search(
    query: str,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    High-level RAG search entrypoint for backend internals (fusion, synthesis).

    Responsibilities:
        - call the low-level FAISS search (rag_index_search)
        - normalize each hit into a stable schema
        - return a small, JSON-safe dict

    NOTE:
        This function does *not* perform safety analysis or session writes.
        Those concerns are layered in higher orchestration layers.
    """
    base = rag_index_search(
        query=query,
        data_dir=Path(settings.data_dir),
        top_k=top_k,
    )

    if not base.get("ok"):
        # bubble up the existing error/shape
        return base

    raw_results = list(base.get("results") or [])
    norm_results: List[Dict[str, Any]] = [normalize_rag_hit(r) for r in raw_results]

    out: Dict[str, Any] = {
        **_jsonable(base),
        "results": norm_results,
        "k": top_k,
    }
    return out


def fuse_image_hits_with_text(
    image_hits: List[Dict[str, Any]],
    data_dir: Path,
) -> Dict[str, Any]:
    """
    For each image neighbor:
        - ensure pdf_path is filled (via registry / fallback)
        - attach nearby text snippet(s)
        - run safety advisories over all snippets

    Never throws; always returns a dict with ok:True/False.
    """
    try:
        # 1) Load text meta (we only need meta for page → text mapping)
        _index, meta, _norm = load_index_and_meta(data_dir)
        if not meta:
            return {"ok": False, "reason": "RAG text index not built."}

        meta_d = {"chunks": [vars(c) for c in meta.chunks]}
        page_map = build_page_map(meta_d)

        # 2) Registry (doc_hash -> pdf_path)
        page_root = Path(settings.page_images_dir)
        pdf_dir = Path(settings.pdf_dir)

        try:
            reg = load_registry(page_root)
        except Exception:
            reg = {}

        if not reg:
            reg = build_registry_from_pdfs(pdf_dir)

        fused: List[Dict[str, Any]] = []
        all_snips: List[str] = []

        for e in image_hits:
            doc_hash = str(e.get("doc_hash", ""))
            pdf = e.get("pdf_path") or reg.get(doc_hash, "")
            page = int(e.get("page_index", 0))

            snips: List[str] = []
            if pdf:
                snips = nearby_chunks(page_map, pdf, page, window=1, max_return=2)

            all_snips.extend(snips)
            fused.append(
                {
                    **e,
                    "pdf_path": pdf,
                    "snippets": snips,
                }
            )

        # 3) Safety advisories over snippets
        safety_report = analyze_snippets(snippets=all_snips)

        # Normalize SafetyReport → dict
        if hasattr(safety_report, "model_dump"):  # pydantic v2
            safety_dict = safety_report.model_dump()
        elif hasattr(safety_report, "dict"):  # pydantic v1
            safety_dict = safety_report.dict()
        elif isinstance(safety_report, dict):
            safety_dict = safety_report
        else:
            safety_dict = {
                "per_item_flags": getattr(safety_report, "per_item_flags", []),
                "advisory": getattr(safety_report, "advisory", []),
                "blocked": bool(getattr(safety_report, "blocked", False)),
                "notes": getattr(safety_report, "notes", ""),
            }

        advisory = safety_dict.get("advisory") or []
        blocked = bool(safety_dict.get("blocked", False))
        safety_notes = safety_dict.get("notes") or ""

        warnings: List[str] = []
        if blocked:
            warnings.append("Safety: blocking advisory present.")
        if safety_notes:
            warnings.append(f"Safety notes: {safety_notes}")
        for w in advisory:
            if w not in warnings:
                warnings.append(w)

        return {"ok": True, "results": fused, "warnings": warnings}

    except Exception as e:  # pragma: no cover - defensive guardrail
        return {"ok": False, "reason": f"fusion-failed: {e!r}"}


# ---------------------------------------------------------------------------
# RAG INDEX: PDF → text chunks → embeddings → FAISS index
# ---------------------------------------------------------------------------


def _import_pypdf():
    """
    Import PdfReader on demand so that import-time failures do not
    prevent the app from starting.
    """
    try:
        from pypdf import PdfReader  # type: ignore

        return PdfReader
    except Exception as e:
        raise ImportError(f"pypdf import failed: {e!r}") from e


def _import_st():
    """
    Import SentenceTransformer lazily to avoid import-time crashes
    (e.g., missing torch or incompatible CUDA).
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        return SentenceTransformer
    except Exception as e:
        raise ImportError(f"SentenceTransformer import failed: {e!r}") from e


def _import_faiss():
    """
    Lazy import for faiss to keep import-time failures from breaking startup.
    """
    try:
        import faiss  # type: ignore

        return faiss
    except Exception as e:
        raise ImportError(f"faiss import failed: {e!r}") from e


def _extract_pdf_texts(pdf_path: Path) -> List[tuple[int, str]]:
    """
    Extract (page_index, text) pairs from a PDF.
    """
    PdfReader = _import_pypdf()
    reader = PdfReader(str(pdf_path))
    out: List[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        out.append((i, txt))
    return out


def _chunk_page_text(
    text: str,
    max_chars: int = 800,
    overlap: int = 80,
) -> List[str]:
    """
    Very simple fixed-window text chunker.
    """
    text = (text or "").strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def get_embedder(model_name: str):
    """
    Construct a SentenceTransformer model for embeddings on demand.

    We keep this in services so higher layers can re-use the same pattern.
    """
    SentenceTransformer = _import_st()
    return SentenceTransformer(model_name)


def build_faiss_index(embeddings: np.ndarray, use_ip: bool = True):
    """
    Build a FAISS index from a 2D embedding matrix.

    Args:
        embeddings: [N, D] matrix of float32 embeddings.
        use_ip: if True, use inner product with L2-normalized vectors
                (cosine similarity); else use L2.
    """
    faiss = _import_faiss()
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) if use_ip else faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index


# ---------------------------
# Build pipeline
# ---------------------------


def build_rag_index(
    pdf_dir: Union[Path, str],
    data_dir: Union[Path, str],
    embed_model_name: str,
    max_pdfs: int | None = None,
) -> Dict[str, Any]:
    """
    Walk pdf_dir, extract+chunk text, embed, build FAISS, persist artifacts to data_dir.

    Robustness features:
      - per-PDF try/except (skip bad files, record errors)
      - optional max_pdfs limit for debugging
      - batch embedding to avoid memory spikes

    Returns:
        Summary dict including counts, timestamps, and any errors.
    """
    pdf_dir = Path(pdf_dir)
    data_dir = Path(data_dir)
    paths = RagPaths.from_data_dir(data_dir)

    embedder = get_embedder(embed_model_name)

    pdfs = sorted(pdf_dir.glob("**/*.pdf"))
    if max_pdfs is not None:
        pdfs = pdfs[:max_pdfs]

    chunk_records: List[ChunkRecord] = []
    skipped: List[str] = []
    errors: List[str] = []

    # --- Extract + chunk ---
    for doc_id, pdf in enumerate(pdfs):
        try:
            page_texts = _extract_pdf_texts(pdf)  # may raise for malformed PDFs
            for (page_i, page_text) in page_texts:
                for piece in _chunk_page_text(page_text):
                    if piece.strip():
                        chunk_records.append(
                            ChunkRecord(
                                doc_id=doc_id,
                                file_path=str(pdf.resolve()),
                                page_from=page_i,
                                page_to=page_i,
                                text=piece,
                            )
                        )
        except Exception as e:
            skipped.append(str(pdf.resolve()))
            errors.append(f"{pdf.name}: {e!r}")
            continue

    # --- Handle empty corpus gracefully ---
    if not chunk_records:
        meta = RagMeta(
            embed_model=embed_model_name,
            dim=0,
            chunks=[],
            built_at=datetime.utcnow().isoformat(timespec="seconds"),
        )
        paths.meta_path.write_text(json.dumps(meta.to_json(), ensure_ascii=False, indent=2))
        # remove any stale index
        if paths.index_path.exists():
            paths.index_path.unlink()
        np.save(paths.norm_path, np.array([True], dtype=bool))
        return {
            "ok": True,
            "pdfs_scanned": len(pdfs),
            "chunks": 0,
            "skipped": skipped,
            "errors": errors,
            "note": "No text extracted.",
            "built_at": meta.built_at,
        }

    # --- Embed in batches to avoid large memory spikes ---
    texts = [c.text for c in chunk_records]
    batch_size = 32
    embs_all: List[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            embs = embedder.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as e:
            # if an embedding batch fails, record and skip that batch
            errors.append(f"embed-batch {i}-{i+len(batch)}: {e!r}")
            continue

        if not isinstance(embs, np.ndarray):
            embs = np.array(embs)
        if embs.ndim == 1:
            embs = embs.reshape(1, -1)

        embs_all.append(embs)

    # --- All embedding batches failed ---
    if not embs_all:
        meta = RagMeta(
            embed_model=embed_model_name,
            dim=0,
            chunks=[],
            built_at=datetime.utcnow().isoformat(timespec="seconds"),
        )
        paths.meta_path.write_text(json.dumps(meta.to_json(), ensure_ascii=False, indent=2))
        if paths.index_path.exists():
            paths.index_path.unlink()
        np.save(paths.norm_path, np.array([True], dtype=bool))
        return {
            "ok": False,
            "reason": "all-embeddings-failed",
            "pdfs_scanned": len(pdfs),
            "skipped": skipped,
            "errors": errors,
            "built_at": meta.built_at,
        }

    # --- Build FAISS + persist artifacts ---
    embs = np.vstack(embs_all)
    embs = l2_normalize(embs)  # cosine via IP

    index = build_faiss_index(embs, use_ip=True)
    faiss = _import_faiss()
    faiss.write_index(index, str(paths.index_path))

    meta = RagMeta(
        embed_model=embed_model_name,
        dim=int(embs.shape[1]),
        chunks=chunk_records,
        built_at=datetime.utcnow().isoformat(timespec="seconds"),
    )
    paths.meta_path.write_text(json.dumps(meta.to_json(), ensure_ascii=False, indent=2))
    np.save(paths.norm_path, np.array([True], dtype=bool))

    return {
        "ok": True,
        "pdfs_scanned": len(pdfs),
        "chunks": len(chunk_records),
        "dim": meta.dim,
        "skipped": skipped,
        "errors": errors,
        "built_at": meta.built_at,
    }


# ---------------------------
# Load & search
# ---------------------------


def load_index_and_meta(data_dir: Path):
    """
    Load FAISS index + RagMeta from data_dir.

    Returns:
        (index | None, meta | None, use_norm: bool)
    """
    data_dir = Path(data_dir)
    paths = RagPaths.from_data_dir(data_dir)
    if not (paths.index_path.exists() and paths.meta_path.exists()):
        return None, None, False

    faiss = _import_faiss()
    index = faiss.read_index(str(paths.index_path))

    meta_d = json.loads(paths.meta_path.read_text())
    chunks = [ChunkRecord(**c) for c in meta_d.get("chunks", [])]
    meta = RagMeta(
        embed_model=meta_d["embed_model"],
        dim=meta_d["dim"],
        chunks=chunks,
        built_at=meta_d.get("built_at")
        or datetime.fromtimestamp(paths.meta_path.stat().st_mtime).isoformat(
            timespec="seconds"
        ),
    )

    use_norm = False
    if paths.norm_path.exists():
        arr = np.load(paths.norm_path)
        use_norm = bool(arr[0])

    return index, meta, use_norm


def rag_index_search(query: str, data_dir: Path, top_k: int = 5) -> Dict[str, Any]:
    """
    Low-level RAG search over the FAISS index in data_dir.

    Returns raw hits in the shape expected by rag_utils.normalize_rag_hit:

        {
            "score": float,
            "file_path": str,
            "page_from": int,
            "page_to": int,
            "snippet": str,
        }
    """
    data_dir = Path(data_dir)
    index, meta, norm = load_index_and_meta(data_dir)
    if index is None or meta is None or not meta.chunks:
        return {"ok": False, "reason": "Index not built."}

    embedder = get_embedder(meta.embed_model)
    q_vec = embedder.encode([query], convert_to_numpy=True)
    if norm:
        q_vec = l2_normalize(q_vec)

    D, I = index.search(q_vec.astype(np.float32), top_k)
    scores = D[0].tolist()
    idxs = I[0].tolist()

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores, idxs):
        if idx < 0 or idx >= len(meta.chunks):
            continue
        c = meta.chunks[idx]
        snippet = c.text[:400].replace("\n", " ").strip()
        results.append(
            {
                "score": float(score),
                "file_path": c.file_path,
                "page_from": c.page_from,
                "page_to": c.page_to,
                "snippet": snippet,
            }
        )

    return {"ok": True, "query": query, "k": top_k, "results": results}


# ---------------------------
# Status helper
# ---------------------------


def get_rag_index_status(data_dir: Path) -> Dict[str, Any]:
    """
    Return small status dict about the current RAG index.
    """
    data_dir = Path(data_dir)
    paths = RagPaths.from_data_dir(data_dir)
    exists = paths.index_path.exists() and paths.meta_path.exists()
    status: Dict[str, Any] = {
        "exists": exists,
        "chunks": 0,
        "built_at": None,
        "index_path": str(paths.index_path),
        "meta_path": str(paths.meta_path),
    }
    if not exists:
        return status

    try:
        meta_d = json.loads(paths.meta_path.read_text())
        status["chunks"] = len(meta_d.get("chunks", []))
        status["built_at"] = meta_d.get("built_at")
        if not status["built_at"]:
            ts = max(
                paths.index_path.stat().st_mtime,
                paths.meta_path.stat().st_mtime,
            )
            status["built_at"] = datetime.fromtimestamp(ts).isoformat(
                timespec="seconds"
            )
    except Exception as e:
        status["error"] = f"status-read-failed: {e!r}"

    return status


# ---------------------------------------------------------------------------
# HIGH-LEVEL WRAPPERS (used by routers / scripts)
# ---------------------------------------------------------------------------


def run_rag_build_index(
    force: bool = False,  # reserved for future freshness checks
    max_pdfs: int | None = None,
) -> Dict[str, Any]:
    """
    Build (or rebuild) the FAISS index from PDFs pointed to by settings.pdf_dir.

    This is a pure Python wrapper around build_rag_index()
    and replaces /api/v1/rag/build from the legacy router.
    """
    # NOTE: `force` is not currently used; we keep it in the signature so
    # future freshness checks (mtime, hash, etc.) can hook in.
    summary = build_rag_index(
        pdf_dir=Path(settings.pdf_dir),
        data_dir=Path(settings.data_dir),
        embed_model_name=settings.embed_model_name,
        max_pdfs=max_pdfs,
    )
    return summary


def run_rag_search_with_safety(
    q: str,
    k: int = 5,
) -> Dict[str, Any]:
    """
    Semantic search over the built index plus safety advisories.

    Pipeline:
        - run RAG search (run_rag_search)
        - extract snippets
        - run safety analyzer over snippets
        - attach per-result safety flags and high-level warnings

    Returns a JSON-safe dict with:
      - ok: bool
      - results: list[...] with optional 'safety_flags'
      - warnings: list[str]
      - other fields from the base RAG search (query, k, etc.)
    """
    base = run_rag_search(query=q, top_k=k)

    if not base.get("ok"):
        # Bubble up base error (e.g. index not built)
        return base

    results = list(base.get("results", []))  # ensure mutable list
    snippets = [r.get("snippet", "") for r in results]

    # --- Safety over result snippets (advisory; hard blocks are elsewhere) ---
    safety_report = analyze_snippets(snippets=snippets)

    # Normalize SafetyReport → dict
    if hasattr(safety_report, "model_dump"):      # pydantic v2
        safety_dict = safety_report.model_dump()
    elif hasattr(safety_report, "dict"):          # pydantic v1
        safety_dict = safety_report.dict()
    elif isinstance(safety_report, dict):
        safety_dict = safety_report
    else:
        safety_dict = {
            "per_item_flags": getattr(safety_report, "per_item_flags", []),
            "advisory": getattr(safety_report, "advisory", []),
            "blocked": bool(getattr(safety_report, "blocked", False)),
            "notes": getattr(safety_report, "notes", ""),
        }

    flags_by_idx = safety_dict.get("per_item_flags") or []
    advisory = safety_dict.get("advisory") or []
    blocked = bool(safety_dict.get("blocked", False))
    safety_notes = safety_dict.get("notes") or ""

    # Attach flags to each result item (NOT to strings)
    for i, item in enumerate(results):
        if i < len(flags_by_idx) and flags_by_idx[i]:
            item["safety_flags"] = flags_by_idx[i]

    # Build warnings list
    warnings: List[str] = []
    if blocked:
        warnings.append("Safety: blocking advisory present.")
    if safety_notes:
        warnings.append(f"Safety notes: {safety_notes}")
    # Optional: de-dup in case upstream repeats
    for w in advisory:
        if w not in warnings:
            warnings.append(w)

    out: Dict[str, Any] = {
        **_jsonable(base),
        "results": results,
        "warnings": warnings,
    }
    return out


def run_rag_status() -> Dict[str, Any]:
    """
    Lightweight index status probe for dashboards/ops.

    This wraps get_rag_index_status(data_dir=settings.data_dir)
    and replaces /api/v1/rag/status.
    """
    return get_rag_index_status(data_dir=Path(settings.data_dir))

