# app/models/images_class.py

# app/models/images.py
"""
IMAGE DATA MODELS

This module centralizes the core data structures used by the image
indexing and retrieval stack.

It replaces ad-hoc classes that previously lived in:
    - app/rag/image_indexer.py
    - app/rag/images.py

High-level usage:

    - ImgEntry
        Represents a single image embedding entry in the FAISS index.
        Typically used inside services/images.py and any code that
        manages the in-memory / on-disk image index.

    - ImgMeta
        Rich metadata for an image associated with a PDF page, including
        document hash, page index, and resolved file paths.

    - PageImage
        Lightweight structure for direct page-image rendering and
        listing operations (e.g., for API layer /sources endpoints).

Design notes:

    - Keep these dataclasses simple and serializable.
    - Use pathlib.Path for local file paths.
    - Avoid heavy logic or IO in this module; it is for data only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

@dataclass
class ImgMeta:
    """
    Metadata for a single page image derived from a PDF.

    Typical fields:
        - doc_hash: stable hash representing the PDF document
        - page_index: zero-based page index within the PDF
        - pdf_path: absolute path to the source PDF
        - image_path: absolute path to the rendered page image (e.g. JPG)
        - extra: optional dict for additional metadata

    This is intended to be the canonical metadata structure for both the
    image indexer and page-image rendering code.
    """
    doc_hash: str
    page_index: int
    pdf_path: Path
    image_path: Path
    extra: Optional[dict[str, Any]] = None


@dataclass
class ImgEntry:
    """
    A single image-embedding entry in the image index.

    Fields:
        - embedding: numpy vector produced by an image encoder (e.g. CLIP)
        - meta: ImgMeta describing the underlying page/image

    This mirrors the logical structure of the old ImgEntry in the
    image indexer, but centralizes it for reuse.

    Example usage:

        from app.models.images import ImgEntry, ImgMeta

        meta = ImgMeta(
            doc_hash="abc123",
            page_index=0,
            pdf_path=Path("/data/pdfs/doc.pdf"),
            image_path=Path("/data/page_images/abc123/page_0000.jpg"),
        )
        entry = ImgEntry(embedding=vec, meta=meta)
    """
    embedding: np.ndarray
    meta: ImgMeta


@dataclass
class PageImage:
    """
    Lightweight representation of a rendered page image.

    This is used primarily by:
        - page listing / browsing
        - API endpoints that serve thumbnails or full-resolution page images

    Fields:
        - doc_hash: stable document identifier
        - page_index: zero-based page index
        - image_path: absolute path to the rendered image file
    """
    doc_hash: str
    page_index: int
    image_path: Path
