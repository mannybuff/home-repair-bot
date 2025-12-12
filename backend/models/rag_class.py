# app/models/rag_class.py
"""
RAG DATA MODELS

This module centralizes the core data structures used by the local
text RAG indexer.

It is extracted from:
    - app/rag/indexer.py

High-level usage:

    - RagPaths
        Holds the paths for the RAG FAISS index and metadata artifacts.

    - ChunkRecord
        Describes a single text chunk extracted from a PDF (file path,
        page range, text).

    - RagMeta
        Metadata header for the RAG index, including model name, dim,
        and a list of ChunkRecord entries.

Design notes:

    - Keep this module limited to dataclasses and very small helpers
      that are strictly data-oriented (e.g., to_json()).
    - All heavy logic (FAISS, embedding, PDF parsing) lives in:
        * app/rag/indexer.py
        * app/utils/rag_utils.py
        * app/services/rag.py
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime


@dataclass
class RagPaths:
    """
    Bundle of key filesystem paths used by the RAG index.

    Typical layout (under data_dir):
        rag.index.faiss       - FAISS index
        rag.index_meta.json   - metadata
        rag.index_norm.npy    - normalization flag
    """
    base_dir: Path
    index_path: Path
    meta_path: Path
    norm_path: Path

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> "RagPaths":
        """
        Build a RagPaths instance from a base data directory, ensuring
        the directory exists.
        """
        data_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            base_dir=data_dir,
            index_path=data_dir / "rag.index.faiss",
            meta_path=data_dir / "rag.index_meta.json",
            norm_path=data_dir / "rag.index_norm.npy",
        )


@dataclass
class ChunkRecord:
    """
    A single text chunk extracted from a PDF.

    Fields:
        - doc_id: integer ID of the document within the corpus
        - file_path: absolute/relative path to the PDF file
        - page_from/page_to: page range covered by this chunk
        - text: the actual text content
    """
    doc_id: int
    file_path: str
    page_from: int
    page_to: int
    text: str


@dataclass
class RagMeta:
    """
    Metadata for a built RAG index.

    Fields:
        - embed_model: embedding model identifier
        - dim: embedding dimensionality
        - chunks: list of ChunkRecord
        - built_at: ISO8601 timestamp when index was built
    """
    embed_model: str
    dim: int
    chunks: List[ChunkRecord]
    built_at: str

    def to_json(self) -> Dict[str, Any]:
        """
        JSON-serializable representation.

        The indexer uses this to persist meta to rag.index_meta.json
        without pulling in any heavy dependencies.
        """
        return {
            "embed_model": self.embed_model,
            "dim": self.dim,
            "built_at": self.built_at,
            "chunks": [asdict(c) for c in self.chunks],
        }
