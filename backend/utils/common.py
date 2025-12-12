# app/utils/common.py

"""
 - This file contains generic helpers used across
 - various functions and features i.e. file parsing
 - read/write, IO helpers, etc.

"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Callable, TypeVar, Optional
from pathlib import Path
import json
import datetime

def jsonable(o: Any) -> Any:
    """
    Recursively convert common non-JSON types (Pydantic, HttpUrl, Path, datetime, sets)
    into JSON-serializable primitives.
    """
    # Primitives
    if o is None or isinstance(o, (bool, int, float, str)):
        return o

    # Pydantic v2 / v1 models
    if hasattr(o, "model_dump"):
        return jsonable(o.model_dump())
    if hasattr(o, "dict"):
        return jsonable(o.dict())

    # Url-like (pydantic HttpUrl/AnyUrl) or pathlib paths
    cls = o.__class__.__name__
    if "Url" in cls or isinstance(o, Path):
        return str(o)

    # datetime/date
    if isinstance(o, (datetime.datetime, datetime.date)):
        try:
            return o.isoformat()
        except Exception:
            return str(o)

    # Containers
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, set):
        return [jsonable(v) for v in o]

    # Fallback
    try:
        return json.loads(json.dumps(o))
    except Exception:
        return str(o)

T = TypeVar("T")
U = TypeVar("U")


def safe_read_upload(upload: Any) -> Optional[bytes]:
    """
    Safely read an uploaded file-like object into bytes.

    - Accepts FastAPI UploadFile, Django-like UploadedFile, or a raw file-like object.
    - Returns None if upload is None or reading fails.
    - If the underlying file supports seek/tell, the position is restored after reading.
    """
    if upload is None:
        return None

    # FastAPI UploadFile has .file, others may be direct file-like
    file_obj = getattr(upload, "file", upload)

    try:
        pos = None
        if hasattr(file_obj, "tell") and hasattr(file_obj, "seek"):
            pos = file_obj.tell()

        data = file_obj.read()
        # normalize to bytes
        if isinstance(data, str):
            data = data.encode("utf-8", errors="ignore")

        # restore position if we moved it
        if pos is not None and hasattr(file_obj, "seek"):
            file_obj.seek(pos)

        return data
    except Exception:
        return None


def map_list(fn: Callable[[T], U], seq: Iterable[T]) -> list[U]:
    """
    Safe, eager map over any iterable.
    Returns a concrete list; returns [] for None/empty input.
    """
    if not seq:
        return []
    return [fn(x) for x in seq]

def load_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def safe_write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_registry(page_root: Path) -> Dict[str, Any]:
    """
    Load a lightweight registry for page images under a document folder.
    Accepts either registry.json (preferred) or registry.jsonl (legacy).
    If neither exists, builds a minimal in-memory registry by scanning
    page_*.jpg files. Never raises; returns {} on failure.
    """
    try:
        json_path = page_root / "registry.json"
        jsonl_path = page_root / "registry.jsonl"

        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as f:
                return json.load(f)

        if jsonl_path.exists():
            # Convert jsonl to dict { "pages": [ ... ] }
            pages: List[Any] = []
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pages.append(json.loads(line))
            return {"pages": pages}

        # Fallback: build minimal registry from filenames
        pages = []
        for p in sorted(page_root.glob("page_*.jpg")):
            # Try to extract index from 'page_XXXX.jpg'
            name = p.stem  # page_0001
            try:
                idx = int(name.split("_")[1])
            except Exception:
                idx = None
            pages.append({
                "file": p.name,
                "page_index": idx,
                "path": str(p),
            })
        return {"pages": pages}
    except Exception:
        # Defensive: never break fusion on registry issues
        return {"pages": []}

def _hash_path(p: Path) -> str:
    return hashlib.sha1(str(p.resolve()).encode("utf-8")).hexdigest()[:16]

# --------------------------------------
# FOR LOADING RAG/IMAGE INDEXES
# --------------------------------------

def load_index(index_dir: Path) -> Tuple[faiss.Index | None, Dict[str, Any] | None, bool]:
    index_path = index_dir / "img.index.faiss"
    meta_path = index_dir / "img.index_meta.json"
    norm_path = index_dir / "img.index_norm.npy"
    if not (index_path.exists() and meta_path.exists()):
        return None, None, False
    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text())
    norm = bool(np.load(norm_path)[0]) if norm_path.exists() else False
    return index, meta, norm
