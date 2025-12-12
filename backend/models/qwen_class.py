# app/models/qwen_class.py
"""
Qwen configuration dataclasses.

Shared between:
  - utils.qwen_utils (text/VL model loading)
  - services.qwen_use (high-level behaviors)
  - any future components that want explicit config objects.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class QwenTextCfg:
    """
    Text / VL model configuration used by the shared Qwen loader.

    Mirrors the structure previously defined in llm_qwen.QwenTextCfg.
    """
    model_id: str
    device_map: str = "auto"
    dtype: str = "bfloat16"
    max_new_tokens: int = 200
    temperature: float = 0.3
    top_p: float = 0.9
    offload_folder: str = "./data/offload"
    max_memory: Optional[Dict[str, str]] = None
    attn_impl: Optional[str] = "flash_attention_2"


@dataclass
class QwenCaptionCfg:
    """
    Captioner configuration (mirrors qwen_captioner.QwenCfg).

    Kept separate in case caption vs. text configs diverge more later.
    """
    model_path: str
    model_id: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    max_new_tokens: int = 48
    temperature: float = 0.2
    top_p: float = 0.95
    offload_folder: str = "./data/offload"
    gpu_max_gb: float = 10.0
    cpu_max_gb: float = 8.0
    attn_impl: str = "flash_attention_2"
    context_tokens: int = 2048
