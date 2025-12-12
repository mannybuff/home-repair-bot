# app/utils/qwen_utils.py
"""
Shared Qwen loader + chat helper.

Responsibilities:
  - Build a QwenTextCfg from settings
  - Lazily load the (text/VL) Qwen model + tokenizer/processor
  - Provide a simple _llm_chat(messages, ...) interface
  - Expose meta + last_error for diagnostics

High-level behaviors (queries, sections, intent) live in services.qwen_use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List
import json
import threading
import os
import logging

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    AutoModelForVision2Seq,
    AutoProcessor,
)

from app.core.settings import settings
from app.models.qwen_class import QwenTextCfg

logger = logging.getLogger("qwen.text")
logger.setLevel(logging.INFO)

# Defaults / env knobs
_LLMQ_DEFAULT_TEMP = float(os.getenv("QWEN_TEMP", "0.3"))

# Singleton state
_TXT_LOCK = threading.Lock()
_TXT_MODEL = None
_TXT_TOK = None
_TXT_META: Dict[str, Any] = {}
_LAST_LOAD_ERROR: Optional[str] = None


def get_text_llm_last_error() -> Optional[str]:
    """
    Last load error message (if any), for health/debug endpoints.
    """
    return _LAST_LOAD_ERROR


def _record_meta(**kw: Any) -> None:
    _TXT_META.update({k: v for k, v in kw.items() if v is not None})


def get_text_llm_meta() -> Dict[str, Any]:
    """
    Metadata about the shared Qwen instance: model_id, tokenizer_src, is_vl, etc.
    """
    return dict(_TXT_META)


def _torch_dtype_from_str(s: Optional[str]):
    if not s:
        return None
    s = str(s).lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    return None


def _maybe_flash_kw(attn_impl: Optional[str]) -> Dict[str, Any]:
    if attn_impl and str(attn_impl).lower() == "flash_attention_2":
        return {"attn_implementation": "flash_attention_2"}
    return {}


def _build_cfg_from_settings() -> QwenTextCfg:
    """
    Construct a QwenTextCfg from the global Settings instance.

    Respects settings.resolved_vlm_model() so that captioner/text paths
    remain aligned when you point to a local directory.
    """
    model_id = getattr(settings, "qwen_model_id", None) or getattr(settings, "qwen_model_name", None)
    if hasattr(settings, "resolved_vlm_model"):
        try:
            rid = settings.resolved_vlm_model()
            if rid:
                model_id = rid
        except Exception:
            pass

    return QwenTextCfg(
        model_id=model_id,
        device_map=getattr(settings, "qwen_device_map", "auto") or "auto",
        dtype=getattr(settings, "qwen_torch_dtype", "bfloat16") or "bfloat16",
        max_new_tokens=int(getattr(settings, "qwen_max_new_tokens", 64)),
        temperature=float(getattr(settings, "qwen_temperature", 0.2)),
        top_p=float(getattr(settings, "qwen_top_p", 0.9)),
        offload_folder=str(getattr(settings, "qwen_offload_folder", "./data/offload")),
        max_memory=getattr(settings, "qwen_max_memory_or_built", None),
        attn_impl=getattr(settings, "qwen_attn_impl", "flash_attention_2"),
    )


def _get_text_lm():
    """
    Lazily load (or return) the shared Qwen model + tokenizer/processor.

    Supports both:
      - VL configs (Qwen2VLConfig) via AutoModelForVision2Seq + AutoProcessor
      - CLM configs via AutoModelForCausalLM + AutoTokenizer
    """
    global _TXT_MODEL, _TXT_TOK, _LAST_LOAD_ERROR

    if _TXT_MODEL is not None and _TXT_TOK is not None:
        return _TXT_MODEL, _TXT_TOK

    with _TXT_LOCK:
        if _TXT_MODEL is not None and _TXT_TOK is not None:
            return _TXT_MODEL, _TXT_TOK

        cfg = _build_cfg_from_settings()
        dtype = _torch_dtype_from_str(cfg.dtype)

        logger.info("Qwen text LLM: loading model…")

        # Normalize local dir vs. Hub ID
        raw_id = cfg.model_id
        p = Path(str(raw_id)).expanduser()
        try:
            if "/" in str(p) or str(p).startswith(".") or str(p).startswith("~"):
                p = p.resolve()
        except Exception:
            pass
        model_dir = str(p)
        is_local_dir = os.path.isdir(model_dir)
        mdl_src = model_dir if is_local_dir else raw_id
        logger.info("Resolved model_id=%s -> mdl_src=%s (is_local_dir=%s)", raw_id, mdl_src, is_local_dir)

        try:
            # 1) Inspect config to detect VL vs CLM
            cfg_hf = AutoConfig.from_pretrained(
                mdl_src,
                local_files_only=is_local_dir,
                trust_remote_code=True,
            )
            is_vl = (cfg_hf.__class__.__name__ == "Qwen2VLConfig")
            logger.info("Detected config: %s (is_vl=%s)", cfg_hf.__class__.__name__, is_vl)

            # 2) TOKENIZER / PROCESSOR
            if is_vl:
                proc_id = getattr(settings, "qwen_tokenizer_fallback_id", "Qwen/Qwen2-VL-2B-Instruct")
                logger.info("VL path: using AutoProcessor from %s", proc_id)
                tok_or_proc = AutoProcessor.from_pretrained(proc_id, trust_remote_code=True)
                _record_meta(tokenizer_src=proc_id, is_vl=True)
            else:
                tok_id = getattr(settings, "qwen_tokenizer_fallback_id", raw_id)
                logger.info("CLM path: using AutoTokenizer from %s", tok_id)
                tok_or_proc = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
                if tok_or_proc.pad_token_id is None and tok_or_proc.eos_token_id is not None:
                    tok_or_proc.pad_token = tok_or_proc.eos_token
                _record_meta(tokenizer_src=tok_id, is_vl=False)

            # 3) MODEL
            load_kwargs: Dict[str, Any] = {
                "device_map": cfg.device_map,
                "torch_dtype": dtype,
                "trust_remote_code": True,
            }
            if cfg.max_memory:
                load_kwargs["max_memory"] = cfg.max_memory
            if cfg.offload_folder:
                load_kwargs["offload_folder"] = cfg.offload_folder
            load_kwargs.update(_maybe_flash_kw(cfg.attn_impl))

            if is_vl:
                model = AutoModelForVision2Seq.from_pretrained(
                    mdl_src,
                    local_files_only=is_local_dir,
                    **load_kwargs,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    mdl_src,
                    local_files_only=is_local_dir,
                    **load_kwargs,
                )
            model.eval()

            _TXT_MODEL, _TXT_TOK = model, tok_or_proc
            _record_meta(
                model_id=mdl_src,
                model_dir=model_dir if is_local_dir else None,
                is_vl=is_vl,
                device=str(getattr(model, "device", "auto")),
                device_map=cfg.device_map,
                dtype=str(dtype) if dtype is not None else "auto",
                attn_impl=cfg.attn_impl,
                offload_folder=cfg.offload_folder,
            )
            logger.info(
                "Qwen text LLM: loaded OK as %s",
                "VL(Vision2Seq)" if is_vl else "CLM(CausalLM)",
            )
            return _TXT_MODEL, _TXT_TOK

        except Exception as e:
            _LAST_LOAD_ERROR = f"{type(e).__name__}: {e}"
            logger.exception("Qwen text LLM: load failed: %s", e)
            raise


def _llm_chat(
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Small wrapper around the shared Qwen2 text/VL model.

    - Uses .generate(...) for both VL and CLM configs.
    - Returns the decoded assistant string, or "" on failure.
    """
    try:
        mdl, tok = _get_text_lm()
        cfg = _build_cfg_from_settings()
        meta = get_text_llm_meta()
        is_vl = bool(meta.get("is_vl"))

        # ----- NEW: build prompt using the model's chat template when available
        # For Qwen2-VL, tok is an AutoProcessor; its tokenizer usually exposes
        # .apply_chat_template(...). For pure text models, tok may *be* the tokenizer.
        tokenizer_like = getattr(tok, "tokenizer", tok)

        if hasattr(tokenizer_like, "apply_chat_template"):
            # messages: [{"role": "system"|"user"|"assistant", "content": "..."}]
            prompt = tokenizer_like.apply_chat_template(
                messages or [],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # Fallback: old simple ROLE: content flattening
            parts: List[str] = []
            for m in messages or []:
                role = str(m.get("role") or "user").upper()
                content = str(m.get("content") or "")
                parts.append(f"{role}: {content}")
            prompt = "\n".join(parts).strip()

        if not prompt:
            return ""

        # ----- existing tokenization + generation logic -----
        if is_vl:
            inputs = tok(text=prompt, images=None, return_tensors="pt")
            inputs = {k: v.to(mdl.device) for k, v in inputs.items()}
        else:
            inputs = tok(prompt, return_tensors="pt").to(mdl.device)

        max_new = max_tokens or cfg.max_new_tokens
        gen = mdl.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True if temperature and temperature > 0 else False,
            temperature=temperature,
            top_p=cfg.top_p,
            pad_token_id=(
                (tok.tokenizer.pad_token_id if is_vl else tok.pad_token_id)
                or (tok.tokenizer.eos_token_id if is_vl else tok.eos_token_id)
            ),
            eos_token_id=(tok.tokenizer.eos_token_id if is_vl else tok.eos_token_id),
        )

        if is_vl:
            gen_text = tok.batch_decode(gen, skip_special_tokens=True)[0]
        else:
            # Drop the prompt portion for CLM decoding
            gen_text = tok.decode(
                gen[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

        logger.info(
            "Qwen text LLM chat: generated (max_new=%s, temp=%s, top_p=%s)",
            max_new,
            temperature,
            cfg.top_p,
        )
        return gen_text.strip()

    except Exception as e:
        logger.exception("Qwen text LLM chat failed: %s", e)
        return ""

# ---------------------------------------------------------------------------
# Dialogue message builder for multi-turn step guidance
# ---------------------------------------------------------------------------

def build_step_dialogue_messages(
    dialogue_ctx: Dict[str, Any],
    new_user_text: str,
    new_image_caption: Optional[str] = None,
    max_turns: int = 4,
) -> List[Dict[str, str]]:
    """
    Build a messages list for Qwen to generate the *next step* in a
    multi-turn repair session.

    This is a lower-level helper: it does not call the LLM, it just
    prepares the messages list consumed by _llm_chat.

    Expected dialogue_ctx (from build_dialogue_context):

        {
          "dialog_id": "...",
          "topic": "plumbing",
          "topic_intent": {...},
          "summary": "...",
          "hazards_seen": [...],
          "recent_turns": [
             {
               "timestamp_utc": "...",
               "user_text": "...",
               "image_caption": "...",
               "answer": "...",
               "first_step": "...",
             },
             ...
          ],
        }
    """
    topic = (dialogue_ctx.get("topic") or "home repair").strip()
    topic_intent = dialogue_ctx.get("topic_intent") or {}
    summary = (dialogue_ctx.get("summary") or "").strip()
    hazards_seen = dialogue_ctx.get("hazards_seen") or []
    recent_turns = dialogue_ctx.get("recent_turns") or []

    if not isinstance(recent_turns, list):
        recent_turns = []

    # Limit to last N turns, oldest -> newest for readability
    max_turns = max(1, int(max_turns or 4))
    recent_turns = list(reversed(recent_turns))[:max_turns]
    recent_turns = list(reversed(recent_turns))  # back to oldest->newest

    # Build a textual representation of prior steps
    prior_block_lines: List[str] = []
    for idx, t in enumerate(recent_turns, start=1):
        ut = (t.get("user_text") or "").strip()
        cap = (t.get("image_caption") or "").strip()
        ans = (t.get("answer") or "").strip()
        fst = (t.get("first_step") or "").strip()

        prior_block_lines.append(f"Turn {idx} user_text: {ut or '(none)'}")
        if cap:
            prior_block_lines.append(f"Turn {idx} image_caption: {cap}")
        if fst:
            prior_block_lines.append(f"Turn {idx} first_step_given: {fst}")
        elif ans:
            prior_block_lines.append(f"Turn {idx} answer_given: {ans}")
        prior_block_lines.append("")  # blank line between turns

    prior_block = "\n".join(prior_block_lines).strip()

    # Hazards summary (if any)
    hz_text = ""
    if hazards_seen:
        hz_text = "Known hazards so far: " + "; ".join(str(h) for h in hazards_seen)

    # System prompt: re-usable across sessions
    sys_lines = [
        "You are a careful, practical home-repair assistant.",
        f"The current topic is: {topic}.",
        "You are in the middle of a multi-step repair. The goal is to guide the user",
        "through the repair one safe, concrete step at a time.",
        "",
        "You must now decide the NEXT step (or clarify what the user should do now),",
        "based on the project summary, prior turns, and the user's new message.",
        "",
        "Rules:",
        "- Assume previous steps you recommended have been followed unless the user says otherwise.",
        "- Focus on the next actionable step, not the entire project.",
        "- Emphasize safety if there is any risk.",
        "- If the user appears stuck on a prior step, clarify that step rather than moving on.",
        "- If the user indicates the task is complete, you may confirm and summarize.",
    ]
    if hz_text:
        sys_lines.append("")
        sys_lines.append(hz_text)

    system_prompt = "\n".join(sys_lines)

    # User content: project summary + prior turns + new message
    user_lines: List[str] = [
        "Project summary:",
        summary or "(no prior summary available)",
        "",
        "Prior turns (oldest to newest):",
        prior_block or "(no prior turns).",
        "",
        "New user message:",
        (new_user_text or "").strip() or "(no text provided)",
    ]

    if new_image_caption:
        user_lines.extend(
            [
                "",
                "New image caption:",
                new_image_caption.strip(),
            ]
        )

    user_content = "\n".join(user_lines)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

# ---------------------------------------------------------------------------
# Captioner support (VL path) – used by services.qwen_use.get_qwen_captioner
# ---------------------------------------------------------------------------

class _CaptionerAdapter:
    """
    Normalizes a callable into an object with both .caption(.) and __call__(.).
    Caption generation uses the same Qwen2-VL model loaded by _get_text_lm().
    """
    def __init__(self, max_new_tokens: int = 48, temperature: float = 0.2, top_p: float = 0.95):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def _fallback(self, prompt: Optional[str]) -> str:
        base = "Photo of a wall; possible drywall seam or defect visible."
        return f"{base} {prompt}".strip() if prompt else base

    def caption(self, image, prompt: Optional[str] = None) -> str:
        """
        Generate a short caption focusing on visible, home-repair-relevant details.

        image: PIL.Image.Image or compatible
        prompt: optional extra instruction (usually omitted)
        """
        try:
            mdl, proc = _get_text_lm()
            meta = get_text_llm_meta()
            is_vl = bool(meta.get("is_vl"))
            if not is_vl:
                # Model is not a VL variant; fall back to a safe generic caption.
                return self._fallback(prompt)

            # Instruction for the VL model – can be long and descriptive
            instruction = (
                prompt
                or "Describe this home repair photo in one or two short sentences, "
                   "focusing on visible materials, fixtures, damage, and clues about the cause."
            )

            # Qwen2-VL chat-style message with an image placeholder
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]

            # Let the processor inject the proper vision tokens into the text prompt
            chat_prompt = proc.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Tokenize prompt + attach image
            inputs = proc(
                text=[chat_prompt],
                images=[image],
                return_tensors="pt",
            )
            inputs = {k: v.to(mdl.device) for k, v in inputs.items()}

            # Generate caption
            gen = mdl.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True if self.temperature and self.temperature > 0 else False,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            # Only decode newly generated tokens (strip off the prompt part)
            input_len = inputs["input_ids"].shape[1]
            gen_ids = gen[:, input_len:]
            text = proc.batch_decode(gen_ids, skip_special_tokens=True)[0]
            text = text.strip()

            return text or self._fallback(prompt)

        except Exception:
            logger.exception("Qwen captioner: failed to generate caption")
            return self._fallback(prompt)

    def __call__(self, image, prompt: Optional[str] = None) -> str:
        return self.caption(image, prompt)



_CAPTIONER_SINGLETON: Optional[_CaptionerAdapter] = None


def get_image_captioner() -> _CaptionerAdapter:
    """
    Return a singleton captioner adapter backed by the shared Qwen2-VL model.

    All consumers should go via this function (usually through services.qwen_use).
    """
    global _CAPTIONER_SINGLETON
    if _CAPTIONER_SINGLETON is not None:
        return _CAPTIONER_SINGLETON

    # Reuse the text cfg for basic knobs (max_new_tokens, temperature, top_p)
    cfg = _build_cfg_from_settings()
    _CAPTIONER_SINGLETON = _CaptionerAdapter(
        max_new_tokens=min(cfg.max_new_tokens, 64),
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )
    return _CAPTIONER_SINGLETON
