"""In-container tuned Gemma serving via llama-cpp-python (CPU).

The DPO-tuned Gemma 3 4B ships in the image as a Q4_K_M GGUF. This module mirrors
stylize.generate()'s interface so pipeline `tuned` modes swap in transparently.
Env: GEMMA_GGUF (path), GEMMA_THREADS (default: all cores), GEMMA_CTX (default 4096).
"""

import os
from pathlib import Path

from . import config
from .stylize import SYSTEM, _user_prompt

GGUF_PATH = Path(os.getenv("GEMMA_GGUF", "/app/models/styleforge-gemma-q4.gguf"))

_llm = None


def available() -> bool:
    return GGUF_PATH.exists()


def _get_llm():
    global _llm
    if _llm is None:
        from llama_cpp import Llama  # lazy: not installed in dev envs without the extra

        _llm = Llama(
            model_path=str(GGUF_PATH),
            n_ctx=int(os.getenv("GEMMA_CTX", "4096")),
            n_threads=int(os.getenv("GEMMA_THREADS", "0")) or None,
            verbose=False,
        )
    return _llm


def generate(description: str, transcript: str, style: str, k: int = 1) -> list[str]:
    """Same contract as stylize.generate, served by the in-container tuned model."""
    if style not in config.STYLES:
        raise ValueError(f"unknown style {style!r}")
    if config.MOCK:
        return [f"[mock tuned {style} caption #{i + 1}]" for i in range(k)]
    llm = _get_llm()
    out: list[str] = []
    for i in range(k):
        resp = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": _user_prompt(description, transcript, style)},
            ],
            max_tokens=80,
            temperature=0.4 if k == 1 else 0.9,
            seed=1234 + i,
        )
        text = resp["choices"][0]["message"]["content"] or ""
        text = text.strip().strip('"')
        if text:
            out.append(text)
    # llama-cpp is in-process: identical seeds dedupe naturally via temperature
    seen: set[str] = set()
    return [c for c in out if not (c in seen or seen.add(c))] or out
