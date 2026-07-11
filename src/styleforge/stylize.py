"""F3 / Stage B: description -> K caption candidates per style.

This module talks to the base stylizer via Fireworks. After fine-tuning, the same
interface points at the tuned Gemma (env STYLIZER_MODEL swap).
"""

from . import config, fw

SYSTEM = (
    "You write short captions for video clips. You are given a factual description of the "
    "clip and a target style. Write ONE caption: at most 2 sentences, at most 40 words, "
    "standalone (no preamble, no quotes, no hashtags, no emoji). The caption must be "
    "faithful to what actually happens in the clip AND land the requested style hard."
)


def _user_prompt(description: str, transcript: str, style: str) -> str:
    return (
        f"CLIP DESCRIPTION:\n{description}\n\n"
        f"TRANSCRIPT EXCERPT: {transcript[:400] or '[no audio]'}\n\n"
        f"TARGET STYLE — {style}: {config.STYLES[style]}\n\n"
        f"Write the caption now."
    )


def generate(
    description: str,
    transcript: str,
    style: str,
    k: int = 1,
    temperature: float = 0.9,
) -> list[str]:
    if style not in config.STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {list(config.STYLES)}")
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _user_prompt(description, transcript, style)},
    ]
    def _one(i: int) -> str:
        text = fw.chat(
            messages,
            model=config.STYLIZER_MODEL,
            mock_response=f"[mock {style} caption #{i + 1}]",
            # Low temp for the single-candidate baseline; high temp for BoN diversity.
            temperature=0.4 if k == 1 else temperature,
            max_tokens=300,
            reasoning_effort="none",
        )
        return text.strip().strip('"')

    if k == 1:
        candidates = [_one(0)]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(k, 8)) as ex:
            candidates = list(ex.map(_one, range(k)))
    # Dedupe while preserving order — identical candidates waste judge calls.
    seen: set[str] = set()
    unique = [c for c in candidates if not (c in seen or seen.add(c))]
    return unique
