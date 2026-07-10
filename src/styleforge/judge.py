"""F4: rubric-scored judging with disk cache.

Tier (a): single mid-size judge for high-volume training-set scoring.
Tier (b): ensemble (env JUDGE_ENSEMBLE, comma-separated model IDs) for eval/reranking.
"""

import hashlib
import json
import os

from . import config, fw

RUBRIC_SYSTEM = (
    "You are a strict caption judge. You receive a factual description of a video clip, a "
    "target style, and a candidate caption. Score the caption on two axes:\n"
    "- accuracy (1-10): is the caption faithful to what actually happens in the clip? "
    "Penalize invented events, wrong subjects, or vagueness that could describe any clip.\n"
    "- tone (1-10): does it land the target style unmistakably? Penalize captions that "
    "could pass for a different style (e.g. sarcastic vs humorous), style labels stated "
    "outright, and jokes that don't land.\n"
    'Respond with JSON only: {"accuracy": int, "tone": int, "rationale": "<one sentence>"}'
)

MOCK_SCORE = '{"accuracy": 7, "tone": 6, "rationale": "mock score"}'


def _cache_path(key: str):
    return config.CACHE_DIR / "judge" / f"{key}.json"


def score(
    description: str,
    caption: str,
    style: str,
    judge_model: str | None = None,
) -> dict:
    """Return {"accuracy": int, "tone": int, "overall": float, "rationale": str}."""
    model = judge_model or config.JUDGE_MODEL
    key = hashlib.sha256(
        json.dumps([model, description, caption, style]).encode()
    ).hexdigest()[:32]
    config.ensure_dirs()
    cached = _cache_path(key)
    # Mock scores must never enter (or be served from) the real cache.
    if config.MOCK:
        cached = None
    if cached is not None and cached.exists():
        return json.loads(cached.read_text())

    user = (
        f"CLIP DESCRIPTION:\n{description}\n\n"
        f"TARGET STYLE — {style}: {config.STYLES[style]}\n\n"
        f"CANDIDATE CAPTION:\n{caption}"
    )
    raw = fw.chat(
        [{"role": "system", "content": RUBRIC_SYSTEM}, {"role": "user", "content": user}],
        model=model,
        mock_response=MOCK_SCORE,
        temperature=0.0,
        max_tokens=400,
        json_mode=True,
        # gpt-oss rejects "none"; "low" is its minimum. Kimi accepts "none".
        reasoning_effort="low" if "gpt-oss" in model else "none",
    )
    try:
        parsed = json.loads(raw)
        result = {
            "accuracy": int(parsed["accuracy"]),
            "tone": int(parsed["tone"]),
            "rationale": str(parsed.get("rationale", "")),
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        raise RuntimeError(f"judge returned unparseable output: {raw[:200]}") from e
    result["overall"] = round((result["accuracy"] + result["tone"]) / 2, 2)
    if cached is not None:
        cached.write_text(json.dumps(result))
    return result


def ensemble_models() -> list[str]:
    raw = os.getenv("JUDGE_ENSEMBLE", "")
    return [m.strip() for m in raw.split(",") if m.strip()] or [config.JUDGE_MODEL]


def ensemble_score(description: str, caption: str, style: str) -> dict:
    """Mean scores across the ensemble (tier b). Falls back to single judge if unset."""
    models = ensemble_models()
    scores = [score(description, caption, style, judge_model=m) for m in models]
    n = len(scores)
    return {
        "accuracy": round(sum(s["accuracy"] for s in scores) / n, 2),
        "tone": round(sum(s["tone"] for s in scores) / n, 2),
        "overall": round(sum(s["overall"] for s in scores) / n, 2),
        "rationale": scores[0]["rationale"],
        "n_judges": n,
    }
