"""F7: end-to-end orchestration with modes baseline | bon | tuned | tuned+bon.

`tuned` modes differ from base modes only via env STYLIZER_MODEL pointing at the
fine-tuned Gemma — the pipeline code is identical by design.
"""

import json
import time
from pathlib import Path

from . import config, ingest, judge, perception, stylize
from .costs import tracker

MODES = ("baseline", "bon", "tuned", "tuned+bon")
DEFAULT_K = 4


def caption_clip(
    path: Path, mode: str = "bon", k: int = DEFAULT_K, styles: list[str] | None = None
) -> dict:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    use_bon = mode.endswith("bon")

    t0 = time.time()
    info = ingest.probe(path)
    frames = ingest.extract_frames(path, info)
    transcript = ingest.transcribe(path, info)
    description = perception.describe(frames, transcript)

    # tuned modes use the in-container Gemma when its GGUF is present; if the
    # model file is missing (e.g. dev checkout) they fall back to the API stylizer.
    use_tuned = mode.startswith("tuned")
    if use_tuned:
        from . import local_gemma

        gen = local_gemma.generate if local_gemma.available() else stylize.generate
    else:
        gen = stylize.generate

    captions: dict[str, dict] = {}
    # None means "all styles"; an explicit empty list means exactly that — no work.
    for style in (list(config.STYLES) if styles is None else styles):
        try:
            cands = gen(description, transcript, style, k=k if use_bon else 1)
            if use_bon and len(cands) > 1:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=min(len(cands), 8)) as ex:
                    scored = list(
                        ex.map(
                            lambda c: (judge.ensemble_score(description, c, style), c),
                            cands,
                        )
                    )
                scored.sort(key=lambda sc: sc[0]["overall"], reverse=True)
                best_score, best = scored[0]
                captions[style] = {
                    "caption": best,
                    "score": best_score,
                    "n_candidates": len(cands),
                }
            else:
                captions[style] = {"caption": cands[0], "n_candidates": len(cands)}
        except Exception as e:  # noqa: BLE001 — NFR-3: never ship a clip with a missing style
            try:
                fallback = stylize.generate(description, transcript, style, k=1)
                captions[style] = {"caption": fallback[0], "fallback": True, "error": str(e)}
            except Exception:  # noqa: BLE001 — API fully down: canned caption beats a zero
                captions[style] = {
                    "caption": config.FALLBACK_CAPTIONS[style],
                    "fallback": True,
                    "error": str(e),
                }

    return {
        "clip": str(path),
        "mode": mode,
        "duration_sec": info.duration,
        "n_frames": len(frames),
        "transcript": transcript,
        "description": description,
        "captions": captions,
        "elapsed_sec": round(time.time() - t0, 1),
        "cost": tracker.summary(),
    }


def save_result(result: dict, out: Path | None = None) -> Path:
    config.ensure_dirs()
    if out is None:
        stem = Path(result["clip"]).stem
        out = config.OUT_DIR / f"{stem}.{result['mode']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    return out
