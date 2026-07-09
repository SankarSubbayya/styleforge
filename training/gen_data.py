"""Generate DPO training data on Fireworks (no GPU needed — run locally overnight).

Phase 1: N diverse synthetic scene descriptions (the hidden eval set is unseen, so we
train for generalization across scenes/settings/subjects, per the Participant Guide).
Phase 2: K caption candidates per (description, style), judge-scored. Resumable.

Usage:  uv run python training/gen_data.py [--n-desc 300] [--k 6]
Output: data/train/descriptions.jsonl, data/train/candidates.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from styleforge import config, fw, judge, stylize  # noqa: E402
from styleforge.costs import tracker  # noqa: E402

TRAIN_DIR = config.DATA_DIR / "train"

DESC_SYSTEM = (
    "You invent factual descriptions of short video clips (30s-2min), as a video analyst "
    "would write them: setting, subjects, sequence of actions with rough timing, on-screen "
    "text, audio cues, overall arc. 120-200 words each. Vary widely: animals, sports, "
    "cooking, offices, streets, nature, fails, kids, machines, weather, crowds, crafts. "
    "Mundane clips, funny clips, dramatic clips. No two alike."
)


def gen_descriptions(n: int) -> list[str]:
    out_path = TRAIN_DIR / "descriptions.jsonl"
    descs: list[str] = []
    if out_path.exists():
        descs = [json.loads(line)["description"] for line in out_path.open()]
        print(f"resuming: {len(descs)} descriptions already on disk")
    with out_path.open("a") as f:
        while len(descs) < n:
            batch = min(5, n - len(descs))
            raw = fw.chat(
                [
                    {"role": "system", "content": DESC_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"Write {batch} new clip descriptions. Respond as JSON: "
                            f'{{"descriptions": ["...", ...]}}. Already covered themes to '
                            f"avoid repeating: {', '.join(d[:40] for d in descs[-8:]) or 'none'}"
                        ),
                    },
                ],
                model=config.PERCEPTION_MODEL,
                temperature=1.0,
                max_tokens=3000,
                json_mode=True,
                reasoning_effort="none",
                mock_response=json.dumps({"descriptions": [
                    "A mock scene description long enough to pass the word-count "
                    "filter: a golden retriever chases a frisbee across a foggy "
                    "beach at dawn, misjudges the catch, tumbles into the surf, "
                    "shakes off dramatically, and trots back looking triumphant "
                    "while a child laughs off-camera and waves join the audio."
                ] * batch}),
            )
            for d in json.loads(raw).get("descriptions", []):
                if d and len(d.split()) > 30:
                    descs.append(d)
                    f.write(json.dumps({"description": d}) + "\n")
            f.flush()
            print(f"{len(descs)}/{n} descriptions | {tracker.summary()}")
    return descs[:n]


def gen_candidates(descs: list[str], k: int) -> None:
    out_path = TRAIN_DIR / "candidates.jsonl"
    done: set[tuple[int, str]] = set()
    if out_path.exists():
        for line in out_path.open():
            rec = json.loads(line)
            done.add((rec["desc_idx"], rec["style"]))
        print(f"resuming: {len(done)} (description, style) cells already scored")
    with out_path.open("a") as f:
        for i, desc in enumerate(descs):
            for style in config.STYLES:
                if (i, style) in done:
                    continue
                try:
                    cands = stylize.generate(desc, "", style, k=k)
                    scored = [
                        {"caption": c, **judge.score(desc, c, style)} for c in cands
                    ]
                    f.write(
                        json.dumps(
                            {"desc_idx": i, "description": desc, "style": style,
                             "candidates": scored}
                        )
                        + "\n"
                    )
                    f.flush()
                except Exception as e:  # noqa: BLE001 — resumable: skip and continue
                    print(f"cell ({i},{style}) failed: {e}", file=sys.stderr)
            if i % 10 == 0:
                print(f"desc {i}/{len(descs)} | {tracker.summary()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-desc", type=int, default=300)
    ap.add_argument("--k", type=int, default=6)
    args = ap.parse_args()
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    descriptions = gen_descriptions(args.n_desc)
    gen_candidates(descriptions, args.k)
    print("DONE |", tracker.summary())
