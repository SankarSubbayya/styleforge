"""Turn judge-scored candidates into TRL-ready datasets.

- dpo_pairs.jsonl: conversational {prompt, chosen, rejected} where chosen/rejected are
  the top/bottom candidates with a score gap >= MIN_GAP (weak pairs teach nothing).
- sft_top.jsonl: top candidate per cell (rejection-sampling SFT warm start).

Usage: uv run python training/build_pairs.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from styleforge import config  # noqa: E402
from styleforge.stylize import SYSTEM, _user_prompt  # noqa: E402

TRAIN_DIR = config.DATA_DIR / "train"
# v3 (post-round-2 eval): STYLE-CONDITIONAL weighting. Global accuracy-weighting (v2)
# fixed formal (+0.50) but cost the humor styles (-0.25/-0.19 vs base) — the judge
# rewards the joke more than the fact inventory there. Weights per style:
MIN_GAP = 1.0
STYLE_WEIGHTS = {  # (accuracy, tone)
    "formal": (0.6, 0.4),
    "sarcastic": (0.5, 0.5),
    "humorous_tech": (0.4, 0.6),
    "humorous_non_tech": (0.4, 0.6),
}
STYLE_MIN_ACC = {  # humor may take small liberties; formal may not
    "formal": 8, "sarcastic": 8, "humorous_tech": 7, "humorous_non_tech": 7,
}


def candidate_accuracy(c: dict) -> float:
    return float(c.get("accuracy", c.get("overall", 0)))


def rank_score(c: dict, style: str = "formal") -> float:
    wa, wt = STYLE_WEIGHTS[style]
    accuracy = candidate_accuracy(c)
    tone = float(c.get("tone", c.get("overall", accuracy)))
    return wa * accuracy + wt * tone


def main() -> None:
    pairs, sft, skipped = 0, 0, 0
    cand_path = TRAIN_DIR / "candidates.jsonl"
    with (
        (TRAIN_DIR / "dpo_pairs.jsonl").open("w") as fp,
        (TRAIN_DIR / "sft_top.jsonl").open("w") as fs,
    ):
        for line in cand_path.open():
            rec = json.loads(line)
            style = rec["style"]
            min_acc = STYLE_MIN_ACC[style]
            cands = sorted(
                rec["candidates"], key=lambda c: rank_score(c, style), reverse=True
            )
            if not cands:
                continue
            prompt_msgs = [
                {"role": "system", "content": SYSTEM},
                {
                    "role": "user",
                    "content": _user_prompt(rec["description"], "", rec["style"]),
                },
            ]
            top, bottom = cands[0], cands[-1]
            # SFT teaches only from captions that are BOTH faithful and on-tone.
            if candidate_accuracy(top) >= min_acc:
                fs.write(
                    json.dumps(
                        {"messages": prompt_msgs
                         + [{"role": "assistant", "content": top["caption"]}]}
                    )
                    + "\n"
                )
                sft += 1
            # DPO pair: chosen must be accurate, and must beat rejected on the
            # accuracy-weighted score AND not lose on accuracy itself.
            if (
                candidate_accuracy(top) >= min_acc
                and rank_score(top, style) - rank_score(bottom, style) >= MIN_GAP
                and candidate_accuracy(top) >= candidate_accuracy(bottom)
            ):
                fp.write(
                    json.dumps(
                        {
                            "prompt": prompt_msgs,
                            "chosen": [{"role": "assistant", "content": top["caption"]}],
                            "rejected": [
                                {"role": "assistant", "content": bottom["caption"]}
                            ],
                        }
                    )
                    + "\n"
                )
                pairs += 1
            else:
                skipped += 1
    print(f"dpo pairs: {pairs} | sft rows: {sft} | skipped: {skipped}")


if __name__ == "__main__":
    main()
