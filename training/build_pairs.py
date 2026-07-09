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
MIN_GAP = 1.5


def main() -> None:
    pairs, sft, skipped = 0, 0, 0
    cand_path = TRAIN_DIR / "candidates.jsonl"
    with (
        (TRAIN_DIR / "dpo_pairs.jsonl").open("w") as fp,
        (TRAIN_DIR / "sft_top.jsonl").open("w") as fs,
    ):
        for line in cand_path.open():
            rec = json.loads(line)
            cands = sorted(rec["candidates"], key=lambda c: c["overall"], reverse=True)
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
            fs.write(
                json.dumps(
                    {"messages": prompt_msgs
                     + [{"role": "assistant", "content": top["caption"]}]}
                )
                + "\n"
            )
            sft += 1
            if top["overall"] - bottom["overall"] >= MIN_GAP:
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
    print(f"dpo pairs: {pairs} | sft rows: {sft} | skipped (gap<{MIN_GAP}): {skipped}")


if __name__ == "__main__":
    main()
