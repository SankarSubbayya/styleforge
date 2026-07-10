"""Generate FRESH held-out descriptions for the G2 eval.

These are generated after training and never appear in any training file —
the clean text-side proxy for the hidden clip set.

Usage: uv run python eval/make_eval_set.py [--n 30]
Output: data/eval/eval_descriptions.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from styleforge import config, fw  # noqa: E402
from training.gen_data import DESC_SYSTEM  # noqa: E402


def main(n: int) -> None:
    out_dir = config.DATA_DIR / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_descriptions.jsonl"
    descs: list[str] = []
    if out_path.exists():
        descs = [json.loads(x)["description"] for x in out_path.open()]
    with out_path.open("a") as f:
        while len(descs) < n:
            raw = fw.chat(
                [
                    {"role": "system", "content": DESC_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            'Write 5 new clip descriptions. Respond as JSON: '
                            '{"descriptions": ["...", ...]}. These are for a HELD-OUT '
                            "evaluation set: favor scene types a hackathon might pick "
                            "(pets, sports moments, street scenes, cooking, office life, "
                            "small fails and surprises). Avoid: "
                            f"{', '.join(d[:40] for d in descs[-6:]) or 'nothing yet'}"
                        ),
                    },
                ],
                model=config.PERCEPTION_MODEL,
                temperature=1.05,
                max_tokens=3000,
                json_mode=True,
                reasoning_effort="none",
            )
            for d in json.loads(raw).get("descriptions", []):
                if d and len(d.split()) > 30 and len(descs) < n:
                    descs.append(d)
                    f.write(json.dumps({"description": d}) + "\n")
            f.flush()
            print(f"{len(descs)}/{n}")
    print(f"eval set ready: {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    main(ap.parse_args().n)
