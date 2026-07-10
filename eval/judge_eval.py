"""G2 eval: tuned Gemma vs base Gemma vs prompted Kimi, judged independently.

The reward judge during training was Kimi; scoring here uses gpt-oss-120b — a
third model family (not Kimi, not Gemma) — so nobody grades their own homework.

Inputs:  data/eval/eval_descriptions.jsonl
         data/eval/eval_captions_gemma.jsonl  (from the droplet)
Output:  data/eval/eval_report.json + printed per-style table + failure list

Usage: uv run python eval/judge_eval.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from styleforge import config, judge, stylize  # noqa: E402

EVAL_DIR = config.DATA_DIR / "eval"
INDEP_JUDGE = "accounts/fireworks/models/gpt-oss-120b"


def kimi_captions(descs: list[str]) -> list[dict]:
    """Arm 3: the prompted-frontier baseline (current product stylizer)."""
    out_path = EVAL_DIR / "eval_captions_kimi.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.open():
            r = json.loads(line)
            done.add((r["desc_idx"], r["style"]))
    rows = []
    with out_path.open("a") as f:
        for i, desc in enumerate(descs):
            for style in config.STYLES:
                if (i, style) in done:
                    continue
                cap = stylize.generate(desc, "", style, k=1)[0]
                rec = {"desc_idx": i, "style": style, "arm": "kimi", "caption": cap}
                rows.append(rec)
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if i % 10 == 0:
                print(f"kimi captions: {i + 1}/{len(descs)}")
    return [json.loads(x) for x in out_path.open()]


def main() -> None:
    descs = [json.loads(x)["description"] for x in (EVAL_DIR / "eval_descriptions.jsonl").open()]
    gemma_rows = [json.loads(x) for x in (EVAL_DIR / "eval_captions_gemma.jsonl").open()]
    rows = gemma_rows + kimi_captions(descs)

    scored = []
    for n, r in enumerate(rows):
        s = judge.score(descs[r["desc_idx"]], r["caption"], r["style"], judge_model=INDEP_JUDGE)
        scored.append({**r, **s})
        if n % 40 == 0:
            print(f"judged {n}/{len(rows)}")

    # aggregate: arm x style
    agg: dict = defaultdict(lambda: defaultdict(list))
    for r in scored:
        agg[r["arm"]][r["style"]].append(r["overall"])
        agg[r["arm"]]["__all__"].append(r["overall"])

    print(f"\n{'arm':<8}", *[f"{s[:12]:>14}" for s in config.STYLES], f"{'ALL':>8}")
    for arm in ("kimi", "base", "tuned"):
        if arm not in agg:
            continue
        cells = [
            f"{sum(agg[arm][s]) / len(agg[arm][s]):>14.2f}" if agg[arm][s] else f"{'-':>14}"
            for s in config.STYLES
        ]
        overall = sum(agg[arm]["__all__"]) / len(agg[arm]["__all__"])
        print(f"{arm:<8}", *cells, f"{overall:>8.2f}")

    # head-to-head win rate: tuned vs kimi on identical (desc, style) cells
    by_cell: dict = defaultdict(dict)
    for r in scored:
        by_cell[(r["desc_idx"], r["style"])][r["arm"]] = r
    wins = ties = losses = 0
    failures = []
    for cell, arms in by_cell.items():
        if "tuned" in arms and "kimi" in arms:
            d = arms["tuned"]["overall"] - arms["kimi"]["overall"]
            wins, ties, losses = wins + (d > 0), ties + (d == 0), losses + (d < 0)
            if d <= -1.5:  # inspectable failures, per decision discipline
                failures.append({
                    "style": cell[1], "desc": descs[cell[0]][:100],
                    "tuned": arms["tuned"]["caption"], "tuned_score": arms["tuned"]["overall"],
                    "kimi": arms["kimi"]["caption"], "kimi_score": arms["kimi"]["overall"],
                    "rationale": arms["tuned"]["rationale"],
                })
    total = wins + ties + losses
    print(f"\ntuned vs kimi head-to-head: {wins}W / {ties}T / {losses}L "
          f"({100 * wins / max(1, total):.0f}% wins of {total} cells)")
    print(f"failure cases (tuned worse by >=1.5): {len(failures)} — see report")

    (EVAL_DIR / "eval_report.json").write_text(json.dumps({
        "means": {a: {s: (sum(v) / len(v) if (v := agg[a][s]) else None)
                      for s in list(config.STYLES) + ["__all__"]} for a in agg},
        "head_to_head": {"wins": wins, "ties": ties, "losses": losses},
        "failures": failures,
    }, indent=2))
    print(f"report: {EVAL_DIR / 'eval_report.json'}")


if __name__ == "__main__":
    main()
