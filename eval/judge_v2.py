"""Round-2 verdict: score v2 captions and compare all four arms per-cell.

v1/base/kimi scores come from the judge cache (free); only v2's 120 are new.
Usage: uv run python eval/judge_v2.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from styleforge import config, judge  # noqa: E402

EVAL_DIR = config.DATA_DIR / "eval"
INDEP_JUDGE = "accounts/fireworks/models/gpt-oss-120b"

descs = [json.loads(x)["description"] for x in (EVAL_DIR / "eval_descriptions.jsonl").open()]

rows = []
for line in (EVAL_DIR / "eval_captions_gemma.jsonl").open():
    r = json.loads(line)
    r["arm"] = {"tuned": "v1", "base": "base"}[r["arm"]]
    rows.append(r)
for line in (EVAL_DIR / "eval_captions_kimi.jsonl").open():
    r = json.loads(line)
    r["arm"] = "kimi"
    rows.append(r)
for line in (EVAL_DIR / "eval_captions_v2.jsonl").open():
    r = json.loads(line)
    r["arm"] = "v2"
    rows.append(r)
for line in (EVAL_DIR / "eval_captions_v3.jsonl").open():
    r = json.loads(line)
    r["arm"] = "v3"
    rows.append(r)

scored = []
for n, r in enumerate(rows):
    s = judge.score(descs[r["desc_idx"]], r["caption"], r["style"], judge_model=INDEP_JUDGE)
    scored.append({**r, **s})
    if n % 60 == 0:
        print(f"scored {n}/{len(rows)}", file=sys.stderr)

agg = defaultdict(lambda: defaultdict(list))
for r in scored:
    agg[r["arm"]][r["style"]].append(r["overall"])
    agg[r["arm"]]["__all__"].append(r["overall"])

print(f"{'arm':<6}", *[f"{s[:12]:>14}" for s in config.STYLES], f"{'ALL':>8}")
for arm in ("kimi", "base", "v1", "v2", "v3"):
    cells = [f"{sum(agg[arm][s]) / len(agg[arm][s]):>14.2f}" for s in config.STYLES]
    print(f"{arm:<6}", *cells, f"{sum(agg[arm]['__all__']) / len(agg[arm]['__all__']):>8.2f}")

by_cell = defaultdict(dict)
for r in scored:
    by_cell[(r["desc_idx"], r["style"])][r["arm"]] = r["overall"]
for a, b in [("v3", "v2"), ("v3", "base"), ("v3", "kimi")]:
    w = sum(1 for c in by_cell.values() if a in c and b in c and c[a] > c[b])
    t = sum(1 for c in by_cell.values() if a in c and b in c and c[a] == c[b])
    l = sum(1 for c in by_cell.values() if a in c and b in c and c[a] < c[b])
    print(f"{a} vs {b}: {w}W/{t}T/{l}L")

(EVAL_DIR / "eval_report_v2.json").write_text(json.dumps({
    "means": {a: {s: sum(v) / len(v) for s, v in styles.items()}
              for a, styles in agg.items()},
}, indent=2))
