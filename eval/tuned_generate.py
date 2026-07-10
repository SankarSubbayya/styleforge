"""Runs ON the droplet (inside the styletrain container).

Generates captions for the held-out eval set from two Gemma arms:
  - tuned: the DPO-merged model
  - base:  stock google/gemma-3-4b-it (isolates what training added)

Uses the EXACT same prompt construction as the product (styleforge.stylize),
so the comparison is apples-to-apples.

Inputs:  /train/eval_descriptions.jsonl, /train/src (styleforge package)
Output:  /train/eval_captions_gemma.jsonl  {desc_idx, style, arm, caption}
"""

import json
import sys

import torch

sys.path.insert(0, "/train/src")
from styleforge.config import STYLES  # noqa: E402
from styleforge.stylize import SYSTEM, _user_prompt  # noqa: E402

MERGED = "/train/styleforge-gemma-dpo-merged"
BASE = "google/gemma-3-4b-it"
OUT = "/train/eval_captions_gemma.jsonl"


def load(path):
    """Load exactly as train_dpo.py did (that path is proven on this container).
    No silent fallback — if the primary path fails we want the real traceback."""
    import traceback

    from transformers import AutoTokenizer

    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
    except Exception:
        print(f"--- AutoModelForCausalLM failed for {path}: ---")
        traceback.print_exc()
        from transformers import Gemma3ForCausalLM

        model = Gemma3ForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
    model = model.to("cuda")
    tok = AutoTokenizer.from_pretrained(path)
    return model, tok


def caption(model, tok, description: str, style: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": _user_prompt(description, "", style)},
    ]
    try:
        ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
    except Exception:  # template without system role: fold into user turn
        merged = [{"role": "user", "content": SYSTEM + "\n\n" + messages[1]["content"]}]
        ids = tok.apply_chat_template(
            merged, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
    out = model.generate(
        ids, max_new_tokens=80, do_sample=True, temperature=0.4, top_p=0.95,
        pad_token_id=tok.eos_token_id,
    )
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return text.strip().strip('"')


def main() -> None:
    descs = [json.loads(x)["description"] for x in open("/train/eval_descriptions.jsonl")]
    done = set()
    try:
        for line in open(OUT):
            r = json.loads(line)
            done.add((r["desc_idx"], r["style"], r["arm"]))
    except FileNotFoundError:
        pass
    with open(OUT, "a") as f:
        for arm, path in [("tuned", MERGED), ("base", BASE)]:
            if all((i, s, arm) in done for i in range(len(descs)) for s in STYLES):
                print(f"{arm}: already complete")
                continue
            model, tok = load(path)
            for i, desc in enumerate(descs):
                for style in STYLES:
                    if (i, style, arm) in done:
                        continue
                    f.write(json.dumps({
                        "desc_idx": i, "style": style, "arm": arm,
                        "caption": caption(model, tok, desc, style),
                    }) + "\n")
                    f.flush()
                print(f"{arm}: {i + 1}/{len(descs)}")
            del model
            torch.cuda.empty_cache()
    print("DONE")


if __name__ == "__main__":
    main()
