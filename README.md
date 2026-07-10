# StyleForge 🎬

**One clip, four voices — a video-captioning agent whose sense of tone was *trained*, not prompted.**

AMD Developer Hackathon: ACT II — Track 2 (Video Captioning)
Team: Sankar Subbayya × Claude (Anthropic)

🔗 **Live demo:** https://sankarsubbayya.github.io/styleforge/
📦 **Submission image:** `ghcr.io/sankarsubbayya/styleforge:latest`

## What it does

Given a video clip, StyleForge writes captions in four styles — `formal`, `sarcastic`,
`humorous_tech`, `humorous_non_tech` — via a three-stage pipeline:

```
clip ──► INGEST          ──► EYES                ──► VOICE                    ──► TASTE
        ffmpeg 1-pass        Kimi K2.6 vision        Gemma 3 4B, DPO-tuned        LLM judge scores
        frames + Whisper     dense description       on AMD MI300X — runs         K candidates,
        transcript           (Fireworks API)         on CPU inside this image     best per style
```

The **tuned Gemma is the differentiator**: we generated 1,200 scene×style cells of
candidate captions, scored them with an LLM judge, built accuracy-weighted preference
pairs, and ran SFT + DPO on an AMD Instinct MI300X — twice, with a failure-inspection
loop between rounds (round 1: sarcastic +0.54 over base; round 2: formal +0.50).
Full numbers: [live demo](https://sankarsubbayya.github.io/styleforge/) or the deck in
[presentation/](presentation/).

## Running it (organizer contract)

The container implements the Track 2 harness contract: reads `/input/tasks.json`,
writes `/output/results.json`, exits 0, finishes well inside the 10-minute budget
(3 UHD example clips: **33 seconds**).

```bash
mkdir -p io/input io/output
cat > io/input/tasks.json << 'EOF'
[{"task_id": "v1",
  "video_url": "https://storage.googleapis.com/amd-hackathon-clips/13825391-uhd_3840_2160_30fps.mp4",
  "styles": ["formal", "sarcastic", "humorous_tech", "humorous_non_tech"]}]
EOF

docker run --rm --platform linux/amd64 \
  -v $(pwd)/io/input:/input -v $(pwd)/io/output:/output \
  ghcr.io/sankarsubbayya/styleforge:latest

cat io/output/results.json
```

No environment variables required — credentials ship in the image per the Track 2
environment spec. Optional knobs: `HARNESS_MODE` (`bon` default | `baseline` |
`tuned` | `tuned+bon`), `HARNESS_K`, `TIME_BUDGET_SEC`.

## Reliability design (built for the 10-minute wall)

- A **schema-valid `results.json` is written at startup** and atomically upgraded as
  each task completes — a SIGKILL at any moment still leaves a scoreable submission.
- **Three-tier degrade** per task and per style: best-of-N → single-shot → canned
  fallback. Dead URLs, 500ing models, malformed tasks — nothing sinks the run.
- Download deadlines beat trickling servers; frames extract in one ffmpeg pass;
  Whisper loads once per process.

## Local development

```bash
uv sync                                  # deps (Python 3.12+, ffmpeg required)
cp .env.example .env                     # add your FIREWORKS_API_KEY
uv run styleforge caption clip.mp4 --mode bon   # caption any clip
uv run pytest tests/                     # 12 tests incl. harness-contract simulation
STYLEFORGE_MOCK=1 uv run styleforge caption clip.mp4  # offline mock mode
```

## Reproducing the training (AMD MI300X)

```
training/gen_data.py       # 300 scenes × 4 styles × 6 candidates, judge-scored (Fireworks)
training/build_pairs.py    # accuracy-weighted DPO pairs + SFT set
training/train_dpo.py      # SFT warm start (merged) + DPO, TRL + LoRA, bf16
training/merge_and_quantize.py  # merge → GGUF Q4_K_M for CPU serving
training/RUNBOOK.md        # machine choice, timings, hard-won ROCm lessons
```

Trained on a single MI300X droplet (AMD Developer Cloud, ROCm 7.2.4, PyTorch 2.10):
~8 minutes per full round, ~$4 of GPU credit.

## Repo layout

```
src/styleforge/     pipeline: ingest, perception, stylize, judge, local_gemma, harness
training/           data factory + DPO training package (ROCm)
eval/               held-out eval harness (independent judge)
tests/              unit + integration (organizer-contract simulation)
frontend/ + docs/   demo page (GitHub Pages)
presentation/       slide deck (PPTX + PDF)
PLAN.md, PRD.md     the planning trail
```

## License

MIT
