# StyleForge 🎬

**One clip, four voices — a video-captioning agent whose sense of tone was *trained*, not prompted.**

AMD Developer Hackathon: ACT II — Track 2 (Video Captioning)
Team: Sankar Subbayya × Claude (Anthropic)

| | |
|---|---|
| 🏆 **Official result** | **0.86 — top 10 of 38+ teams** (peak 0.88 / #6 during the event) |
| 🔗 **Live demo** | https://sankarsubbayya.github.io/styleforge/ |
| 📦 **Submission image** | `ghcr.io/sankarsubbayya/styleforge:latest` (linux/amd64, 2.9 GB) |
| 🧠 **Trained artifact** | Gemma 3 4B, 3 rounds of DPO on AMD Instinct MI300X, Q4 GGUF in-container |

## The problem

This is Track 2 of the AMD Developer Hackathon ACT II: build an agent that
watches a short video clip (30 s – 2 min) and writes a caption for it in four
required styles — **formal**, **sarcastic**, **humorous_tech**, and
**humorous_non_tech**. Every caption is scored by an LLM judge on two axes:
**accuracy** (does it describe what actually happens?) and **style match**
(does it land the requested tone?). Evaluation runs on a *hidden* set of
clips, inside a Docker container, on unknown CPU-only hardware, with a hard
10-minute limit.

The hard part isn't describing a video — modern vision models do that well.
The hard part is **tone**: prompted models blur the line between sarcastic
and humorous, drift into generic captions that could fit any clip, and a
single missing style or malformed output file scores zero. So the real
problems are (1) making four voices genuinely distinct and faithful, and
(2) never, ever returning nothing.

## The solution

StyleForge splits the job in two. A frontier vision model (Kimi K2.6) does
the *seeing* — it turns the clip's frames and audio transcript into a dense
factual description. Writing the caption is then pure **style transfer over
text** — a narrow task, small enough to *train*: we fine-tuned a **Gemma 3
4B** with DPO on 1,200+ judge-ranked caption pairs, across three measured
rounds on an AMD Instinct MI300X, until it matched the frontier model on
factual accuracy and beat it on non-tech humor. At runtime, several candidate
captions per style are generated and an LLM judge picks the best one.

Wrapped around all of it is a survival-first harness: a valid results file
exists from the first second of execution and only ever gets better, so
infrastructure failures (including the organizers' own, of which there were
several) degrade the score instead of zeroing it.

## How it works

Given a video clip, StyleForge writes captions in four styles via a staged
pipeline:

```
clip ──► INGEST                ──► EYES                    ──► VOICE                       ──► TASTE
        ffmpeg single-pass         Kimi K2.6 vision            Gemma 3 4B, DPO-tuned           LLM judge scores
        frame sampling +           (Fireworks API) →           on AMD MI300X — runs            K candidates per
        Whisper transcript         dense factual               on CPU inside this              style, best one
        (tiny, int8, CPU)          description                 image (llama.cpp Q4)            ships
```

| Stage | Component | Detail |
|---|---|---|
| Ingest | ffmpeg + faster-whisper | One decode pass; evenly-spaced frames (default 12 @ 512 px); transcript skipped for silent clips |
| Eyes | Kimi K2.6 (Fireworks serverless) | Dense factual description with timeline; `reasoning_effort=none` for clean fast output |
| Voice | Kimi K2.6 (default `bon` mode) or tuned Gemma (`tuned` modes) | K candidates per style at temperature 0.9 |
| Taste | Kimi K2.6 judge | Rubric-scored accuracy + tone per candidate; best per style ships; scores disk-cached by content hash |
| Armor | harness.py | See "Built to survive" below |

## What makes it strong

- **A schema-valid `results.json` exists from the first second** — written at
  startup with fallback captions, atomically upgraded as each task completes.
  A SIGKILL at any moment still leaves a scoreable submission. (The organizers'
  scoring infrastructure failed repeatedly during the event; this design was
  built for exactly that and never returned a zero.)
- **Three-tier degrade per task and per style**: best-of-N → single-shot →
  canned fallback. Dead URLs, 500ing models, malformed task entries — nothing
  sinks the run. Malformed `tasks.json` still produces valid output.
- **Download deadlines** beat trickling servers; per-read timeouts alone
  cannot (a server dripping bytes never triggers them).
- **Fail-fast API client**: 90 s hard timeout, own retry loop, no SDK
  auto-retries — a hung connection costs seconds, not the 10-minute budget.
- **The tuned Gemma has zero runtime dependencies** — quantized Q4_K_M GGUF
  served in-process by llama.cpp on CPU. No GPU, no endpoint to keep alive.
- **Fast**: the 3 organizer example clips (UHD) complete in **33 seconds**
  against the 10-minute budget.

## Measured honestly

### Internal eval — 30 held-out scenes × 4 styles, judged by gpt-oss-120b
(an independent model family: it neither generated candidates nor was tuned)

| Arm | formal | sarcastic | humorous_tech | humorous_non_tech | ALL |
|---|---|---|---|---|---|
| Kimi K2.6 prompted | **8.92** | **7.72** | **8.23** | 7.87 | **8.18** |
| Gemma 3 4B base | 7.88 | 6.78 | 8.18 | 8.02 | 7.72 |
| Gemma tuned v1 | 7.90 | 7.32 | 7.82 | 7.97 | 7.75 |
| Gemma tuned v2 | 8.40 | 7.13 | 7.93 | 7.83 | 7.83 |
| **Gemma tuned v3** | 8.00 | 7.28 | 8.22 | **8.12** | 7.90 |
| *v3 vs base* | *+0.12* | *+0.50* | *+0.04* | *+0.10* | ***+0.18*** |

Three targeted DPO rounds, each aimed at the previous round's diagnosed
failure axis: round 1 lifted sarcastic tone (+0.54 vs base); round 2's
accuracy-weighted pairs lifted formal (+0.50) and brought the 4B to **frontier
parity on factual accuracy (7.62 vs Kimi's 7.62)**; round 3's
style-conditional pair weights (accuracy-heavy for formal, tone-heavy for
humor) went green across all four styles — and **v3 outscores the ~1T-class
frontier model on humorous_non_tech (8.12 vs 7.87)**. The residual overall gap
is wit: comedy resists distillation into 4B parameters. So best-of-N ships as
the scored default and the tuned Gemma rides in-container.

### Official score timeline (and what it taught us)

| When | Image | Official score |
|---|---|---|
| Jul 11 05:20 IST | baseline pipeline (digest `370d68ce`) | **0.88** (#8, first submission) |
| Jul 11 12:39 IST | +4 simultaneous "improvements" | **0.83** (#17) |
| Jul 13 12:38 IST | rollback to digest `370d68ce` | **0.86** (#10, final) |

The identical image scored 0.88 on Friday and 0.86 on Sunday — that's the
hidden evaluator's own run-to-run drift (±0.02), measured for free by the
rollback. See **What we learned** below for the 0.83 post-mortem.

## Running it (organizer contract)

The container implements the Track 2 harness contract: reads
`/input/tasks.json`, writes `/output/results.json`, exits 0.

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

No environment variables required — credentials ship in the image per the
Track 2 environment spec (no env injection at eval time).

## Configuration reference

All knobs are environment variables with safe defaults.

| Variable | Default | Purpose |
|---|---|---|
| `FIREWORKS_API_KEY` | baked in image | API key for all Fireworks calls |
| `FIREWORKS_BASE_URL` | `https://api.fireworks.ai/inference/v1` | OpenAI-compatible endpoint (point at any vLLM server for the all-Gemma mode) |
| `PERCEPTION_MODEL` | `kimi-k2p6` | Vision model for scene description |
| `STYLIZER_MODEL` | `kimi-k2p6` | Caption generator (API modes) |
| `JUDGE_MODEL` | `kimi-k2p6` | Rerank judge |
| `JUDGE_ENSEMBLE` | unset | Comma-separated judge list for mean-score reranking (see What we learned before using) |
| `HARNESS_MODE` | `bon` | `baseline` \| `bon` \| `tuned` \| `tuned+bon` |
| `HARNESS_K` | `3` | Candidates per style in BoN modes |
| `HARNESS_WORKERS` | `4` | Concurrent clips |
| `TIME_BUDGET_SEC` | `540` | Self-imposed wall (under the 10-min external kill) |
| `DEGRADE_THRESHOLD_SEC` | `75` | Remaining-time floor below which tasks run `baseline` |
| `DOWNLOAD_MAX_SEC` | `150` | Hard per-download deadline (anti-trickle) |
| `MAX_FRAMES` / `FRAME_WIDTH` | `12` / `512` | Frame sampling |
| `WHISPER_MODEL` | `tiny` | faster-whisper size (CPU, int8) |
| `GEMMA_GGUF` | `/app/models/styleforge-gemma-q4.gguf` | Tuned-model path for `tuned` modes |
| `STYLEFORGE_MOCK` | `0` | `1` = full offline smoke mode, no API calls |
| `REASONING_EFFORT_OVERRIDE` | unset | Force a reasoning effort (vLLM endpoints reject `none`; use `low`) |

## Local development

```bash
uv sync                                   # Python 3.12+, ffmpeg required
cp .env.example .env                      # add FIREWORKS_API_KEY
uv run styleforge caption clip.mp4 --mode bon   # caption any clip, pretty table
uv run styleforge probe clip.mp4          # debug: frames + transcript
uv run pytest tests/                      # 12 tests incl. organizer-contract simulation
STYLEFORGE_MOCK=1 uv run styleforge caption clip.mp4  # offline
```

## Reproducing the training (AMD MI300X)

```
training/gen_data.py            # 300 scenes × 4 styles × 6 candidates, judge-scored (~$8 Fireworks)
training/build_pairs.py         # style-conditional accuracy/tone-weighted DPO pairs + SFT set
training/train_dpo.py           # SFT warm start (merged before DPO — see RUNBOOK) + DPO, TRL + LoRA r=32 bf16
training/merge_and_quantize.py  # merge → GGUF Q4_K_M (run llama.cpp deps in a THROWAWAY container)
training/RUNBOOK.md             # machine choice, timings, and the ROCm traps that cost us hours
```

One full round on a single MI300X droplet (AMD Developer Cloud, ROCm 7.2.4,
PyTorch 2.10): **~8 minutes, ~$4 of GPU credit.** The eval harness
(`eval/`) scores held-out scenes with an independent judge and caches every
score, so re-evals are nearly free.

## What we learned (the 0.83 post-mortem)

After scoring 0.88 we shipped four simultaneous "obviously good" changes:
K=3→5 candidates, a second rerank judge (gpt-oss-120b), a detail-density
prompt, and 12→16 frames. It scored **0.83**. Forensics on identical clips
showed the mechanism: the density prompt generated longer candidates *and*
the added judge (accuracy-biased) preferentially selected the densest —
a compounding loop that produced 56-word captions with raw timestamps
against our own 40-word rule. Style purity died; the official judge noticed.

What saved us: every scored image was banked by digest, so the rollback was
one `docker tag` — and the rescored original measured the evaluator's drift
(±0.02) as a bonus. The rules we re-learned, now encoded in this repo's
history: **one variable per submission; judge ensembles reduce variance, not
bias — an added judge must correlate with the target judge or it actively
hurts; and internal evals only measure what they measure.** The winning
team's repo (0.92) independently confirms the better design: verify facts at
the perception layer, enforce style with deterministic guardrails, and skip
LLM reranking entirely.

## Repo layout

```
src/styleforge/     pipeline: ingest, perception, stylize, judge, local_gemma, harness, cli
training/           data factory + DPO training package (ROCm) + RUNBOOK
eval/               held-out eval harness (independent judge, cached scoring)
tests/              unit + integration (organizer-contract simulation, hostile inputs)
frontend/ + docs/   demo page (GitHub Pages)
presentation/       deck (PPTX/PDF), demo video, all-Gemma run artifact
PLAN.md, PRD.md     the planning trail, kept honest
```

## License

MIT
