# AMD Developer Hackathon: ACT II — RL Fine-Tuning Plan

**Deadline:** July 11, 2026, 9:00 AM PDT (~4 days from July 7)
**Resources:** AMD Developer Cloud credits (MI300X-class GPUs, 192GB HBM), Fireworks AI API credits, $6,000 Gemma side-prize pool
**Reference:** [amd_hackathon_readme.md](amd_hackathon_readme.md) — full scraped hackathon page + original analysis

## The tracks

| Track | Challenge | Judging | Prizes |
|---|---|---|---|
| 1 | Hybrid Token-Efficient Routing Agent | Leaderboard: token count + accuracy | $2,500 / $1,500 / $1,000 (+$1k Gemma) |
| 2 | Video Captioning (formal, sarcastic, humorous-tech, humorous-non-tech) | LLM judge: accuracy + tone | $2,500 / $1,500 / $1,000 (+**$3k Gemma**) |
| 3 | Unicorn (open, startup-oriented) | Judges: creativity, completeness, AMD use, market potential | $2,500 / $1,500 / $1,000 (+$2k Gemma) |

Fine-tuning is explicitly permitted in Tracks 1 and 2. All submissions must be containerized, with a public GitHub repo and runnable instructions.

## Lessons from ACT I — the actual winning projects

- 🤖 **AI Agents Track**: 🥇 REPOMIND (repo-scale coding agent) · 🥇 CatalystMD (AI drug
  discovery) · 🥉 Boardroom (summarization with structured debate + verification)
- ⚡ **Fine-Tuning Track**: 🥇 BrainConnect-ASD (early autism diagnosis) · 🥈 AgentReady
  (OWASP agent security) · 🥉 Eidolon (LoRA adapter weights on demand)
- 🎨 **Vision & Multi-Modal Track**: 🥇 Xiao-field-copilot (edge multimodal assistant) ·
  🥈 Strike Lab (biomechanical football analytics) · 🥉 Automato (vision-assisted RPA)
- 🏆 **AMD + Akash Grand Prize**: **Chaos Economy — Multi-Agent RL Market Simulator**
- 🤗 **Hugging Face Community Prize**: Lumi Voice Companion · StudioMI300 · AtlasOps

**What the winners tell us:**

1. **RL wins here** — the Grand Prize went to a multi-agent RL simulator. Judges reward
   ambitious RL work; our RL fine-tuning angle has direct precedent.
2. **Fine-tuning winners paired technique with a story** — real-world impact (autism
   diagnosis), security (OWASP), or infrastructure novelty (LoRA-on-demand serving).
   A technique demo alone didn't place.
3. **Impact narratives dominate judged tracks** — drug discovery, healthcare, sports
   analytics, edge hardware. Relevant for ACT II's Track 3 (the only judged track).
4. **AMD-hardware-native plays get recognized** (StudioMI300, Akash grand prize).

## Competitive intel — current ACT II submissions (in amd_hackathon_readme.md)

The submissions listed on the event page are **this hackathon's competitors**, not past
winners. What they show:

- **Track 1 rivals** (TokenForge, TERA, VoxRouter, TokenOptimizer, Frugal Router): all
  rule-based/prompted — zero-token deterministic solvers, free local tier, confidence-based
  escalation, prompt compression, semantic caching. **None are doing RL.**
- **Track 2 rivals** (CineScribe, ClipForger, Stryvo): prompted VLM pipelines with frame
  timeline + Whisper transcripts and fallback reliability. **None are fine-tuning, let
  alone RL fine-tuning.**
- These engineering patterns are table stakes to match; RL on top is the edge.

## Project ideas (RL fine-tuning focus)

### Idea 1 — RECOMMENDED: RLAIF-tuned Gemma captioner (Track 2)

Track 2 is judged by an LLM judge on accuracy and tone — RLAIF trains against the same kind of signal we're scored on.

- Frame sampling + audio transcript (ffmpeg + Whisper) → visual/audio context
- Fine-tune **Gemma 3 4B/12B** (multimodal) with **GRPO** on MI300X: N candidate captions
  per clip per style, scored by a strong LLM judge (via Fireworks) on faithfulness + style
  adherence, scores used as group-relative rewards
- Prompted Fireworks fallback for reliability

**Why it wins:** nearly everyone prompts a VLM; an RL-tuned model that reliably separates *sarcastic* from *humorous-tech* is a real differentiator. Prize stacking: Track 2 placement + $3,000 Gemma captioning prize (largest partner prize) + meaningful AMD GPU training use. TRL's `GRPOTrainer` runs on ROCm PyTorch; LoRA GRPO on 4B–12B fits one MI300X.

**4-day plan:**

| Day | Deliverable |
|---|---|
| 1 | Data pipeline (clips → frames/transcripts) + prompted baseline producing all 4 styles, containerized — *valid submission on its own (de-risk)* |
| 2 | Judge/reward design + SFT warm-start on judge-filtered outputs |
| 3 | GRPO run on MI300X + side-by-side eval vs baseline |
| 4 | Demo video, README, submission |

### Idea 2 — RL-trained escalation policy (Track 1)

RL-train the escalation policy over the whole winning architecture — "TERA's architecture, but the thresholds are learned, not guessed."

- Action space: {deterministic solver, free local Gemma, Fireworks-tiny/mid/large} + prompt-compression level
- Reward: correctness − λ · Fireworks tokens, over the 8 task categories (factual QA, math, sentiment, summarization, NER, code debugging, logic, codegen)
- Router runs locally (free); only routed Fireworks calls cost tokens

**Risk:** most crowded track; betting a learned policy beats hand-tuned heuristics.

### Idea 3 — "GRPO-on-ROCm in a box" (Track 3)

One-command containerized RL fine-tuning stack for AMD GPUs (pick base model, define reward, get GRPO run + vLLM-ROCm endpoint). Maximal AMD-platform use, credible startup story. ACT I precedent is encouraging: the Grand Prize went to a multi-agent RL simulator, and Eidolon (LoRA-on-demand infra) placed in Fine-Tuning — judges reward RL ambition and training-infrastructure novelty. Remaining risk: needs production polish in 4 days; consider pairing the infra with one compelling impact demo (ACT I fine-tuning winners all had a story).

## Reuse map — existing projects in ~/projects

> ⚠️ `cancer_research` is **off-limits** — do not reuse anything from it.

| Project | What it has | Reuse for |
|---|---|---|
| **CancerLLM** | Complete LoRA fine-tuning pipeline: `trl` trainer, PEFT LoRA (r=64), 4-bit quantization, multi-GPU torchrun; LLM-as-judge eval script (`3_llm_judge_eval.py`) | GRPO training scaffold (swap `SFTTrainer`→`GRPOTrainer`, Mistral→Gemma); judge/reward-function starting point |
| **vision_language** | CLIP embedding + cosine-similarity scoring; opencv | Frame extraction; CLIP-based caption-faithfulness reward component |
| **wiki_search** | Real PyTorch training loop + eval infra | Training-loop boilerplate, reward-model calibration |
| **agentic_economy** | Real-time quality gating with mid-stream token cutoff; multi-model routing | Track 1 only: token-efficient escalation core |

## Decision

**Go with Idea 1 (Track 2 RLAIF captioner).** Prompted pipeline is the day-1 safe submission; the RL-tuned model is strict upside — and no current Track 2 competitor is fine-tuning. Track 1 is the fallback if we prefer an objective leaderboard. Track 3 with an RL angle is more viable than first assessed (ACT I Grand Prize was multi-agent RL), but the 4-day polish bar keeps it third.
