# PRD — StyleForge: Judge-Optimized Four-Style Video Captioner
### AMD Developer Hackathon ACT II · Track 2 · with RL fine-tuned Gemma stylizer

| | |
|---|---|
| **Author** | Sankar (with Claude) |
| **Date** | 2026-07-09 |
| **Status** | Draft v1 — post-adversarial-review restructure |
| **Deadline** | **2026-07-11, 9:00 AM PDT = 9:30 PM IST** (~60 hours away) |
| **Related** | [README.md](README.md) (plan), [amd_hackathon_readme.md](amd_hackathon_readme.md) (event page) |

---

## 1. One-liner

A containerized pipeline that watches a short video clip and produces four captions
(formal, sarcastic, humorous-tech, humorous-non-tech), where caption *tone* — the thing
the LLM judge scores and prompted models blur — is optimized by RL fine-tuning a Gemma
stylizer against an LLM-judge reward, trained on AMD MI300X.

---

## 2. Adversarial review of the original proposal

The original proposal ("GRPO-tune multimodal Gemma 3 end-to-end, 4-day plan") was
red-teamed. Verdict: **the goal survives; the architecture and schedule do not.**
Each attack below produced a design change.

| # | Attack | Verdict | Design response |
|---|---|---|---|
| A1 | **The 4-day plan is already dead.** It assumed a Jul-8 baseline; it's Jul 9 and nothing is built. Solo developer, ~60h left, credits possibly unredeemed. | **Lands** | Re-planned to 2.5 days with hard decision gates (§9). Baseline must exist tonight. |
| A2 | **Multimodal GRPO is the riskiest possible training config**: vision inputs in TRL's GRPOTrainer + ROCm quirks + judge-in-the-loop, debugged overnight, solo. Likely total training failure. | **Lands** | **Two-stage architecture (§6):** big prompted VLM handles *perception* (dense factual description); RL fine-tuning applies only to a **text-only Gemma stylizer**. Vision is removed from the training loop entirely. Tone is where the differentiation is anyway. |
| A3 | **Reward hacking / judge mismatch.** We train against *our* judge but are scored by the organizers' *unknown* judge. RLAIF can overfit our judge's quirks. | **Lands partially** | Judge ensemble (2–3 different model families), rubric anchored to the published criteria (accuracy + tone), KL penalty in training, held-out clips never used in training, human spot-check of top/bottom captions. |
| A4 | **Fine-tuning may be unnecessary to win.** Track 2 has *no token penalty* — inference-time compute is scoring-free. Best-of-N + judge reranking at inference approximates the RL objective with zero training risk. | **Lands — and helps us** | Best-of-N reranking is now the **guaranteed quality layer** (F7). The tuned model raises the ceiling, composes with BoN, and carries the differentiation + Gemma-prize story. If all training fails, BoN alone is still a strong submission. |
| A5 | **Online GRPO with a judge API in the loop is fragile** (rate limits, cost, no resumability) for a solo overnight run. | **Lands** | **Offline preference optimization is primary**: generate K candidates → judge-score once (cached) → preference pairs → **DPO**; rejection-sampling SFT as warm start. GRPO is a stretch goal only if DPO lands early (§9). Framing stays honest: DPO/GRPO are both RL-from-AI-feedback methods. |
| A6 | **MI300X access is unconfirmed** ($100 AMD credit is in 2–3 day manual approval). | **Lands** | Track 2's required compute is only Fireworks API — AMD GPU training is a story bonus, not a rule. Fallback ladder (§10, R1) + hard gate G1: no MI300X by Jul 10 noon IST → switch to fallback. |
| A7 | **Judges must be able to run the container** — they may have no GPU. A 12B local model in the container fails the "runnable per instructions" requirement. | **Lands** | Container's default path is CPU-only and calls Fireworks for all inference. Tuned weights: try Fireworks LoRA serving; else optional `--local-gpu` path with HF-hosted weights. Demo never requires an MI300X. |
| A8 | **We haven't read the Participant Guide.** Clip-set access, submission format, and the organizer judge rubric are unknowns we've been planning around. | **Lands** | Open questions (§11) with owner + resolve-by **tonight**. Highest-priority action. |
| A9 | **$50 Fireworks budget could be eaten by dataset generation** (thousands of judge calls). | **Survivable** | Cost model in §8: mid-size judge, aggressive caching, batching; ~$25–35 projected, $10 reserved for demo. Live cost logging (NFR-4). |
| A10 | **Demo video + slides + README are required deliverables** and always take longer than planned. | **Lands** | Friday 14:00–18:00 IST hard-reserved; submit by 19:00 IST, not 21:29 (§9). |
| A11 | **Gemma-prize eligibility** could be diluted if perception uses a non-Gemma VLM. | **Partial** | Gemma is the *headline*: the tuned stylizer. If Fireworks serves a Gemma multimodal variant, use it for perception too; verify tonight (§11 Q4). |
| A12 | **MIT-compliance rule** vs. model licenses and reused code. | **Minor** | Our code MIT-licensed; Gemma weights used under Gemma terms (permitted); reused code is from Sankar's own repos (cancer_research excluded per prior decision). |

**Post-review architecture in one sentence:** *prompted frontier VLM for eyes, RL-tuned
Gemma for voice, judge-ensemble reranking for insurance.*

---

## 3. Goals

1. **Place on the Track 2 leaderboard** — maximize LLM-judge score on accuracy + tone.
2. **Win "Best Use of Gemma in Video Captioning" ($3,000)** — Gemma as the tuned stylizer, trained on AMD hardware.
3. **Be the only team with RL fine-tuning** — observed differentiator: as of Jul 8, none of the ~35 submissions visible on the event page mention fine-tuning (observed, not guaranteed — teams may be building quietly). Honest caveat from A4: RL's *leaderboard* delta over best-of-N may be small; its value is the score ceiling plus the Gemma-prize/training narrative. Gate G2 measures the delta and we ship whichever mode actually wins.
4. **Always have a submittable artifact** — every layer (baseline → BoN → tuned) is independently a valid submission.

## 4. Non-goals

- Training any vision component (perception is prompted, not tuned).
- Online/live GRPO with judge-in-the-loop as the primary method (stretch only).
- Real-time/streaming captioning, UI polish beyond a minimal demo, multi-language support.
- Competing in Tracks 1 or 3.

---

## 5. Users & win conditions

| "User" | What they do | What winning looks like |
|---|---|---|
| Organizer LLM judge | Scores each caption for accuracy + tone per style | Distinct, unmistakable tone per style; no factual drift from the clip |
| Gemma prize judges | Assess how meaningfully Gemma is used | Gemma is the trained artifact, with training curves + before/after evals on AMD hardware |
| Submission reviewers | Run the container per README | One command, CPU-only, produces 4 captions per clip |
| Future us (Fri 6 PM IST) | Assemble demo video + slides | Eval table + training story ready to screenshot |

---

## 6. System architecture

### 6.1 Inference pipeline (the submission)

```
clip.mp4 (30s–2min)
  ├─ ffmpeg → frames (1 fps, cap 16, keyframe-biased)
  ├─ Whisper → transcript (faster-whisper, CPU ok)
  ▼
[Stage A: PERCEPTION — prompted, not trained]
  Gemma 3 27B (multimodal) via Fireworks — ALL-GEMMA product policy, decision 2026-07-09:
  Gemma watches, Gemma writes, Gemma judges (maximizes Gemma-prize centrality)
  → dense factual scene description + timeline + notable objects/actions/audio cues
  ▼
[Stage B: STYLIZATION — the trained artifact]
  Gemma stylizer (RL fine-tuned; text-only)
  → K candidates × 4 styles   (K configurable, default 8)
  ▼
[Stage C: RERANK — insurance layer]
  Gemma judge scores candidates on accuracy+tone rubric → top-1 per style
  (in-product judging is Gemma per the all-Gemma policy; the multi-family ensemble
  from A3 is used only in our OFFLINE eval harness, to check we aren't fooling
  ourselves with Gemma-judges-Gemma self-bias — it never ships in the product)
  ▼
4 captions: formal | sarcastic | humorous-tech | humorous-non-tech
```

Inference modes (F7): `baseline` (prompted only) · `bon` (baseline + rerank) ·
`tuned` (fine-tuned stylizer) · `tuned+bon` (default).

### 6.2 Training pipeline (offline, on MI300X)

```
open video-caption datasets + our clip descriptions
  → Stage-A descriptions (reuse inference code)
  → base Gemma generates K candidates per (description, style)
  → judge ensemble scores once, cached to disk        [~$; see §8]
  → build: (a) top-scored SFT set  (b) chosen/rejected DPO pairs
  → RS-SFT warm start → DPO (TRL, LoRA r=16–64, bf16)  [primary]
  → GRPO short run                                      [stretch only]
  → merge LoRA → push to HF Hub → eval vs baseline on held-out clips
```

Base model: **Gemma 3 4B-it** (text). 12B only if 4B converges early and MI300X headroom
allows. ROCm stack: prebuilt `rocm/pytorch` image + TRL; **no BitsAndBytes** (CUDA-centric
— use bf16 LoRA instead; MI300X's 192 GB makes quantization unnecessary).

### 6.3 Reuse map

- **CancerLLM** → TRL/PEFT training harness skeleton + LLM-judge script pattern
- **vision_language** → opencv/CLIP utilities (frame handling; optional CLIP-similarity signal in the reward mix)
- **wiki_search** → training-loop/eval boilerplate
- `cancer_research` → **excluded** (off-limits per prior decision)

---

## 7. Product requirements

### Functional

| ID | Requirement | Priority |
|---|---|---|
| F1 | Ingest a 30s–2min clip → frames (1 fps, ≤16) + Whisper transcript | P0 |
| F2 | Stage-A perception: dense factual description via Fireworks VLM | P0 |
| F3 | Stage-B generation: 4 styles, K candidates each, from description (+transcript) | P0 |
| F4 | Judging, two tiers: **(a)** training-set scoring by a single mid-size judge (high volume, disk-cached); **(b)** eval + inference reranking by a 2–3-family ensemble (low volume). Ensemble calibrated against single judge on a 5% subsample | P0 |
| F5 | Dataset builder: candidates+scores → SFT set + DPO preference pairs | P1 |
| F6 | Training: RS-SFT + DPO via TRL/LoRA on ROCm; GRPO stretch; resumable; W&B or CSV curves | P1 |
| F7 | Inference modes: `--mode baseline\|bon\|tuned\|tuned+bon` | P0 |
| F8 | Eval harness: all modes on held-out clips → comparison table (the demo centerpiece) | P0 |
| F9 | Packaging: Dockerfile (CPU-only default), README with one-command run, public repo, zero secrets. Judge/user supplies their own `FIREWORKS_API_KEY` via env var — documented up front in README (every hackathon participant and judge has Fireworks credits for this event) | P0 |
| F10 | Demo application URL (submission requirement): minimal hosted Streamlit page — upload/pick a clip → 4 captions + the eval comparison table | P0 |

### Non-functional

- **NFR-1** Default container path runs on CPU-only hosts; all inference via Fireworks using a user-supplied `FIREWORKS_API_KEY` (no bundled secrets).
- **NFR-2** Every batch job (candidate gen, judging, training) is resumable from cache/checkpoint.
- **NFR-3** Any single caption failure falls back to `baseline` mode with retries — 100% output reliability *given Fireworks availability* (a full API outage is out of scope; retry with exponential backoff is in scope).
- **NFR-4** All API spend logged per call; running total printed per batch.
- **NFR-5** Seeds fixed where the API allows; eval results reproducible from cached scores.

---

## 8. Budget ($50 Fireworks + $100 AMD Cloud if approved)

| Item | Estimate |
|---|---|
| Stage-A perception, ~40 eval clips + ~300 training descriptions | ~$5 |
| Candidate generation (300 desc × 4 styles × 8 cand × ~150 tok) | ~$3 |
| Judge scoring, training set (**single** mid-size judge per F4a: ~9,600 calls × ~800 tok, cached) | ~$10–15 |
| Ensemble calibration subsample (5% × 2 extra judges) + eval runs (4 modes × 40 clips × 4 styles × ensemble) | ~$6 |
| Inference-time reranking during dev/demo (ensemble × K=8 × 4 styles per clip run) | ~$3 |
| **Reserve for demo day** | **$8** |
| **Projected total** | **~$27–32 of $50** |
| MI300X training (~6–10 h × ~$2–3/h) | within $100 AMD credit |

Kill-switch: if actual spend hits $35 before Friday, drop to single judge everywhere + K=4.
Note: training-set scoring uses ONE judge by design (F4a) — the 2–3-family ensemble applies
only to low-volume eval/reranking, so ensemble cost does not scale with dataset size.

---

## 9. Timeline & decision gates (IST)

### Tonight — Jul 9
- [ ] **Read the Participant Guide; resolve §11 Q1–Q5** ← blocks everything
- [ ] Sankar: redeem the Fireworks coupon (code in credentials.local.md, not committed) → `FIREWORKS_API_KEY` in `.env`; check AMD credit status
- [ ] Scaffold repo, Dockerfile; F1+F2 working on 2–3 sample clips
- [ ] Prompted baseline (F3, K=1) end-to-end → **valid submission exists tonight**
- [ ] Kick off overnight: descriptions + candidate generation for training data

### Jul 10
- [ ] Judge rubric + ensemble (F4); score overnight candidates; build datasets (F5)
- [ ] BoN reranking layer (F7 `bon` mode) — guaranteed quality lift, zero training risk
- [ ] **Gate G1 @ 12:00** — MI300X reachable? **Yes** → env smoke test now, RS-SFT + DPO evening; GRPO stretch runs **overnight Jul 10→11 only if DPO completes by 22:00** (this is the only window — Jul 11 has no training time). **No** → fallback ladder (R1), no further GPU work
- [ ] Eval harness (F8) on held-out clips; baseline vs bon numbers banked

### Jul 11
- [ ] **Gate G2 @ 10:00** — best tuned checkpoint (DPO or overnight GRPO) beats baseline on held-out? **Yes** → merge, upload, `tuned+bon` becomes default. **No** → ship `bon`, story = "RL data pipeline + honest negative result"
- [ ] 12:00–14:00 final eval table; container smoke test from clean clone; deploy demo app (F10)
- [ ] **14:00–18:00 hard-reserved**: demo video, slides, README, cover image
- [ ] **19:00 submit** (2.5 h buffer before the 21:30 IST deadline — leaderboard/upload failures happen)

---

## 10. Risk register

| ID | Risk | L | I | Mitigation / trigger |
|---|---|---|---|---|
| R1 | AMD MI300X credits never arrive | M | H | G1 ladder: Fireworks fine-tuning service (if Gemma supported) → any cloud GPU for 4B LoRA → ship `bon` only. Track 2 doesn't require AMD compute — but be explicit: **each fallback step weakens Goal 2** (the "trained-on-AMD" Gemma-prize narrative); the last step forfeits it. Leaderboard placement (Goal 1) is unaffected. |
| R2 | ROCm/TRL env burns the training window | M | H | Prebuilt `rocm/pytorch` image; bf16 LoRA (no bnb); 30-min smoke test at G1 before committing the night. |
| R3 | Tuned model overfits our judge, flops on theirs | M | M | Ensemble judges, KL penalty, held-out evals, human spot checks; `bon` fallback at G2. |
| R4 | Organizer clip set / submission format differs from assumptions | M | H | §11 Q1–Q3 resolved tonight before heavy build. |
| R5 | Budget blowout on judge calls | L | M | §8 cost model, caching, kill-switch at $40. |
| R6 | Solo-developer time crunch / burnout | H | H | Every layer independently submittable; gates force scope cuts instead of heroics; Friday afternoon untouchable. |
| R7 | Fireworks can't serve our LoRA; judges can't run tuned model | M | M | Demo `tuned` via recorded eval + HF weights + optional `--local-gpu`; container default stays `bon` against Fireworks. |

---

## 11. Open questions — resolve TONIGHT (Jul 9)

| # | Question | Why it matters | Owner |
|---|---|---|---|
| Q1 | Do we get the fixed clip set now, or are scoring clips hidden? *Contingency: if hidden, all "eval clip" counts in §8/§9 substitute open-dataset clips as proxy held-out set.* | Decides train/held-out split & overfitting strategy | Sankar (Participant Guide / Discord) |
| Q2 | Exact submission mechanics — repo + container only, or a scored API endpoint? | Shapes F9 packaging | Sankar |
| Q3 | The *published track blurb* says only "LLM-Judge for accuracy and tone" (that's our rubric anchor, and the basis for "no token penalty" in A4 — both need confirming). Is the detailed judge model/rubric disclosed in the Participant Guide? | Directly shapes our reward rubric; confirms A4 | Sankar |
| Q4 | Which Gemma variants does Fireworks serve (multimodal? which sizes)? | Stage-A Gemma option + Gemma-prize strength | Claude (API check once key exists) |
| Q5 | Does Fireworks support serving custom LoRA adapters for Gemma? | R7 mitigation | Claude |
| Q6 | AMD Developer Cloud credit approval status? | G1 input | Sankar (ADP dashboard) |
| Q7 | Confirm Gemma terms permit publishing fine-tuned weights to HF Hub (expected yes, with notice + terms passthrough) | F6 weight hosting, A12 | Claude |

---

## 12. Deliverables checklist (submission requirements)

- [ ] Public GitHub repo, MIT license, README with setup + one-command usage
- [ ] Containerized app (CPU-only default path) — **runnable by judges**
- [ ] Project title, short + long descriptions, tech/category tags
- [ ] Cover image · video presentation · slide presentation
- [ ] Demo application URL (F10: hosted Streamlit page — clip in, 4 captions + eval table out)
- [ ] No secrets in repo (`credentials.local.md`, `.env` gitignored — verified)

---

## Appendix A — Glossary (for non-ML readers of this PRD)

**RLAIF** — RL from AI feedback: an AI judge, not humans, provides the training reward.
**DPO** — Direct Preference Optimization: trains directly on (better, worse) answer pairs; offline, robust.
**GRPO** — Group Relative Policy Optimization: online RL comparing groups of sampled answers; stronger but fragile under deadline.
**RS-SFT** — rejection-sampling supervised fine-tuning: keep only the judge's top-scored outputs, fine-tune on those.
**BoN** — best-of-N: generate N candidates at inference, let the judge pick the best.
**LoRA** — low-rank adapters: fine-tune a small add-on instead of all weights.
**KL penalty** — keeps the tuned model close to the original so it can't drift into judge-gaming gibberish.
**TRL / ROCm / bf16** — Hugging Face's training library / AMD's GPU software stack / the 16-bit number format used for training.

## Appendix B — Minor contingencies

- **Silent clips:** Whisper stage emits an empty transcript; Stage A prompt marks audio as absent (visual-only captioning). No pipeline failure.
- **Whisper speed on judge machines:** `faster-whisper` small/int8 on CPU handles a 2-min clip in well under a minute; acceptable.
