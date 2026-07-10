# StyleForge Demo Video — Script & Shot List

**Target: 2:15–2:30.** Record with QuickTime (File → New Screen Recording, mic ON) or Loom.
Do one full rehearsal pass first. Speak slower than feels natural. Upload to YouTube as
Public or Unlisted, paste the link into the lablab form.

---

## Scene 1 — The hook (0:00–0:20)
**On screen:** the demo page — https://sankarsubbayya.github.io/styleforge/ (scroll slowly)

> "This is StyleForge — our Track 2 entry. One clip, four voices: formal, sarcastic,
> humorous tech, and humorous non-tech. Most teams prompt a big model and hope.
> We took a different bet: we *trained* the sense of tone into a small one."

## Scene 2 — Live run (0:20–0:55)
**On screen:** terminal, run this (pre-type it before recording):

```bash
docker run --rm --platform linux/amd64 \
  -v $(pwd)/io/input:/input -v $(pwd)/io/output:/output \
  ghcr.io/sankarsubbayya/styleforge:latest && cat io/output/results.json | head -30
```

> "This is the exact container the judges run. Three ultra-HD clips, four styles each —
> done in thirty-three seconds, well inside the ten-minute limit."
> *(when results print, read one aloud:)* "Here's the sarcastic take on the kitten clip:
> 'A wild beast embarks on a breathtaking quest across three inches of dirt.'"

## Scene 3 — The training story (0:55–1:35)
**On screen:** deck slides 4 → 5 → 6 (architecture, data factory, AMD training)

> "Under the hood: a frontier vision model watches the clip and writes a factual
> description. Then our star — a Gemma 3 4B we fine-tuned with DPO on an AMD MI300X.
> We built twelve hundred training cells — synthetic scenes times four styles times six
> candidate captions, every one scored by an LLM judge — and turned them into preference
> pairs. Each full training round took eight minutes and about four dollars of GPU time
> on AMD Developer Cloud."

## Scene 4 — Measured honestly (1:35–2:10)
**On screen:** deck slide 8 (the eval table)

> "And we measured it the hard way: held-out scenes, an independent judge, no model
> grading its own family. Two targeted rounds. Round one lifted sarcastic tone by
> point-five-four. Round two lifted formal by point-five — and brought our four-billion-
> parameter model to *exact parity* with the trillion-class frontier model on factual
> accuracy. Seven-point-six-two versus seven-point-six-two. What remains is wit — comedy
> resists distillation. So we ship the mode that measured best, and the tuned Gemma rides
> in the container as a zero-dependency fallback. Measured, not mythologized."

## Scene 5 — All-Gemma + close (2:10–2:30)
**On screen:** terminal with the all-Gemma run output (I'll stage this), then deck slide 10

> "One more thing: the same pipeline runs end-to-end on pure Gemma — Gemma eyes, Gemma
> voice, Gemma judge — served by vLLM on the same AMD MI300X we trained on. Everything's
> open: the repo, the demo, the container. StyleForge — one clip, four voices."

---

## Pre-record checklist
- [ ] `io/input/tasks.json` staged (Scene 2) — command below prepares it
- [ ] Demo page open in a clean browser window (no bookmarks bar)
- [ ] Deck open in presentation mode, slides 4–6, 8, 10 bookmarked
- [ ] All-Gemma terminal output staged (waiting on vLLM — I'll provide)
- [ ] Mic check: record 5 seconds, play back
