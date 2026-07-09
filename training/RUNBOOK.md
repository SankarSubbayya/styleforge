# AMD MI300X Training Runbook — StyleForge Gemma DPO

Budget: $5 AMD credit + up to $40 Sankar's money. Every idle GPU minute is wasted money —
this runbook front-loads everything that can happen BEFORE the droplet exists.

## Machine choice (decided — see rationale)

| Setting | Choice | Why |
|---|---|---|
| Instance | **Single MI300X GPU droplet** (1 GPU) | Gemma 3 4B LoRA DPO needs <30GB VRAM; one 192GB MI300X is already 6x overkill. Multi-GPU nodes cost 8x and buy nothing. |
| Image | AMD ROCm **PyTorch preinstalled** image (pick the latest ROCm 6.x + PyTorch option in the droplet creation menu) | Installing torch-ROCm from scratch costs 15-30 GPU-minutes; the prebaked image costs zero. |
| Disk | Default | Model (8GB) + dataset (<50MB) + llama.cpp fits easily. |
| Region | Whichever has MI300X availability | No data-locality concern. |
| Est. cost | ~$2/hr x ~2h = **~$4** (rehearsal included) | Within the $5 credit alone if nothing goes wrong. |

## Before creating the droplet (local, free)

1. `uv run python training/gen_data.py` — overnight Fireworks run (descriptions + scored candidates)
2. `uv run python training/build_pairs.py` — emits `data/train/dpo_pairs.jsonl` + `sft_top.jsonl`
3. Sanity-check pair counts (want >= 500 DPO pairs; more descriptions if short)
4. Have ready: HF account with **Gemma license accepted** (gated repo) + `HF_TOKEN`
5. SSH key added to AMD Developer Cloud account

## On the droplet (GPU clock running)

```bash
# from laptop — FLAT layout: scripts and jsonl files must land in the SAME directory
# (run_on_amd.sh cd's to its own dir and expects dpo_pairs.jsonl next to itself):
ssh root@<droplet-ip> mkdir -p train
scp training/*.py training/*.sh data/train/dpo_pairs.jsonl data/train/sft_top.jsonl \
    root@<droplet-ip>:~/train/
ssh root@<droplet-ip>
# on droplet:
cd ~/train && export HF_TOKEN=hf_...
bash run_on_amd.sh          # sanity -> deps -> 5-step smoke -> SFT+DPO -> merge+GGUF
```

Expected timeline: sanity+deps ~10min · smoke ~5min · SFT+DPO on ~1-4k pairs ~30-60min ·
merge+GGUF ~15min. **Abort decision point:** if the smoke step fails on model-class or
ROCm errors and isn't fixed in 20 minutes, STOP the clock (snapshot notes, destroy
droplet), debug locally, retry later.

## After

```bash
scp root@<droplet-ip>:~/train/styleforge-gemma-q4.gguf data/models/
scp -r root@<droplet-ip>:~/train/styleforge-gemma-dpo data/models/lora/
# then IMMEDIATELY destroy the droplet in the console — it bills while stopped too
```

Then: wire GGUF into the container (llama-cpp-python, `--mode tuned+bon`), run the eval
harness (baseline vs bon vs tuned+bon), Gate G2 decides what ships.

## Known risks

- **Gemma 3 4B = multimodal checkpoint**: `train_dpo.py` tries CausalLM then
  ImageTextToText. If both misbehave under TRL, fallback base: `google/gemma-3-1b-it`
  (text-only, trains anywhere) — smaller but the story holds.
- **HF gating**: accept the Gemma license on huggingface.co BEFORE droplet creation.
- **ROCm/TRL mismatch**: the 5-step smoke run catches this inside the abort window.
