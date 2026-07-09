#!/usr/bin/env bash
# One-shot runbook for the MI300X droplet. Run from /root after scp-ing the training/
# dir + data/train/*.jsonl. Every step prints timing — GPU minutes are money.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== [0] GPU sanity ($(date +%T)) ==="
rocm-smi --showproductname || { echo "no ROCm GPU visible"; exit 1; }
python3 -c "import torch; assert torch.cuda.is_available(), 'torch does not see the GPU'; print('torch OK:', torch.version.hip)"

echo "=== [1] deps ($(date +%T)) ==="
pip install --quiet "transformers>=4.53" "trl>=0.19" peft datasets accelerate sentencepiece
[ -n "${HF_TOKEN:-}" ] || { echo "set HF_TOKEN (Gemma is gated on HF)"; exit 1; }

echo "=== [2] smoke: model loads + 5 training steps ($(date +%T)) ==="
head -n 40 dpo_pairs.jsonl > smoke_pairs.jsonl
DPO_DATA=smoke_pairs.jsonl RUN_SFT=0 EPOCHS=1 OUT_DIR=smoke-out python3 train_dpo.py

echo "=== [3] full run: SFT warm start + DPO ($(date +%T)) ==="
python3 train_dpo.py

echo "=== [4] merge LoRA + quantize to GGUF for CPU serving ($(date +%T)) ==="
python3 merge_and_quantize.py

echo "=== [5] DONE ($(date +%T)) — artifacts: ==="
ls -lh styleforge-gemma-dpo* *.gguf || true
echo "Now scp the .gguf + LoRA dir back, then DESTROY THIS DROPLET."
