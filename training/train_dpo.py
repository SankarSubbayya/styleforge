"""DPO fine-tune Gemma 3 4B on judge-preference pairs. Target: 1x AMD MI300X (ROCm).

Env: BASE_MODEL (default google/gemma-3-4b-it), DPO_DATA, SFT_DATA, OUT_DIR, EPOCHS.
Order: optional SFT warm start (RS-SFT), then DPO. bf16 LoRA — no quantization needed
on a 192GB card, and BitsAndBytes is CUDA-centric (avoid on ROCm).
"""

import dataclasses
import os

import torch
from datasets import load_dataset
from peft import LoraConfig


def make_cfg(cls, **kwargs):
    """Build a TRL config, dropping kwargs this TRL version doesn't know —
    the API shifts between releases (e.g. max_prompt_length removal)."""
    valid = {f.name for f in dataclasses.fields(cls)}
    dropped = [k for k in kwargs if k not in valid]
    if dropped:
        print(f"[cfg] {cls.__name__} dropped unsupported args: {dropped}")
    return cls(**{k: v for k, v in kwargs.items() if k in valid})

BASE = os.getenv("BASE_MODEL", "google/gemma-3-4b-it")
DPO_DATA = os.getenv("DPO_DATA", "dpo_pairs.jsonl")
SFT_DATA = os.getenv("SFT_DATA", "sft_top.jsonl")
OUT = os.getenv("OUT_DIR", "styleforge-gemma-dpo")
RUN_SFT = os.getenv("RUN_SFT", "1") == "1"

PEFT_CFG = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)


def load_model_and_tok():
    """Gemma 3 4B is a multimodal checkpoint; we train text-only. Try the causal-LM
    path first (recent transformers exposes the text backbone), fall back to the
    image-text class. Verify this loads in the FIRST 10 minutes of GPU time."""
    from transformers import AutoProcessor, AutoTokenizer

    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            BASE, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
    except Exception:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            BASE, torch_dtype=torch.bfloat16, attn_implementation="eager"
        )
    try:
        tok = AutoTokenizer.from_pretrained(BASE)
    except Exception:
        tok = AutoProcessor.from_pretrained(BASE).tokenizer
    return model, tok


def main() -> None:
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

    model, tok = load_model_and_tok()

    if RUN_SFT and os.path.exists(SFT_DATA):
        sft_ds = load_dataset("json", data_files=SFT_DATA, split="train")
        sft_cfg = make_cfg(
            SFTConfig,
            output_dir=OUT + "-sft",
            per_device_train_batch_size=8,
            gradient_accumulation_steps=2,
            learning_rate=1e-5,
            num_train_epochs=1,
            logging_steps=10,
            bf16=True,
            max_length=1024,
            save_strategy="epoch",
            report_to=[],
        )
        sft_trainer = SFTTrainer(
            model=model, args=sft_cfg, train_dataset=sft_ds,
            processing_class=tok, peft_config=PEFT_CFG,
        )
        sft_trainer.train()
        sft_trainer.save_model(OUT + "-sft")
        # Bake the SFT adapter into the base weights before DPO — passing the raw
        # `model` to DPOTrainer would re-inject a fresh LoRA and discard the warm
        # start (verified: TRL only continues a PeftModel, not an injected tree).
        model = sft_trainer.model.merge_and_unload()
        print("SFT warm start complete and merged into base")

    dpo_ds = load_dataset("json", data_files=DPO_DATA, split="train")
    dpo_cfg = make_cfg(
        DPOConfig,
        output_dir=OUT,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=int(os.getenv("EPOCHS", "1")),
        logging_steps=10,
        bf16=True,
        beta=0.1,
        max_length=1024,
        max_prompt_length=768,
        save_strategy="epoch",
        report_to=[],
    )
    trainer = DPOTrainer(
        model=model, args=dpo_cfg, train_dataset=dpo_ds,
        processing_class=tok, peft_config=PEFT_CFG,
    )
    trainer.train()
    trainer.save_model(OUT)
    print(f"saved LoRA to {OUT}")


if __name__ == "__main__":
    main()
