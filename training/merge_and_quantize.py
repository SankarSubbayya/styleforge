"""Merge the DPO LoRA into the base model and quantize to GGUF (Q4_K_M) so the tuned
Gemma serves on CPU inside the submission container."""

import os
import subprocess

import torch
from peft import PeftModel

BASE = os.getenv("BASE_MODEL", "google/gemma-3-4b-it")
LORA = os.getenv("OUT_DIR", "styleforge-gemma-dpo")
MERGED = LORA + "-merged"
GGUF = os.getenv("GGUF_OUT", "styleforge-gemma-q4.gguf")


def main() -> None:
    from transformers import AutoTokenizer

    try:
        from transformers import AutoModelForCausalLM as Cls
        base = Cls.from_pretrained(BASE, torch_dtype=torch.bfloat16)
    except Exception:
        from transformers import AutoModelForImageTextToText as Cls
        base = Cls.from_pretrained(BASE, torch_dtype=torch.bfloat16)

    merged = PeftModel.from_pretrained(base, LORA).merge_and_unload()
    merged.save_pretrained(MERGED)
    AutoTokenizer.from_pretrained(BASE).save_pretrained(MERGED)
    print(f"merged -> {MERGED}")

    # llama.cpp conversion (clone if absent; convert bf16 -> GGUF -> quantize Q4_K_M)
    if not os.path.isdir("llama.cpp"):
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/ggml-org/llama.cpp.git"], check=True,
        )
        subprocess.run(["pip", "install", "-q", "-r",
                        "llama.cpp/requirements/requirements-convert_hf_to_gguf.txt"],
                       check=True)
    subprocess.run(
        ["python3", "llama.cpp/convert_hf_to_gguf.py", MERGED,
         "--outfile", "styleforge-f16.gguf", "--outtype", "f16"], check=True,
    )
    subprocess.run(["cmake", "-B", "llama.cpp/build", "-S", "llama.cpp"], check=True)
    subprocess.run(
        ["cmake", "--build", "llama.cpp/build", "--target", "llama-quantize", "-j"],
        check=True,
    )
    subprocess.run(
        ["llama.cpp/build/bin/llama-quantize", "styleforge-f16.gguf", GGUF, "Q4_K_M"],
        check=True,
    )
    print(f"quantized -> {GGUF}")


if __name__ == "__main__":
    main()
