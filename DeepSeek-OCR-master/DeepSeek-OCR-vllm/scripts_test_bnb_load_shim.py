#!/usr/bin/env python3
"""
Quick test: load only the language model of Deepseek in 8-bit using bitsandbytes.
This version includes compatibility shims for:
 - LlamaFlashAttention2 (aliased to LlamaAttention if missing)
 - is_torch_fx_available in transformers.utils.import_utils (defined if missing)

Usage:
  python scripts_test_bnb_load_shim2.py --model deepseek-ai/DeepSeek-OCR
"""
import argparse
import importlib
import sys

# -------------------------
# Shim 1: ensure transformers.models.llama.modeling_llama has LlamaFlashAttention2
# -------------------------
try:
    mod_llama = importlib.import_module("transformers.models.llama.modeling_llama")
    if not hasattr(mod_llama, "LlamaFlashAttention2"):
        if hasattr(mod_llama, "LlamaAttention"):
            setattr(mod_llama, "LlamaFlashAttention2", getattr(mod_llama, "LlamaAttention"))
            print("Shim: aliased LlamaFlashAttention2 -> LlamaAttention")
        else:
            # provide a stub that will raise if used (keeps imports working)
            class _StubFlashAttention2:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("Stub LlamaFlashAttention2 used: full feature not available.")
            setattr(mod_llama, "LlamaFlashAttention2", _StubFlashAttention2)
            print("Shim: created stub LlamaFlashAttention2 (will raise if actually invoked).")
except Exception as e:
    print("Shim warning (llama): could not setup LlamaFlashAttention2 shim:", e)

# -------------------------
# Shim 2: ensure transformers.utils.import_utils has is_torch_fx_available
# -------------------------
try:
    mod_import_utils = importlib.import_module("transformers.utils.import_utils")
    if not hasattr(mod_import_utils, "is_torch_fx_available"):
        def is_torch_fx_available() -> bool:
            try:
                import torch
                # check torch.fx availability (basic)
                # torch.fx exists in torch >=1.8 typically, but some builds may not include full FX functionality
                return hasattr(torch, "fx")
            except Exception:
                return False
        setattr(mod_import_utils, "is_torch_fx_available", is_torch_fx_available)
        print("Shim: added is_torch_fx_available() to transformers.utils.import_utils")
except Exception as e:
    print("Shim warning (import_utils): could not setup is_torch_fx_available shim:", e)

# Now safe to import HF and proceed
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HF model id or local path")
    parser.add_argument("--prompt", type=str, default="Hello world", help="Test prompt")
    args = parser.parse_args()

    model_id = args.model

    print("Checking bitsandbytes import...")
    try:
        import bitsandbytes as bnb  # noqa: F401
        print("bitsandbytes version:", bnb.__version__)
    except Exception as e:
        print("bitsandbytes import FAILED:", e)
        return

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print("Loading model in 8-bit (this may take a while)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print("Error loading model with transformers.from_pretrained():", e)
        # Provide a hint for next steps
        print("\nHINTS:")
        print("- If this mentions missing native libs (flash_attn / triton / xformers), consider installing them.")
        print("- If it mentions other missing symbols from transformers, we can either add shims or install a matching transformers version.")
        return

    device = next(model.parameters()).device
    print("Model loaded on device:", device)

    print("Tokenizing prompt and generating a short output...")
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    with torch.no_grad():
        out_ids = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    out_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print("Generation result:")
    print(out_text)

if __name__ == "__main__":
    main()