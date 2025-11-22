#!/usr/bin/env python3
"""
Quick test: load only the language model of Deepseek in 8-bit using bitsandbytes.
Usage:
  python scripts/test_bnb_load.py --model deepseek-ai/DeepSeek-OCR
Or provide a local path: --model /path/to/local/checkpoint
"""
import argparse
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
    # load_in_8bit requires bitsandbytes and transformers support
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True,
    )

    # quick check: move tokenizer input to device where model lives
    device = next(model.parameters()).device
    print("Model loaded on device:", device)

    print("Tokenizing prompt and generating a short output...")
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)

    # generate (use small tokens to test)
    with torch.no_grad():
        out_ids = model.generate(input_ids, max_new_tokens=20, do_sample=False)
    out_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print("Generation result:")
    print(out_text)

if __name__ == "__main__":
    main()