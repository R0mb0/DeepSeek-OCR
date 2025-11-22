#!/usr/bin/env python3
# name=scripts/remap_and_create_hf_lang_ckpt.py
"""
Crea una cartella HF contenente solo i pesi "language" rimappati (tolto prefisso 'model.')
e salva la config (AutoConfig.from_pretrained(..., trust_remote_code=True)).

Uso:
  python scripts/remap_and_create_hf_lang_ckpt.py \
    --src saved_states/DeepseekOCRForCausalLM_full_state_language_state.pt \
    --model-id deepseek-ai/DeepSeek-OCR \
    --out-dir ./hf_lang_ckpt
"""
import argparse
from pathlib import Path
import torch
from transformers import AutoConfig

def load_state(path):
    sd = torch.load(str(path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    return sd

def remap_keys(lang_sd):
    new = {}
    for k, v in lang_sd.items():
        if k.startswith("model."):
            nk = k[len("model."):]
        else:
            nk = k
        new[nk] = v
    return new

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--model-id", required=True, help="HuggingFace model id to copy config from")
    p.add_argument("--out-dir", default="./hf_lang_ckpt")
    args = p.parse_args()

    src = Path(args.src)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading language state:", src)
    sd = load_state(src)
    print("Total keys:", len(sd))

    print("Remapping keys (dropping leading 'model.' where present)...")
    sd_remap = remap_keys(sd)
    print("Keys after remap:", len(sd_remap))

    print("Saving pytorch_model.bin to", out)
    torch.save(sd_remap, out / "pytorch_model.bin")

    # Save config (AutoConfig from HF model id)
    try:
        cfg = AutoConfig.from_pretrained(args.model_id, trust_remote_code=True)
        cfg.save_pretrained(out)
        print("Saved config from", args.model_id, "->", out / "config.json")
    except Exception as e:
        print("Warning: could not fetch AutoConfig from HF:", e)
        print("You may need to provide a config.json manually in", out)

    print("Done. Output folder:", out)

if __name__ == "__main__":
    main()
