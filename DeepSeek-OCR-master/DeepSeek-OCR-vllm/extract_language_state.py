#!/usr/bin/env python3
"""
Estrae la porzione "language" da un full_state .pt usando regole conservative:
- considera vision-like le chiavi che contengono esplicitamente i prefissi
  'vision_model', 'sam_model', 'projector', 'patch_embed', 'pos_embed',
  'image_newline', 'view_seperator', 'sam', 'neck', 'patch_embedding'
- considera language le altre chiavi sotto 'model.' (incluso model.layers.*)
- salva two files: <orig>_language_state.pt e <orig>_vision_state.pt

Uso:
  python extract_language_state_fixed.py saved_states/DeepseekOCRForCausalLM_full_state.pt --out-dir ./saved_states
"""
import argparse
from pathlib import Path
import torch

VISION_KEYWORDS = [
    "vision_model", "sam_model", "projector", "patch_embed", "patch_embedding",
    "pos_embed", "image_newline", "view_seperator", "sam_model", "sam", "neck",
    "patch_embedding", "patch_embed", "class_embedding"
]

def is_vision_key(k: str) -> bool:
    kl = k.lower()
    # explicit vision prefixes
    for v in VISION_KEYWORDS:
        if v in kl:
            return True
    # also treat explicit top-level vision.* or sam_model.* as vision
    if kl.startswith("vision.") or kl.startswith("sam_model.") or kl.startswith("projector.") or kl.startswith("vision_model."):
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("state_file", help="Path to full state_dict .pt")
    parser.add_argument("--out-dir", default=".", help="Output directory")
    args = parser.parse_args()

    p = Path(args.state_file)
    if not p.exists():
        print("File not found:", p)
        return 2

    print("Loading:", p)
    sd = torch.load(str(p), map_location="cpu")
    # common wrapper handling
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        print("Loaded object is not a state_dict (dict). Type:", type(sd))
        return 3

    lang_sd = {}
    vis_sd = {}
    other_sd = {}

    for k, v in sd.items():
        # if clearly vision, mark as vision
        if is_vision_key(k):
            vis_sd[k] = v
            continue
        # if key is lm_head or embed_tokens -> language
        low = k.lower()
        if low.startswith("lm_head") or "embed_tokens" in low or low.startswith("model.layers") or low.startswith("model.embed_tokens") or low.startswith("model.lm_head") or low.startswith("model.layers."):
            lang_sd[k] = v
            continue
        # if key under model.* but not vision -> treat as language (covers most LM keys)
        if k.startswith("model."):
            # if it contains explicit vision substrings we already filtered above
            lang_sd[k] = v
            continue
        # fallback: other
        other_sd[k] = v

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lang_path = out_dir / (p.stem + "_language_state.pt")
    vis_path = out_dir / (p.stem + "_vision_state.pt")
    other_path = out_dir / (p.stem + "_other_state.pt")

    torch.save(lang_sd, str(lang_path))
    torch.save(vis_sd, str(vis_path))
    torch.save(other_sd, str(other_path))

    print(f"Total keys in full state: {len(sd)}")
    print(f"Saved language-like keys: {len(lang_sd)} -> {lang_path}")
    print(f"Saved vision-like keys: {len(vis_sd)} -> {vis_path}")
    print(f"Saved other keys: {len(other_sd)} -> {other_path}")

    # print small samples
    print("\nSample language keys (first 50):")
    for k in list(lang_sd.keys())[:50]:
        print("  ", k)

    print("\nSample vision keys (first 50):")
    for k in list(vis_sd.keys())[:50]:
        print("  ", k)

    return 0

if __name__ == "__main__":
    main()