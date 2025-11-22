#!/usr/bin/env python3
"""
Mostra i prefissi e un campione di chiavi da un pytorch state_dict salvato.
Uso:
  python list_state_keys.py /path/to/state.pt

Esempio:
  python list_state_keys.py saved_states/DeepseekOCRForCausalLM_full_state.pt
"""
import sys
from collections import Counter
import torch
from pathlib import Path

def sample_keys(state_dict, n=40):
    keys = list(state_dict.keys())
    print(f"Total keys: {len(keys)}")
    print("\nSample keys (first {}):".format(min(n, len(keys))))
    for k in keys[:n]:
        v = state_dict[k]
        shape = getattr(v, "shape", type(v))
        print("  ", k, shape)
    return keys

def prefix_stats(keys, topk=40):
    prefixes = [k.split('.')[0] if '.' in k else k for k in keys]
    cnt = Counter(prefixes)
    print("\nTop prefixes:")
    for p, c in cnt.most_common(topk):
        print(f"  {p}: {c}")
    return cnt

def heuristic_lists(keys):
    lang_like = [k for k in keys if k.startswith("language") or k.startswith("lm_head") or ("embed_tokens" in k and "vision" not in k.lower())]
    vis_like = [k for k in keys if any(x in k for x in ("vision", "sam_model", "projector", "image_newline", "view_seperator", "patch_embed", "patch_embedding", "sam"))]
    other = [k for k in keys if k not in lang_like and k not in vis_like]
    print(f"\nHeuristic counts: language-like={len(lang_like)}, vision-like={len(vis_like)}, other={len(other)}")
    print("\nHeuristic language-like keys (sample up to 30):")
    for k in lang_like[:30]:
        print("  ", k)
    print("\nHeuristic vision-like keys (sample up to 30):")
    for k in vis_like[:30]:
        print("  ", k)
    return lang_like, vis_like, other

def main():
    if len(sys.argv) != 2:
        print("Usage: python list_state_keys.py /path/to/state.pt")
        sys.exit(1)
    path = Path(sys.argv[1])
    if not path.exists():
        print("File not found:", path)
        sys.exit(1)
    print("Loading state dict (map_location='cpu'):", path)
    sd = torch.load(str(path), map_location="cpu")
    # if file contains dict with 'state_dict' key (common), use it
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        print("Loaded object is not a state_dict (dict). Type:", type(sd))
        sys.exit(1)
    keys = sample_keys(sd, n=60)
    prefix_stats(keys, topk=60)
    heuristic_lists(keys)
    print("\nDone.")

if __name__ == "__main__":
    main()