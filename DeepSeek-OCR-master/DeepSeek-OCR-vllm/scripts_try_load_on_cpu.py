#!/usr/bin/env python3
"""
Try to load candidate model classes from the HF cached dynamic modules
into CPU (low memory) so we can inspect which class is usable.

Usage:
  python scripts_try_load_on_cpu.py --model deepseek-ai/DeepSeek-OCR
"""
import argparse
import importlib.util
import glob
import os
import sys
import traceback
import importlib

CACHE_BASE = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")

CANDIDATES = [
    "DeepseekV2ForCausalLM",
    "DeepseekV2Model",
    "DeepseekV2PreTrainedModel",
    "DeepseekOCRForCausalLM",
    "DeepseekOCRModel",
]

def find_modeling_files(owner, repo):
    pattern = os.path.join(CACHE_BASE, "**", "modeling*.py")
    files = [p for p in glob.glob(pattern, recursive=True) if owner.lower() in p.lower() or repo.lower() in p.lower()]
    return files

def import_module_with_name(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"Error importing {path} as {module_name}: {e}")
        traceback.print_exc(limit=3)
        sys.modules.pop(module_name, None)
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    owner, repo = args.model.split("/", 1)

    # ensure shims already applied by other scripts, but try to import transformers to avoid surprises
    try:
        import transformers  # noqa: F401
    except Exception as e:
        print("Warning: transformers import failed:", e)

    files = find_modeling_files(owner, repo)
    print("Found modeling files:", files)
    if not files:
        print("No modeling files found in cache. Run AutoConfig.from_pretrained(...) once with trust_remote_code=True to download them.")
        return

    # import all modeling modules under package-like names so their internal relative imports work
    imported = []
    for idx, path in enumerate(files):
        module_name = f"hf_dynamic_model_{idx}"
        mod = import_module_with_name(path, module_name)
        if mod is not None:
            imported.append((module_name, mod, path))

    # list available classes
    classes = []
    for module_name, mod, path in imported:
        for name, obj in list(mod.__dict__.items()):
            try:
                if isinstance(obj, type):
                    classes.append((module_name, name, obj, path))
            except Exception:
                continue

    print(f"Discovered {len(classes)} classes in modeling files (showing candidates):")
    for mname, name, obj, path in classes:
        if name in CANDIDATES:
            print("  Candidate:", name, "in", path)

    # Try to instantiate candidates on CPU with low_cpu_mem_usage (no load_in_8bit)
    from transformers import AutoTokenizer
    print("\nLoading tokenizer (trust_remote_code=True)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        print("Tokenizer loaded.")
    except Exception as e:
        print("Tokenizer load failed:", e)

    import torch
    successes = []
    for mname, name, obj, path in classes:
        if name not in CANDIDATES:
            continue
        print("\n--- Trying class:", name, "from", path)
        if not hasattr(obj, "from_pretrained"):
            print("Class has no from_pretrained; skipping.")
            continue
        try:
            # try safe kwargs (CPU low memory)
            print("Calling from_pretrained(..., device_map='cpu', low_cpu_mem_usage=True, trust_remote_code=True)")
            model = obj.from_pretrained(args.model, device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True)
            try:
                dev = next(model.parameters()).device
                dtype = next(model.parameters()).dtype
            except Exception:
                dev = "no-params"
                dtype = None
            print(f"SUCCESS: {name} loaded on {dev}, dtype={dtype}")
            successes.append((name, path, dev, dtype))
            # optionally save a small manifest file with keys count
            try:
                n_params = sum(p.numel() for p in model.parameters())
                print(f"  total params: {n_params}")
            except Exception:
                pass
            # free
            del model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"Failed to load {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)

    print("\nDone. Successful loads (if any):", successes)

if __name__ == "__main__":
    main()