#!/usr/bin/env python3
"""
Patched: load dynamic modeling files from HF cache as proper package modules,
apply shims, instantiate candidate model class on CPU (low mem) and save state_dict.

Usage:
  python scripts_save_model_states_patched.py --model deepseek-ai/DeepSeek-OCR --out-dir ./saved_states
"""
import argparse
import glob
import importlib.util
import importlib
import inspect
import os
import sys
import traceback
import types
from pathlib import Path

CACHE_BASE = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")

# ---------------- shims ----------------
def ensure_shims():
    try:
        mod_llama = importlib.import_module("transformers.models.llama.modeling_llama")
        if not hasattr(mod_llama, "LlamaFlashAttention2"):
            if hasattr(mod_llama, "LlamaAttention"):
                setattr(mod_llama, "LlamaFlashAttention2", getattr(mod_llama, "LlamaAttention"))
                print("Shim: aliased LlamaFlashAttention2 -> LlamaAttention")
            else:
                class _StubFlashAttention2:
                    def __init__(self, *a, **k):
                        raise RuntimeError("Stub LlamaFlashAttention2 used.")
                setattr(mod_llama, "LlamaFlashAttention2", _StubFlashAttention2)
                print("Shim: created stub LlamaFlashAttention2")
    except Exception as e:
        print("Shim (llama) notice:", e)

    try:
        mod_import_utils = importlib.import_module("transformers.utils.import_utils")
        if not hasattr(mod_import_utils, "is_torch_fx_available"):
            def is_torch_fx_available() -> bool:
                try:
                    import torch
                    return hasattr(torch, "fx")
                except Exception:
                    return False
            setattr(mod_import_utils, "is_torch_fx_available", is_torch_fx_available)
            print("Shim: added is_torch_fx_available()")
    except Exception as e:
        print("Shim (import_utils) notice:", e)

# ---------------- module helpers ----------------
def find_modeling_files():
    pattern = os.path.join(CACHE_BASE, "**", "modeling*.py")
    files = sorted(glob.glob(pattern, recursive=True))
    return files

def make_module_name_from_cache_path(path: str) -> str:
    rel = os.path.relpath(path, CACHE_BASE)
    parts = rel.split(os.sep)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    module_name = "transformers_modules." + ".".join(parts)
    return module_name

def ensure_package_modules(module_name: str, file_path: str):
    parts = module_name.split(".")
    for i in range(1, len(parts)):
        pkg_name = ".".join(parts[:i])
        if pkg_name in sys.modules:
            continue
        mod = types.ModuleType(pkg_name)
        # try to set __path__ to candidate dir
        if pkg_name.startswith("transformers_modules"):
            rel_parts = parts[1:i]
            cand_dir = os.path.join(CACHE_BASE, *rel_parts) if rel_parts else CACHE_BASE
            if os.path.isdir(cand_dir):
                mod.__path__ = [cand_dir]
            else:
                mod.__path__ = []
        else:
            mod.__path__ = []
        sys.modules[pkg_name] = mod
        if i > 1:
            parent = ".".join(parts[:i-1])
            parent_mod = sys.modules.get(parent)
            if parent_mod:
                setattr(parent_mod, parts[i-1], mod)

def import_module_with_package_name(path: str):
    module_name = make_module_name_from_cache_path(path)
    ensure_package_modules(module_name, path)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            print("Could not create spec for", path)
            return None, module_name
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, module_name
    except Exception as e:
        print(f"Error importing {path} as {module_name}: {e}")
        traceback.print_exc(limit=5)
        sys.modules.pop(module_name, None)
        return None, module_name

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id")
    parser.add_argument("--out-dir", default="./saved_states")
    args = parser.parse_args()
    model_id = args.model
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_shims()

    print("Searching modeling_*.py under cache...")
    files = find_modeling_files()
    if not files:
        print("No modeling_*.py found under", CACHE_BASE)
        print("Run AutoConfig.from_pretrained(..., trust_remote_code=True) or copy files into cache.")
        return 2

    print("Found modeling files:")
    for f in files:
        print(" -", f)

    imported = []
    for p in files:
        mod, name = import_module_with_package_name(p)
        if mod is not None:
            imported.append((mod, name, p))

    if not imported:
        print("No modules imported successfully.")
        return 3

    # Try to load tokenizer (metadata)
    try:
        from transformers import AutoTokenizer
        print("Loading tokenizer (trust_remote_code=True)...")
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("Tokenizer load attempted.")
    except Exception as e:
        print("Tokenizer attempt failed (non-fatal):", e)

    # Find candidate classes in imported modules
    candidates = []
    for mod, modname, path in imported:
        for n, o in inspect.getmembers(mod, inspect.isclass):
            if getattr(o, "__module__", "").startswith(mod.__name__):
                if n in ("DeepseekOCRForCausalLM", "DeepseekV2ForCausalLM", "DeepseekOCRModel", "DeepseekV2Model"):
                    candidates.append((n, o, path, modname))

    if not candidates:
        print("No candidate classes found in modeling files.")
        return 4

    import torch
    for name, cls, path, modname in candidates:
        print(f"\n--- Trying class {name} from {path}")
        if not hasattr(cls, "from_pretrained"):
            print("Class has no from_pretrained; skipping.")
            continue
        try:
            print("Calling from_pretrained(device_map='cpu', low_cpu_mem_usage=True, trust_remote_code=True)")
            model = cls.from_pretrained(model_id, device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True)
            try:
                n_params = sum(p.numel() for p in model.parameters())
                print(f"Loaded OK. total params: {n_params:,}")
            except Exception:
                print("Loaded OK (could not inspect parameters).")
            # save full state_dict
            full_path = out_dir / f"{name}_full_state.pt"
            try:
                torch.save(model.state_dict(), full_path)
                print("Saved full state_dict to:", full_path)
            except Exception as e:
                print("Error saving full state_dict:", e)
            # try to get language submodule
            lang = None
            if hasattr(model, "get_language_model"):
                try:
                    lang = model.get_language_model()
                except Exception:
                    lang = None
            elif hasattr(model, "language_model"):
                lang = getattr(model, "language_model")
            if lang is not None:
                lang_path = out_dir / f"{name}_language_state.pt"
                try:
                    torch.save(lang.state_dict(), lang_path)
                    print("Saved language state_dict to:", lang_path)
                except Exception as e:
                    print("Error saving language state_dict:", e)
            else:
                print("Language submodule not found; skipping language-only save.")
            # cleanup
            del model
            import gc; gc.collect()
        except Exception as e:
            print(f"Failed to load {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=6)

    print("Done.")

if __name__ == "__main__":
    main()