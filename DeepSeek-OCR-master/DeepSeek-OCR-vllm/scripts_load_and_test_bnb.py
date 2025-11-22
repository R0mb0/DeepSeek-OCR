#!/usr/bin/env python3
"""
Patched discovery + test script (v2)

- Ensures shims for LlamaFlashAttention2 and is_torch_fx_available.
- Finds modeling_*.py under ~/.cache/huggingface/modules/transformers_modules recursively.
- Imports each modeling_*.py using a module name that mirrors its cache path,
  creating intermediate package entries in sys.modules so relative imports work.
- Lists classes and tries to load candidates via .from_pretrained(..., load_in_8bit=True, device_map="auto", trust_remote_code=True).

Usage:
  python scripts_load_and_test_bnb_patched2.py --model deepseek-ai/DeepSeek-OCR
"""
import argparse
import importlib
import importlib.util
import inspect
import os
import sys
import glob
import traceback
import types

CACHE_BASE = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")

# -------------------------
# Small shims to mitigate missing symbols in some transformers versions
# -------------------------
def ensure_shims():
    try:
        mod_llama = importlib.import_module("transformers.models.llama.modeling_llama")
        if not hasattr(mod_llama, "LlamaFlashAttention2"):
            if hasattr(mod_llama, "LlamaAttention"):
                setattr(mod_llama, "LlamaFlashAttention2", getattr(mod_llama, "LlamaAttention"))
                print("Shim: aliased LlamaFlashAttention2 -> LlamaAttention")
            else:
                class _StubFlashAttention2:
                    def __init__(self, *args, **kwargs):
                        raise RuntimeError("Stub LlamaFlashAttention2 used.")
                setattr(mod_llama, "LlamaFlashAttention2", _StubFlashAttention2)
                print("Shim: created stub LlamaFlashAttention2")
    except Exception as e:
        print("Shim (llama) warning:", e)

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
        print("Shim (import_utils) warning:", e)


# -------------------------
# Helpers to import a file as a package-style module
# -------------------------
def make_module_name_from_cache_path(path):
    """
    Given a full path like:
      ~/.cache/huggingface/modules/transformers_modules/deepseek_hyphen_ai/DeepSeek_hyphen_OCR/<hash>/modeling_deepseekv2.py
    produce a module name:
      transformers_modules.deepseek_hyphen_ai.DeepSeek_hyphen_OCR.<hash>.modeling_deepseekv2
    """
    rel = os.path.relpath(path, CACHE_BASE)
    parts = rel.split(os.sep)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]  # remove .py
    module_name = "transformers_modules." + ".".join(parts)
    return module_name

def ensure_package_modules(module_name, file_path):
    """
    Ensure intermediate package modules exist in sys.modules with correct __path__.
    For example, for module_name = transformers_modules.a.b.c, ensure:
      sys.modules['transformers_modules'] exists and __path__ contains the cache base dir
      sys.modules['transformers_modules.a'] exists and __path__ contains its directory, etc.
    """
    parts = module_name.split(".")
    for i in range(1, len(parts)):
        pkg_name = ".".join(parts[:i])
        if pkg_name in sys.modules:
            continue
        mod = types.ModuleType(pkg_name)
        # set __path__ for packages: determine directory on disk for this package if possible
        # compute candidate dir by mapping package to cache path
        candidate_dir = os.path.join(CACHE_BASE, *parts[1:i])
        if os.path.isdir(candidate_dir):
            mod.__path__ = [candidate_dir]
        else:
            # fallback: use directory of file_path for deepest package
            if i == len(parts) - 1:
                mod.__path__ = [os.path.dirname(file_path)]
            else:
                mod.__path__ = []
        sys.modules[pkg_name] = mod
        # also attach to parent module attribute for attribute-style access
        if i > 1:
            parent = ".".join(parts[:i-1])
            parent_mod = sys.modules.get(parent)
            if parent_mod:
                setattr(parent_mod, parts[i-1], mod)

def import_module_with_name(path):
    module_name = make_module_name_from_cache_path(path)
    ensure_package_modules(module_name, path)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            print("Could not create spec for", path)
            return None
        module = importlib.util.module_from_spec(spec)
        # put module in sys.modules under the chosen name BEFORE exec to allow intra-module imports to work
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, module_name
    except Exception as e:
        print(f"Error importing {path} as {module_name}: {e}")
        traceback.print_exc(limit=3)
        # cleanup partial module if present
        sys.modules.pop(module_name, None)
        return None, module_name

# -------------------------
# Main discovery / test
# -------------------------
def find_modeling_files_recursive(owner, repo):
    pattern = os.path.join(CACHE_BASE, "**", "modeling*.py")
    all_files = glob.glob(pattern, recursive=True)
    filtered = []
    owner_l = owner.lower()
    repo_l = repo.lower()
    for f in all_files:
        low = f.lower()
        if owner_l in low or repo_l in low:
            filtered.append(f)
    return filtered or all_files

def list_classes_in_module(mod):
    classes = []
    for n, o in inspect.getmembers(mod, inspect.isclass):
        # only include classes defined in that module
        if getattr(o, "__module__", "").startswith(mod.__name__):
            classes.append((n, o))
    return classes

def try_load_candidates(model_id, candidates):
    import torch
    from transformers import AutoTokenizer
    successes = []
    print("Loading tokenizer (trust_remote_code=True)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print("Tokenizer load failed (warning):", e)
        tokenizer = None
    for name, obj, path in candidates:
        print("\n--- Trying:", name, "from", path)
        try:
            if hasattr(obj, "from_pretrained"):
                print("Attempting .from_pretrained(... load_in_8bit=True, device_map='auto', trust_remote_code=True)")
                model = obj.from_pretrained(model_id, load_in_8bit=True, device_map="auto", trust_remote_code=True)
                dev = next(model.parameters()).device
                dtype = next(model.parameters()).dtype
                print(f"SUCCESS: loaded {name} -> device={dev}, dtype={dtype}")
                successes.append((name, path, dev, dtype))
                # cleanup
                try:
                    del model
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                print("Class has no from_pretrained; skipping.")
        except Exception as e:
            print(f"Failed to load {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=3)
    return successes

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id (e.g. deepseek-ai/DeepSeek-OCR)")
    args = parser.parse_args()
    model_id = args.model
    owner, repo = model_id.split("/", 1)

    ensure_shims()

    print("Step 1: try AutoConfig.from_pretrained (may fail but will download code)...")
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print("AutoConfig loaded. model_type:", getattr(cfg, "model_type", None))
    except Exception as e:
        print("AutoConfig attempt error (continuing):", e)

    print("\nStep 2: search modeling_*.py under cache...")
    files = find_modeling_files_recursive(owner, repo)
    if not files:
        print("No modeling files found under cache:", CACHE_BASE)
        return
    print(f"Found {len(files)} files:")
    for f in files:
        print(" -", f)

    candidates = []
    for f in files:
        mod, modname = import_module_with_name(f)
        if mod is None:
            print("Import failed for", f)
            continue
        classes = list_classes_in_module(mod)
        print(f"Module {modname} classes:", [c[0] for c in classes])
        for n, o in classes:
            candidates.append((n, o, f))

    if not candidates:
        print("No candidate classes discovered.")
        return

    print("\nStep 3: try loading candidate classes in 8-bit (this may allocate GPU memory)")
    succ = try_load_candidates(model_id, candidates)
    if succ:
        print("\nSuccessful loads:")
        for s in succ:
            print(" -", s)
    else:
        print("\nNo candidate class loaded successfully. See above errors for native lib / API issues.")

if __name__ == "__main__":
    import argparse
    main()