#!/usr/bin/env python3
"""
Importa i modeling_*.py dalla cache HF creando i package intermedi,
applica shim necessari e prova a caricare le classi candidate in CPU
(low memory). Uso:

  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:64"
  export CUDA_VISIBLE_DEVICES=""
  python scripts_try_load_cpu_no8bit_fixed.py --model deepseek-ai/DeepSeek-OCR

"""
import argparse
import os
import glob
import importlib
import importlib.util
import sys
import types
import traceback
import inspect

CACHE_BASE = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")

CANDIDATE_CLASS_NAMES = [
    "DeepseekOCRForCausalLM",
    "DeepseekOCRModel",
    "DeepseekV2ForCausalLM",
    "DeepseekV2Model",
    "DeepseekV2PreTrainedModel",
]

def ensure_shims():
    # Shim 1: LlamaFlashAttention2 -> LlamaAttention (se manca)
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

    # Shim 2: is_torch_fx_available in transformers.utils.import_utils
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


def find_all_modeling_files():
    pattern = os.path.join(CACHE_BASE, "**", "modeling*.py")
    files = sorted(glob.glob(pattern, recursive=True))
    return files

def make_module_name_from_cache_path(path):
    rel = os.path.relpath(path, CACHE_BASE)
    parts = rel.split(os.sep)
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    module_name = "transformers_modules." + ".".join(parts)
    return module_name

def ensure_package_modules(module_name, file_path):
    parts = module_name.split(".")
    for i in range(1, len(parts)):
        pkg_name = ".".join(parts[:i])
        if pkg_name in sys.modules:
            continue
        mod = types.ModuleType(pkg_name)
        # try to set a reasonable __path__ for the package
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
            if parent_mod is not None:
                setattr(parent_mod, parts[i-1], mod)

def import_module_with_package_name(path):
    module_name = make_module_name_from_cache_path(path)
    ensure_package_modules(module_name, path)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        # register BEFORE exec to allow relative imports inside module
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, module_name
    except Exception as e:
        print(f"Error importing {path} as {module_name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=6)
        sys.modules.pop(module_name, None)
        return None, module_name

def try_load_class(cls, model_id):
    print(f"\n>>> Trying {cls.__name__}.from_pretrained(...) (CPU, low mem)")
    kwargs = dict(device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True)
    try:
        if hasattr(cls, "from_pretrained"):
            model = cls.from_pretrained(model_id, **kwargs)
        else:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
            model = cls(cfg)
        try:
            n_params = sum(p.numel() for p in model.parameters())
            print(f" -> Loaded OK. total params: {n_params:,}")
        except Exception:
            print(" -> Loaded OK (no parameter introspection available).")
        try:
            sd = model.state_dict()
            keys = list(sd.keys())[:20]
            print(" -> state_dict keys sample (first 20):")
            for k in keys:
                v = sd[k]
                print("    ", k, getattr(v, "shape", type(v)))
        except Exception as e:
            print(" -> Could not read state_dict sample:", e)
        # cleanup
        try:
            del model
            import gc; gc.collect()
        except Exception:
            pass
        return True, None
    except Exception as e:
        tb = traceback.format_exc(limit=8)
        print(" -> Load FAILED:", type(e).__name__, e)
        print(tb)
        return False, tb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    model_id = args.model

    ensure_shims()

    files = find_all_modeling_files()
    if not files:
        print("No modeling_*.py found under", CACHE_BASE)
        print("Run AutoConfig.from_pretrained(..., trust_remote_code=True) first or copy files into cache.")
        return 2

    print(f"Found {len(files)} modeling files (sample):")
    for f in files:
        print(" -", f)

    imported = []
    for path in files:
        mod, modname = import_module_with_package_name(path)
        if mod is not None:
            imported.append((modname, mod, path))

    # Load tokenizer safely (only metadata)
    try:
        from transformers import AutoTokenizer
        print("\nLoading tokenizer (trust_remote_code=True)...")
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print("Tokenizer load attempted.")
    except Exception as e:
        print("Tokenizer attempt failed (non-fatal):", e)

    # For each module try candidate classes
    for modname, mod, path in imported:
        print(f"\n=== Inspect module {modname} (path: {path}) ===")
        for cls_name, cls_obj in inspect.getmembers(mod, inspect.isclass):
            if cls_name in CANDIDATE_CLASS_NAMES:
                print("Found candidate class:", cls_name)
                ok, tb = try_load_class(cls_obj, model_id)
                if ok:
                    print(f"SUCCESS loading {cls_name} from {path}")
                else:
                    print(f"FAILED loading {cls_name} (see traceback above)")

    print("\nDone.")

if __name__ == "__main__":
    main()