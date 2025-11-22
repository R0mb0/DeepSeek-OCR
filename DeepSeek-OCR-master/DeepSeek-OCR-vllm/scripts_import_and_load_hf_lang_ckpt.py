#!/usr/bin/env python3
"""
Import dynamic modeling files saved under ~/.cache/huggingface/modules/transformers_modules/hf_lang_ckpt
as package modules (so relative imports work), apply shims, and try to load the language-only
checkpoint in ./hf_lang_ckpt using the custom class.

Usage:
  python scripts_import_and_load_hf_lang_ckpt.py
"""
import importlib
import importlib.util
import sys
import types
import traceback
from pathlib import Path

# ---------------------
# Settings
# ---------------------
CACHE_BASE = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"
HF_PKG_NAME = "hf_lang_ckpt"
HF_PKG_DIR = CACHE_BASE / HF_PKG_NAME
MODULE_FILES = ["modeling_deepseekv2.py", "modeling_deepseekocr.py"]
CANDIDATE_CLASSES = ("DeepseekOCRForCausalLM", "DeepseekV2ForCausalLM", "DeepseekOCRModel", "DeepseekV2Model")

# ---------------------
# Shims (must exist BEFORE importing dynamic code)
# ---------------------
def apply_shims():
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
                print("Shim: created stub LlamaFlashAttention2 (stub)")
    except Exception as e:
        print("Shim (llama) warning:", e)
        traceback.print_exc(limit=2)

    try:
        import transformers.utils.import_utils as import_utils_mod
        if not hasattr(import_utils_mod, "is_torch_fx_available"):
            def is_torch_fx_available() -> bool:
                try:
                    import torch
                    return hasattr(torch, "fx")
                except Exception:
                    return False
            setattr(import_utils_mod, "is_torch_fx_available", is_torch_fx_available)
            print("Shim: added is_torch_fx_available() to transformers.utils.import_utils")
        else:
            print("is_torch_fx_available already present")
    except Exception as e:
        print("Shim (import_utils) warning:", e)
        traceback.print_exc(limit=2)

# ---------------------
# Ensure package modules
# ---------------------
def ensure_package_modules_for_hf_pkg():
    """
    Create sys.modules entries for:
      'transformers_modules' and 'transformers_modules.hf_lang_ckpt'
    and set __path__ so relative imports inside the downloaded modeling files work.
    """
    root_pkg = "transformers_modules"
    pkg_name = f"{root_pkg}.{HF_PKG_NAME}"
    # create root package if missing
    if root_pkg not in sys.modules:
        mod = types.ModuleType(root_pkg)
        # __path__ should include CACHE_BASE so subpackages can be found
        mod.__path__ = [str(CACHE_BASE)]
        sys.modules[root_pkg] = mod
        print(f"Created package module: {root_pkg} -> path {CACHE_BASE}")
    # create hf_lang_ckpt package
    if pkg_name not in sys.modules:
        mod = types.ModuleType(pkg_name)
        # set __path__ to the actual directory where HF saved the modeling files
        mod.__path__ = [str(HF_PKG_DIR)]
        sys.modules[pkg_name] = mod
        # attach to parent module as attribute for normal import semantics
        parent = sys.modules[root_pkg]
        setattr(parent, HF_PKG_NAME, mod)
        print(f"Created package module: {pkg_name} -> path {HF_PKG_DIR}")

# ---------------------
# Import modeling modules (package-qualified)
# ---------------------
def import_modeling_modules():
    imported = {}
    for fname in MODULE_FILES:
        fpath = HF_PKG_DIR / fname
        if not fpath.exists():
            print(f"File not found in cache: {fpath}  (skipping)")
            continue
        module_name = f"transformers_modules.{HF_PKG_NAME}.{fpath.stem}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, str(fpath))
            module = importlib.util.module_from_spec(spec)
            # register BEFORE exec to make relative imports work
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            imported[module_name] = module
            print(f"Imported dynamic module: {module_name}")
        except Exception as e:
            print(f"Import failed for {fpath}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=6)
            # cleanup partial module
            if module_name in sys.modules:
                del sys.modules[module_name]
    return imported

# ---------------------
# Find candidate classes and try load
# ---------------------
def try_load_candidates(imported_modules):
    from transformers import AutoTokenizer  # ensure tokenizers available
    # attempt to load tokenizer metadata (non-fatal)
    try:
        AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-OCR", trust_remote_code=True)
        print("Tokenizer metadata load attempted (ok).")
    except Exception as e:
        print("Tokenizer metadata load warning:", e)

    for mod_name, mod in imported_modules.items():
        print(f"\n=== Inspect module {mod_name} ===")
        for cls_name in CANDIDATE_CLASSES:
            if hasattr(mod, cls_name):
                cls = getattr(mod, cls_name)
                print(f"Found candidate class: {cls_name} in {mod_name}")
                if not hasattr(cls, "from_pretrained"):
                    print(f"Class {cls_name} has no from_pretrained; skipping.")
                    continue
                try:
                    print(f"Attempting {cls_name}.from_pretrained('./hf_lang_ckpt', device_map='cpu', low_cpu_mem_usage=True, trust_remote_code=True)")
                    model = cls.from_pretrained("./hf_lang_ckpt", device_map="cpu", low_cpu_mem_usage=True, trust_remote_code=True)
                    try:
                        n = sum(p.numel() for p in model.parameters())
                        print(f"Loaded OK. total params: {n:,}")
                    except Exception:
                        print("Loaded OK (could not count params).")
                    try:
                        sd = model.state_dict()
                        print("State dict sample keys:")
                        for k in list(sd.keys())[:20]:
                            v = sd[k]
                            print("  ", k, getattr(v, "shape", type(v)))
                    except Exception:
                        pass
                    # cleanup
                    del model
                    import gc; gc.collect()
                except Exception as e:
                    print(f"Failed to load {cls_name}: {type(e).__name__}: {e}")
                    traceback.print_exc(limit=8)

def main():
    if not HF_PKG_DIR.exists():
        print(f"HF package dir not found: {HF_PKG_DIR}")
        print("Ensure you have ./hf_lang_ckpt created from previous remap step and that cache files exist.")
        return 2

    apply_shims()
    ensure_package_modules_for_hf_pkg()
    imported = import_modeling_modules()
    if not imported:
        print("No modules imported. Exiting.")
        return 3
    try_load_candidates(imported)
    return 0

if __name__ == "__main__":
    sys.exit(main())