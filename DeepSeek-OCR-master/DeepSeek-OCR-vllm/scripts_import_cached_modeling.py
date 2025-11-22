#!/usr/bin/env python3
"""
Importa i modeling_*.py presenti nella cache ~/.cache/huggingface/modules/transformers_modules
creando i package intermedi in sys.modules e applicando shim per LlamaFlashAttention2 / is_torch_fx_available.

Uso:
  python scripts/import_cached_modeling.py --list-only   # mostra i file trovati
  python scripts/import_cached_modeling.py              # prova ad importarli e stampa le classi

Output:
  - lista dei modeling_*.py trovati
  - per ogni file: tentativo di import e lista delle classi definite, oppure traceback dell'errore
"""
import argparse
import os
import glob
import importlib.util
import importlib
import sys
import types
import traceback

CACHE_BASE = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")

def ensure_shims():
    # Shim 1: alias LlamaFlashAttention2 -> LlamaAttention se manca
    try:
        mod_llama = importlib.import_module("transformers.models.llama.modeling_llama")
        if not hasattr(mod_llama, "LlamaFlashAttention2"):
            if hasattr(mod_llama, "LlamaAttention"):
                setattr(mod_llama, "LlamaFlashAttention2", getattr(mod_llama, "LlamaAttention"))
                print("Shim: aliased LlamaFlashAttention2 -> LlamaAttention")
            else:
                # definisci stub che alzerà se usato (ma evita ImportError all'import)
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
            print("Shim: added is_torch_fx_available() to transformers.utils.import_utils")
    except Exception as e:
        print("Shim (import_utils) notice:", e)

def find_modeling_files():
    pattern = os.path.join(CACHE_BASE, "**", "modeling*.py")
    files = sorted(glob.glob(pattern, recursive=True))
    return files

def make_module_name_from_cache_path(path):
    """Costruisce un nome di modulo coerente con la struttura della cache."""
    rel = os.path.relpath(path, CACHE_BASE)
    parts = rel.split(os.sep)
    # rimuovi estensione .py dall'ultimo pezzo
    if parts[-1].endswith(".py"):
        parts[-1] = parts[-1][:-3]
    # prefisso 'transformers_modules' (come fa HF quando registra moduli dinamici)
    module_name = "transformers_modules." + ".".join(parts)
    # sostituisci caratteri non validi (già ok) e ritorna
    return module_name

def ensure_package_modules(module_name, file_path):
    """Crea voci package intermedie in sys.modules con __path__ correttamente impostato."""
    parts = module_name.split(".")
    for i in range(1, len(parts)):
        pkg_name = ".".join(parts[:i])
        if pkg_name in sys.modules:
            continue
        mod = types.ModuleType(pkg_name)
        # tenta determinare una directory per __path__
        # mappa transformers_modules.<owner>/<repo>/... -> candidate dir
        if pkg_name.startswith("transformers_modules"):
            # calcola relativo rispetto a CACHE_BASE
            rel_parts = parts[1:i]
            cand_dir = os.path.join(CACHE_BASE, *rel_parts) if rel_parts else CACHE_BASE
            if os.path.isdir(cand_dir):
                mod.__path__ = [cand_dir]
            else:
                mod.__path__ = []
        else:
            mod.__path__ = []
        sys.modules[pkg_name] = mod
        # collega al parent come attributo (opzionale)
        if i > 1:
            parent = ".".join(parts[:i-1])
            parent_mod = sys.modules.get(parent)
            if parent_mod is not None:
                setattr(parent_mod, parts[i-1], mod)

def import_module_with_name(path):
    module_name = make_module_name_from_cache_path(path)
    ensure_package_modules(module_name, path)
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            print("Could not create spec for", path)
            return None, module_name
        module = importlib.util.module_from_spec(spec)
        # inserisci in sys.modules *prima* di eseguire per supportare import relativi
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module, module_name
    except Exception as e:
        print(f"Error importing {path} as {module_name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=5)
        # pulisci eventuale entry parziale
        sys.modules.pop(module_name, None)
        return None, module_name

def main(list_only=False):
    print("CACHE_BASE:", CACHE_BASE)
    ensure_shims()
    files = find_modeling_files()
    if not files:
        print("No modeling_*.py files found under", CACHE_BASE)
        return 1

    print(f"Found {len(files)} modeling files:")
    for f in files:
        print(" -", f)

    if list_only:
        return 0

    for f in files:
        print("\n--- Importing:", f)
        mod, modname = import_module_with_name(f)
        if mod is None:
            print("Import failed for", f)
            continue
        # elenca le classi definite nel modulo
        classes = [n for n,o in mod.__dict__.items() if isinstance(o, type) and getattr(o, "__module__", "").startswith(mod.__name__)]
        print(f"Imported as {modname}, classes found: {classes}")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-only", action="store_true", help="Mostra solo i modeling_*.py trovati")
    args = parser.parse_args()
    sys.exit(main(list_only=args.list_only))