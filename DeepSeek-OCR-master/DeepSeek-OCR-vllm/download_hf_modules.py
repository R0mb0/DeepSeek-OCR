#!/usr/bin/env python3
"""
Forza il download dei moduli dinamici di un repo HF e mostra i modeling_*.py
Uso:
  python download_hf_modules.py --model deepseek-ai/DeepSeek-OCR
"""
import argparse
import importlib
import importlib.util
import os
import sys
import traceback
import glob

CACHE_BASE = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")

def ensure_shims():
    # shim LlamaFlashAttention2 -> LlamaAttention se manca
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
        print("Shim (llama) warning:", e)

    # shim is_torch_fx_available se manca
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


def force_download_model_code(model_id):
    # forza il download della config / codice dinamico HF
    try:
        from transformers import AutoConfig
    except Exception as e:
        print("Errore import transformers:", e)
        return False
    try:
        print("Chiamando AutoConfig.from_pretrained(..., trust_remote_code=True) per", model_id)
        AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print("AutoConfig OK (download provato).")
        return True
    except Exception as e:
        print("AutoConfig.from_pretrained() ha lanciato un'eccezione:")
        traceback.print_exc(limit=5)
        return False

def list_modeling_files(owner, repo):
    pattern = os.path.join(CACHE_BASE, "**", "modeling*.py")
    files = [p for p in glob.glob(pattern, recursive=True) if owner.lower() in p.lower() or repo.lower() in p.lower()]
    return files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id (es. deepseek-ai/DeepSeek-OCR)")
    args = parser.parse_args()
    model_id = args.model
    owner, repo = model_id.split("/", 1)

    # 1) applica shim prima di importare moduli dinamici
    ensure_shims()

    # 2) forza download del codice dinamico
    ok = force_download_model_code(model_id)
    if not ok:
        print("Attenzione: AutoConfig ha fallito. Se vedi ImportError relativi a simboli mancanti, incollami il traceback.")
        print("Comunque tento di elencare eventuali file già scaricati in cache...")
    # 3) elenca i modeling_*.py nella cache (ricorsivo)
    files = list_modeling_files(owner, repo)
    if not files:
        print("Nessun modeling_*.py trovato nella cache:", CACHE_BASE)
    else:
        print(f"Trovati {len(files)} modeling_*.py nella cache:")
        for f in files:
            print(" -", f)
    print("\nSe sono stati scaricati i file, puoi ora eseguire lo script di discovery / caricamento.\n")

if __name__ == "__main__":
    main()