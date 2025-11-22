#!/usr/bin/env python3
"""
Verifica accesso al modello su Hugging Face e prova a scaricare / importare il codice dinamico.

Cosa fa:
- verifica che il repo esista e che tu abbia accesso (huggingface_hub.HfApi().repo_info)
- elenca i file del repo (list_repo_files)
- se trova modeling_*.py prova a scaricarne uno (hf_hub_download)
- applica due "shim" che risolvono errori noti durante l'import dinamico (LlamaFlashAttention2 e is_torch_fx_available)
- prova a chiamare AutoConfig.from_pretrained(..., trust_remote_code=True) e AutoTokenizer.from_pretrained(...)
  (non carica i pesi del modello, quindi non usa GPU)

Uso:
  # Assicurati di avere huggingface_hub e transformers installati nell'ambiente
  python test_model_access.py --model deepseek-ai/DeepSeek-OCR

Se il repo è privato:
  export HUGGINGFACE_HUB_TOKEN="hf_..."
  python test_model_access.py --model user/private-model
"""
import argparse
import importlib
import importlib.util
import os
import sys
import tempfile
import traceback

from huggingface_hub import HfApi, list_repo_files, hf_hub_download, hf_hub_url
from transformers import AutoConfig, AutoTokenizer

def apply_shims():
    # Shim 1: alias LlamaFlashAttention2 -> LlamaAttention if missing
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
        # transformers may not expose the module yet; ignore
        print("Shim (llama) notice:", e)

    # Shim 2: add is_torch_fx_available to transformers.utils.import_utils if missing
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


def check_repo_access(repo_id):
    api = HfApi()
    try:
        info = api.repo_info(repo_id)
        print(f"Repo found: {repo_id} (private={info.private}, sha={getattr(info, 'sha', 'N/A')})")
        return True
    except Exception as e:
        print(f"ERROR: cannot access repo '{repo_id}': {e}")
        print("Hint: if the repo is private, run `huggingface-cli login` or set HUGGINGFACE_HUB_TOKEN env var.")
        return False


def list_and_download_modeling(repo_id):
    print("Listing files in repo (this may take a moment)...")
    try:
        files = list_repo_files(repo_id)
    except Exception as e:
        print("Failed to list repo files:", e)
        return []
    print(f"Total files in repo: {len(files)}")
    modeling_files = [f for f in files if f.lower().startswith("modeling") or "modeling" in os.path.basename(f).lower() or "deepseek" in os.path.basename(f).lower()]
    print("modeling / related .py files found (sample):")
    for f in modeling_files:
        print("  -", f)
    downloaded = []
    for f in modeling_files:
        try:
            print(f"Attempting to download '{f}'...")
            local = hf_hub_download(repo_id=repo_id, filename=f)
            print("  downloaded to:", local)
            downloaded.append(local)
        except Exception as e:
            print(f"  could not download '{f}': {e}")
    return downloaded


def try_autoconfig_and_tokenizer(repo_id):
    # Apply shims first
    apply_shims()

    print("Trying AutoConfig.from_pretrained(..., trust_remote_code=True) ...")
    try:
        cfg = AutoConfig.from_pretrained(repo_id, trust_remote_code=True)
        print("  AutoConfig loaded. model_type:", getattr(cfg, "model_type", None))
    except Exception as e:
        print("  AutoConfig FAILED:", e)
        traceback.print_exc(limit=2)

    print("Trying AutoTokenizer.from_pretrained(..., trust_remote_code=True) ...")
    try:
        tok = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
        print("  AutoTokenizer loaded. Example tokens: bos_token:", getattr(tok, "bos_token", None))
    except Exception as e:
        print("  AutoTokenizer FAILED:", e)
        traceback.print_exc(limit=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id, e.g. owner/ModelName")
    args = parser.parse_args()
    repo_id = args.model

    print("Checking access to:", repo_id)
    ok = check_repo_access(repo_id)
    if not ok:
        sys.exit(2)

    print("\nListing and attempting to download modeling files (if any)...")
    downloaded = list_and_download_modeling(repo_id)
    if not downloaded:
        print("No modeling files downloaded (repo may not contain modelling code, or files are in subfolders).")
    else:
        print(f"Downloaded {len(downloaded)} file(s). They are stored in the HF cache (printed above).")

    print("\nNow attempting to load config/tokenizer with trust_remote_code=True (will import model code but not weights).")
    try_autoconfig_and_tokenizer(repo_id)

    print("\nTEST COMPLETE. If AutoConfig/AutoTokenizer succeeded, you have access and the model's code can be imported (trust_remote_code=True).")
    print("If any step failed, paste the error here and I will help interpret it.")


if __name__ == "__main__":
    main()