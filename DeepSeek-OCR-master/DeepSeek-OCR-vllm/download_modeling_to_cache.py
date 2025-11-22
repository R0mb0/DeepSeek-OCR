#!/usr/bin/env python3
"""
Scarica i file .py dal model repo su Hugging Face e copia i modeling_*.py
(ed eventuali file correlati) nella cache transformers_modules in una
sottocartella consistente con il nome del repo.

Uso:
  python download_modeling_to_cache.py --repo deepseek-ai/DeepSeek-OCR

Note:
- Se il repo è privato fai 'huggingface-cli login' prima o esporta HUGGINGFACE_HUB_TOKEN.
- Lo script cerca file che contengono "modeling" o che finiscono con .py e li copia
  nella cache sotto una directory consistente (deepseek_hyphen_ai/DeepSeek_hyphen_OCR/<stamp>).
"""
import os
import argparse
from huggingface_hub import list_repo_files, hf_hub_download
import shutil
from pathlib import Path

CACHE_BASE = Path.home() / ".cache" / "huggingface" / "modules" / "transformers_modules"

def safe_mkdir(p: Path):
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Repo model id, e.g. owner/modelname")
    parser.add_argument("--subdir-token", default=None, help="Optional token to pick specific subdir name")
    args = parser.parse_args()

    repo_id = args.repo
    print("Repo:", repo_id)

    print("Listing files on HF hub...")
    files = list_repo_files(repo_id)
    print(f"Found {len(files)} files in repo")

    # choose files to download: modeling_*.py plus other .py that the model may import
    to_get = [f for f in files if f.endswith(".py") and ("modeling" in f or "deepencoder" in f or "conversation" in f or "configuration" in f or f in ("modeling_deepseekocr.py","modeling_deepseekv2.py"))]
    # also include any module files often used (e.g. deepencoder.py)
    if not to_get:
        print("No candidate .py files found matching pattern. Aborting.")
        return

    print("Files to download (samples):")
    for f in to_get:
        print(" -", f)

    # Build target cache path similar to what transformers expects:
    # Use owner and repo but replace '-' with '_hyphen_' in owner/repo tokens to match earlier cache naming style if necessary.
    owner, repo = repo_id.split("/", 1)
    owner_token = owner.replace("-", "_hyphen_")
    repo_token = repo.replace("-", "_hyphen_")
    # Use a deterministic 'localstamp' folder name to avoid collisions
    target_dir = CACHE_BASE / owner_token / (repo_token + "/manual_download")
    target_dir = CACHE_BASE / owner_token / (repo_token + "/manual_download")
    # Some systems had DeepSeek_hyphen_OCR style; to be safe create both
    target_dir = CACHE_BASE / owner_token / repo_token / "manual_download"
    safe_mkdir(target_dir)

    print("Target cache directory:", target_dir)

    for f in to_get:
        try:
            print("Downloading:", f)
            local_path = hf_hub_download(repo_id=repo_id, filename=f)
            dest = target_dir / Path(f).name
            shutil.copy(local_path, dest)
            print("  -> copied to", dest)
        except Exception as e:
            print("  ERROR downloading", f, ":", e)

    print("\nDone. Files copied into:", target_dir)
    print("Now verify with:")
    print(f"  find {target_dir} -maxdepth 1 -type f -name 'modeling*.py' -print")
    print("Then re-run the discovery/test script (scripts_load_and_test_bnb_patched2.py).")

if __name__ == "__main__":
    main()