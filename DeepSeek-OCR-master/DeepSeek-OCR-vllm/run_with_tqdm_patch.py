#!/usr/bin/env python3
"""
Wrapper che applica una monkeypatch a tqdm.asyncio.tqdm_asyncio
poi esegue uno script Python passato come primo argomento (path).
Uso:
  python run_with_tqdm_patch.py path/to/your_script.py [arg1 arg2 ...]
Esempio:
  python run_with_tqdm_patch.py test_vllm_gpt2.py
  python run_with_tqdm_patch.py run_one_debug_skip_tokenizer.py /path/to/image.png
"""
import importlib, sys, runpy

# Applica monkeypatch PRIMA di caricare qualunque altro modulo
try:
    tqa = importlib.import_module("tqdm.asyncio")
    from tqdm import tqdm
    # Sostituisci la funzione/classe con tqdm standard per evitare mismatch di signature.
    tqa.tqdm_asyncio = tqdm
    # alias per sicurezza
    setattr(tqa, "tqdm_asyncio", tqdm)
    print("Applied monkeypatch: tqdm.asyncio.tqdm_asyncio -> tqdm.tqdm")
except Exception as e:
    # se non esiste il submodule, ignora ma mostra info
    print("Could not import tqdm.asyncio or apply patch:", e)

if len(sys.argv) < 2:
    print("Usage: python run_with_tqdm_patch.py path/to/script.py [args...]")
    sys.exit(2)

script_path = sys.argv[1]
script_args = sys.argv[2:]
# imposta sys.argv per lo script che eseguiamo
sys.argv = [script_path] + script_args

# Esegui lo script target come __main__
runpy.run_path(script_path, run_name="__main__")