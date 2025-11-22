# sitecustomize.py
# Questo file viene caricato automaticamente all'avvio dell'interprete Python (se presente in sys.path).
# Applica una monkeypatch che sostituisce tqdm.asyncio.tqdm_asyncio con tqdm.tqdm
# così anche i processi figli (EngineCore) non incontrano il TypeError sulla firma.

import importlib, sys

def _patch_tqdm_asyncio():
    try:
        tqa = importlib.import_module("tqdm.asyncio")
        from tqdm import tqdm as _tqdm
        # Sostituisci la funzione/classe problematica con tqdm standard
        tqa.tqdm_asyncio = _tqdm
        setattr(tqa, "tqdm_asyncio", _tqdm)
        # Messaggio diagnostico che comparirà nei log dei processi figli
        sys.stderr.write("sitecustomize: patched tqdm.asyncio.tqdm_asyncio -> tqdm.tqdm\n")
    except Exception:
        # non vogliamo fallire l'avvio dell'interprete se il patch fallisce
        pass

_patch_tqdm_asyncio()