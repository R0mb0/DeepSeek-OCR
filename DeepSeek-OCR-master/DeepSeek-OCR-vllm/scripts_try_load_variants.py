#!/usr/bin/env python3
"""
Try multiple from_pretrained variants on candidate classes discovered in the
downloaded dynamic modeling files for deepseek-ai/DeepSeek-OCR.

Usage:
  python scripts/try_load_variants.py --model deepseek-ai/DeepSeek-OCR
"""
import argparse, glob, importlib.util, os, sys, traceback, inspect
from transformers import AutoConfig, AutoTokenizer

CACHE_BASE = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")

def find_modeling_files(owner, repo):
    pattern = os.path.join(CACHE_BASE, "**", "modeling*.py")
    files = [p for p in glob.glob(pattern, recursive=True) if owner.lower() in p.lower() or repo.lower() in p.lower()]
    return files

def import_mod_from_path(path, name_hint):
    spec = importlib.util.spec_from_file_location(name_hint, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def try_calls(cls, model_id):
    results = []
    calls = [
        {"kwargs": {"trust_remote_code": True}},
        {"kwargs": {"trust_remote_code": True, "device_map": "cpu"}},
        {"kwargs": {"trust_remote_code": True, "device_map": "cpu", "low_cpu_mem_usage": True}},
        {"kwargs": {"trust_remote_code": True, "torch_dtype": "auto", "device_map": "cpu", "low_cpu_mem_usage": True}},
    ]
    for c in calls:
        try:
            # Some wrappers expect particular kw names; we try to call from_pretrained if present
            if hasattr(cls, "from_pretrained"):
                print(f"  Trying {cls.__name__}.from_pretrained with kwargs: {c['kwargs']}")
                # map torch_dtype string to actual dtype if needed
                kw = dict(c["kwargs"])
                if "torch_dtype" in kw and kw["torch_dtype"] == "auto":
                    # pass nothing or try float16
                    kw.pop("torch_dtype", None)
                    # try float16 as separate attempt below
                try:
                    model = cls.from_pretrained(model_id, **kw)
                except TypeError as e:
                    # try float16 variant
                    try:
                        import torch
                        model = cls.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cpu", trust_remote_code=True, low_cpu_mem_usage=True)
                    except Exception as e2:
                        raise e2
                # if we get here, loaded something
                dev = None
                try:
                    import torch
                    dev = next(model.parameters()).device
                except Exception:
                    dev = "no-params"
                results.append(("SUCCESS", cls.__name__, str(dev)))
                # cleanup
                try:
                    del model
                    import gc
                    gc.collect()
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            else:
                results.append(("NO_FROM_PRETRAINED", cls.__name__, ""))
        except Exception as e:
            tb = traceback.format_exc(limit=2)
            results.append(("ERROR", cls.__name__, str(e).splitlines()[-1], tb))
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    owner, repo = args.model.split("/",1)
    print("Finding modeling files...")
    files = find_modeling_files(owner, repo)
    print("Found:", files)
    all_classes = []
    for f in files:
        try:
            mod = import_mod_from_path(f, f"mod_{os.path.basename(f)}")
        except Exception as e:
            print(f"Failed to import module {f}: {e}")
            traceback.print_exc(limit=2)
            continue
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            # only include classes defined in this module
            if getattr(obj, "__module__", "").startswith(mod.__name__):
                all_classes.append((name, obj, f))
    print(f"Discovered {len(all_classes)} classes")
    # try to load each class
    for name, cls, path in all_classes:
        print("\n=== CLASS:", name, "from", path)
        res = try_calls(cls, args.model)
        for r in res:
            if r[0] == "SUCCESS":
                print("  => SUCCESS:", r)
            elif r[0] == "NO_FROM_PRETRAINED":
                print("  => SKIP: no from_pretrained")
            else:
                print("  => ERROR:", r[1], r[2])
                print(r[3])
    print("\nDone.")

if __name__ == "__main__":
    main()
